"""Rank-element Top-k selection and tail-score intervention helpers.

For every prompt, layer, and head, the 120 strict-lower ``S_asym`` values
from tail16 are sorted by signed value in ascending order.  The selection
unit is the resulting one-based ``(layer, head, rank)`` identity.  Runtime
intervention writes each selected rank's replacement back to the strict-
lower position occupied by that rank before intervention, then mirrors the
delta into the upper triangle with the opposite sign.
"""

from __future__ import annotations

import statistics
import types
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from transformers.models.llama import modeling_llama

from utils.asym_score_intervention import SUPPORTED_TRANSFORMERS_VERSION
from utils.artifact_io import read_json, read_jsonl, sha256_file
from utils.gcg_position_element_topk_intervention import validate_top_k_values
from utils.gcg_wasserstein_reference import (
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    TAIL_TOKENS,
    strict_lower_indices,
    validate_score_tensor,
    verify_declared_file_hashes,
)
from utils.reference_asym_score_intervention import (
    ReferenceInterventionRuntimeError,
    _patched_llama_attention_forward,
    validate_alpha,
    validate_intervention_timing,
)


TOP_K = 500
TOP_K_VALUES = (TOP_K,)
RANK_SORT = "signed_value_ascending"
RANK_TIE_BREAK = "stable strict-lower row-major position order"
RANKING_METRIC = (
    "abs(mean_p(sorted_jailbreak_asym-sorted_original_asym))"
)
SIGN_CONSISTENCY_DEFINITION = (
    "fraction of paired prompt deltas whose sign agrees with the signed mean; "
    "when the signed mean is exactly zero, fraction of exactly-zero deltas"
)
RANK_ARTIFACT_FILES = (
    "rank_selection.pt",
    "selected_rank_elements.jsonl",
    "selection_diagnostics.json",
    "selection_config.json",
)


def _validate_paired_scores(
    original: torch.Tensor,
    jailbreak: torch.Tensor,
) -> None:
    validate_score_tensor(original, prompt_count=SELECTION_COUNT)
    validate_score_tensor(jailbreak, prompt_count=SELECTION_COUNT)
    if original.shape != jailbreak.shape:
        raise ValueError("original/jailbreak score shapes differ")


def sort_signed_rank_values(values: torch.Tensor) -> torch.Tensor:
    """Return signed ascending order statistics for each prompt/head.

    Stable sorting makes ties deterministic: tied values retain the existing
    strict-lower row-major position order.
    """

    validate_score_tensor(values, prompt_count=SELECTION_COUNT)
    sorted_values = torch.sort(values, dim=-1, stable=True).values.contiguous()
    if tuple(sorted_values.shape) != tuple(values.shape):
        raise AssertionError("rank-sorted score shape mismatch")
    if sorted_values.dtype != torch.float32 or sorted_values.device.type != "cpu":
        raise AssertionError("rank-sorted values must be CPU float32")
    if bool((sorted_values[..., 1:] < sorted_values[..., :-1]).any().item()):
        raise AssertionError("rank-sorted values are not ascending")
    return sorted_values


def compute_original_rank_reference(original: torch.Tensor) -> torch.Tensor:
    """Mean Selection20 original order statistic for every head and rank."""

    sorted_original = sort_signed_rank_values(original)
    reference = sorted_original.mean(dim=0, dtype=torch.float32).contiguous()
    expected = (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)
    if tuple(reference.shape) != expected:
        raise AssertionError("original rank reference shape mismatch")
    if reference.dtype != torch.float32 or reference.device.type != "cpu":
        raise AssertionError("original rank reference must be CPU float32")
    if not bool(torch.isfinite(reference).all().item()):
        raise ValueError("original rank reference contains NaN or infinity")
    if bool((reference[..., 1:] < reference[..., :-1]).any().item()):
        raise AssertionError("mean original rank reference is not ascending")
    return reference


def compute_rank_element_ranking(
    original: torch.Tensor,
    jailbreak: torch.Tensor,
) -> list[dict]:
    """Rank all ``(layer, head, rank)`` features by absolute mean delta.

    Original and GCG values are independently sorted by signed value for each
    paired prompt and head.  Overall ranking ties are resolved by lower layer,
    then lower head, then lower within-head rank.
    """

    _validate_paired_scores(original, jailbreak)
    sorted_original = sort_signed_rank_values(original)
    sorted_jailbreak = sort_signed_rank_values(jailbreak)
    delta = sorted_jailbreak - sorted_original
    mean_original = sorted_original.mean(dim=0, dtype=torch.float32)
    mean_gcg = sorted_jailbreak.mean(dim=0, dtype=torch.float32)
    mean_delta = delta.mean(dim=0, dtype=torch.float32)
    mean_absolute_delta = delta.abs().mean(dim=0, dtype=torch.float32)
    std_delta = delta.std(dim=0, correction=1).float()
    positive_rate = (delta > 0).float().mean(dim=0)
    negative_rate = (delta < 0).float().mean(dim=0)
    zero_rate = (delta == 0).float().mean(dim=0)
    sign_consistency = torch.where(
        mean_delta > 0,
        positive_rate,
        torch.where(mean_delta < 0, negative_rate, zero_rate),
    )
    dominant_sign_consistency = torch.maximum(
        torch.maximum(positive_rate, negative_rate), zero_rate
    )

    ranking = []
    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            for rank_index in range(ELEMENTS_PER_HEAD):
                signed = float(mean_delta[layer, head, rank_index].item())
                ranking.append(
                    {
                        "flat_index": (
                            (layer * NUM_HEADS + head) * ELEMENTS_PER_HEAD
                            + rank_index
                        ),
                        "layer": layer,
                        "head": head,
                        "rank": rank_index + 1,
                        "rank_index": rank_index,
                        "mean_original": float(
                            mean_original[layer, head, rank_index].item()
                        ),
                        "original_rank_reference": float(
                            mean_original[layer, head, rank_index].item()
                        ),
                        "mean_gcg": float(
                            mean_gcg[layer, head, rank_index].item()
                        ),
                        "mean_delta": signed,
                        "ranking_score": abs(signed),
                        "mean_absolute_delta": float(
                            mean_absolute_delta[layer, head, rank_index].item()
                        ),
                        "std_delta": float(
                            std_delta[layer, head, rank_index].item()
                        ),
                        "positive_rate": float(
                            positive_rate[layer, head, rank_index].item()
                        ),
                        "negative_rate": float(
                            negative_rate[layer, head, rank_index].item()
                        ),
                        "zero_rate": float(
                            zero_rate[layer, head, rank_index].item()
                        ),
                        "mean_delta_sign": (
                            1 if signed > 0 else -1 if signed < 0 else 0
                        ),
                        "sign_consistency": float(
                            sign_consistency[layer, head, rank_index].item()
                        ),
                        "dominant_sign_consistency": float(
                            dominant_sign_consistency[
                                layer, head, rank_index
                            ].item()
                        ),
                    }
                )
    ranking.sort(
        key=lambda record: (
            -record["ranking_score"],
            record["layer"],
            record["head"],
            record["rank"],
        )
    )
    for selection_rank, record in enumerate(ranking, 1):
        record["selection_rank"] = selection_rank
    return ranking


def build_topk_rank_masks(
    ranking: Sequence[Mapping[str, object]],
    *,
    k_values: Sequence[int] = TOP_K_VALUES,
) -> dict[int, torch.Tensor]:
    """Build nested CPU boolean masks over ``layer, head, rank_index``."""

    total = NUM_LAYERS * NUM_HEADS * ELEMENTS_PER_HEAD
    values = validate_top_k_values(k_values, total_elements=total)
    if len(ranking) != total:
        raise ValueError(f"ranking must contain exactly {total} elements")
    masks = {
        k: torch.zeros(
            (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD), dtype=torch.bool
        )
        for k in values
    }
    observed_flat_indices = []
    for position, record in enumerate(ranking):
        selection_rank = record.get("selection_rank")
        layer = record.get("layer")
        head = record.get("head")
        rank = record.get("rank")
        rank_index = record.get("rank_index")
        flat_index = record.get("flat_index")
        if selection_rank != position + 1:
            raise ValueError(
                "ranking rows must have contiguous one-based selection ranks"
            )
        if not all(
            type(value) is int
            for value in (layer, head, rank, rank_index, flat_index)
        ):
            raise TypeError("ranking indices must be integers")
        expected_flat = (
            (layer * NUM_HEADS + head) * ELEMENTS_PER_HEAD + rank_index
        )
        if (
            not 0 <= layer < NUM_LAYERS
            or not 0 <= head < NUM_HEADS
            or not 0 <= rank_index < ELEMENTS_PER_HEAD
            or rank != rank_index + 1
            or flat_index != expected_flat
        ):
            raise ValueError("ranking contains an invalid rank-element identity")
        observed_flat_indices.append(flat_index)
        for k, mask in masks.items():
            if selection_rank <= k:
                mask[layer, head, rank_index] = True
    if len(set(observed_flat_indices)) != total:
        raise ValueError("ranking contains duplicate or missing rank elements")
    for k, mask in masks.items():
        if int(mask.sum().item()) != k:
            raise AssertionError(f"Top-k rank mask cardinality mismatch for k={k}")
    for smaller, larger in zip(values, values[1:]):
        if bool((masks[smaller] & ~masks[larger]).any().item()):
            raise AssertionError("Top-k rank masks are not nested")
    return {k: mask.contiguous() for k, mask in masks.items()}


@dataclass(frozen=True)
class RankElementSelection:
    """In-memory, CPU-only rank selection/reference bundle."""

    ranking: tuple[dict, ...]
    masks: Mapping[int, torch.Tensor]
    original_rank_reference: torch.Tensor
    k_values: tuple[int, ...]

    def mask_for_k(self, k: int) -> torch.Tensor:
        if type(k) is not int or k not in self.masks:
            raise ValueError(f"k must be one of {self.k_values}, found {k!r}")
        return self.masks[k]

    def topk_records(self, k: int) -> tuple[dict, ...]:
        self.mask_for_k(k)
        return self.ranking[:k]


def build_rank_element_selection(
    original: torch.Tensor,
    jailbreak: torch.Tensor,
    *,
    k_values: Sequence[int] = TOP_K_VALUES,
) -> RankElementSelection:
    """Construct rank ranking, masks, and Selection20 original reference."""

    values = validate_top_k_values(k_values)
    ranking = compute_rank_element_ranking(original, jailbreak)
    masks = build_topk_rank_masks(ranking, k_values=values)
    reference = compute_original_rank_reference(original)
    return RankElementSelection(
        ranking=tuple(ranking),
        masks=masks,
        original_rank_reference=reference,
        k_values=values,
    )


def _rank_band_counts(records: Sequence[Mapping[str, object]]) -> list[dict]:
    output = []
    for start in range(1, ELEMENTS_PER_HEAD + 1, 12):
        end = min(start + 11, ELEMENTS_PER_HEAD)
        output.append(
            {
                "rank_start": start,
                "rank_end": end,
                "selected_rank_element_count": sum(
                    start <= int(record["rank"]) <= end for record in records
                ),
            }
        )
    return output


def selection_diagnostics(selection: RankElementSelection) -> dict:
    """Return scalar and count diagnostics for selection review."""

    conditions = []
    for k in selection.k_values:
        records = selection.topk_records(k)
        sign_values = [float(record["sign_consistency"]) for record in records]
        selected_ranks = [int(record["rank"]) for record in records]
        layer_counts: dict[int, int] = {}
        head_counts: dict[tuple[int, int], int] = {}
        rank_counts: dict[int, int] = {}
        for record in records:
            layer = int(record["layer"])
            head = int(record["head"])
            rank = int(record["rank"])
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
            head_counts[(layer, head)] = head_counts.get((layer, head), 0) + 1
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        conditions.append(
            {
                "k": k,
                "selected_rank_element_count": int(
                    selection.mask_for_k(k).sum().item()
                ),
                "selected_head_count": len(head_counts),
                "distinct_within_head_rank_count": len(rank_counts),
                "selected_rank_min": min(selected_ranks),
                "selected_rank_max": max(selected_ranks),
                "selected_rank_mean": sum(selected_ranks) / len(selected_ranks),
                "selected_rank_median": float(statistics.median(selected_ranks)),
                "positive_mean_delta_count": sum(
                    int(row["mean_delta_sign"]) > 0 for row in records
                ),
                "negative_mean_delta_count": sum(
                    int(row["mean_delta_sign"]) < 0 for row in records
                ),
                "zero_mean_delta_count": sum(
                    int(row["mean_delta_sign"]) == 0 for row in records
                ),
                "sign_consistency_mean": sum(sign_values) / len(sign_values),
                "sign_consistency_min": min(sign_values),
                "ranking_score_min": min(
                    float(row["ranking_score"]) for row in records
                ),
                "ranking_score_max": max(
                    float(row["ranking_score"]) for row in records
                ),
                "rank_band_counts": _rank_band_counts(records),
                "within_head_rank_counts": [
                    {
                        "rank": rank,
                        "selected_rank_element_count": rank_counts.get(rank, 0),
                    }
                    for rank in range(1, ELEMENTS_PER_HEAD + 1)
                ],
                "layer_selection_counts": [
                    {
                        "layer": layer,
                        "selected_rank_element_count": layer_counts.get(layer, 0),
                    }
                    for layer in range(NUM_LAYERS)
                ],
                "head_selection_counts": [
                    {
                        "layer": layer,
                        "head": head,
                        "selected_rank_element_count": head_counts.get(
                            (layer, head), 0
                        ),
                    }
                    for layer in range(NUM_LAYERS)
                    for head in range(NUM_HEADS)
                ],
            }
        )
    return {
        "method": "rank_element_topk_v1",
        "rank_sort": RANK_SORT,
        "rank_tie_break": RANK_TIE_BREAK,
        "rank_numbering": "one_based_1_to_120",
        "ranking_metric": RANKING_METRIC,
        "sign_consistency_definition": SIGN_CONSISTENCY_DEFINITION,
        "selection_prompt_count": SELECTION_COUNT,
        "candidate_rank_element_count": len(selection.ranking),
        "tail_tokens": TAIL_TOKENS,
        "strict_lower_elements_per_head": ELEMENTS_PER_HEAD,
        "k_values": list(selection.k_values),
        "conditions": conditions,
    }


def offline_rank_crossing_diagnostics(
    values: torch.Tensor,
    selection: RankElementSelection,
    *,
    k: int = TOP_K,
    alpha: float = 1.0,
) -> dict:
    """Measure adjacent-rank inversions on stored Selection20 values.

    This diagnostic applies all selected replacements simultaneously to the
    already extracted rank values.  It does not change selection and does not
    correct or re-sort any crossing.
    """

    alpha = validate_alpha(alpha)
    sorted_values = sort_signed_rank_values(values)
    mask = selection.mask_for_k(k)
    reference = selection.original_rank_reference
    updated = sorted_values + torch.where(
        mask.unsqueeze(0),
        alpha * (reference.unsqueeze(0) - sorted_values),
        0.0,
    )
    adjacent_crossings = updated[..., :-1] > updated[..., 1:]
    relevant_pairs = mask[..., :-1] | mask[..., 1:]
    adjacent_crossings &= relevant_pairs.unsqueeze(0)
    selected_crossing = torch.zeros_like(sorted_values, dtype=torch.bool)
    selected_crossing[..., 1:] |= adjacent_crossings
    selected_crossing[..., :-1] |= adjacent_crossings
    selected_crossing &= mask.unsqueeze(0)
    selected_count = values.shape[0] * int(mask.sum().item())
    selected_crossing_count = int(selected_crossing.sum().item())
    adjacent_pair_count = int(adjacent_crossings.sum().item())
    prompt_has_crossing = selected_crossing.reshape(values.shape[0], -1).any(dim=1)
    return {
        "prompt_count": values.shape[0],
        "selected_rank_intervention_count": selected_count,
        "selected_value_crossing_count": selected_crossing_count,
        "selected_value_crossing_fraction": (
            selected_crossing_count / selected_count
        ),
        "adjacent_rank_pair_crossing_count": adjacent_pair_count,
        "prompt_with_crossing_count": int(prompt_has_crossing.sum().item()),
        "definition": (
            "stored Selection20 S_asym values; after all selected replacements, "
            "a selected rank participates in a strict ascending-order inversion "
            "with an adjacent rank; no correction or re-sorting"
        ),
    }


def validate_rank_selection_payload(payload: Mapping[str, object]) -> None:
    """Validate the tensor payload used by runtime evaluation."""

    required = {
        "schema_version",
        "method",
        "k",
        "rank_mask",
        "original_rank_reference",
        "selected_features",
        "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"rank selection artifact missing fields: {sorted(missing)}")
    if payload["schema_version"] != 1 or payload["method"] != "rank_element_topk_v1":
        raise ValueError("rank selection artifact schema/method mismatch")
    if payload["k"] != TOP_K:
        raise ValueError(f"rank selection artifact must use k={TOP_K}")
    expected = (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)
    reference = payload["original_rank_reference"]
    mask = payload["rank_mask"]
    features = payload["selected_features"]
    if (
        not isinstance(reference, torch.Tensor)
        or tuple(reference.shape) != expected
        or reference.dtype != torch.float32
        or reference.device.type != "cpu"
        or not bool(torch.isfinite(reference).all().item())
    ):
        raise ValueError("original rank reference shape/dtype/device/finite mismatch")
    if bool((reference[..., 1:] < reference[..., :-1]).any().item()):
        raise ValueError("original rank reference is not ascending")
    if (
        not isinstance(mask, torch.Tensor)
        or tuple(mask.shape) != expected
        or mask.dtype != torch.bool
        or mask.device.type != "cpu"
        or int(mask.sum().item()) != TOP_K
    ):
        raise ValueError("rank mask shape/dtype/device/cardinality mismatch")
    if (
        not isinstance(features, torch.Tensor)
        or tuple(features.shape) != (TOP_K, 4)
        or features.dtype != torch.int64
        or features.device.type != "cpu"
    ):
        raise ValueError("selected_features must be a CPU int64 [500,4] tensor")
    if not isinstance(payload["metadata"], Mapping):
        raise ValueError("rank selection artifact metadata must be an object")
    expected_rows = []
    for selection_rank in range(1, TOP_K + 1):
        matches = (features[:, 0] == selection_rank).nonzero(as_tuple=False)
        if tuple(matches.shape) != (1, 1):
            raise ValueError("selected feature selection ranks are not unique 1..500")
        row = features[int(matches[0, 0].item())]
        layer, head, rank = (int(value.item()) for value in row[1:])
        if not 0 <= layer < NUM_LAYERS or not 0 <= head < NUM_HEADS:
            raise ValueError("selected feature layer/head is out of range")
        if not 1 <= rank <= ELEMENTS_PER_HEAD:
            raise ValueError("selected feature rank is out of range")
        expected_rows.append((layer, head, rank - 1))
    expected_mask = torch.zeros_like(mask)
    for layer, head, rank_index in expected_rows:
        expected_mask[layer, head, rank_index] = True
    if not torch.equal(mask, expected_mask):
        raise ValueError("selected_features and rank_mask disagree")


def selection_from_payload(payload: Mapping[str, object]) -> RankElementSelection:
    """Construct the runtime selection subset from a validated artifact."""

    validate_rank_selection_payload(payload)
    records = []
    for row in payload["selected_features"]:
        selection_rank, layer, head, rank = (
            int(value.item()) for value in row
        )
        records.append(
            {
                "selection_rank": selection_rank,
                "layer": layer,
                "head": head,
                "rank": rank,
                "rank_index": rank - 1,
                "flat_index": (
                    (layer * NUM_HEADS + head) * ELEMENTS_PER_HEAD + rank - 1
                ),
            }
        )
    return RankElementSelection(
        ranking=tuple(records),
        masks={TOP_K: payload["rank_mask"].contiguous()},
        original_rank_reference=payload[
            "original_rank_reference"
        ].contiguous(),
        k_values=TOP_K_VALUES,
    )


def load_rank_selection_artifact(
    artifact_dir: str | Path,
) -> tuple[RankElementSelection, dict, dict, dict]:
    """Verify and load a frozen Rank Element Top-500 artifact directory."""

    artifact_dir = Path(artifact_dir)
    manifest = read_json(artifact_dir / "artifact_manifest.json")
    if manifest.get("experiment") != "llama2_gcg_rank_element_topk_v1":
        raise ValueError("rank artifact experiment mismatch")
    if manifest.get("required_files") != list(RANK_ARTIFACT_FILES):
        raise ValueError("rank artifact required file declaration mismatch")
    verify_declared_file_hashes(artifact_dir, manifest)
    payload = torch.load(
        artifact_dir / "rank_selection.pt",
        map_location="cpu",
        weights_only=True,
    )
    validate_rank_selection_payload(payload)
    config = read_json(artifact_dir / "selection_config.json")
    diagnostics = read_json(artifact_dir / "selection_diagnostics.json")
    selected_rows = read_jsonl(artifact_dir / "selected_rank_elements.jsonl")
    if len(selected_rows) != TOP_K:
        raise ValueError("selected rank element JSONL must contain 500 rows")
    tensor_rows = payload["selected_features"].tolist()
    json_identities = [
        [
            int(row["selection_rank"]),
            int(row["layer"]),
            int(row["head"]),
            int(row["rank"]),
        ]
        for row in selected_rows
    ]
    if tensor_rows != json_identities:
        raise ValueError("rank selection tensor and JSONL identities disagree")
    if config.get("method") != "rank_element_topk_v1" or config.get("k") != TOP_K:
        raise ValueError("rank selection config method/k mismatch")
    conditions = diagnostics.get("conditions")
    if (
        diagnostics.get("method") != "rank_element_topk_v1"
        or not isinstance(conditions, list)
        or len(conditions) != 1
        or conditions[0].get("k") != TOP_K
        or conditions[0].get("selected_rank_element_count") != TOP_K
    ):
        raise ValueError("rank selection diagnostics mismatch")
    payload_metadata = payload["metadata"]
    for field in (
        "source_selection_manifest_sha256",
        "source_score_artifact_sha256",
        "original_score_sha256",
        "jailbreak_score_sha256",
    ):
        if payload_metadata.get(field) != config.get(field):
            raise ValueError(f"rank artifact provenance mismatch: {field}")
    verification = {
        "verified": True,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_manifest_sha256": sha256_file(
            artifact_dir / "artifact_manifest.json"
        ),
        "rank_selection_sha256": sha256_file(
            artifact_dir / "rank_selection.pt"
        ),
        "selected_rank_elements_sha256": sha256_file(
            artifact_dir / "selected_rank_elements.jsonl"
        ),
        "required_file_count": len(RANK_ARTIFACT_FILES),
    }
    return selection_from_payload(payload), config, diagnostics, verification


def _max_abs(value: torch.Tensor) -> float:
    return (
        float(value.detach().abs().amax().float().cpu().item())
        if value.numel()
        else 0.0
    )


class RankElementTopKController:
    """Replace selected order statistics at their pre-intervention positions."""

    def __init__(
        self,
        *,
        reference_mean: torch.Tensor,
        rank_mask: torch.Tensor,
        alpha: float,
        tail_tokens: int = TAIL_TOKENS,
        expected_score_dtype: torch.dtype | None = None,
        intervention_timing: str = "first_token_only",
    ):
        self.alpha = validate_alpha(alpha)
        self.intervention_timing = validate_intervention_timing(
            intervention_timing
        )
        if not isinstance(reference_mean, torch.Tensor):
            raise TypeError("reference_mean must be a torch tensor")
        if not isinstance(rank_mask, torch.Tensor):
            raise TypeError("rank_mask must be a torch tensor")
        if reference_mean.ndim != 3:
            raise ValueError("reference_mean must have shape layer,head,rank")
        if tuple(rank_mask.shape) != tuple(reference_mean.shape):
            raise ValueError("rank_mask/reference_mean shapes differ")
        if reference_mean.dtype != torch.float32 or reference_mean.device.type != "cpu":
            raise ValueError("reference_mean must be CPU float32")
        if rank_mask.dtype != torch.bool or rank_mask.device.type != "cpu":
            raise ValueError("rank_mask must be CPU bool")
        expected_elements = tail_tokens * (tail_tokens - 1) // 2
        if reference_mean.shape[-1] != expected_elements:
            raise ValueError("reference_mean rank dimension mismatch")
        if not bool(torch.isfinite(reference_mean).all().item()):
            raise ValueError("reference_mean contains NaN or infinity")
        if bool((reference_mean[..., 1:] < reference_mean[..., :-1]).any().item()):
            raise ValueError("reference_mean must be ascending across rank")
        if int(rank_mask.sum().item()) <= 0:
            raise ValueError("rank_mask must select at least one rank element")
        self.reference_mean = reference_mean.contiguous()
        self.rank_mask = rank_mask.contiguous()
        self.layer_count, self.head_count, _ = reference_mean.shape
        self.tail_tokens = tail_tokens
        self.expected_score_dtype = expected_score_dtype
        self.selected_layers = tuple(
            layer
            for layer in range(self.layer_count)
            if bool(self.rank_mask[layer].any().item())
        )
        self.selected_rank_element_count = int(self.rank_mask.sum().item())
        self._tril = strict_lower_indices(tail_tokens, device="cpu")
        self._device_cache: dict[
            tuple[int, str], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.clear()

    def clear(self) -> None:
        self.prompt_length: int | None = None
        self.attention_forward_call_count = 0
        self.intervention_call_count = 0
        self.selected_rank_intervention_count = 0
        self.selected_value_crossing_count = 0
        self.adjacent_rank_pair_crossing_count = 0
        self.alpha_zero_bypass_count = 0
        self.short_sequence_skip_count = 0
        self.first_forward_detected = False
        self.first_forward_intervened = False
        self.later_forward_intervention_count = 0
        self.later_forward_skip_count = 0
        self._records: list[dict] = []

    def set_prompt_context(self, *, prompt_length: int) -> None:
        if self.attention_forward_call_count or self.intervention_call_count:
            raise ReferenceInterventionRuntimeError(
                "prompt context must be set before attention forwards"
            )
        if type(prompt_length) is not int or prompt_length < self.tail_tokens:
            raise ValueError(
                f"prompt_length must be an integer >= {self.tail_tokens}"
            )
        self.prompt_length = prompt_length

    def _layer_values(
        self, layer: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer, str(device))
        cached = self._device_cache.get(key)
        if cached is None:
            cached = (
                self.reference_mean[layer].to(device=device),
                self.rank_mask[layer].to(device=device),
            )
            self._device_cache[key] = cached
        return cached

    def apply(self, layer: int, attention_scores: torch.Tensor) -> torch.Tensor:
        if layer not in self.selected_layers:
            return attention_scores
        self.attention_forward_call_count += 1
        if self.alpha == 0.0:
            self.alpha_zero_bypass_count += 1
            return attention_scores
        if not isinstance(attention_scores, torch.Tensor) or attention_scores.ndim != 4:
            raise ReferenceInterventionRuntimeError(
                "attention scores must have shape batch,head,query,key"
            )
        if not attention_scores.is_floating_point():
            raise ReferenceInterventionRuntimeError(
                "attention scores must be floating point"
            )
        if (
            self.expected_score_dtype is not None
            and attention_scores.dtype != self.expected_score_dtype
        ):
            raise ReferenceInterventionRuntimeError(
                "attention score dtype mismatch"
            )
        batch, heads, query_length, key_length = attention_scores.shape
        if batch != 1 or heads != self.head_count or query_length != key_length:
            raise ReferenceInterventionRuntimeError("attention score shape mismatch")
        if self.intervention_timing == "first_token_only":
            if self.prompt_length is None:
                raise ReferenceInterventionRuntimeError(
                    "first_token_only requires prompt context"
                )
            if query_length < self.prompt_length:
                raise ReferenceInterventionRuntimeError(
                    "current sequence length is shorter than prompt length"
                )
            if query_length > self.prompt_length:
                self.later_forward_skip_count += 1
                return attention_scores
            self.first_forward_detected = True
        elif self.prompt_length is not None:
            if query_length < self.prompt_length:
                raise ReferenceInterventionRuntimeError(
                    "current sequence length is shorter than prompt length"
                )
            self.first_forward_detected |= query_length == self.prompt_length
        if query_length < self.tail_tokens:
            self.short_sequence_skip_count += 1
            return attention_scores
        if not bool(torch.isfinite(attention_scores).all().item()):
            raise ReferenceInterventionRuntimeError(
                "pre-intervention attention scores contain NaN or infinity"
            )

        reference, mask = self._layer_values(layer, attention_scores.device)
        rows, cols = self._tril.to(attention_scores.device)
        original_block = attention_scores[
            :, :, -self.tail_tokens :, -self.tail_tokens :
        ].float()
        current_asym = (original_block - original_block.transpose(-1, -2)) / 2
        current_lower = current_asym[:, :, rows, cols]
        sorted_lower, rank_to_position = torch.sort(
            current_lower, dim=-1, stable=True
        )
        sorted_delta = self.alpha * (reference.unsqueeze(0) - sorted_lower)
        sorted_delta = torch.where(mask.unsqueeze(0), sorted_delta, 0.0)
        updated_rank_values = sorted_lower + sorted_delta

        adjacent_crossings = (
            updated_rank_values[..., :-1] > updated_rank_values[..., 1:]
        )
        relevant_adjacent_pairs = mask[..., :-1] | mask[..., 1:]
        adjacent_crossings &= relevant_adjacent_pairs.unsqueeze(0)
        left_crossing = torch.zeros_like(mask).unsqueeze(0)
        right_crossing = torch.zeros_like(mask).unsqueeze(0)
        left_crossing[..., 1:] = adjacent_crossings
        right_crossing[..., :-1] = adjacent_crossings
        selected_value_crossing = mask.unsqueeze(0) & (
            left_crossing | right_crossing
        )

        lower_delta = torch.zeros_like(current_lower).scatter(
            -1, rank_to_position, sorted_delta
        )
        selected_position_mask = torch.zeros_like(
            current_lower, dtype=torch.bool
        ).scatter(-1, rank_to_position, mask.unsqueeze(0))
        constructed_delta = torch.zeros_like(original_block)
        constructed_delta[:, :, rows, cols] = lower_delta
        constructed_delta[:, :, cols, rows] = -lower_delta
        intervened_float32 = original_block + constructed_delta
        intervened_cast = intervened_float32.to(attention_scores.dtype)

        modified = attention_scores.clone()
        modified[:, :, -self.tail_tokens :, -self.tail_tokens :] = intervened_cast
        if not bool(torch.isfinite(modified).all().item()):
            raise ReferenceInterventionRuntimeError(
                "post-intervention attention scores contain NaN or infinity"
            )
        realized_delta = intervened_cast.float() - original_block
        selected_pair_mask = torch.zeros(
            (1, self.head_count, self.tail_tokens, self.tail_tokens),
            dtype=torch.bool,
            device=attention_scores.device,
        )
        selected_pair_mask[:, :, rows, cols] = selected_position_mask
        selected_pair_mask[:, :, cols, rows] = selected_position_mask
        outside_element_change = _max_abs(
            realized_delta.masked_select(~selected_pair_mask)
        )
        prefix = query_length - self.tail_tokens
        outside_tail_change = 0.0
        if prefix:
            outside_tail_change = max(
                _max_abs(
                    modified[:, :, :prefix, :] - attention_scores[:, :, :prefix, :]
                ),
                _max_abs(
                    modified[:, :, -self.tail_tokens :, :prefix]
                    - attention_scores[:, :, -self.tail_tokens :, :prefix]
                ),
            )
        if outside_element_change != 0.0 or outside_tail_change != 0.0:
            raise ReferenceInterventionRuntimeError(
                "intervention changed an unselected position or value outside the tail"
            )

        selected_in_layer = int(mask.sum().item())
        selected_crossing_count = int(selected_value_crossing.sum().item())
        adjacent_crossing_count = int(adjacent_crossings.sum().item())
        self.intervention_call_count += 1
        self.selected_rank_intervention_count += selected_in_layer
        self.selected_value_crossing_count += selected_crossing_count
        self.adjacent_rank_pair_crossing_count += adjacent_crossing_count
        if self.prompt_length is not None:
            if query_length == self.prompt_length:
                self.first_forward_intervened = True
            elif query_length > self.prompt_length:
                self.later_forward_intervention_count += selected_in_layer
        before_sym = (original_block + original_block.transpose(-1, -2)) / 2
        after_sym = (
            intervened_float32 + intervened_float32.transpose(-1, -2)
        ) / 2
        self._records.append(
            {
                "layer": layer,
                "sequence_length": query_length,
                "selected_rank_element_count": selected_in_layer,
                "selected_value_crossing_count": selected_crossing_count,
                "adjacent_rank_pair_crossing_count": adjacent_crossing_count,
                "max_constructed_delta_antisymmetry_error_float32": _max_abs(
                    constructed_delta + constructed_delta.transpose(-1, -2)
                ),
                "max_symmetric_component_error_float32": _max_abs(
                    after_sym - before_sym
                ),
                "max_diagonal_change_float32": _max_abs(
                    torch.diagonal(constructed_delta, dim1=-2, dim2=-1)
                ),
                "max_unselected_element_change": outside_element_change,
                "max_outside_tail_change": outside_tail_change,
                "max_absolute_delta_float32": _max_abs(constructed_delta),
            }
        )
        return modified

    def summary(self) -> dict:
        def maximum(key: str) -> float:
            return max(
                (float(record[key]) for record in self._records), default=0.0
            )

        crossing_fraction = (
            self.selected_value_crossing_count
            / self.selected_rank_intervention_count
            if self.selected_rank_intervention_count
            else 0.0
        )
        return {
            "alpha": self.alpha,
            "intervention_timing": self.intervention_timing,
            "selected_layers": list(self.selected_layers),
            "selected_rank_element_count": self.selected_rank_element_count,
            "prompt_length": self.prompt_length,
            "attention_forward_call_count": self.attention_forward_call_count,
            "intervention_call_count": self.intervention_call_count,
            "selected_rank_intervention_count": self.selected_rank_intervention_count,
            "selected_value_crossing_count": self.selected_value_crossing_count,
            "selected_value_crossing_fraction": crossing_fraction,
            "adjacent_rank_pair_crossing_count": (
                self.adjacent_rank_pair_crossing_count
            ),
            "crossing_definition": (
                "after all selected replacements, a selected rank participates "
                "in a strict ascending-order inversion with an adjacent rank; "
                "no correction or re-sorting is applied"
            ),
            "alpha_zero_bypass_count": self.alpha_zero_bypass_count,
            "short_sequence_skip_count": self.short_sequence_skip_count,
            "first_forward_detected": self.first_forward_detected,
            "first_forward_intervened": self.first_forward_intervened,
            "later_forward_intervention_count": self.later_forward_intervention_count,
            "later_forward_skip_count": self.later_forward_skip_count,
            "max_constructed_delta_antisymmetry_error_float32": maximum(
                "max_constructed_delta_antisymmetry_error_float32"
            ),
            "max_symmetric_component_error_float32": maximum(
                "max_symmetric_component_error_float32"
            ),
            "max_diagonal_change_float32": maximum(
                "max_diagonal_change_float32"
            ),
            "max_unselected_element_change": maximum(
                "max_unselected_element_change"
            ),
            "max_outside_tail_change": maximum("max_outside_tail_change"),
            "max_absolute_delta_float32": maximum("max_absolute_delta_float32"),
            "all_scores_finite": True,
        }


@contextmanager
def patch_llama_rank_element_attention(
    model,
    *,
    selection: RankElementSelection,
    k: int,
    alpha: float,
    intervention_timing: str = "first_token_only",
):
    """Temporarily install Rank Element Top-k intervention hooks."""

    mask = selection.mask_for_k(k)
    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected Transformers {SUPPORTED_TRANSFORMERS_VERSION}, "
            f"found {transformers.__version__}"
        )
    controller = RankElementTopKController(
        reference_mean=selection.original_rank_reference,
        rank_mask=mask,
        alpha=alpha,
        expected_score_dtype=torch.float16,
        intervention_timing=intervention_timing,
    )
    patched = []
    try:
        for layer_index in controller.selected_layers:
            module = model.model.layers[layer_index].self_attn
            if type(module) is not modeling_llama.LlamaAttention:
                raise TypeError(
                    "rank-element intervention requires eager LlamaAttention; "
                    f"layer={layer_index}, found={type(module).__name__}"
                )
            if module.layer_idx != layer_index:
                raise ValueError(
                    f"attention layer_idx mismatch at layer={layer_index}"
                )
            had_instance_forward = "forward" in module.__dict__
            previous_instance_forward = module.__dict__.get("forward")
            replacement = _patched_llama_attention_forward(controller)
            module.forward = types.MethodType(replacement, module)
            patched.append(
                (module, had_instance_forward, previous_instance_forward)
            )
        yield controller
    finally:
        for module, had_instance_forward, previous_instance_forward in reversed(
            patched
        ):
            if had_instance_forward:
                module.forward = previous_instance_forward
            else:
                module.__dict__.pop("forward", None)
