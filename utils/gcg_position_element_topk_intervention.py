"""Position-element Top-k selection and tail-score intervention helpers.

The selection unit is one strict-lower ``S_asym`` element identified by
``(layer, head, row_pos, col_pos)``.  Selection is computed only from the
fixed GCG selection split.  Runtime intervention mirrors every selected
strict-lower edit into the upper triangle with the opposite sign, preserving
the raw score block's symmetric component and diagonal.
"""

from __future__ import annotations

import types
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import transformers
from transformers.models.llama import modeling_llama

from utils.asym_score_intervention import SUPPORTED_TRANSFORMERS_VERSION
from utils.gcg_wasserstein_reference import (
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    TAIL_TOKENS,
    strict_lower_indices,
    validate_score_tensor,
)
from utils.reference_asym_score_intervention import (
    ReferenceInterventionRuntimeError,
    _patched_llama_attention_forward,
    validate_alpha,
    validate_intervention_timing,
)


TOP_K_VALUES = (10, 50, 100, 500)
RANKING_METRIC = "abs(mean_p(jailbreak_asym-original_asym))"
SIGN_CONSISTENCY_DEFINITION = (
    "fraction of paired prompt deltas whose sign agrees with the signed mean; "
    "when the signed mean is exactly zero, fraction of exactly-zero deltas"
)


def validate_top_k_values(
    k_values: Sequence[int],
    *,
    total_elements: int = NUM_LAYERS * NUM_HEADS * ELEMENTS_PER_HEAD,
) -> tuple[int, ...]:
    """Validate an ascending, unique Top-k grid."""

    if isinstance(k_values, (str, bytes)):
        raise TypeError("k_values must be a sequence of integers")
    values = tuple(k_values)
    if not values:
        raise ValueError("k_values must not be empty")
    if any(type(value) is not int for value in values):
        raise TypeError("every k value must be an integer")
    if any(value <= 0 or value > total_elements for value in values):
        raise ValueError(
            f"k values must satisfy 1 <= k <= {total_elements}: {values}"
        )
    if tuple(sorted(set(values))) != values:
        raise ValueError("k_values must be strictly increasing and unique")
    return values


def _validate_paired_scores(
    original: torch.Tensor,
    jailbreak: torch.Tensor,
) -> None:
    validate_score_tensor(original, prompt_count=SELECTION_COUNT)
    validate_score_tensor(jailbreak, prompt_count=SELECTION_COUNT)
    if original.shape != jailbreak.shape:
        raise ValueError("original/jailbreak score shapes differ")


def compute_original_mean_reference(original: torch.Tensor) -> torch.Tensor:
    """Build the deployable element reference from selection20 originals."""

    validate_score_tensor(original, prompt_count=SELECTION_COUNT)
    reference = original.mean(dim=0, dtype=torch.float32).contiguous()
    expected = (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)
    if tuple(reference.shape) != expected:
        raise AssertionError("original mean reference shape mismatch")
    if reference.dtype != torch.float32 or reference.device.type != "cpu":
        raise AssertionError("original mean reference must be CPU float32")
    if not bool(torch.isfinite(reference).all().item()):
        raise ValueError("original mean reference contains NaN or infinity")
    return reference


def compute_position_element_ranking(
    original: torch.Tensor,
    jailbreak: torch.Tensor,
) -> list[dict]:
    """Rank all position elements by absolute paired selection mean delta.

    Ties are deterministic: lower ``layer``, then ``head``, then strict-lower
    ``position`` (the row-major order from :func:`torch.tril_indices`).
    """

    _validate_paired_scores(original, jailbreak)
    delta = jailbreak - original
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

    rows, cols = strict_lower_indices(TAIL_TOKENS, device="cpu")
    ranking = []
    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            for position in range(ELEMENTS_PER_HEAD):
                signed = float(mean_delta[layer, head, position].item())
                ranking.append(
                    {
                        "flat_index": (
                            (layer * NUM_HEADS + head) * ELEMENTS_PER_HEAD
                            + position
                        ),
                        "layer": layer,
                        "head": head,
                        "position": position,
                        "row_pos": int(rows[position].item()),
                        "col_pos": int(cols[position].item()),
                        "lag": int((rows[position] - cols[position]).item()),
                        "mean_delta": signed,
                        "ranking_score": abs(signed),
                        "mean_absolute_delta": float(
                            mean_absolute_delta[layer, head, position].item()
                        ),
                        "std_delta": float(std_delta[layer, head, position].item()),
                        "positive_rate": float(
                            positive_rate[layer, head, position].item()
                        ),
                        "negative_rate": float(
                            negative_rate[layer, head, position].item()
                        ),
                        "zero_rate": float(zero_rate[layer, head, position].item()),
                        "mean_delta_sign": (
                            1 if signed > 0 else -1 if signed < 0 else 0
                        ),
                        "sign_consistency": float(
                            sign_consistency[layer, head, position].item()
                        ),
                        "dominant_sign_consistency": float(
                            dominant_sign_consistency[
                                layer, head, position
                            ].item()
                        ),
                    }
                )
    ranking.sort(
        key=lambda record: (
            -record["ranking_score"],
            record["layer"],
            record["head"],
            record["position"],
        )
    )
    for rank, record in enumerate(ranking, 1):
        record["rank"] = rank
    return ranking


def build_topk_element_masks(
    ranking: Sequence[Mapping[str, object]],
    *,
    k_values: Sequence[int] = TOP_K_VALUES,
) -> dict[int, torch.Tensor]:
    """Build nested CPU boolean masks over ``layer,head,position``."""

    total = NUM_LAYERS * NUM_HEADS * ELEMENTS_PER_HEAD
    values = validate_top_k_values(k_values, total_elements=total)
    if len(ranking) != total:
        raise ValueError(f"ranking must contain exactly {total} elements")
    observed_flat_indices = []
    masks = {
        k: torch.zeros(
            (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD), dtype=torch.bool
        )
        for k in values
    }
    for position, record in enumerate(ranking):
        rank = record.get("rank")
        layer = record.get("layer")
        head = record.get("head")
        element = record.get("position")
        flat_index = record.get("flat_index")
        if rank != position + 1:
            raise ValueError("ranking rows must have contiguous one-based ranks")
        if not all(type(value) is int for value in (layer, head, element, flat_index)):
            raise TypeError("ranking indices must be integers")
        expected_flat = (
            (layer * NUM_HEADS + head) * ELEMENTS_PER_HEAD + element
        )
        if (
            not 0 <= layer < NUM_LAYERS
            or not 0 <= head < NUM_HEADS
            or not 0 <= element < ELEMENTS_PER_HEAD
            or flat_index != expected_flat
        ):
            raise ValueError("ranking contains an invalid element identity")
        observed_flat_indices.append(flat_index)
        for k, mask in masks.items():
            if rank <= k:
                mask[layer, head, element] = True
    if len(set(observed_flat_indices)) != total:
        raise ValueError("ranking contains duplicate or missing elements")
    for k, mask in masks.items():
        if int(mask.sum().item()) != k:
            raise AssertionError(f"Top-k mask cardinality mismatch for k={k}")
    for smaller, larger in zip(values, values[1:]):
        if bool((masks[smaller] & ~masks[larger]).any().item()):
            raise AssertionError("Top-k masks are not nested")
    return {k: mask.contiguous() for k, mask in masks.items()}


@dataclass(frozen=True)
class PositionElementSelection:
    """In-memory, CPU-only selection/reference bundle."""

    ranking: tuple[dict, ...]
    masks: Mapping[int, torch.Tensor]
    original_mean_reference: torch.Tensor
    k_values: tuple[int, ...]

    def mask_for_k(self, k: int) -> torch.Tensor:
        if type(k) is not int or k not in self.masks:
            raise ValueError(f"k must be one of {self.k_values}, found {k!r}")
        return self.masks[k]

    def topk_records(self, k: int) -> tuple[dict, ...]:
        self.mask_for_k(k)
        return self.ranking[:k]


def build_position_element_selection(
    original: torch.Tensor,
    jailbreak: torch.Tensor,
    *,
    k_values: Sequence[int] = TOP_K_VALUES,
) -> PositionElementSelection:
    """Construct ranking, nested masks, and original-only mean reference."""

    values = validate_top_k_values(k_values)
    ranking = compute_position_element_ranking(original, jailbreak)
    masks = build_topk_element_masks(ranking, k_values=values)
    reference = compute_original_mean_reference(original)
    return PositionElementSelection(
        ranking=tuple(ranking),
        masks=masks,
        original_mean_reference=reference,
        k_values=values,
    )


def selection_diagnostics(selection: PositionElementSelection) -> dict:
    """Return scalar-only diagnostics for review and run manifests."""

    conditions = []
    for k in selection.k_values:
        records = selection.topk_records(k)
        sign_values = [float(record["sign_consistency"]) for record in records]
        conditions.append(
            {
                "k": k,
                "selected_element_count": int(selection.mask_for_k(k).sum().item()),
                "selected_head_count": len(
                    {(int(row["layer"]), int(row["head"])) for row in records}
                ),
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
            }
        )
    return {
        "ranking_metric": RANKING_METRIC,
        "sign_consistency_definition": SIGN_CONSISTENCY_DEFINITION,
        "selection_prompt_count": SELECTION_COUNT,
        "candidate_element_count": len(selection.ranking),
        "tail_tokens": TAIL_TOKENS,
        "strict_lower_elements_per_head": ELEMENTS_PER_HEAD,
        "k_values": list(selection.k_values),
        "conditions": conditions,
    }


def _max_abs(value: torch.Tensor) -> float:
    return float(value.detach().abs().amax().float().cpu().item()) if value.numel() else 0.0


class PositionElementTopKController:
    """Interpolate only masked strict-lower elements toward an original mean."""

    def __init__(
        self,
        *,
        reference_mean: torch.Tensor,
        element_mask: torch.Tensor,
        alpha: float,
        tail_tokens: int = TAIL_TOKENS,
        expected_score_dtype: torch.dtype | None = None,
        intervention_timing: str = "first_token_only",
    ):
        self.alpha = validate_alpha(alpha)
        self.intervention_timing = validate_intervention_timing(intervention_timing)
        if not isinstance(reference_mean, torch.Tensor):
            raise TypeError("reference_mean must be a torch tensor")
        if not isinstance(element_mask, torch.Tensor):
            raise TypeError("element_mask must be a torch tensor")
        if reference_mean.ndim != 3:
            raise ValueError("reference_mean must have shape layer,head,position")
        if tuple(element_mask.shape) != tuple(reference_mean.shape):
            raise ValueError("element_mask/reference_mean shapes differ")
        if reference_mean.dtype != torch.float32 or reference_mean.device.type != "cpu":
            raise ValueError("reference_mean must be CPU float32")
        if element_mask.dtype != torch.bool or element_mask.device.type != "cpu":
            raise ValueError("element_mask must be CPU bool")
        expected_elements = tail_tokens * (tail_tokens - 1) // 2
        if reference_mean.shape[-1] != expected_elements:
            raise ValueError("reference_mean position dimension mismatch")
        if not bool(torch.isfinite(reference_mean).all().item()):
            raise ValueError("reference_mean contains NaN or infinity")
        if int(element_mask.sum().item()) <= 0:
            raise ValueError("element_mask must select at least one element")
        self.reference_mean = reference_mean.contiguous()
        self.element_mask = element_mask.contiguous()
        self.layer_count, self.head_count, _ = reference_mean.shape
        self.tail_tokens = tail_tokens
        self.expected_score_dtype = expected_score_dtype
        self.selected_layers = tuple(
            layer
            for layer in range(self.layer_count)
            if bool(self.element_mask[layer].any().item())
        )
        self.selected_element_count = int(self.element_mask.sum().item())
        self._tril = strict_lower_indices(tail_tokens, device="cpu")
        self._device_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self.clear()

    def clear(self) -> None:
        self.prompt_length: int | None = None
        self.attention_forward_call_count = 0
        self.intervention_call_count = 0
        self.selected_element_intervention_count = 0
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
                self.element_mask[layer].to(device=device),
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
            raise ReferenceInterventionRuntimeError("attention scores must be floating point")
        if self.expected_score_dtype is not None and attention_scores.dtype != self.expected_score_dtype:
            raise ReferenceInterventionRuntimeError("attention score dtype mismatch")
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
        lower_delta = self.alpha * (reference.unsqueeze(0) - current_lower)
        lower_delta = torch.where(mask.unsqueeze(0), lower_delta, 0.0)
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
            (self.head_count, self.tail_tokens, self.tail_tokens),
            dtype=torch.bool,
            device=attention_scores.device,
        )
        selected_pair_mask[:, rows, cols] = mask
        selected_pair_mask[:, cols, rows] = mask
        outside_element_change = _max_abs(
            realized_delta.masked_select(~selected_pair_mask.unsqueeze(0))
        )
        prefix = query_length - self.tail_tokens
        outside_tail_change = 0.0
        if prefix:
            outside_tail_change = max(
                _max_abs(modified[:, :, :prefix, :] - attention_scores[:, :, :prefix, :]),
                _max_abs(
                    modified[:, :, -self.tail_tokens :, :prefix]
                    - attention_scores[:, :, -self.tail_tokens :, :prefix]
                ),
            )
        if outside_element_change != 0.0 or outside_tail_change != 0.0:
            raise ReferenceInterventionRuntimeError(
                "intervention changed an unselected element or value outside the tail"
            )

        selected_in_layer = int(mask.sum().item())
        self.intervention_call_count += 1
        self.selected_element_intervention_count += selected_in_layer
        if self.prompt_length is not None:
            if query_length == self.prompt_length:
                self.first_forward_intervened = True
            elif query_length > self.prompt_length:
                self.later_forward_intervention_count += selected_in_layer
        before_sym = (original_block + original_block.transpose(-1, -2)) / 2
        after_sym = (intervened_float32 + intervened_float32.transpose(-1, -2)) / 2
        self._records.append(
            {
                "layer": layer,
                "sequence_length": query_length,
                "selected_element_count": selected_in_layer,
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
            return max((float(record[key]) for record in self._records), default=0.0)

        return {
            "alpha": self.alpha,
            "intervention_timing": self.intervention_timing,
            "selected_layers": list(self.selected_layers),
            "selected_element_count": self.selected_element_count,
            "prompt_length": self.prompt_length,
            "attention_forward_call_count": self.attention_forward_call_count,
            "intervention_call_count": self.intervention_call_count,
            "selected_element_intervention_count": self.selected_element_intervention_count,
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
def patch_llama_position_element_attention(
    model,
    *,
    selection: PositionElementSelection,
    k: int,
    alpha: float,
    intervention_timing: str = "first_token_only",
):
    """Temporarily install Position Element Top-k intervention hooks."""

    mask = selection.mask_for_k(k)
    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected Transformers {SUPPORTED_TRANSFORMERS_VERSION}, "
            f"found {transformers.__version__}"
        )
    controller = PositionElementTopKController(
        reference_mean=selection.original_mean_reference,
        element_mask=mask,
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
                    "position-element intervention requires eager LlamaAttention; "
                    f"layer={layer_index}, found={type(module).__name__}"
                )
            if module.layer_idx != layer_index:
                raise ValueError(f"attention layer_idx mismatch at layer={layer_index}")
            had_instance_forward = "forward" in module.__dict__
            previous_instance_forward = module.__dict__.get("forward")
            replacement = _patched_llama_attention_forward(controller)
            module.forward = types.MethodType(replacement, module)
            patched.append((module, had_instance_forward, previous_instance_forward))
        yield controller
    finally:
        for module, had_instance_forward, previous_instance_forward in reversed(patched):
            if had_instance_forward:
                module.forward = previous_instance_forward
            else:
                module.__dict__.pop("forward", None)
