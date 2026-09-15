"""Selection20 rank-distribution shift and all-rank raw-S cancellation."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import torch

from utils.artifact_io import read_json, sha256_file
from utils.gcg_rank_element_topk_intervention import (
    RANK_SORT,
    RANK_TIE_BREAK,
    RankElementTopKController,
    patch_llama_rank_element_controller,
    sort_signed_rank_values,
)
from utils.gcg_wasserstein_reference import (
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    TAIL_TOKENS,
    validate_score_tensor,
    verify_declared_file_hashes,
)
from utils.reference_asym_score_intervention import validate_alpha


METHOD = "rank_distribution_shift_cancellation_v1"
ALPHA_DEFAULT = 1.0
ALL_RANK_COUNT = NUM_LAYERS * NUM_HEADS * ELEMENTS_PER_HEAD
SHIFT_ARTIFACT_FILES = ("mean_shift.pt", "shift_config.json")


def compute_mean_shift(original: torch.Tensor, gcg: torch.Tensor) -> torch.Tensor:
    """Compute D = mean_p(sorted_GCG - sorted_original), with independent sorts."""

    validate_score_tensor(original, prompt_count=SELECTION_COUNT)
    validate_score_tensor(gcg, prompt_count=SELECTION_COUNT)
    sorted_original = sort_signed_rank_values(original)
    sorted_gcg = sort_signed_rank_values(gcg)
    shift = (sorted_gcg - sorted_original).mean(dim=0, dtype=torch.float32)
    shift = shift.contiguous()
    validate_mean_shift(shift)
    return shift


def validate_mean_shift(shift: torch.Tensor) -> None:
    expected = (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)
    if not isinstance(shift, torch.Tensor) or tuple(shift.shape) != expected:
        raise ValueError(f"mean_shift must have shape {expected}")
    if shift.dtype != torch.float32 or shift.device.type != "cpu":
        raise ValueError("mean_shift must be CPU float32")
    if not bool(torch.isfinite(shift).all().item()):
        raise ValueError("mean_shift contains NaN or infinity")


def offline_shift_crossing_diagnostics(
    values: torch.Tensor, mean_shift: torch.Tensor, *, alpha: float = ALPHA_DEFAULT
) -> dict:
    """Count adjacent inversions after subtraction without changing rank order."""

    alpha = validate_alpha(alpha)
    validate_mean_shift(mean_shift)
    sorted_values = sort_signed_rank_values(values)
    updated = sorted_values - alpha * mean_shift.unsqueeze(0)
    crossing = updated[..., :-1] > updated[..., 1:]
    involved = torch.zeros_like(updated, dtype=torch.bool)
    involved[..., 1:] |= crossing
    involved[..., :-1] |= crossing
    count = int(involved.sum().item())
    total = values.shape[0] * ALL_RANK_COUNT
    return {
        "prompt_count": values.shape[0],
        "rank_element_count": total,
        "rank_value_crossing_count": count,
        "rank_value_crossing_fraction": count / total,
        "adjacent_rank_pair_crossing_count": int(crossing.sum().item()),
        "prompt_with_crossing_count": int(
            involved.reshape(values.shape[0], -1).any(dim=1).sum().item()
        ),
        "definition": "strict adjacent inversion in Q-alpha*D; no correction or re-sort",
    }


def validate_shift_payload(payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("shift payload must be a mapping")
    if payload.get("schema_version") != 1 or payload.get("method") != METHOD:
        raise ValueError("shift payload schema/method mismatch")
    validate_mean_shift(payload.get("mean_shift"))
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("shift payload metadata missing")
    for field in (
        "source_selection_manifest_sha256",
        "source_score_artifact_sha256",
        "original_score_sha256",
        "jailbreak_score_sha256",
    ):
        value = metadata.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"shift payload {field} missing")
    if metadata.get("selection_prompt_count") != SELECTION_COUNT:
        raise ValueError("shift payload must use Selection20")
    if metadata.get("alpha_default") != ALPHA_DEFAULT:
        raise ValueError("shift payload alpha default mismatch")
    if metadata.get("rank_sort") != RANK_SORT or metadata.get("rank_tie_break") != RANK_TIE_BREAK:
        raise ValueError("shift payload rank sort/tie break mismatch")
    if metadata.get("scientific_conditions") is None:
        raise ValueError("shift payload scientific conditions missing")


def load_shift_artifact(artifact_dir: str | Path) -> tuple[torch.Tensor, dict, dict]:
    artifact_dir = Path(artifact_dir)
    manifest = read_json(artifact_dir / "artifact_manifest.json")
    if manifest.get("experiment") != METHOD:
        raise ValueError("shift artifact experiment mismatch")
    if manifest.get("required_files") != list(SHIFT_ARTIFACT_FILES):
        raise ValueError("shift artifact file declaration mismatch")
    verify_declared_file_hashes(artifact_dir, manifest)
    payload = torch.load(artifact_dir / "mean_shift.pt", map_location="cpu", weights_only=True)
    validate_shift_payload(payload)
    config = read_json(artifact_dir / "shift_config.json")
    if config.get("method") != METHOD:
        raise ValueError("shift config method mismatch")
    for field, value in payload["metadata"].items():
        if config.get(field) != value:
            raise ValueError(f"shift config/payload mismatch: {field}")
    verification = {
        "verified": True,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_manifest_sha256": sha256_file(artifact_dir / "artifact_manifest.json"),
        "mean_shift_sha256": sha256_file(artifact_dir / "mean_shift.pt"),
        "shift_config_sha256": sha256_file(artifact_dir / "shift_config.json"),
    }
    return payload["mean_shift"], config, verification


class RankDistributionShiftCancellationController(RankElementTopKController):
    """Subtract a common rank shift at every pre-intervention physical position."""

    def __init__(
        self,
        *,
        mean_shift: torch.Tensor,
        alpha: float = ALPHA_DEFAULT,
        expected_score_dtype: torch.dtype | None = None,
        intervention_timing: str = "first_token_only",
    ):
        validate_mean_shift(mean_shift)
        super().__init__(
            reference_mean=mean_shift,
            rank_mask=torch.ones_like(mean_shift, dtype=torch.bool),
            alpha=alpha,
            tail_tokens=TAIL_TOKENS,
            expected_score_dtype=expected_score_dtype,
            intervention_timing=intervention_timing,
            shift_mode=True,
        )


@contextmanager
def patch_llama_shift_cancellation_attention(
    model, *, mean_shift: torch.Tensor, alpha: float = ALPHA_DEFAULT
):
    controller = RankDistributionShiftCancellationController(
        mean_shift=mean_shift,
        alpha=alpha,
        expected_score_dtype=torch.float16,
        intervention_timing="first_token_only",
    )
    with patch_llama_rank_element_controller(model, controller) as active:
        yield active
