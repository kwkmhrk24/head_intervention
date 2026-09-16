"""Full-prompt reference moments and dynamic pure-asymmetric intervention.

The reference is built only from per-prompt summaries of the strict lower
triangle of ``A_asym = (S - S.T) / 2``.  Consequently every reference prompt
has weight 1/20 regardless of its token length.  At runtime, anomalous
strict-lower elements are selected independently for each prompt by an
explicit z-score threshold and edited as an antisymmetric raw-score pair.
"""

from __future__ import annotations

import math
import types
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from transformers.models.llama import modeling_llama

from utils.asym_score_intervention import SUPPORTED_TRANSFORMERS_VERSION
from utils.gcg_wasserstein_reference import (
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    TOTAL_COUNT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    strict_lower_indices,
    validate_split_manifest,
)
from utils.reference_asym_score_intervention import (
    ReferenceInterventionRuntimeError,
    _patched_llama_attention_forward,
    validate_alpha,
)


REFERENCE_SCHEMA_VERSION = 1
# The square root keeps division by a floored denominator inside float32's
# normal range for ordinary finite attention scores.  This is a numerical
# guard, not a scientifically selected scale or threshold.
NUMERICAL_SIGMA_FLOOR = math.sqrt(torch.finfo(torch.float32).tiny)
VARIANCE_ROUNDOFF_FACTOR = 64.0


def _max_abs(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().abs().amax().cpu().item())


def validate_tau(tau: float) -> float:
    """Validate an explicitly supplied absolute z-score threshold."""

    if isinstance(tau, bool) or not isinstance(tau, (int, float)):
        raise TypeError("tau must be a finite non-negative number")
    tau = float(tau)
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be a finite non-negative number")
    return tau


def full_prompt_asymmetry_moments(
    attention_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return per-head ``m_p`` and ``q_p`` over the full strict lower triangle."""

    if not isinstance(attention_scores, torch.Tensor) or attention_scores.ndim != 4:
        raise TypeError("attention_scores must have shape batch,head,query,key")
    if not attention_scores.is_floating_point():
        raise TypeError("attention_scores must be floating point")
    batch, _, query_length, key_length = attention_scores.shape
    if batch != 1 or query_length != key_length or query_length < 2:
        raise ValueError("full-prompt moments require one square sequence of length >= 2")
    if not bool(torch.isfinite(attention_scores).all().item()):
        raise ValueError("attention_scores contain NaN or infinity")

    rows, cols = strict_lower_indices(query_length, device=attention_scores.device)
    scores = attention_scores.float()
    asymmetry = (scores - scores.transpose(-1, -2)) / 2.0
    lower = asymmetry[0, :, rows, cols].to(torch.float64)
    element_count = query_length * (query_length - 1) // 2
    if lower.shape[-1] != element_count:
        raise AssertionError("full strict-lower extraction cardinality mismatch")
    prompt_mean = lower.mean(dim=-1)
    prompt_mean_square = lower.square().mean(dim=-1)
    if not bool(torch.isfinite(prompt_mean).all().item()) or not bool(
        torch.isfinite(prompt_mean_square).all().item()
    ):
        raise ValueError("full-prompt moments contain NaN or infinity")
    return prompt_mean, prompt_mean_square, element_count


def _roundoff_tolerance(*values: torch.Tensor) -> torch.Tensor:
    scale = torch.ones_like(values[0], dtype=torch.float64)
    for value in values:
        scale = torch.maximum(scale, value.detach().abs().to(torch.float64))
    return VARIANCE_ROUNDOFF_FACTOR * torch.finfo(torch.float64).eps * scale


def compute_equal_prompt_reference_statistics(
    prompt_means: torch.Tensor,
    prompt_mean_squares: torch.Tensor,
    *,
    expected_prompt_count: int = SELECTION_COUNT,
) -> dict[str, torch.Tensor | int | float]:
    """Compute equal-prompt-weight ``mu``, variance, and ``sigma``.

    The two inputs have axes ``prompt,layer,head``.  Prompt token counts never
    enter the aggregation, which makes every prompt contribute exactly the
    same weight.
    """

    for name, value in (
        ("prompt_means", prompt_means),
        ("prompt_mean_squares", prompt_mean_squares),
    ):
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise TypeError(f"{name} must have shape prompt,layer,head")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} contains NaN or infinity")
    if prompt_means.shape != prompt_mean_squares.shape:
        raise ValueError("prompt moment tensor shapes differ")
    if type(expected_prompt_count) is not int or expected_prompt_count <= 0:
        raise ValueError("expected_prompt_count must be a positive integer")
    if prompt_means.shape[0] != expected_prompt_count:
        raise ValueError(
            "reference prompt count mismatch: "
            f"{prompt_means.shape[0]} != {expected_prompt_count}"
        )

    means64 = prompt_means.to(torch.float64)
    squares64 = prompt_mean_squares.to(torch.float64)
    within_prompt_variance = squares64 - means64.square()
    within_tolerance = _roundoff_tolerance(squares64, means64.square())
    if bool((within_prompt_variance < -within_tolerance).any().item()):
        minimum = float(within_prompt_variance.min().item())
        raise ValueError(
            "prompt mean-square is inconsistent with its mean; "
            f"minimum implied variance={minimum}"
        )

    mu64 = means64.mean(dim=0)
    mean_q64 = squares64.mean(dim=0)
    raw_variance64 = mean_q64 - mu64.square()
    variance_tolerance = _roundoff_tolerance(mean_q64, mu64.square())
    materially_negative = raw_variance64 < -variance_tolerance
    if bool(materially_negative.any().item()):
        minimum = float(raw_variance64.min().item())
        raise ValueError(
            "reference variance is materially negative; "
            f"minimum variance={minimum}"
        )
    clamped_negative_count = int(
        ((raw_variance64 < 0) & ~materially_negative).sum().item()
    )
    variance64 = raw_variance64.clamp_min(0.0)
    sigma64 = torch.sqrt(variance64)

    mu = mu64.to(torch.float32).contiguous()
    mean_q = mean_q64.to(torch.float32).contiguous()
    variance = variance64.to(torch.float32).contiguous()
    sigma = sigma64.to(torch.float32).contiguous()
    for name, value in (
        ("mu", mu),
        ("mean_q", mean_q),
        ("variance", variance),
        ("sigma", sigma),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"computed reference {name} contains NaN or infinity")
    if bool((variance < 0).any().item()) or bool((sigma < 0).any().item()):
        raise AssertionError("computed reference variance/sigma is negative")

    return {
        "mu": mu,
        "mean_q": mean_q,
        "variance": variance,
        "sigma": sigma,
        "prompt_means": means64.contiguous(),
        "prompt_mean_squares": squares64.contiguous(),
        "prompt_count": expected_prompt_count,
        "clamped_negative_variance_count": clamped_negative_count,
        "numerical_sigma_floor": NUMERICAL_SIGMA_FLOOR,
        "sigma_floor_head_count": int((sigma < NUMERICAL_SIGMA_FLOOR).sum().item()),
    }


def load_selection20_original_prompts(
    *,
    split_manifest_path: str | Path,
    original_input_path: str | Path,
) -> tuple[list[dict], dict]:
    """Load the frozen Top-500 Selection20 originals with hash validation."""

    split_manifest_path = Path(split_manifest_path)
    original_input_path = Path(original_input_path)
    manifest = read_json(split_manifest_path)
    validate_split_manifest(manifest)
    if sha256_file(original_input_path) != manifest["original_input_sha256"]:
        raise ValueError("original input hash does not match the Selection20 manifest")
    source_rows = read_jsonl(original_input_path)
    if len(source_rows) != TOTAL_COUNT:
        raise ValueError(f"original input must contain exactly {TOTAL_COUNT} rows")

    records = []
    for selection_position, row_position in enumerate(
        manifest["selection_row_positions"]
    ):
        split_record = manifest["records"][row_position]
        if split_record["split"] != "selection":
            raise ValueError("Selection20 manifest row is not labeled selection")
        source = source_rows[row_position]
        dataset_index = split_record["dataset_index"]
        prompt = source.get("prompt")
        if source.get("index") != dataset_index:
            raise ValueError("Selection20 original row/dataset index mismatch")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Selection20 original prompt is missing")
        prompt_sha256 = sha256_text(prompt)
        if prompt_sha256 != split_record["original_prompt_sha256"]:
            raise ValueError("Selection20 original prompt hash mismatch")
        records.append(
            {
                "selection_position": selection_position,
                "row_position": row_position,
                "dataset_index": dataset_index,
                "prompt_id": split_record["prompt_id"],
                "prompt": prompt,
                "original_prompt_sha256": prompt_sha256,
            }
        )
    if len(records) != SELECTION_COUNT:
        raise ValueError("Selection20 original record count mismatch")
    if [row["dataset_index"] for row in records] != list(
        manifest["selection_dataset_indices"]
    ):
        raise ValueError("Selection20 original ordering mismatch")
    return records, manifest


@dataclass(frozen=True)
class DynamicAsymmetryReference:
    mu: torch.Tensor
    sigma: torch.Tensor
    metadata: Mapping[str, object]


def validate_dynamic_reference_payload(payload: Mapping[str, object]) -> None:
    """Fail fast on malformed or scientifically incompatible references."""

    if not isinstance(payload, Mapping):
        raise TypeError("dynamic reference payload must be a mapping")
    required = {
        "schema_version",
        "mu",
        "sigma",
        "variance",
        "mean_q",
        "prompt_means",
        "prompt_mean_squares",
        "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"dynamic reference payload missing fields: {sorted(missing)}")
    if payload["schema_version"] != REFERENCE_SCHEMA_VERSION:
        raise ValueError("dynamic reference schema version mismatch")
    expected = (NUM_LAYERS, NUM_HEADS)
    for name in ("mu", "sigma", "variance", "mean_q"):
        value = payload[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"dynamic reference {name} shape mismatch")
        if value.dtype != torch.float32 or value.device.type != "cpu":
            raise ValueError(f"dynamic reference {name} must be CPU float32")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"dynamic reference {name} contains NaN or infinity")
    for name in ("prompt_means", "prompt_mean_squares"):
        value = payload[name]
        expected_prompt_shape = (SELECTION_COUNT, NUM_LAYERS, NUM_HEADS)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_prompt_shape:
            raise ValueError(f"dynamic reference {name} shape mismatch")
        if value.dtype != torch.float64 or value.device.type != "cpu":
            raise ValueError(f"dynamic reference {name} must be CPU float64")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"dynamic reference {name} contains NaN or infinity")
    if bool((payload["sigma"] < 0).any().item()) or bool(
        (payload["variance"] < 0).any().item()
    ):
        raise ValueError("dynamic reference has negative sigma/variance")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("dynamic reference metadata must be a mapping")
    expected_metadata = {
        "prompt_weighting": "equal_weight_per_prompt",
        "position_scope": "full_prompt_strict_lower_i_gt_j",
        "component": "A_asym=(S-S^T)/2",
        "prompt_count": SELECTION_COUNT,
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"dynamic reference metadata mismatch for {key}")
    condition = metadata.get("reference_condition")
    if condition == "selection20_original_only":
        if metadata.get("jailbreak_reference_data_used") is not False:
            raise ValueError("GCG original reference metadata mismatch")
    elif condition == "autodan_selection20_official_jailbreak_only":
        if (
            metadata.get("jailbreak_reference_data_used") is not True
            or metadata.get("prompt_format") != "official_fastchat_34062e9"
        ):
            raise ValueError("AutoDAN reference metadata mismatch")
    else:
        raise ValueError("dynamic reference condition is unsupported")

    recomputed = compute_equal_prompt_reference_statistics(
        payload["prompt_means"],
        payload["prompt_mean_squares"],
    )
    for name in ("mu", "sigma", "variance", "mean_q"):
        torch.testing.assert_close(
            payload[name], recomputed[name], rtol=1e-6, atol=1e-7
        )


def load_dynamic_reference(path: str | Path) -> DynamicAsymmetryReference:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_dynamic_reference_payload(payload)
    return DynamicAsymmetryReference(
        mu=payload["mu"].contiguous(),
        sigma=payload["sigma"].contiguous(),
        metadata=dict(payload["metadata"]),
    )


class FullPromptAsymmetryMomentCollector:
    """Collect one ``m_p`` and ``q_p`` value per Layer/Head during prefill."""

    def __init__(
        self,
        *,
        layer_count: int = NUM_LAYERS,
        head_count: int = NUM_HEADS,
        expected_score_dtype: torch.dtype | None = None,
    ):
        if type(layer_count) is not int or layer_count <= 0:
            raise ValueError("layer_count must be a positive integer")
        if type(head_count) is not int or head_count <= 0:
            raise ValueError("head_count must be a positive integer")
        self.layer_count = layer_count
        self.head_count = head_count
        self.expected_score_dtype = expected_score_dtype
        self.clear()

    def clear(self) -> None:
        self.prompt_length: int | None = None
        self._means: dict[int, torch.Tensor] = {}
        self._mean_squares: dict[int, torch.Tensor] = {}
        self._element_counts: dict[int, int] = {}

    def set_prompt_context(self, *, prompt_length: int) -> None:
        if self._means:
            raise ReferenceInterventionRuntimeError(
                "collector prompt context must be set before attention forwards"
            )
        if type(prompt_length) is not int or prompt_length < 2:
            raise ValueError("prompt_length must be an integer >= 2")
        self.prompt_length = prompt_length

    def apply(self, layer: int, attention_scores: torch.Tensor) -> torch.Tensor:
        if self.prompt_length is None:
            raise ReferenceInterventionRuntimeError("collector prompt context is missing")
        if type(layer) is not int or not 0 <= layer < self.layer_count:
            raise ReferenceInterventionRuntimeError("collector layer index is invalid")
        if layer in self._means:
            raise ReferenceInterventionRuntimeError("collector saw a layer more than once")
        if not isinstance(attention_scores, torch.Tensor) or attention_scores.ndim != 4:
            raise ReferenceInterventionRuntimeError(
                "collector attention scores must have shape batch,head,query,key"
            )
        if attention_scores.shape[-2:] != (self.prompt_length, self.prompt_length):
            raise ReferenceInterventionRuntimeError(
                "collector score shape does not match the full prompt length"
            )
        if attention_scores.shape[1] != self.head_count:
            raise ReferenceInterventionRuntimeError("collector head count mismatch")
        if (
            self.expected_score_dtype is not None
            and attention_scores.dtype != self.expected_score_dtype
        ):
            raise ReferenceInterventionRuntimeError("collector attention dtype mismatch")
        means, mean_squares, count = full_prompt_asymmetry_moments(attention_scores)
        self._means[layer] = means.detach().cpu()
        self._mean_squares[layer] = mean_squares.detach().cpu()
        self._element_counts[layer] = count
        return attention_scores

    def result(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        expected_layers = set(range(self.layer_count))
        if set(self._means) != expected_layers or set(self._mean_squares) != expected_layers:
            raise ReferenceInterventionRuntimeError(
                "collector did not observe every attention layer exactly once"
            )
        counts = set(self._element_counts.values())
        if len(counts) != 1:
            raise ReferenceInterventionRuntimeError(
                "collector strict-lower element counts differ across layers"
            )
        means = torch.stack([self._means[layer] for layer in range(self.layer_count)])
        mean_squares = torch.stack(
            [self._mean_squares[layer] for layer in range(self.layer_count)]
        )
        expected_shape = (self.layer_count, self.head_count)
        if tuple(means.shape) != expected_shape or tuple(mean_squares.shape) != expected_shape:
            raise AssertionError("collector moment shape mismatch")
        return means.contiguous(), mean_squares.contiguous(), counts.pop()


class DynamicAsymmetryController:
    """Select full-prompt anomalies and apply a pure antisymmetric pair edit."""

    def __init__(
        self,
        *,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        tau: float,
        alpha: float = 1.0,
        expected_score_dtype: torch.dtype | None = None,
        capture_score_change_metrics: bool = False,
    ):
        if not isinstance(mu, torch.Tensor) or not isinstance(sigma, torch.Tensor):
            raise TypeError("mu and sigma must be torch tensors")
        if mu.ndim != 2 or sigma.shape != mu.shape:
            raise ValueError("mu and sigma must have identical layer,head shapes")
        for name, value in (("mu", mu), ("sigma", sigma)):
            if value.dtype != torch.float32 or value.device.type != "cpu":
                raise ValueError(f"{name} must be CPU float32")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} contains NaN or infinity")
        if bool((sigma < 0).any().item()):
            raise ValueError("sigma must be non-negative")
        self.mu = mu.contiguous()
        self.sigma = sigma.contiguous()
        self.tau = validate_tau(tau)
        self.alpha = validate_alpha(alpha)
        self.expected_score_dtype = expected_score_dtype
        if type(capture_score_change_metrics) is not bool:
            raise TypeError("capture_score_change_metrics must be bool")
        self.capture_score_change_metrics = capture_score_change_metrics
        self.layer_count, self.head_count = mu.shape
        self._reference_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._tril_cache: dict[tuple[int, str], torch.Tensor] = {}
        self.sigma_floor_head_count = int(
            (self.sigma < NUMERICAL_SIGMA_FLOOR).sum().item()
        )
        self.clear()

    def clear(self) -> None:
        self.prompt_length: int | None = None
        self.attention_forward_call_count = 0
        self.intervention_call_count = 0
        self.selected_element_intervention_count = 0
        self.alpha_zero_bypass_count = 0
        self.first_forward_detected = False
        self.first_forward_intervened = False
        self.later_forward_intervention_count = 0
        self.later_forward_skip_count = 0
        self._prefill_layers_seen: set[int] = set()
        self._records: list[dict] = []
        self._selected_score_change_abs_sum = 0.0
        self._selected_score_change_sq_sum = 0.0
        self._selected_score_change_max_abs = 0.0

    def set_prompt_context(self, *, prompt_length: int) -> None:
        if self.attention_forward_call_count or self.intervention_call_count:
            raise ReferenceInterventionRuntimeError(
                "prompt context must be set before attention forwards"
            )
        if type(prompt_length) is not int or prompt_length < 2:
            raise ValueError("prompt_length must be an integer >= 2")
        self.prompt_length = prompt_length

    def _layer_reference(
        self, layer: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer, str(device))
        cached = self._reference_cache.get(key)
        if cached is None:
            cached = (
                self.mu[layer].to(device=device),
                self.sigma[layer].to(device=device).clamp_min(NUMERICAL_SIGMA_FLOOR),
            )
            self._reference_cache[key] = cached
        return cached

    def _tril(self, length: int, device: torch.device) -> torch.Tensor:
        key = (length, str(device))
        cached = self._tril_cache.get(key)
        if cached is None:
            cached = strict_lower_indices(length, device=device)
            self._tril_cache[key] = cached
        return cached

    def _selection_mask(
        self,
        *,
        layer: int,
        z_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Return the selected lower-triangle elements for this layer."""

        del layer
        return z_scores.abs() > self.tau

    def apply(self, layer: int, attention_scores: torch.Tensor) -> torch.Tensor:
        self.attention_forward_call_count += 1
        if type(layer) is not int or not 0 <= layer < self.layer_count:
            raise ReferenceInterventionRuntimeError("dynamic intervention layer is invalid")
        if not isinstance(attention_scores, torch.Tensor) or attention_scores.ndim != 4:
            raise ReferenceInterventionRuntimeError(
                "attention scores must have shape batch,head,query,key"
            )
        if not attention_scores.is_floating_point():
            raise ReferenceInterventionRuntimeError("attention scores must be floating point")
        if (
            self.expected_score_dtype is not None
            and attention_scores.dtype != self.expected_score_dtype
        ):
            raise ReferenceInterventionRuntimeError("attention score dtype mismatch")
        batch, heads, query_length, key_length = attention_scores.shape
        if batch != 1 or heads != self.head_count or query_length != key_length:
            raise ReferenceInterventionRuntimeError("attention score shape mismatch")
        if self.prompt_length is None:
            raise ReferenceInterventionRuntimeError(
                "dynamic intervention requires prompt context"
            )
        if query_length < self.prompt_length:
            raise ReferenceInterventionRuntimeError(
                "current sequence length is shorter than prompt length"
            )
        if query_length > self.prompt_length:
            self.later_forward_skip_count += 1
            return attention_scores
        if layer in self._prefill_layers_seen:
            raise ReferenceInterventionRuntimeError(
                "dynamic intervention saw a prefill layer more than once"
            )
        self._prefill_layers_seen.add(layer)
        self.first_forward_detected = True
        if not bool(torch.isfinite(attention_scores).all().item()):
            raise ReferenceInterventionRuntimeError(
                "pre-intervention attention scores contain NaN or infinity"
            )
        if self.alpha == 0.0:
            self.alpha_zero_bypass_count += 1
            return attention_scores

        rows, cols = self._tril(query_length, attention_scores.device)
        expected_elements = query_length * (query_length - 1) // 2
        if rows.numel() != expected_elements or not bool((rows > cols).all().item()):
            raise ReferenceInterventionRuntimeError(
                "dynamic intervention did not obtain the full strict lower triangle"
            )
        original = attention_scores.float()
        current_asym = (original - original.transpose(-1, -2)) / 2.0
        current_lower = current_asym[:, :, rows, cols]
        mu, safe_sigma = self._layer_reference(layer, attention_scores.device)
        z_scores = (current_lower - mu.view(1, -1, 1)) / safe_sigma.view(1, -1, 1)
        if not bool(torch.isfinite(z_scores).all().item()):
            raise ReferenceInterventionRuntimeError(
                "dynamic intervention z-scores contain NaN or infinity"
            )
        selected = self._selection_mask(layer=layer, z_scores=z_scores)
        if selected.dtype != torch.bool or selected.shape != z_scores.shape:
            raise ReferenceInterventionRuntimeError(
                "dynamic intervention selection mask shape/dtype mismatch"
            )
        selected_count = int(selected.sum().item())
        component_delta = self.alpha * (mu.view(1, -1, 1) - current_lower)
        component_delta = torch.where(selected, component_delta, 0.0)
        constructed_delta = torch.zeros_like(original)
        constructed_delta[:, :, rows, cols] = component_delta
        constructed_delta[:, :, cols, rows] = -component_delta
        intervened_float32 = original + constructed_delta
        intervened_cast = intervened_float32.to(attention_scores.dtype)

        original64 = original.to(torch.float64)
        intervened64 = intervened_float32.to(torch.float64)
        constructed_delta64 = constructed_delta.to(torch.float64)
        before_sym64 = (original64 + original64.transpose(-1, -2)) / 2.0
        original_asym64 = (original64 - original64.transpose(-1, -2)) / 2.0
        current_asym_calculation_roundoff = _max_abs(
            current_lower.to(torch.float64) - original_asym64[:, :, rows, cols]
        )
        constructed_sym64 = (
            intervened64 + intervened64.transpose(-1, -2)
        ) / 2.0
        expected_lower64 = current_lower.to(torch.float64) + component_delta.to(
            torch.float64
        )
        constructed_asym64 = (
            intervened64 - intervened64.transpose(-1, -2)
        ) / 2.0
        constructed_lower64 = constructed_asym64[:, :, rows, cols]
        float32_addition_roundoff = _max_abs(
            intervened64 - (original64 + constructed_delta64)
        )
        antisymmetry_error = _max_abs(
            constructed_delta + constructed_delta.transpose(-1, -2)
        )
        symmetric_error = _max_abs(constructed_sym64 - before_sym64)
        target_error = _max_abs(
            (constructed_lower64 - expected_lower64).masked_select(selected)
        )
        if antisymmetry_error != 0.0:
            raise ReferenceInterventionRuntimeError(
                "pure asymmetric float32 construction invariant failed"
            )
        if symmetric_error > float32_addition_roundoff:
            raise ReferenceInterventionRuntimeError(
                "float32 symmetric error exceeds score-addition roundoff"
            )
        if target_error > (
            current_asym_calculation_roundoff + float32_addition_roundoff
        ):
            raise ReferenceInterventionRuntimeError(
                "float32 Asym target error exceeds component plus addition roundoff"
            )

        modified = attention_scores.clone()
        modified.copy_(intervened_cast)
        if not bool(torch.isfinite(modified).all().item()):
            raise ReferenceInterventionRuntimeError(
                "post-intervention attention scores contain NaN or infinity"
            )
        realized_delta = modified.float() - original
        selected_pair_mask = torch.zeros(
            (1, self.head_count, query_length, query_length),
            dtype=torch.bool,
            device=attention_scores.device,
        )
        selected_pair_mask[:, :, rows, cols] = selected
        selected_pair_mask[:, :, cols, rows] = selected
        unselected_change = _max_abs(
            realized_delta.masked_select(~selected_pair_mask)
        )
        diagonal_change = _max_abs(
            torch.diagonal(realized_delta, dim1=-2, dim2=-1)
        )
        if unselected_change != 0.0 or diagonal_change != 0.0:
            raise ReferenceInterventionRuntimeError(
                "dynamic intervention changed an unselected or diagonal score"
            )

        realized = modified.float()
        realized64 = realized.to(torch.float64)
        if self.capture_score_change_metrics and selected_count:
            score_delta = (
                realized[:, :, rows, cols] - original[:, :, rows, cols]
            ).masked_select(selected).to(torch.float64)
            self._selected_score_change_abs_sum += float(score_delta.abs().sum())
            self._selected_score_change_sq_sum += float((score_delta * score_delta).sum())
            self._selected_score_change_max_abs = max(
                self._selected_score_change_max_abs, float(score_delta.abs().max())
            )
        realized_sym64 = (realized64 + realized64.transpose(-1, -2)) / 2.0
        realized_asym = (realized - realized.transpose(-1, -2)) / 2.0
        realized_lower = realized_asym[:, :, rows, cols]
        selected_before_distance = _max_abs(
            (current_lower - mu.view(1, -1, 1)).masked_select(selected)
        )
        selected_after_distance_float32 = _max_abs(
            (
                constructed_lower64
                - mu.view(1, -1, 1).to(torch.float64)
            ).masked_select(selected)
        )
        selected_after_distance_after_cast = _max_abs(
            (realized_lower - mu.view(1, -1, 1)).masked_select(selected)
        )
        cast_quantization_error = _max_abs(realized64 - intervened64)
        symmetric_error_after_cast = _max_abs(realized_sym64 - before_sym64)
        if symmetric_error_after_cast > (
            symmetric_error + cast_quantization_error
        ):
            raise ReferenceInterventionRuntimeError(
                "post-cast symmetric error exceeds construction plus cast roundoff"
            )

        self.intervention_call_count += 1
        self.selected_element_intervention_count += selected_count
        if selected_count:
            self.first_forward_intervened = True
        self._records.append(
            {
                "layer": layer,
                "sequence_length": query_length,
                "strict_lower_element_count_per_head": expected_elements,
                "candidate_element_count": self.head_count * expected_elements,
                "selected_element_count": selected_count,
                "max_absolute_z_score": _max_abs(z_scores),
                "max_constructed_delta_antisymmetry_error_float32": antisymmetry_error,
                "max_float32_score_addition_roundoff": float32_addition_roundoff,
                "max_float32_asym_calculation_roundoff": (
                    current_asym_calculation_roundoff
                ),
                "max_symmetric_component_error_float32": symmetric_error,
                "max_symmetric_component_error_after_cast": symmetric_error_after_cast,
                "max_score_cast_quantization_error": cast_quantization_error,
                "max_asym_expected_error_float32": target_error,
                "max_unselected_element_change": unselected_change,
                "max_diagonal_change_float32": diagonal_change,
                "max_selected_distance_to_mu_before": selected_before_distance,
                "max_selected_distance_to_mu_after_float32": (
                    selected_after_distance_float32
                ),
                "max_selected_distance_to_mu_after_cast": (
                    selected_after_distance_after_cast
                ),
            }
        )
        return modified

    def assert_complete_prefill(self) -> None:
        if self._prefill_layers_seen != set(range(self.layer_count)):
            raise ReferenceInterventionRuntimeError(
                "dynamic intervention did not process every layer during prefill"
            )
        if self.later_forward_intervention_count != 0:
            raise ReferenceInterventionRuntimeError(
                "dynamic intervention unexpectedly ran after prefill"
            )

    def summary(self) -> dict:
        def maximum(key: str) -> float:
            return max((float(row[key]) for row in self._records), default=0.0)

        captured_count = self.selected_element_intervention_count
        score_metrics = None
        if self.capture_score_change_metrics:
            if captured_count <= 0 or self._selected_score_change_sq_sum <= 0.0:
                raise ReferenceInterventionRuntimeError(
                    "captured selected score changes are empty"
                )
            score_metrics = {
                "count": captured_count,
                "mean_absolute": self._selected_score_change_abs_sum / captured_count,
                "rms": math.sqrt(self._selected_score_change_sq_sum / captured_count),
                "l2": math.sqrt(self._selected_score_change_sq_sum),
                "max_absolute": self._selected_score_change_max_abs,
            }
        return {
            "component": "antisymmetric",
            "intervention_variant": "dynamic_full_prompt_zscore_pure_asym",
            "position_scope": "full_prompt_strict_lower_i_gt_j",
            "selection_rule": "abs(z) > tau",
            "delta_definition": "alpha*(mu_lh-A_asym_ij)",
            "raw_score_intervention": "S_ij += delta; S_ji -= delta",
            "tau": self.tau,
            "alpha": self.alpha,
            "intervention_timing": "first_prefill_only",
            "prompt_length": self.prompt_length,
            "layer_count": self.layer_count,
            "head_count": self.head_count,
            "prefill_layer_count": len(self._prefill_layers_seen),
            "attention_forward_call_count": self.attention_forward_call_count,
            "intervention_call_count": self.intervention_call_count,
            "selected_element_intervention_count": self.selected_element_intervention_count,
            "alpha_zero_bypass_count": self.alpha_zero_bypass_count,
            "first_forward_detected": self.first_forward_detected,
            "first_forward_intervened": self.first_forward_intervened,
            "later_forward_intervention_count": self.later_forward_intervention_count,
            "later_forward_skip_count": self.later_forward_skip_count,
            "numerical_sigma_floor": NUMERICAL_SIGMA_FLOOR,
            "sigma_floor_head_count": self.sigma_floor_head_count,
            "strict_lower_element_count_per_head": (
                self.prompt_length * (self.prompt_length - 1) // 2
                if self.prompt_length is not None
                else None
            ),
            "candidate_element_count": sum(
                int(row["candidate_element_count"]) for row in self._records
            ),
            "max_absolute_z_score": maximum("max_absolute_z_score"),
            "max_constructed_delta_antisymmetry_error_float32": maximum(
                "max_constructed_delta_antisymmetry_error_float32"
            ),
            "max_float32_score_addition_roundoff": maximum(
                "max_float32_score_addition_roundoff"
            ),
            "max_float32_asym_calculation_roundoff": maximum(
                "max_float32_asym_calculation_roundoff"
            ),
            "max_symmetric_component_error_float32": maximum(
                "max_symmetric_component_error_float32"
            ),
            "max_symmetric_component_error_after_cast": maximum(
                "max_symmetric_component_error_after_cast"
            ),
            "max_score_cast_quantization_error": maximum(
                "max_score_cast_quantization_error"
            ),
            "max_asym_expected_error_float32": maximum(
                "max_asym_expected_error_float32"
            ),
            "max_unselected_element_change": maximum(
                "max_unselected_element_change"
            ),
            "max_diagonal_change_float32": maximum(
                "max_diagonal_change_float32"
            ),
            "max_selected_distance_to_mu_before": maximum(
                "max_selected_distance_to_mu_before"
            ),
            "max_selected_distance_to_mu_after_float32": maximum(
                "max_selected_distance_to_mu_after_float32"
            ),
            "max_selected_distance_to_mu_after_cast": maximum(
                "max_selected_distance_to_mu_after_cast"
            ),
            "all_scores_finite": True,
            "selected_raw_score_change": score_metrics,
        }


def _patch_llama_attention_modules(model, controller):
    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected Transformers {SUPPORTED_TRANSFORMERS_VERSION}, "
            f"found {transformers.__version__}"
        )
    if len(model.model.layers) != controller.layer_count:
        raise ValueError("model/reference layer count mismatch")
    patched = []
    try:
        for layer_index in range(controller.layer_count):
            module = model.model.layers[layer_index].self_attn
            if type(module) is not modeling_llama.LlamaAttention:
                raise TypeError(
                    "dynamic asymmetry requires eager LlamaAttention; "
                    f"layer={layer_index}, found={type(module).__name__}"
                )
            if (
                module.layer_idx != layer_index
                or module.num_heads != controller.head_count
            ):
                raise ValueError(f"attention structure mismatch at layer={layer_index}")
            had_instance_forward = "forward" in module.__dict__
            previous_instance_forward = module.__dict__.get("forward")
            replacement = _patched_llama_attention_forward(controller)
            module.forward = types.MethodType(replacement, module)
            patched.append((module, had_instance_forward, previous_instance_forward))
    except Exception:
        _restore_llama_attention_modules(patched)
        raise
    return patched


def _restore_llama_attention_modules(patched) -> None:
    for module, had_instance_forward, previous_instance_forward in reversed(patched):
        if had_instance_forward:
            module.forward = previous_instance_forward
        else:
            module.__dict__.pop("forward", None)


@contextmanager
def patch_llama_full_prompt_moment_collection(model):
    """Capture full-prompt reference moments at the intervention score site."""

    controller = FullPromptAsymmetryMomentCollector(
        layer_count=len(model.model.layers),
        head_count=model.config.num_attention_heads,
        expected_score_dtype=torch.float16,
    )
    patched = []
    try:
        patched = _patch_llama_attention_modules(model, controller)
        yield controller
    finally:
        _restore_llama_attention_modules(patched)


@contextmanager
def patch_llama_dynamic_asymmetry_attention(
    model,
    *,
    reference: DynamicAsymmetryReference,
    tau: float,
    alpha: float = 1.0,
):
    """Install the first-prefill-only dynamic pure-Asym intervention."""

    controller = DynamicAsymmetryController(
        mu=reference.mu,
        sigma=reference.sigma,
        tau=tau,
        alpha=alpha,
        expected_score_dtype=torch.float16,
    )
    patched = []
    try:
        patched = _patch_llama_attention_modules(model, controller)
        yield controller
    finally:
        _restore_llama_attention_modules(patched)
