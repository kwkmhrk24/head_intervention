"""Runtime tail-block interpolation toward an original S_asym reference.

This module is intentionally independent from ``utils.asym_score_intervention``'s
beta-suppression patch.  It reproduces the audited Transformers 4.42.3 eager
Llama attention forward and changes only selected heads' final 16x16 raw-score
blocks after RoPE/repeat_kv and before the causal mask and softmax.
"""

from __future__ import annotations

import math
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama import modeling_llama

from utils.asym_score_intervention import (
    SUPPORTED_TRANSFORMERS_VERSION,
    UnsupportedAttentionCacheError,
    validate_layer_head_pairs,
)
from utils.gcg_wasserstein_reference import (
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    TAIL_TOKENS,
    llama_model_config_summary,
    read_json,
    sha256_file,
    strict_lower_indices,
    validate_reference_payload,
    verify_artifact_directory,
)


EXPECTED_SCORE_CAPTURE_POSITION = (
    "post-RoPE, after repeat_kv, immediately after QK^T/sqrt(head_dim), "
    "pre-mask, pre-softmax"
)
INTERVENTION_TIMINGS = ("dynamic_all_steps", "first_token_only")


class ReferenceInterventionRuntimeError(RuntimeError):
    """Raised when a runtime invariant of the reference intervention fails."""


def validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a real number")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must satisfy 0.0 <= alpha <= 1.0")
    return alpha


def validate_intervention_timing(intervention_timing: str) -> str:
    if intervention_timing not in INTERVENTION_TIMINGS:
        raise ValueError(
            f"intervention_timing must be one of {INTERVENTION_TIMINGS}, "
            f"found {intervention_timing!r}"
        )
    return intervention_timing


def _max_abs(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().abs().amax().float().cpu().item())


def build_antisymmetric_reference_matrix(
    lower_values: torch.Tensor,
    *,
    tail_tokens: int = TAIL_TOKENS,
    tril_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand strict-lower values into antisymmetric square matrices."""

    if not isinstance(lower_values, torch.Tensor):
        raise TypeError("lower_values must be a torch tensor")
    if not lower_values.is_floating_point():
        raise TypeError("lower_values must use a floating-point dtype")
    expected = tail_tokens * (tail_tokens - 1) // 2
    if lower_values.ndim < 1 or lower_values.shape[-1] != expected:
        raise ValueError(
            f"strict-lower value shape mismatch: last dimension must be {expected}"
        )
    if not bool(torch.isfinite(lower_values).all().item()):
        raise ValueError("lower_values contain NaN or infinity")
    if tril_indices is None:
        tril_indices = strict_lower_indices(
            tail_tokens,
            device=lower_values.device,
        )
    if not isinstance(tril_indices, torch.Tensor) or tuple(tril_indices.shape) != (
        2,
        expected,
    ):
        raise ValueError("tril_indices shape mismatch")
    expected_indices = strict_lower_indices(
        tail_tokens,
        device=tril_indices.device,
    )
    if not torch.equal(tril_indices, expected_indices):
        raise ValueError("tril_indices order does not match torch.tril_indices")
    if tril_indices.device != lower_values.device:
        tril_indices = tril_indices.to(lower_values.device)

    matrix = torch.zeros(
        (*lower_values.shape[:-1], tail_tokens, tail_tokens),
        dtype=lower_values.dtype,
        device=lower_values.device,
    )
    rows, cols = tril_indices
    matrix[..., rows, cols] = lower_values
    matrix[..., cols, rows] = -lower_values
    return matrix


@dataclass(frozen=True)
class ValidatedReferenceArtifacts:
    artifact_dir: Path
    top5_heads: tuple[tuple[int, int], ...]
    top5_records: tuple[dict, ...]
    reference_mean: torch.Tensor
    reference_metadata: dict
    verification: dict


def validate_runtime_artifact_payloads(
    *,
    verification: Mapping[str, object],
    top5_payload: Mapping[str, object],
    reference_payload: Mapping[str, object],
    expected_heads: Sequence[tuple[int, int]],
    expected_selection_manifest_sha256: str,
    expected_score_artifact_sha256: str,
    expected_model_path: str | Path,
) -> tuple[tuple[int, int], ...]:
    """Validate cross-artifact runtime requirements without loading a model."""

    if verification.get("verified") is not True:
        raise ValueError("artifact directory verification did not succeed")
    for key, expected in (
        ("selection_manifest_sha256", expected_selection_manifest_sha256),
        ("score_artifact_sha256", expected_score_artifact_sha256),
    ):
        if verification.get(key) != expected:
            raise ValueError(
                f"artifact verification {key} mismatch: "
                f"{verification.get(key)!r} != {expected!r}"
            )

    heads = top5_payload.get("heads")
    if not isinstance(heads, list) or len(heads) != 5:
        raise ValueError("top5_heads.json must contain exactly five heads")
    loaded_pairs = []
    for expected_rank, record in enumerate(heads, 1):
        if not isinstance(record, Mapping):
            raise ValueError("top5 head record must be an object")
        layer, head = record.get("layer"), record.get("head")
        if type(layer) is not int or type(head) is not int:
            raise ValueError("top5 layer/head must be integers")
        if record.get("rank") != expected_rank:
            raise ValueError("top5 head ranks must be 1 through 5 in order")
        loaded_pairs.append((layer, head))
    loaded_pairs = tuple(loaded_pairs)
    if len(set(loaded_pairs)) != 5:
        raise ValueError("top5 heads contain duplicates")
    if loaded_pairs != tuple(expected_heads):
        raise ValueError(
            f"top5 heads mismatch: loaded={loaded_pairs}, "
            f"expected={tuple(expected_heads)}"
        )
    if top5_payload.get("selection_manifest_sha256") != (
        expected_selection_manifest_sha256
    ):
        raise ValueError("top5 selection manifest hash mismatch")
    if top5_payload.get("score_artifact_sha256") != expected_score_artifact_sha256:
        raise ValueError("top5 score artifact hash mismatch")

    validate_reference_payload(reference_payload)
    metadata = reference_payload["metadata"]
    if metadata.get("tail_tokens") != TAIL_TOKENS:
        raise ValueError("reference tail_tokens mismatch")
    if metadata.get("selection_manifest_sha256") != (
        expected_selection_manifest_sha256
    ):
        raise ValueError("reference selection manifest hash mismatch")
    if metadata.get("score_artifact_sha256") != expected_score_artifact_sha256:
        raise ValueError("reference score artifact hash mismatch")
    if metadata.get("attention_implementation") != "eager":
        raise ValueError("reference attention implementation is not eager")
    if metadata.get("score_capture_position") != EXPECTED_SCORE_CAPTURE_POSITION:
        raise ValueError("reference score capture position mismatch")
    if Path(metadata.get("model_path", "")).resolve() != Path(
        expected_model_path
    ).resolve():
        raise ValueError("reference model path mismatch")

    expected_tril = strict_lower_indices(TAIL_TOKENS, device="cpu")
    stored_tril = torch.stack(
        (reference_payload["tril_rows"], reference_payload["tril_cols"])
    ).cpu()
    if not torch.equal(stored_tril, expected_tril):
        raise ValueError("reference tril index order mismatch")
    return loaded_pairs


def load_validated_reference_artifacts(
    artifact_dir: str | Path,
    *,
    expected_heads: Sequence[tuple[int, int]],
    expected_selection_manifest_sha256: str,
    expected_score_artifact_sha256: str,
    expected_model_path: str | Path,
) -> ValidatedReferenceArtifacts:
    artifact_dir = Path(artifact_dir).resolve()
    verification = verify_artifact_directory(artifact_dir)
    top5_payload = read_json(artifact_dir / "top5_heads.json")
    reference_payload = torch.load(
        artifact_dir / "original_reference.pt",
        map_location="cpu",
        weights_only=True,
    )
    loaded_pairs = validate_runtime_artifact_payloads(
        verification=verification,
        top5_payload=top5_payload,
        reference_payload=reference_payload,
        expected_heads=expected_heads,
        expected_selection_manifest_sha256=expected_selection_manifest_sha256,
        expected_score_artifact_sha256=expected_score_artifact_sha256,
        expected_model_path=expected_model_path,
    )
    reference_mean = reference_payload["mean"].contiguous()
    return ValidatedReferenceArtifacts(
        artifact_dir=artifact_dir,
        top5_heads=loaded_pairs,
        top5_records=tuple(dict(record) for record in top5_payload["heads"]),
        reference_mean=reference_mean,
        reference_metadata=dict(reference_payload["metadata"]),
        verification=dict(verification),
    )


def validate_loaded_llama_model(
    model,
    artifacts: ValidatedReferenceArtifacts,
) -> dict:
    """Validate model configuration/dtype/eager class against the reference."""

    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected Transformers {SUPPORTED_TRANSFORMERS_VERSION}, "
            f"found {transformers.__version__}"
        )
    config_summary = llama_model_config_summary(model)
    if config_summary != artifacts.reference_metadata["model_config"]:
        raise ValueError("loaded model config does not match reference metadata")
    if len(model.model.layers) != NUM_LAYERS:
        raise ValueError("loaded model layer count mismatch")
    if model.config._attn_implementation != "eager":
        raise ValueError("loaded model attention implementation is not eager")
    if model.config.use_cache is not False:
        raise ValueError("loaded model config.use_cache must be False")
    parameter_dtype = next(model.parameters()).dtype
    if parameter_dtype != torch.float16:
        raise ValueError(f"loaded model dtype is not float16: {parameter_dtype}")
    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        if type(attention) is not modeling_llama.LlamaAttention:
            raise TypeError(
                f"expected eager LlamaAttention at layer={layer_index}, "
                f"found={type(attention).__name__}"
            )
        if attention.layer_idx != layer_index:
            raise ValueError(f"attention layer_idx mismatch at layer={layer_index}")
        if attention.num_heads != NUM_HEADS:
            raise ValueError(f"attention head count mismatch at layer={layer_index}")
    return {
        "transformers_version": transformers.__version__,
        "model_config": config_summary,
        "model_parameter_dtype": str(parameter_dtype),
        "attention_implementation": model.config._attn_implementation,
        "attention_class": type(model.model.layers[0].self_attn).__name__,
    }


class ReferenceAsymmetryController:
    """Apply and diagnose selected-layer reference interpolation."""

    def __init__(
        self,
        *,
        reference_mean: torch.Tensor,
        layer_head_pairs: Sequence[tuple[int, int]],
        alpha: float,
        tail_tokens: int = TAIL_TOKENS,
        expected_score_dtype: torch.dtype | None = None,
        intervention_timing: str = "dynamic_all_steps",
    ):
        self.alpha = validate_alpha(alpha)
        self.intervention_timing = validate_intervention_timing(
            intervention_timing
        )
        if not isinstance(reference_mean, torch.Tensor):
            raise TypeError("reference_mean must be a torch tensor")
        if reference_mean.ndim != 3:
            raise ValueError("reference_mean must have shape layer,head,position")
        if reference_mean.dtype != torch.float32 or reference_mean.device.type != "cpu":
            raise ValueError("reference_mean must be CPU float32")
        if reference_mean.shape[-1] != tail_tokens * (tail_tokens - 1) // 2:
            raise ValueError("reference_mean position dimension mismatch")
        if not bool(torch.isfinite(reference_mean).all().item()):
            raise ValueError("reference_mean contains NaN or infinity")
        self.tail_tokens = tail_tokens
        self.layer_count = int(reference_mean.shape[0])
        self.head_count = int(reference_mean.shape[1])
        self.selected_by_layer = validate_layer_head_pairs(
            layer_head_pairs,
            layer_count=self.layer_count,
            head_count=self.head_count,
        )
        self.layer_head_pairs = tuple(
            (layer, head)
            for layer, heads in self.selected_by_layer.items()
            for head in heads
        )
        self.expected_score_dtype = expected_score_dtype
        full_reference = build_antisymmetric_reference_matrix(
            reference_mean,
            tail_tokens=tail_tokens,
        )
        self._reference_cpu = {
            (layer, head): full_reference[layer, head].contiguous()
            for layer, head in self.layer_head_pairs
        }
        self._reference_cache: dict[tuple[int, int, str], torch.Tensor] = {}
        self._reference_cache_transfer_count = 0
        self.clear()

    def clear(self) -> None:
        self.attention_forward_call_count = 0
        self.intervention_call_count = 0
        self.alpha_zero_bypass_count = 0
        self.short_sequence_skip_count = 0
        self.prompt_length: int | None = None
        self.first_forward_detected = False
        self.first_forward_intervened = False
        self.later_forward_intervention_count = 0
        self.later_forward_skip_count = 0
        self._observed_sequence_lengths: set[int] = set()
        self._per_head_intervention_count = {
            (layer, head): 0 for layer, head in self.layer_head_pairs
        }
        self._head_records: list[dict] = []
        self._layer_records: list[dict] = []
        self._short_sequence_skip_by_head = {
            (layer, head): 0 for layer, head in self.layer_head_pairs
        }
        self._cache_transfer_count_at_clear = self._reference_cache_transfer_count

    def set_prompt_context(
        self,
        *,
        prompt_length: int,
        intervention_timing: str | None = None,
    ) -> None:
        """Set prompt-specific timing state before any selected-layer forward."""

        if self.attention_forward_call_count != 0 or self.intervention_call_count != 0:
            raise ReferenceInterventionRuntimeError(
                "prompt context must be set before attention forwards"
            )
        if type(prompt_length) is not int:
            raise TypeError("prompt_length must be an integer")
        if prompt_length < self.tail_tokens:
            raise ValueError(
                f"prompt_length must be at least {self.tail_tokens}, "
                f"found {prompt_length}"
            )
        if intervention_timing is not None:
            self.intervention_timing = validate_intervention_timing(
                intervention_timing
            )
        self.prompt_length = prompt_length

    def head_metric_records(self) -> list[dict]:
        """Return detached scalar-only call diagnostics for external aggregation."""

        return [dict(record) for record in self._head_records]

    def _reference(self, layer: int, head: int, device: torch.device) -> torch.Tensor:
        key = (layer, head, str(device))
        value = self._reference_cache.get(key)
        if value is None:
            value = self._reference_cpu[(layer, head)].to(
                device=device,
                dtype=torch.float32,
            )
            self._reference_cache[key] = value
            self._reference_cache_transfer_count += 1
        return value

    def apply(self, layer: int, attention_scores: torch.Tensor) -> torch.Tensor:
        selected_heads = self.selected_by_layer.get(layer)
        if selected_heads is None:
            return attention_scores
        self.attention_forward_call_count += 1

        # Exact identity path: no shape checks, clone, dtype conversion, or arithmetic.
        if self.alpha == 0.0:
            if (
                isinstance(attention_scores, torch.Tensor)
                and attention_scores.ndim >= 2
            ):
                current_length = int(attention_scores.shape[-1])
                self._observed_sequence_lengths.add(current_length)
                if self.prompt_length is not None:
                    if current_length < self.prompt_length:
                        raise ReferenceInterventionRuntimeError(
                            "current sequence length is shorter than prompt length"
                        )
                    if current_length == self.prompt_length:
                        self.first_forward_detected = True
            self.alpha_zero_bypass_count += 1
            return attention_scores

        if not isinstance(attention_scores, torch.Tensor) or attention_scores.ndim != 4:
            raise ReferenceInterventionRuntimeError(
                "attention scores must have shape batch,head,query,key"
            )
        if not attention_scores.is_floating_point():
            raise ReferenceInterventionRuntimeError("attention scores must be floating point")
        if self.expected_score_dtype is not None and attention_scores.dtype != (
            self.expected_score_dtype
        ):
            raise ReferenceInterventionRuntimeError(
                f"attention score dtype mismatch: {attention_scores.dtype} != "
                f"{self.expected_score_dtype}"
            )
        batch, heads, query_length, key_length = attention_scores.shape
        if (batch, heads) != (1, self.head_count):
            raise ReferenceInterventionRuntimeError(
                f"attention score batch/head shape mismatch: {(batch, heads)}"
            )
        if query_length != key_length:
            raise ReferenceInterventionRuntimeError(
                f"attention score matrix is not square: {query_length}x{key_length}"
            )
        self._observed_sequence_lengths.add(query_length)
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
                self.later_forward_skip_count += len(selected_heads)
                return attention_scores
            self.first_forward_detected = True
            duplicate_heads = [
                head
                for head in selected_heads
                if self._per_head_intervention_count[(layer, head)] != 0
            ]
            if duplicate_heads:
                raise ReferenceInterventionRuntimeError(
                    "first_token_only duplicate intervention for "
                    f"layer={layer}, heads={duplicate_heads}"
                )
        elif self.prompt_length is not None:
            if query_length < self.prompt_length:
                raise ReferenceInterventionRuntimeError(
                    "current sequence length is shorter than prompt length"
                )
            if query_length == self.prompt_length:
                self.first_forward_detected = True
        if query_length < self.tail_tokens:
            self.short_sequence_skip_count += 1
            for head in selected_heads:
                self._short_sequence_skip_by_head[(layer, head)] += 1
            return attention_scores
        if not bool(torch.isfinite(attention_scores).all().item()):
            raise ReferenceInterventionRuntimeError(
                "pre-intervention attention scores contain NaN or infinity"
            )

        selected = list(selected_heads)
        block_original = attention_scores[
            :, selected, -self.tail_tokens :, -self.tail_tokens :
        ].float()
        current_asym = (
            block_original - block_original.transpose(-1, -2)
        ) / 2
        reference = torch.stack(
            [self._reference(layer, head, attention_scores.device) for head in selected]
        ).unsqueeze(0)
        constructed_delta = self.alpha * (reference - current_asym)
        block_float32 = block_original + constructed_delta
        block_cast = block_float32.to(attention_scores.dtype)

        modified = attention_scores.clone()
        modified[:, selected, -self.tail_tokens :, -self.tail_tokens :] = block_cast
        if not bool(torch.isfinite(modified).all().item()):
            raise ReferenceInterventionRuntimeError(
                "post-intervention attention scores contain NaN or infinity"
            )

        realized_float32_delta = block_float32 - block_original
        cast_block_float32 = block_cast.float()
        cast_delta = cast_block_float32 - block_original
        original_sym = (
            block_original + block_original.transpose(-1, -2)
        ) / 2
        float32_sym = (
            block_float32 + block_float32.transpose(-1, -2)
        ) / 2
        cast_sym = (
            cast_block_float32 + cast_block_float32.transpose(-1, -2)
        ) / 2

        head_mask = torch.ones(
            heads,
            dtype=torch.bool,
            device=attention_scores.device,
        )
        head_mask[selected] = False
        unselected_change = _max_abs(
            modified[:, head_mask] - attention_scores[:, head_mask]
        )
        prefix = query_length - self.tail_tokens
        outside_parts = []
        if prefix > 0:
            outside_parts.extend(
                [
                    modified[:, selected, :prefix, :] - attention_scores[:, selected, :prefix, :],
                    modified[:, selected, -self.tail_tokens :, :prefix]
                    - attention_scores[:, selected, -self.tail_tokens :, :prefix],
                ]
            )
        outside_tail_change = max(
            (_max_abs(value) for value in outside_parts),
            default=0.0,
        )
        if unselected_change != 0.0 or outside_tail_change != 0.0:
            raise ReferenceInterventionRuntimeError(
                "intervention changed an unselected head or a value outside the tail"
            )

        self.intervention_call_count += 1
        for head in selected:
            self._per_head_intervention_count[(layer, head)] += 1
        if self.prompt_length is not None:
            if query_length == self.prompt_length:
                self.first_forward_intervened = True
            elif query_length > self.prompt_length:
                self.later_forward_intervention_count += len(selected)
        self._layer_records.append(
            {
                "layer": layer,
                "sequence_length": query_length,
                "block_start": query_length - self.tail_tokens,
                "block_end": query_length,
                "selected_heads": list(selected),
                "max_unselected_head_change": unselected_change,
                "max_outside_tail_change": outside_tail_change,
            }
        )
        for local_index, head in enumerate(selected):
            index = (slice(None), local_index)
            current = current_asym[index]
            ref = reference[index]
            current_to_reference = ref - current
            intervened_float32 = (
                block_float32[index] - block_float32[index].transpose(-1, -2)
            ) / 2
            intervened_cast = (
                cast_block_float32[index]
                - cast_block_float32[index].transpose(-1, -2)
            ) / 2
            current_norm = float(torch.linalg.vector_norm(current).item())
            reference_norm = float(torch.linalg.vector_norm(ref).item())
            intervened_norm = float(torch.linalg.vector_norm(intervened_cast).item())
            self._head_records.append(
                {
                    "layer": layer,
                    "head": head,
                    "sequence_length": query_length,
                    "score_dtype": str(attention_scores.dtype),
                    "current_asym_frobenius_norm": current_norm,
                    "reference_asym_frobenius_norm": reference_norm,
                    "current_to_reference_frobenius_distance": float(
                        torch.linalg.vector_norm(current_to_reference).item()
                    ),
                    "current_to_reference_mean_absolute_distance": float(
                        current_to_reference.abs().mean().item()
                    ),
                    "intervened_asym_frobenius_norm": intervened_norm,
                    "intervened_to_current_norm_ratio": (
                        intervened_norm / current_norm if current_norm > 0 else None
                    ),
                    "mean_absolute_delta": float(cast_delta[index].abs().mean().item()),
                    "max_absolute_delta": _max_abs(cast_delta[index]),
                    "mean_absolute_delta_float32": float(
                        realized_float32_delta[index].abs().mean().item()
                    ),
                    "max_absolute_delta_float32": _max_abs(
                        realized_float32_delta[index]
                    ),
                    "current_min": float(current.amin().item()),
                    "current_max": float(current.amax().item()),
                    "intervened_min": float(intervened_cast.amin().item()),
                    "intervened_max": float(intervened_cast.amax().item()),
                    "max_constructed_delta_antisymmetry_error_float32": _max_abs(
                        constructed_delta[index]
                        + constructed_delta[index].transpose(-1, -2)
                    ),
                    "max_delta_antisymmetry_error_float32": _max_abs(
                        realized_float32_delta[index]
                        + realized_float32_delta[index].transpose(-1, -2)
                    ),
                    "max_delta_antisymmetry_error": _max_abs(
                        cast_delta[index] + cast_delta[index].transpose(-1, -2)
                    ),
                    "max_symmetric_component_error_float32": _max_abs(
                        float32_sym[index] - original_sym[index]
                    ),
                    "max_symmetric_component_error": _max_abs(
                        cast_sym[index] - original_sym[index]
                    ),
                    "max_diagonal_change_float32": _max_abs(
                        torch.diagonal(realized_float32_delta[index], dim1=-2, dim2=-1)
                    ),
                    "max_diagonal_change": _max_abs(
                        torch.diagonal(cast_delta[index], dim1=-2, dim2=-1)
                    ),
                    "max_reference_asymmetry_error_float32": _max_abs(
                        intervened_float32
                        - ((1.0 - self.alpha) * current + self.alpha * ref)
                    ),
                    "max_unselected_head_change": unselected_change,
                    "max_outside_tail_change": outside_tail_change,
                    "finite": True,
                }
            )
        return modified

    def summary(self) -> dict:
        def maximum(key: str) -> float:
            return max(
                (float(record[key]) for record in self._head_records),
                default=0.0,
            )

        per_head = []
        for layer, head in self.layer_head_pairs:
            records = [
                record
                for record in self._head_records
                if record["layer"] == layer and record["head"] == head
            ]
            ratios = [
                float(record["intervened_to_current_norm_ratio"])
                for record in records
                if record["intervened_to_current_norm_ratio"] is not None
            ]

            def values(key: str) -> list[float]:
                return [float(record[key]) for record in records]

            per_head.append(
                {
                    "layer": layer,
                    "head": head,
                    "call_count": len(records),
                    "short_sequence_skip_count": self._short_sequence_skip_by_head[
                        (layer, head)
                    ],
                    "current_asym_frobenius_norm_min": min(
                        values("current_asym_frobenius_norm"), default=None
                    ),
                    "current_asym_frobenius_norm_max": max(
                        values("current_asym_frobenius_norm"), default=None
                    ),
                    "reference_asym_frobenius_norm": (
                        values("reference_asym_frobenius_norm")[0]
                        if records
                        else float(
                            torch.linalg.vector_norm(
                                self._reference_cpu[(layer, head)]
                            ).item()
                        )
                    ),
                    "current_to_reference_frobenius_distance_min": min(
                        values("current_to_reference_frobenius_distance"),
                        default=None,
                    ),
                    "current_to_reference_frobenius_distance_max": max(
                        values("current_to_reference_frobenius_distance"),
                        default=None,
                    ),
                    "current_to_reference_mean_absolute_distance_min": min(
                        values("current_to_reference_mean_absolute_distance"),
                        default=None,
                    ),
                    "current_to_reference_mean_absolute_distance_max": max(
                        values("current_to_reference_mean_absolute_distance"),
                        default=None,
                    ),
                    "intervened_asym_frobenius_norm_min": min(
                        values("intervened_asym_frobenius_norm"), default=None
                    ),
                    "intervened_asym_frobenius_norm_max": max(
                        values("intervened_asym_frobenius_norm"), default=None
                    ),
                    "intervened_to_current_norm_ratio_min": min(
                        ratios, default=None
                    ),
                    "intervened_to_current_norm_ratio_max": max(
                        ratios, default=None
                    ),
                    "mean_absolute_delta_max": max(
                        values("mean_absolute_delta"), default=0.0
                    ),
                    "max_absolute_delta": max(
                        values("max_absolute_delta"), default=0.0
                    ),
                    "max_constructed_delta_antisymmetry_error_float32": max(
                        values(
                            "max_constructed_delta_antisymmetry_error_float32"
                        ),
                        default=0.0,
                    ),
                    "max_delta_antisymmetry_error_float32": max(
                        values("max_delta_antisymmetry_error_float32"),
                        default=0.0,
                    ),
                    "max_delta_antisymmetry_error": max(
                        values("max_delta_antisymmetry_error"), default=0.0
                    ),
                    "max_symmetric_component_error_float32": max(
                        values("max_symmetric_component_error_float32"),
                        default=0.0,
                    ),
                    "max_symmetric_component_error": max(
                        values("max_symmetric_component_error"), default=0.0
                    ),
                    "max_diagonal_change_float32": max(
                        values("max_diagonal_change_float32"), default=0.0
                    ),
                    "max_diagonal_change": max(
                        values("max_diagonal_change"), default=0.0
                    ),
                    "max_unselected_head_change": max(
                        values("max_unselected_head_change"), default=0.0
                    ),
                    "max_outside_tail_change": max(
                        values("max_outside_tail_change"), default=0.0
                    ),
                    "max_reference_asymmetry_error_float32": max(
                        values("max_reference_asymmetry_error_float32"),
                        default=0.0,
                    ),
                    "current_min": min(values("current_min"), default=None),
                    "current_max": max(values("current_max"), default=None),
                    "intervened_min": min(
                        values("intervened_min"), default=None
                    ),
                    "intervened_max": max(
                        values("intervened_max"), default=None
                    ),
                }
            )
        return {
            "alpha": self.alpha,
            "intervention_timing": self.intervention_timing,
            "prompt_length": self.prompt_length,
            "current_sequence_lengths": sorted(self._observed_sequence_lengths),
            "first_forward_detected": self.first_forward_detected,
            "first_forward_intervened": self.first_forward_intervened,
            "later_forward_intervention_count": (
                self.later_forward_intervention_count
            ),
            "later_forward_skip_count": self.later_forward_skip_count,
            "selected_heads": [
                {"layer": layer, "head": head}
                for layer, head in self.layer_head_pairs
            ],
            "attention_forward_call_count": self.attention_forward_call_count,
            "intervention_call_count": self.intervention_call_count,
            "total_intervention_count": sum(
                self._per_head_intervention_count.values()
            ),
            "per_head_intervention_count": [
                {
                    "layer": layer,
                    "head": head,
                    "count": self._per_head_intervention_count[(layer, head)],
                }
                for layer, head in self.layer_head_pairs
            ],
            "alpha_zero_bypass_count": self.alpha_zero_bypass_count,
            "short_sequence_skip_count": self.short_sequence_skip_count,
            "reference_cache_entry_count": len(self._reference_cache),
            "reference_cache_transfer_count": (
                self._reference_cache_transfer_count
                - self._cache_transfer_count_at_clear
            ),
            "reference_cache_transfer_count_total": self._reference_cache_transfer_count,
            "block_start": (
                self._layer_records[0]["block_start"]
                if self._layer_records
                else None
            ),
            "block_end": (
                self._layer_records[0]["block_end"]
                if self._layer_records
                else None
            ),
            "intervention_blocks": [
                {"start": start, "end": end}
                for start, end in sorted(
                    {
                        (record["block_start"], record["block_end"])
                        for record in self._layer_records
                    }
                )
            ],
            "block_changed_across_interventions": len(
                {
                    (record["block_start"], record["block_end"])
                    for record in self._layer_records
                }
            )
            > 1,
            "max_constructed_delta_antisymmetry_error_float32": maximum(
                "max_constructed_delta_antisymmetry_error_float32"
            ),
            "max_delta_antisymmetry_error_float32": maximum(
                "max_delta_antisymmetry_error_float32"
            ),
            "max_delta_antisymmetry_error": maximum(
                "max_delta_antisymmetry_error"
            ),
            "max_symmetric_component_error_float32": maximum(
                "max_symmetric_component_error_float32"
            ),
            "max_symmetric_component_error": maximum(
                "max_symmetric_component_error"
            ),
            "max_diagonal_change_float32": maximum(
                "max_diagonal_change_float32"
            ),
            "max_diagonal_change": maximum("max_diagonal_change"),
            "max_unselected_head_change": max(
                (
                    float(record["max_unselected_head_change"])
                    for record in self._layer_records
                ),
                default=0.0,
            ),
            "max_outside_tail_change": max(
                (
                    float(record["max_outside_tail_change"])
                    for record in self._layer_records
                ),
                default=0.0,
            ),
            "max_reference_asymmetry_error_float32": maximum(
                "max_reference_asymmetry_error_float32"
            ),
            "all_scores_finite": all(
                bool(record["finite"]) for record in self._head_records
            ),
            "per_head": per_head,
        }


def _patched_llama_attention_forward(
    controller: ReferenceAsymmetryController,
):
    """Return Transformers 4.42.3 eager forward plus one tail-score edit."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if use_cache:
            raise UnsupportedAttentionCacheError(
                "reference score intervention requires use_cache=False"
            )
        if past_key_value is not None:
            raise UnsupportedAttentionCacheError(
                "reference score intervention does not support past_key_value"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp,
                dim=0,
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
            query_states = torch.cat(
                [
                    F.linear(hidden_states, query_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            key_states = torch.cat(
                [
                    F.linear(hidden_states, key_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            value_states = torch.cat(
                [
                    F.linear(hidden_states, value_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = modeling_llama.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )
        key_states = modeling_llama.repeat_kv(
            key_states,
            self.num_key_value_groups,
        )
        value_states = modeling_llama.repeat_kv(
            value_states,
            self.num_key_value_groups,
        )
        attn_weights = torch.matmul(
            query_states,
            key_states.transpose(2, 3),
        ) / math.sqrt(self.head_dim)

        # Sole semantic change: selected raw scores, post-RoPE/repeat_kv and
        # pre-mask/pre-softmax, are interpolated toward the original reference.
        attn_weights = controller.apply(self.layer_idx, attn_weights)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights,
            p=self.attention_dropout,
            training=self.training,
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (
            bsz,
            self.num_heads,
            q_len,
            self.head_dim,
        ):
            raise ValueError(
                "`attn_output` should be of size "
                f"{(bsz, self.num_heads, q_len, self.head_dim)}, but is "
                f"{attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp,
                dim=2,
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp,
                dim=1,
            )
            attn_output = sum(
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            )
        else:
            attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    return forward


@contextmanager
def patch_llama_reference_attention(
    model,
    *,
    artifacts: ValidatedReferenceArtifacts,
    alpha: float,
    intervention_timing: str = "dynamic_all_steps",
):
    """Temporarily patch only artifact-selected eager Llama attention layers."""

    alpha = validate_alpha(alpha)
    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected Transformers {SUPPORTED_TRANSFORMERS_VERSION}, "
            f"found {transformers.__version__}"
        )
    controller = ReferenceAsymmetryController(
        reference_mean=artifacts.reference_mean,
        layer_head_pairs=artifacts.top5_heads,
        alpha=alpha,
        tail_tokens=TAIL_TOKENS,
        expected_score_dtype=torch.float16,
        intervention_timing=intervention_timing,
    )
    patched = []
    try:
        for layer_index in controller.selected_by_layer:
            module = model.model.layers[layer_index].self_attn
            if type(module) is not modeling_llama.LlamaAttention:
                raise TypeError(
                    "reference intervention requires eager LlamaAttention; "
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
        for module, had_instance_forward, previous_instance_forward in reversed(
            patched
        ):
            if had_instance_forward:
                module.forward = previous_instance_forward
            else:
                module.__dict__.pop("forward", None)
