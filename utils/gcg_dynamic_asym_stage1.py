"""Selection plans and runtime controller for dynamic Asym Stage 1 tuning."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Mapping

import torch

from utils.gcg_dynamic_asym_intervention import (
    DynamicAsymmetryController,
    DynamicAsymmetryReference,
    _patch_llama_attention_modules,
    _restore_llama_attention_modules,
)
from utils.gcg_dynamic_asym_selection_diagnostics import (
    distribution_summary,
    selection_count_summary,
    stable_top_indices,
)
from utils.reference_asym_score_intervention import (
    ReferenceInterventionRuntimeError,
)


CAP_NUMERATOR = 5
RATE_DENOMINATOR = 1000
CAP_RATE = CAP_NUMERATOR / RATE_DENOMINATOR
FIXED_TOP_VALUES = (250, 500, 750)
PROPORTIONAL_RATE_NUMERATORS = (1, 2, 4)
THRESHOLD_VALUES = (3.0, 2.8, 2.7)
SELECTION_DIRECTIONS = ("absolute", "positive", "negative")


def stage1_conditions() -> tuple[dict, ...]:
    conditions = []
    for top_n in FIXED_TOP_VALUES:
        conditions.append(
            {
                "name": f"fixed_top_{top_n}",
                "family": "fixed_top_n",
                "top_n": top_n,
                "rate": None,
                "rate_numerator": None,
                "tau": None,
            }
        )
    for numerator in PROPORTIONAL_RATE_NUMERATORS:
        conditions.append(
            {
                "name": f"proportional_{numerator / 10:.1f}pct".replace(".", "p"),
                "family": "proportional_top_n",
                "top_n": None,
                "rate": numerator / RATE_DENOMINATOR,
                "rate_numerator": numerator,
                "tau": None,
            }
        )
    for tau in THRESHOLD_VALUES:
        conditions.append(
            {
                "name": f"threshold_tau_{tau:.1f}".replace(".", "p"),
                "family": "z_score_threshold",
                "top_n": None,
                "rate": None,
                "rate_numerator": None,
                "tau": tau,
            }
        )
    return tuple(conditions)


def validate_stage1_conditions(conditions) -> None:
    expected_names = (
        "fixed_top_250",
        "fixed_top_500",
        "fixed_top_750",
        "proportional_0p1pct",
        "proportional_0p2pct",
        "proportional_0p4pct",
        "threshold_tau_3p0",
        "threshold_tau_2p8",
        "threshold_tau_2p7",
    )
    if tuple(condition.get("name") for condition in conditions) != expected_names:
        raise ValueError("Stage 1 condition names/order mismatch")
    for condition in conditions:
        if condition.get("family") not in {
            "fixed_top_n",
            "proportional_top_n",
            "z_score_threshold",
        }:
            raise ValueError("Stage 1 condition family is invalid")


def cap_count(candidate_count: int) -> int:
    if type(candidate_count) is not int or candidate_count <= 0:
        raise ValueError("candidate_count must be a positive integer")
    return candidate_count * CAP_NUMERATOR // RATE_DENOMINATOR


def _condition_selection_count(condition: Mapping[str, object], candidate_count: int):
    family = condition["family"]
    if family == "fixed_top_n":
        requested = int(condition["top_n"])
        return requested, requested
    if family == "proportional_top_n":
        numerator = int(condition["rate_numerator"])
        requested = candidate_count * numerator // RATE_DENOMINATOR
        return requested, requested
    if family == "z_score_threshold":
        return None, None
    raise ValueError("unknown Stage 1 selection family")


def build_stage1_selection_plan(
    abs_z: torch.Tensor,
    *,
    condition: Mapping[str, object],
    layer_count: int,
    head_count: int,
    elements_per_head: int,
    eligible_mask: torch.Tensor | None = None,
    cap_candidate_count: int | None = None,
    signed_z: torch.Tensor | None = None,
    selection_direction: str = "absolute",
) -> dict:
    """Build one deterministic global Layer×Head×position selection plan.

    ``eligible_mask`` restricts the candidate set before applying the selection
    rule.  ``cap_candidate_count`` only controls the 0.5% cap denominator.  The
    defaults retain the original Stage 1 behavior exactly.
    """

    if not isinstance(abs_z, torch.Tensor) or abs_z.ndim != 1:
        raise TypeError("abs_z must be a rank-1 tensor")
    if abs_z.device.type != "cpu" or not abs_z.is_floating_point():
        raise ValueError("abs_z must be a floating-point CPU tensor")
    if not bool(torch.isfinite(abs_z).all().item()):
        raise ValueError("abs_z contains NaN or infinity")
    if selection_direction not in SELECTION_DIRECTIONS:
        raise ValueError("selection_direction must be absolute, positive, or negative")
    expected_candidates = layer_count * head_count * elements_per_head
    if abs_z.numel() != expected_candidates:
        raise ValueError("abs_z candidate count does not match its structure")
    if signed_z is not None:
        if (
            not isinstance(signed_z, torch.Tensor)
            or signed_z.ndim != 1
            or signed_z.device.type != "cpu"
            or not signed_z.is_floating_point()
            or signed_z.numel() != expected_candidates
        ):
            raise ValueError("signed_z must be a floating-point CPU candidate vector")
        if not bool(torch.isfinite(signed_z).all().item()):
            raise ValueError("signed_z contains NaN or infinity")
        if not torch.equal(abs_z, signed_z.abs()):
            raise ValueError("abs_z does not exactly match abs(signed_z)")
    elif selection_direction != "absolute":
        raise ValueError("signed_z is required for directional selection")
    if eligible_mask is None:
        eligible_indices = torch.arange(expected_candidates, dtype=torch.int64)
    else:
        if (
            not isinstance(eligible_mask, torch.Tensor)
            or eligible_mask.dtype != torch.bool
            or eligible_mask.device.type != "cpu"
            or eligible_mask.ndim != 1
            or eligible_mask.numel() != expected_candidates
        ):
            raise ValueError("eligible_mask must be rank-1 CPU bool with candidate shape")
        eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).flatten()
    eligible_candidate_count = int(eligible_indices.numel())
    if eligible_candidate_count <= 0:
        raise ValueError("selection candidate set is empty")
    if cap_candidate_count is None:
        cap_candidate_count = expected_candidates
    if (
        type(cap_candidate_count) is not int
        or not eligible_candidate_count <= cap_candidate_count <= expected_candidates
    ):
        raise ValueError("cap_candidate_count is outside the valid candidate range")
    cap = cap_count(cap_candidate_count)
    if cap <= 0:
        raise ValueError("0.5% cap produced no selectable elements")

    family = condition.get("family")
    requested_count, eligible_count = _condition_selection_count(
        condition, eligible_candidate_count
    )
    if selection_direction == "absolute":
        sign_eligible_indices = eligible_indices
        ranking_values = abs_z
    elif selection_direction == "positive":
        sign_eligible_indices = eligible_indices[
            signed_z[eligible_indices] > 0
        ].contiguous()
        ranking_values = signed_z
    else:
        sign_eligible_indices = eligible_indices[
            signed_z[eligible_indices] < 0
        ].contiguous()
        ranking_values = -signed_z
    sign_eligible_candidate_count = int(sign_eligible_indices.numel())
    if family in ("fixed_top_n", "proportional_top_n"):
        selected_count = min(requested_count, cap, sign_eligible_candidate_count)
        if selected_count:
            local_indices = stable_top_indices(
                ranking_values[sign_eligible_indices], selected_count
            )
            selected_indices = sign_eligible_indices[local_indices].contiguous()
        else:
            selected_indices = torch.empty(0, dtype=torch.int64)
    elif family == "z_score_threshold":
        tau = condition.get("tau")
        if isinstance(tau, bool) or not isinstance(tau, (int, float)):
            raise TypeError("threshold tau must be a real number")
        tau = float(tau)
        if not math.isfinite(tau) or tau < 0:
            raise ValueError("threshold tau must be finite and non-negative")
        threshold_indices = sign_eligible_indices[
            ranking_values[sign_eligible_indices] > tau
        ].contiguous()
        eligible_count = int(threshold_indices.numel())
        requested_count = eligible_count
        if eligible_count > cap:
            local_indices = stable_top_indices(ranking_values[threshold_indices], cap)
            selected_indices = threshold_indices[local_indices].contiguous()
            if not bool((ranking_values[selected_indices] > tau).all().item()):
                raise AssertionError("threshold cap selected an ineligible element")
        else:
            selected_indices = threshold_indices
        selected_count = int(selected_indices.numel())
    else:
        raise ValueError("unknown Stage 1 selection family")

    pre_cap_selected_count = min(requested_count, sign_eligible_candidate_count)
    cap_activated = pre_cap_selected_count > cap
    cap_excluded_count = max(0, pre_cap_selected_count - cap)
    selection_shortfall_count = (
        max(0, min(requested_count, cap) - selected_count)
        if family in ("fixed_top_n", "proportional_top_n")
        else 0
    )
    if selected_count > cap:
        raise AssertionError("Stage 1 plan exceeds the 0.5% cap")
    if cap_activated != (cap_excluded_count > 0):
        raise AssertionError("Stage 1 cap activation accounting mismatch")
    if selected_count:
        counts = selection_count_summary(
            selected_indices,
            layer_count=layer_count,
            head_count=head_count,
            elements_per_head=elements_per_head,
        )
    else:
        counts = {
            "layer_counts": [0] * layer_count,
            "head_counts": [0] * head_count,
            "layer_head_counts": [[0] * head_count for _ in range(layer_count)],
        }
    selected_values = abs_z[selected_indices]
    selected_signed_z = signed_z[selected_indices] if signed_z is not None else None
    if selection_direction == "positive" and selected_count and not bool(
        (selected_signed_z > 0).all().item()
    ):
        raise AssertionError("positive direction selected a non-positive z-score")
    if selection_direction == "negative" and selected_count and not bool(
        (selected_signed_z < 0).all().item()
    ):
        raise AssertionError("negative direction selected a non-negative z-score")
    return {
        "condition": dict(condition),
        "selection_direction": selection_direction,
        "candidate_count": expected_candidates,
        "eligible_candidate_count": eligible_candidate_count,
        "excluded_candidate_count": expected_candidates - eligible_candidate_count,
        "cap_candidate_count": cap_candidate_count,
        "cap_rate": CAP_RATE,
        "cap_count": cap,
        "requested_count": requested_count,
        "requested_or_threshold_eligible_count": requested_count,
        "sign_eligible_candidate_count": sign_eligible_candidate_count,
        "threshold_eligible_count": eligible_count if family == "z_score_threshold" else None,
        "selection_shortfall_count": selection_shortfall_count,
        "actual_selected_count": selected_count,
        "selected_count": selected_count,
        "selected_fraction": selected_count / expected_candidates,
        "cap_activated": cap_activated,
        "cap_excluded_count": cap_excluded_count,
        "selected_indices": selected_indices,
        "selected_abs_z": (
            distribution_summary(selected_values) if selected_count else None
        ),
        "selected_z": (
            distribution_summary(selected_signed_z)
            if selected_signed_z is not None and selected_count
            else None
        ),
        "positive_selected_count": (
            int((selected_signed_z > 0).sum().item())
            if selected_signed_z is not None
            else None
        ),
        "negative_selected_count": (
            int((selected_signed_z < 0).sum().item())
            if selected_signed_z is not None
            else None
        ),
        **counts,
    }


class PlannedDynamicAsymmetryController(DynamicAsymmetryController):
    """Apply a precomputed global selection plan with the existing Pure Asym edit."""

    def __init__(
        self,
        *,
        reference: DynamicAsymmetryReference,
        condition: Mapping[str, object],
        alpha: float = 1.0,
        expected_score_dtype: torch.dtype | None = None,
    ):
        if float(alpha) != 1.0:
            raise ValueError("Stage 1 alpha is fixed at 1.0")
        self.condition = dict(condition)
        super().__init__(
            mu=reference.mu,
            sigma=reference.sigma,
            tau=0.0,
            alpha=alpha,
            expected_score_dtype=expected_score_dtype,
        )

    def clear(self) -> None:
        super().clear()
        self._planned_masks_cpu: dict[int, torch.Tensor] = {}
        self._plan_metadata: dict | None = None

    def set_prompt_plan(self, *, prompt_length: int, plan: Mapping[str, object]) -> None:
        self.set_prompt_context(prompt_length=prompt_length)
        expected_elements = prompt_length * (prompt_length - 1) // 2
        expected_candidates = self.layer_count * self.head_count * expected_elements
        if plan.get("condition") != self.condition:
            raise ValueError("Stage 1 runtime plan condition mismatch")
        if plan.get("candidate_count") != expected_candidates:
            raise ValueError("Stage 1 runtime plan candidate count mismatch")
        indices = plan.get("selected_indices")
        if (
            not isinstance(indices, torch.Tensor)
            or indices.dtype != torch.int64
            or indices.device.type != "cpu"
            or indices.ndim != 1
        ):
            raise ValueError("Stage 1 selected indices must be rank-1 CPU int64")
        if indices.numel() != plan.get("selected_count"):
            raise ValueError("Stage 1 selected index count mismatch")
        mask = torch.zeros(expected_candidates, dtype=torch.bool)
        mask[indices] = True
        if int(mask.sum().item()) != indices.numel():
            raise ValueError("Stage 1 selected indices contain duplicates")
        mask = mask.reshape(
            self.layer_count, self.head_count, expected_elements
        )
        self._planned_masks_cpu = {
            layer: mask[layer].unsqueeze(0).contiguous()
            for layer in range(self.layer_count)
        }
        self._plan_metadata = {
            key: value
            for key, value in plan.items()
            if key not in {"selected_indices", "layer_head_counts"}
        }

    def _selection_mask(
        self,
        *,
        layer: int,
        z_scores: torch.Tensor,
    ) -> torch.Tensor:
        if self._plan_metadata is None or layer not in self._planned_masks_cpu:
            raise ReferenceInterventionRuntimeError("Stage 1 selection plan is missing")
        mask = self._planned_masks_cpu[layer].to(device=z_scores.device)
        if mask.shape != z_scores.shape:
            raise ReferenceInterventionRuntimeError(
                "Stage 1 selection plan layer shape mismatch"
            )
        return mask

    def assert_complete_prefill(self) -> None:
        super().assert_complete_prefill()
        if self._plan_metadata is None:
            raise ReferenceInterventionRuntimeError("Stage 1 selection plan is missing")
        if self.selected_element_intervention_count != self._plan_metadata[
            "selected_count"
        ]:
            raise ReferenceInterventionRuntimeError(
                "Stage 1 applied selection count differs from its plan"
            )
        observed_by_layer = [0] * self.layer_count
        for record in self._records:
            observed_by_layer[int(record["layer"])] = int(
                record["selected_element_count"]
            )
        if observed_by_layer != self._plan_metadata["layer_counts"]:
            raise ReferenceInterventionRuntimeError(
                "Stage 1 applied layer counts differ from its plan"
            )

    def summary(self) -> dict:
        summary = super().summary()
        if self._plan_metadata is None:
            raise ReferenceInterventionRuntimeError("Stage 1 selection plan is missing")
        summary.update(
            {
                "intervention_variant": "dynamic_full_prompt_stage1_planned_pure_asym",
                "selection_rule": self.condition["family"],
                "condition": self.condition,
                "tau": self.condition.get("tau"),
                "layer0_included": True,
                "cap_rate": CAP_RATE,
                "cap_count": self._plan_metadata["cap_count"],
                "selection_direction": self._plan_metadata["selection_direction"],
                "requested_count": self._plan_metadata["requested_count"],
                "sign_eligible_candidate_count": self._plan_metadata[
                    "sign_eligible_candidate_count"
                ],
                "selection_shortfall_count": self._plan_metadata[
                    "selection_shortfall_count"
                ],
                "selected_z": self._plan_metadata["selected_z"],
                "positive_selected_count": self._plan_metadata[
                    "positive_selected_count"
                ],
                "negative_selected_count": self._plan_metadata[
                    "negative_selected_count"
                ],
                "cap_activated": self._plan_metadata["cap_activated"],
                "cap_excluded_count": self._plan_metadata["cap_excluded_count"],
                "planned_selected_count": self._plan_metadata["selected_count"],
                "planned_layer_counts": self._plan_metadata["layer_counts"],
            }
        )
        return summary


@contextmanager
def patch_llama_planned_dynamic_asymmetry_attention(
    model,
    *,
    reference: DynamicAsymmetryReference,
    condition: Mapping[str, object],
    alpha: float = 1.0,
):
    controller = PlannedDynamicAsymmetryController(
        reference=reference,
        condition=condition,
        alpha=alpha,
        expected_score_dtype=torch.float16,
    )
    patched = []
    try:
        patched = _patch_llama_attention_modules(model, controller)
        yield controller
    finally:
        _restore_llama_attention_modules(patched)
