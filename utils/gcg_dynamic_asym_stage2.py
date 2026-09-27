"""Frozen conditions and runtime controller for dynamic Asym Stage 2 tuning."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Mapping

import torch

from utils.gcg_dynamic_asym_intervention import (
    DynamicAsymmetryController,
    DynamicAsymmetryReference,
    _patch_llama_attention_modules,
    _restore_llama_attention_modules,
)
from utils.gcg_dynamic_asym_stage1 import (
    CAP_RATE,
    PlannedDynamicAsymmetryController,
    build_stage1_selection_plan,
    stage1_conditions,
)
from utils.gcg_dynamic_asym_selection_diagnostics import (
    distribution_summary,
    selection_count_summary,
)
from utils.reference_asym_score_intervention import ReferenceInterventionRuntimeError


STAGE2_SELECTION_NAMES = (
    "fixed_top_500",
    "proportional_0p1pct",
    "proportional_0p4pct",
    "threshold_tau_3p0",
)
STAGE2_NEW_VARIANTS = (
    {
        "name": "alpha_0p5_layer0_included",
        "alpha": 0.5,
        "layer0_included": True,
        "reused_from_stage1": False,
    },
    {
        "name": "alpha_1p0_layer0_excluded",
        "alpha": 1.0,
        "layer0_included": False,
        "reused_from_stage1": False,
    },
    {
        "name": "alpha_0p5_layer0_excluded",
        "alpha": 0.5,
        "layer0_included": False,
        "reused_from_stage1": False,
    },
)
STAGE2_REUSED_VARIANT = {
    "name": "alpha_1p0_layer0_included",
    "alpha": 1.0,
    "layer0_included": True,
    "reused_from_stage1": True,
}
ABSOLUTE_INTERVENTION_TARGETS = (
    "all",
    "positive",
    "negative",
    "positive-count-matched",
    "negative-count-matched",
)


def stage2_selection_conditions() -> tuple[dict, ...]:
    by_name = {row["name"]: row for row in stage1_conditions()}
    return tuple(dict(by_name[name]) for name in STAGE2_SELECTION_NAMES)


def stage2_new_conditions() -> tuple[dict, ...]:
    return tuple(
        {
            "selection": dict(selection),
            **variant,
            "variant_name": variant["name"],
            "name": f"{selection['name']}__{variant['name']}",
        }
        for selection in stage2_selection_conditions()
        for variant in STAGE2_NEW_VARIANTS
    )


def validate_stage2_conditions(conditions) -> None:
    expected = tuple(
        f"{selection}__{variant['name']}"
        for selection in STAGE2_SELECTION_NAMES
        for variant in STAGE2_NEW_VARIANTS
    )
    if tuple(row.get("name") for row in conditions) != expected:
        raise ValueError("Stage 2 condition names/order mismatch")
    if len(conditions) != 12:
        raise ValueError("Stage 2 must contain exactly 12 new GPU conditions")
    for row in conditions:
        if row.get("alpha") not in (0.5, 1.0):
            raise ValueError("Stage 2 alpha must be 0.5 or 1.0")
        if type(row.get("layer0_included")) is not bool:
            raise ValueError("Stage 2 layer0 flag must be boolean")
        if row.get("alpha") == 1.0 and row.get("layer0_included") is True:
            raise ValueError("Stage 1 reused condition must not be scheduled on GPU")


def layer_eligibility_mask(
    *,
    layer_count: int,
    head_count: int,
    elements_per_head: int,
    layer0_included: bool,
) -> torch.Tensor:
    if min(layer_count, head_count, elements_per_head) <= 0:
        raise ValueError("candidate dimensions must be positive")
    mask = torch.ones(
        layer_count * head_count * elements_per_head, dtype=torch.bool
    )
    if not layer0_included:
        mask[: head_count * elements_per_head] = False
    return mask


def build_stage2_selection_plan(
    abs_z: torch.Tensor,
    *,
    selection: Mapping[str, object],
    layer_count: int,
    head_count: int,
    elements_per_head: int,
    layer0_included: bool,
    signed_z: torch.Tensor | None = None,
    selection_direction: str = "absolute",
) -> dict:
    """Apply the Stage 1 rule after the frozen Layer 0 eligibility filter.

    Proportional N uses the filtered candidate set.  The cap denominator remains
    the original full candidate count for both included and excluded conditions.
    """

    full_count = layer_count * head_count * elements_per_head
    mask = layer_eligibility_mask(
        layer_count=layer_count,
        head_count=head_count,
        elements_per_head=elements_per_head,
        layer0_included=layer0_included,
    )
    plan = build_stage1_selection_plan(
        abs_z,
        condition=selection,
        layer_count=layer_count,
        head_count=head_count,
        elements_per_head=elements_per_head,
        eligible_mask=mask,
        cap_candidate_count=full_count,
        signed_z=signed_z,
        selection_direction=selection_direction,
    )
    plan["layer0_included"] = layer0_included
    plan["selection_candidate_definition"] = (
        "all_layers" if layer0_included else "layers_1_through_31"
    )
    plan["cap_candidate_definition"] = "original_all_layer_candidate_count"
    if not layer0_included:
        if plan["layer_counts"][0] != 0:
            raise AssertionError("Layer 0 excluded plan selected Layer 0 elements")
        expected_eligible = (layer_count - 1) * head_count * elements_per_head
        if plan["eligible_candidate_count"] != expected_eligible:
            raise AssertionError("Layer 0 excluded candidate count mismatch")
    return plan


def restrict_absolute_plan_to_intervention_sign(
    plan: Mapping[str, object],
    signed_z: torch.Tensor,
    *,
    intervention_target_sign: str,
    layer_count: int,
    head_count: int,
    elements_per_head: int,
) -> dict:
    """Restrict an already-ranked absolute Top-N plan without re-ranking."""

    if intervention_target_sign not in ABSOLUTE_INTERVENTION_TARGETS:
        raise ValueError("absolute intervention target is invalid")
    if plan.get("selection_direction") != "absolute":
        raise ValueError("sign-restricted intervention requires an absolute selection plan")
    expected_candidates = layer_count * head_count * elements_per_head
    if (
        not isinstance(signed_z, torch.Tensor)
        or signed_z.ndim != 1
        or signed_z.device.type != "cpu"
        or not signed_z.is_floating_point()
        or signed_z.numel() != expected_candidates
        or not bool(torch.isfinite(signed_z).all().item())
    ):
        raise ValueError("signed_z must be a finite rank-1 CPU candidate vector")
    absolute_indices = plan.get("selected_indices")
    if (
        not isinstance(absolute_indices, torch.Tensor)
        or absolute_indices.dtype != torch.int64
        or absolute_indices.device.type != "cpu"
        or absolute_indices.ndim != 1
        or absolute_indices.numel() != plan.get("selected_count")
    ):
        raise ValueError("absolute plan selected indices are invalid")
    absolute_z = signed_z[absolute_indices]
    positive_mask = absolute_z > 0
    negative_mask = absolute_z < 0
    if not bool((positive_mask | negative_mask).all().item()):
        raise AssertionError("absolute Top-N contains a zero z-score")
    positive_indices = absolute_indices[positive_mask].contiguous()
    negative_indices = absolute_indices[negative_mask].contiguous()
    if torch.isin(positive_indices, negative_indices).any().item():
        raise AssertionError("absolute sign subsets overlap")
    if positive_indices.numel() + negative_indices.numel() != absolute_indices.numel():
        raise AssertionError("absolute sign subsets do not partition the Top-N plan")

    absolute_count = int(absolute_indices.numel())
    absolute_positive_count = int(positive_indices.numel())
    absolute_negative_count = int(negative_indices.numel())
    matched_count = min(absolute_positive_count, absolute_negative_count)
    if (
        plan.get("positive_selected_count") != absolute_positive_count
        or plan.get("negative_selected_count") != absolute_negative_count
    ):
        raise AssertionError("absolute plan sign counts disagree with signed z-scores")

    restricted = dict(plan)
    restricted.update(
        {
            "intervention_target_sign": intervention_target_sign,
            "absolute_top_n_count": absolute_count,
            "absolute_positive_selected_count": absolute_positive_count,
            "absolute_negative_selected_count": absolute_negative_count,
            "absolute_selected_fraction": absolute_count / expected_candidates,
            "matched_count": matched_count,
        }
    )
    if intervention_target_sign == "all":
        restricted["intervention_subset_excluded_count"] = 0
        return restricted

    target_is_positive = intervention_target_sign.startswith("positive")
    selected_indices = positive_indices if target_is_positive else negative_indices
    if intervention_target_sign.endswith("count-matched"):
        selected_indices = selected_indices[:matched_count].contiguous()
    selected_z = signed_z[selected_indices]
    selected_count = int(selected_indices.numel())
    if selected_count:
        counts = selection_count_summary(
            selected_indices,
            layer_count=layer_count,
            head_count=head_count,
            elements_per_head=elements_per_head,
        )
        selected_abs_z_summary = distribution_summary(selected_z.abs())
        selected_z_summary = distribution_summary(selected_z)
    else:
        counts = {
            "layer_counts": [0] * layer_count,
            "head_counts": [0] * head_count,
            "layer_head_counts": [[0] * head_count for _ in range(layer_count)],
        }
        selected_abs_z_summary = None
        selected_z_summary = None
    restricted.update(
        {
            "selected_indices": selected_indices,
            "actual_selected_count": selected_count,
            "selected_count": selected_count,
            "selected_fraction": selected_count / expected_candidates,
            "selected_abs_z": selected_abs_z_summary,
            "selected_z": selected_z_summary,
            "positive_selected_count": (
                selected_count if target_is_positive else 0
            ),
            "negative_selected_count": (
                0 if target_is_positive else selected_count
            ),
            "intervention_subset_excluded_count": absolute_count - selected_count,
            **counts,
        }
    )
    if selected_count > int(plan["cap_count"]):
        raise AssertionError("sign-restricted intervention exceeds the absolute cap")
    if target_is_positive and not bool(
        (selected_z > 0).all().item()
    ):
        raise AssertionError("positive-only intervention contains a non-positive z-score")
    if not target_is_positive and not bool(
        (selected_z < 0).all().item()
    ):
        raise AssertionError("negative-only intervention contains a non-negative z-score")
    if intervention_target_sign.endswith("count-matched") and selected_count != matched_count:
        raise AssertionError("count-matched intervention count mismatch")
    return restricted


class Stage2PlannedDynamicAsymmetryController(PlannedDynamicAsymmetryController):
    """Apply Stage 2 plans while reusing the existing Pure Asym implementation."""

    def __init__(
        self,
        *,
        reference: DynamicAsymmetryReference,
        selection: Mapping[str, object],
        alpha: float,
        layer0_included: bool,
        expected_score_dtype: torch.dtype | None = None,
        capture_score_change_metrics: bool = False,
        component: str = "antisymmetric",
    ):
        if float(alpha) not in (0.5, 1.0):
            raise ValueError("Stage 2 alpha must be exactly 0.5 or 1.0")
        if type(layer0_included) is not bool:
            raise TypeError("layer0_included must be boolean")
        self.condition = dict(selection)
        self.layer0_included = layer0_included
        DynamicAsymmetryController.__init__(
            self,
            mu=reference.mu,
            sigma=reference.sigma,
            tau=0.0,
            alpha=float(alpha),
            expected_score_dtype=expected_score_dtype,
            capture_score_change_metrics=capture_score_change_metrics,
            component=component,
        )

    def set_prompt_plan(self, *, prompt_length: int, plan: Mapping[str, object]) -> None:
        if plan.get("layer0_included") is not self.layer0_included:
            raise ValueError("Stage 2 plan Layer 0 condition mismatch")
        if not self.layer0_included and plan.get("layer_counts", [None])[0] != 0:
            raise ValueError("Stage 2 excluded plan contains Layer 0 selections")
        super().set_prompt_plan(prompt_length=prompt_length, plan=plan)

    def summary(self) -> dict:
        summary = DynamicAsymmetryController.summary(self)
        if self._plan_metadata is None:
            raise ReferenceInterventionRuntimeError("Stage 2 selection plan is missing")
        summary.update(
            {
                "intervention_variant": (
                    "dynamic_full_prompt_stage2_planned_pure_asym"
                    if self.component == "antisymmetric"
                    else "dynamic_full_prompt_stage2_planned_pure_sym"
                ),
                "selection_rule": self.condition["family"],
                "selection": self.condition,
                "tau": self.condition.get("tau"),
                "layer0_included": self.layer0_included,
                "cap_rate": CAP_RATE,
                "cap_count": self._plan_metadata["cap_count"],
                "cap_candidate_count": self._plan_metadata["cap_candidate_count"],
                "eligible_candidate_count": self._plan_metadata[
                    "eligible_candidate_count"
                ],
                "selection_direction": self._plan_metadata["selection_direction"],
                "intervention_target_sign": self._plan_metadata.get(
                    "intervention_target_sign", "all"
                ),
                "absolute_top_n_count": self._plan_metadata.get(
                    "absolute_top_n_count"
                ),
                "absolute_positive_selected_count": self._plan_metadata.get(
                    "absolute_positive_selected_count"
                ),
                "absolute_negative_selected_count": self._plan_metadata.get(
                    "absolute_negative_selected_count"
                ),
                "matched_count": self._plan_metadata.get("matched_count"),
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
def patch_llama_stage2_dynamic_asymmetry_attention(
    model,
    *,
    reference: DynamicAsymmetryReference,
    selection: Mapping[str, object],
    alpha: float,
    layer0_included: bool,
    capture_score_change_metrics: bool = False,
    component: str = "antisymmetric",
):
    controller = Stage2PlannedDynamicAsymmetryController(
        reference=reference,
        selection=selection,
        alpha=alpha,
        layer0_included=layer0_included,
        expected_score_dtype=torch.float16,
        capture_score_change_metrics=capture_score_change_metrics,
        component=component,
    )
    patched = []
    try:
        patched = _patch_llama_attention_modules(model, controller)
        yield controller
    finally:
        _restore_llama_attention_modules(patched)
