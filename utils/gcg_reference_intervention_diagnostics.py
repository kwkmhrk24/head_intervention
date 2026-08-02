"""Analysis helpers for the fixed GCG reference-intervention evaluation.

The helpers in this module do not select heads or alter intervention behavior.
They compare saved responses, validate diagnostic head subsets, and reduce
per-generation-step logit measurements into finite scalar summaries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


LOGIT_MODES = ("teacher_forced_common_prefix", "free_running")


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the number of equal leading items."""

    length = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        length += 1
    return length


def first_different_position(
    left: Sequence[int],
    right: Sequence[int],
) -> int | None:
    """Return the zero-based first differing position, including length changes."""

    prefix = common_prefix_length(left, right)
    if prefix == len(left) == len(right):
        return None
    return prefix


def token_edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Compute Levenshtein distance without retaining the full DP matrix."""

    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_item in enumerate(right, 1):
        current = [row]
        for column, left_item in enumerate(left, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def compare_token_sequences(
    baseline_token_ids: Sequence[int],
    intervention_token_ids: Sequence[int],
) -> dict:
    baseline = [int(value) for value in baseline_token_ids]
    intervention = [int(value) for value in intervention_token_ids]
    prefix = common_prefix_length(baseline, intervention)
    return {
        "response_equal": baseline == intervention,
        "baseline_token_count": len(baseline),
        "intervention_token_count": len(intervention),
        "common_prefix_token_count": prefix,
        "first_different_token_position": first_different_position(
            baseline,
            intervention,
        ),
        "token_edit_distance": token_edit_distance(baseline, intervention),
    }


def summarize_response_changes(rows: Sequence[Mapping[str, object]]) -> dict:
    """Summarize response/token and attack-success changes."""

    if not rows:
        raise ValueError("response comparison rows must not be empty")
    required = {
        "dataset_index",
        "response_equal",
        "baseline_success",
        "intervention_success",
        "baseline_token_count",
        "intervention_token_count",
    }
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"response comparison row missing fields: {sorted(missing)}")
    dataset_indices = [int(row["dataset_index"]) for row in rows]
    if len(dataset_indices) != len(set(dataset_indices)):
        raise ValueError("response comparison contains duplicate dataset indices")
    changed = [row for row in rows if not bool(row["response_equal"])]
    success_changed = [
        row
        for row in changed
        if bool(row["baseline_success"]) != bool(row["intervention_success"])
    ]
    token_count_changed = [
        row
        for row in rows
        if int(row["baseline_token_count"])
        != int(row["intervention_token_count"])
    ]
    return {
        "prompt_count": len(rows),
        "response_equal_count": len(rows) - len(changed),
        "response_changed_count": len(changed),
        "success_changed_count": len(success_changed),
        "success_same_response_changed_count": len(changed) - len(success_changed),
        "response_token_count_changed_count": len(token_count_changed),
        "response_changed_dataset_indices": [
            int(row["dataset_index"]) for row in changed
        ],
        "success_changed_dataset_indices": [
            int(row["dataset_index"]) for row in success_changed
        ],
        "response_token_count_changed_dataset_indices": [
            int(row["dataset_index"]) for row in token_count_changed
        ],
    }


def validate_head_subset(
    heads: Sequence[tuple[int, int]],
    *,
    expected_heads: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    expected = tuple(expected_heads)
    values = tuple(heads)
    if not values:
        raise ValueError("head subset must not be empty")
    if len(values) != len(set(values)):
        raise ValueError("head subset contains duplicates")
    unexpected = [head for head in values if head not in expected]
    if unexpected:
        raise ValueError(f"head subset contains unexpected heads: {unexpected}")
    return values


def build_head_subset_conditions(
    expected_heads: Sequence[tuple[int, int]],
) -> list[dict]:
    """Build fixed full, single-head, and leave-one-out diagnostics."""

    expected = tuple(expected_heads)
    if len(expected) != 5 or len(set(expected)) != 5:
        raise ValueError("expected_heads must contain five unique heads")
    conditions = [{"name": "full_top5", "selected_heads": expected}]
    conditions.extend(
        {
            "name": f"single_{layer}_{head}",
            "selected_heads": ((layer, head),),
        }
        for layer, head in expected
    )
    conditions.extend(
        {
            "name": f"without_{layer}_{head}",
            "selected_heads": tuple(
                pair for pair in expected if pair != (layer, head)
            ),
            "omitted_head": (layer, head),
        }
        for layer, head in expected
    )
    for condition in conditions:
        condition["selected_heads"] = validate_head_subset(
            condition["selected_heads"],
            expected_heads=expected,
        )
    return conditions


def _finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} is NaN or infinity")
    return value


def compute_logit_step_metrics(
    baseline_logits: torch.Tensor,
    intervention_logits: torch.Tensor,
    *,
    step: int,
) -> dict:
    """Compare two vocab-logit vectors using float32 probabilities."""

    if not isinstance(baseline_logits, torch.Tensor) or not isinstance(
        intervention_logits,
        torch.Tensor,
    ):
        raise TypeError("logits must be torch tensors")
    if baseline_logits.ndim != 1 or intervention_logits.ndim != 1:
        raise ValueError("logits must be one-dimensional vocab vectors")
    if baseline_logits.shape != intervention_logits.shape:
        raise ValueError("baseline/intervention logit shapes differ")
    baseline = baseline_logits.float()
    intervention = intervention_logits.float()
    if not bool(torch.isfinite(baseline).all().item()) or not bool(
        torch.isfinite(intervention).all().item()
    ):
        raise ValueError("logits contain NaN or infinity")

    difference = intervention - baseline
    baseline_log_probs = torch.log_softmax(baseline, dim=-1)
    intervention_log_probs = torch.log_softmax(intervention, dim=-1)
    baseline_probs = torch.exp(baseline_log_probs)
    kl = torch.sum(
        baseline_probs * (baseline_log_probs - intervention_log_probs)
    )
    # Tiny negative round-off is not a negative KL divergence in substance.
    kl_value = max(0.0, float(kl.detach().cpu().item()))
    baseline_top = torch.topk(baseline, k=2)
    intervention_top = torch.topk(intervention, k=2)
    # Greedy generation uses torch.argmax.  torch.topk can return a different
    # index when several tokens have exactly tied maximum logits.
    baseline_top1_id = int(torch.argmax(baseline).item())
    intervention_top1_id = int(torch.argmax(intervention).item())
    baseline_top1_logit = float(baseline[baseline_top1_id].item())
    intervention_at_baseline_top1 = float(
        intervention[baseline_top1_id].item()
    )
    baseline_margin = float(
        (baseline_top.values[0] - baseline_top.values[1]).item()
    )
    intervention_margin = float(
        (intervention_top.values[0] - intervention_top.values[1]).item()
    )
    result = {
        "step": int(step),
        "max_absolute_logit_difference": float(difference.abs().max().item()),
        "mean_absolute_logit_difference": float(difference.abs().mean().item()),
        "kl_baseline_to_intervention": kl_value,
        "baseline_top1_token_id": baseline_top1_id,
        "intervention_top1_token_id": intervention_top1_id,
        "argmax_equal": baseline_top1_id == intervention_top1_id,
        "baseline_top1_logit": baseline_top1_logit,
        "intervention_logit_at_baseline_top1": intervention_at_baseline_top1,
        "top1_logit_change": intervention_at_baseline_top1 - baseline_top1_logit,
        "baseline_top1_top2_margin": baseline_margin,
        "intervention_top1_top2_margin": intervention_margin,
        "margin_change": intervention_margin - baseline_margin,
    }
    for name, value in result.items():
        if name not in {
            "step",
            "baseline_top1_token_id",
            "intervention_top1_token_id",
            "argmax_equal",
        }:
            _finite_scalar(value, name)
    return result


def aggregate_logit_steps(steps: Sequence[Mapping[str, object]]) -> dict:
    if not steps:
        raise ValueError("logit steps must not be empty")
    required = {
        "step",
        "max_absolute_logit_difference",
        "mean_absolute_logit_difference",
        "kl_baseline_to_intervention",
        "argmax_equal",
        "top1_logit_change",
        "margin_change",
    }
    for record in steps:
        missing = required - set(record)
        if missing:
            raise ValueError(f"logit step missing fields: {sorted(missing)}")
    changed_steps = [
        int(record["step"]) for record in steps if not bool(record["argmax_equal"])
    ]
    mean_logit = [
        _finite_scalar(record["mean_absolute_logit_difference"], "mean logit diff")
        for record in steps
    ]
    kl_values = [
        _finite_scalar(record["kl_baseline_to_intervention"], "KL")
        for record in steps
    ]
    top1_changes = [
        _finite_scalar(record["top1_logit_change"], "top1 logit change")
        for record in steps
    ]
    margin_changes = [
        _finite_scalar(record["margin_change"], "margin change")
        for record in steps
    ]
    return {
        "generation_step_count": len(steps),
        "argmax_change_step_count": len(changed_steps),
        "argmax_changed_steps": changed_steps,
        "first_argmax_change_step": changed_steps[0] if changed_steps else None,
        "max_logit_difference_max": max(
            float(record["max_absolute_logit_difference"]) for record in steps
        ),
        "mean_logit_difference_mean": sum(mean_logit) / len(mean_logit),
        "kl_mean": sum(kl_values) / len(kl_values),
        "kl_max": max(kl_values),
        "top1_logit_change_mean": sum(top1_changes) / len(top1_changes),
        "top1_logit_change_max_absolute": max(abs(value) for value in top1_changes),
        "margin_change_mean": sum(margin_changes) / len(margin_changes),
        "all_finite": True,
    }


def validate_logit_diagnostic_record(
    record: Mapping[str, object],
    *,
    mode: str,
) -> None:
    if mode not in LOGIT_MODES:
        raise ValueError(f"unsupported logit diagnostic mode: {mode}")
    if record.get("diagnostic_mode") != mode:
        raise ValueError("logit diagnostic mode/schema mismatch")
    required = {
        "dataset_index",
        "alpha",
        "steps",
        "summary",
        "baseline_response_token_ids",
        "intervention_response_token_ids",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"logit diagnostic record missing fields: {sorted(missing)}")
    if not isinstance(record["steps"], list):
        raise ValueError("logit diagnostic steps must be a list")
    aggregate_logit_steps(record["steps"])
