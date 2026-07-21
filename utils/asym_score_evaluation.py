"""Model-independent reporting core for asymmetric-score experiments."""

from __future__ import annotations

from typing import Mapping, Sequence


TRANSITIONS = (
    "defended",
    "remained_success",
    "regressed",
    "remained_failure",
)


def classify_transition(
    baseline_success: bool | None,
    current_success: bool | None,
    *,
    generation_failed: bool = False,
) -> str | None:
    if generation_failed or baseline_success is None or current_success is None:
        return None
    if baseline_success and current_success:
        return "remained_success"
    if baseline_success and not current_success:
        return "defended"
    if not baseline_success and current_success:
        return "regressed"
    return "remained_failure"


def attach_baseline_transitions(
    rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> list[dict]:
    """Attach transitions using only the no-intervention rows from this run."""

    baseline = {}
    for row in baseline_rows:
        index = row.get("source_index")
        if index in baseline:
            raise ValueError(f"duplicate baseline source_index: {index}")
        baseline[index] = row
    output = []
    for row in rows:
        index = row.get("source_index")
        if index not in baseline:
            raise ValueError(f"missing run baseline source_index: {index}")
        baseline_success = baseline[index].get("success")
        current_success = row.get("success")
        failed = bool(row.get("generation_failed"))
        if baseline_success is not None and type(baseline_success) is not bool:
            raise ValueError(f"invalid baseline success for source_index {index}")
        if current_success is not None and type(current_success) is not bool:
            raise ValueError(f"invalid success for source_index {index}")
        output.append(
            {
                **row,
                "baseline_success": baseline_success,
                "transition": classify_transition(
                    baseline_success,
                    current_success,
                    generation_failed=failed,
                ),
            }
        )
    return output


def summarize_prompt_results(
    rows: Sequence[Mapping[str, object]],
    *,
    condition: str,
    heads: Sequence[tuple[int, int]],
    beta: float | None,
) -> dict:
    """Summarize results; generation failures are excluded from ASR."""

    evaluated = [row for row in rows if not row.get("generation_failed")]
    success_count = sum(row.get("success") is True for row in evaluated)
    baseline_known = [
        row for row in rows if type(row.get("baseline_success")) is bool
    ]
    baseline_success_count = sum(
        row["baseline_success"] is True for row in baseline_known
    )
    baseline_failure_count = len(baseline_known) - baseline_success_count
    transition_counts = {
        name: sum(row.get("transition") == name for row in rows)
        for name in TRANSITIONS
    }
    lengths = [
        row.get("response_length_tokens")
        for row in evaluated
        if isinstance(row.get("response_length_tokens"), int)
    ]
    observed = [
        float(row["mean_observed_asym_scale"])
        for row in evaluated
        if row.get("mean_observed_asym_scale") is not None
    ]
    return {
        "condition": condition,
        "heads": [{"layer": layer, "head": head} for layer, head in heads],
        "beta": beta,
        "total_count": len(rows),
        "evaluated_count": len(evaluated),
        "asr_denominator": len(evaluated),
        "success_count": success_count,
        "asr": success_count / len(evaluated) if evaluated else None,
        "baseline_success_count": baseline_success_count,
        "baseline_failure_count": baseline_failure_count,
        "defended_count": transition_counts["defended"],
        "remained_success_count": transition_counts["remained_success"],
        "regressed_count": transition_counts["regressed"],
        "remained_failure_count": transition_counts["remained_failure"],
        "defense_success_rate": (
            transition_counts["defended"] / baseline_success_count
            if baseline_success_count
            else None
        ),
        "regression_rate": (
            transition_counts["regressed"] / baseline_failure_count
            if baseline_failure_count
            else None
        ),
        "generation_failure_count": len(rows) - len(evaluated),
        "empty_response_count": sum(
            not row.get("generation_failed") and row.get("response") == ""
            for row in rows
        ),
        "mean_response_length_tokens": (
            sum(lengths) / len(lengths) if lengths else None
        ),
        "mean_observed_asym_scale": (
            sum(observed) / len(observed) if observed else None
        ),
        "max_intervention_identity_error": max(
            (
                float(row.get("max_intervention_identity_error", 0.0))
                for row in rows
            ),
            default=0.0,
        ),
        "max_symmetric_component_error": max(
            (
                float(row.get("max_symmetric_component_error", 0.0))
                for row in rows
            ),
            default=0.0,
        ),
        "max_unselected_head_error": max(
            (float(row.get("max_unselected_head_error", 0.0)) for row in rows),
            default=0.0,
        ),
    }
