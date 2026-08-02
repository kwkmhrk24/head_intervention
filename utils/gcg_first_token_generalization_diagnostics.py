"""CPU helpers for selection/holdout first-token generalization diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path

from utils.gcg_reference_intervention_diagnostics import (
    build_head_subset_conditions,
)
from utils.gcg_wasserstein_reference import read_jsonl
from utils.gcg_wasserstein_reference_evaluation import distribution_statistics


ALPHAS = (0.0, 0.1, 0.3, 0.5)
PRIMARY_ALPHA = 0.5
EXPECTED_TOP5 = ((31, 22), (13, 4), (5, 30), (9, 18), (27, 24))
GROUP_COUNTS = {"selection20": 20, "holdout48": 48}
LOGIT_DISTRIBUTION_FIELDS = (
    "max_absolute_logit_difference",
    "kl_baseline_to_intervention",
    "baseline_top1_top2_margin",
    "intervention_top1_top2_margin",
)
SCORE_FIELDS = (
    "current_asym_frobenius_norm",
    "reference_asym_frobenius_norm",
    "intervened_asym_frobenius_norm",
    "current_to_reference_frobenius_distance",
    "current_to_reference_mean_absolute_distance",
    "mean_absolute_delta",
    "max_absolute_delta",
    "intervened_to_current_norm_ratio",
)


def _finite(value: object, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} is NaN or infinity")
    return value


def normalize_logit_metric(
    *,
    dataset_group: str,
    dataset_index: int,
    baseline_success: bool,
    alpha: float,
    metric: Mapping[str, object],
) -> dict:
    if dataset_group not in GROUP_COUNTS:
        raise ValueError(f"unknown dataset group: {dataset_group}")
    required = {
        "baseline_top1_token_id",
        "intervention_top1_token_id",
        "argmax_equal",
        "max_absolute_logit_difference",
        "mean_absolute_logit_difference",
        "kl_baseline_to_intervention",
        "baseline_top1_logit",
        "intervention_logit_at_baseline_top1",
        "top1_logit_change",
        "baseline_top1_top2_margin",
        "intervention_top1_top2_margin",
        "margin_change",
    }
    missing = required - set(metric)
    if missing:
        raise ValueError(f"saved first-step metric is missing: {sorted(missing)}")
    record = {
        "dataset_group": dataset_group,
        "dataset_index": int(dataset_index),
        "baseline_success": bool(baseline_success),
        "alpha": float(alpha),
        "baseline_top1_token_id": int(metric["baseline_top1_token_id"]),
        "intervention_top1_token_id": int(metric["intervention_top1_token_id"]),
        "argmax_equal": bool(metric["argmax_equal"]),
        "max_absolute_logit_difference": _finite(
            metric["max_absolute_logit_difference"], "maximum logit difference"
        ),
        "mean_absolute_logit_difference": _finite(
            metric["mean_absolute_logit_difference"], "mean logit difference"
        ),
        "kl_baseline_to_intervention": _finite(
            metric["kl_baseline_to_intervention"], "KL"
        ),
        "baseline_top1_logit": _finite(metric["baseline_top1_logit"], "top1 logit"),
        "intervention_logit_for_baseline_top1": _finite(
            metric["intervention_logit_at_baseline_top1"],
            "intervention baseline-top1 logit",
        ),
        "baseline_top1_logit_change": _finite(
            metric["top1_logit_change"], "top1 logit change"
        ),
        "baseline_top1_top2_margin": _finite(
            metric["baseline_top1_top2_margin"], "baseline margin"
        ),
        "intervention_top1_top2_margin": _finite(
            metric["intervention_top1_top2_margin"], "intervention margin"
        ),
        "margin_change": _finite(metric["margin_change"], "margin change"),
    }
    if record["kl_baseline_to_intervention"] < 0:
        raise ValueError("KL must not be negative")
    return record


def load_group_logit_records(
    *,
    selection_run: str | Path,
    holdout_run: str | Path,
) -> list[dict]:
    selection_run = Path(selection_run)
    holdout_run = Path(holdout_run)
    selection_baseline = {
        int(row["dataset_index"]): bool(row["baseline_success"])
        for row in read_jsonl(selection_run / "baseline_selection_results.jsonl")
    }
    if len(selection_baseline) != 20 or not all(selection_baseline.values()):
        raise ValueError("selection baseline map must contain 20 attack successes")
    records = []
    for row in read_jsonl(selection_run / "first_token_logit_diagnostics.jsonl"):
        records.append(
            normalize_logit_metric(
                dataset_group="selection20",
                dataset_index=row["dataset_index"],
                baseline_success=selection_baseline[int(row["dataset_index"])],
                alpha=row["alpha"],
                metric=row["metrics"],
            )
        )
    for alpha in ALPHAS:
        for row in read_jsonl(holdout_run / f"alpha_{alpha:.1f}/prompt_results.jsonl"):
            records.append(
                normalize_logit_metric(
                    dataset_group="holdout48",
                    dataset_index=row["dataset_index"],
                    baseline_success=row["baseline_success"],
                    alpha=alpha,
                    metric=row["first_token_logit_metrics"],
                )
            )
    for group, expected_count in GROUP_COUNTS.items():
        selected = [record for record in records if record["dataset_group"] == group]
        if len(selected) != expected_count * len(ALPHAS):
            raise ValueError(f"{group} first-step record count mismatch")
        keys = {(record["dataset_index"], record["alpha"]) for record in selected}
        if len(keys) != len(selected):
            raise ValueError(f"{group} first-step records contain duplicates")
        for dataset_index in {record["dataset_index"] for record in selected}:
            baseline_values = {
                (
                    record["baseline_top1_token_id"],
                    record["baseline_top1_logit"],
                    record["baseline_top1_top2_margin"],
                )
                for record in selected
                if record["dataset_index"] == dataset_index
            }
            if len(baseline_values) != 1:
                raise ValueError("baseline first-step values differ across alpha")
    return records


def _distribution(values: Sequence[float]) -> dict:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    return distribution_statistics([_finite(value, "distribution value") for value in values])


def summarize_group_logits(records: Sequence[Mapping[str, object]]) -> dict:
    conditions = []
    for group, expected_count in GROUP_COUNTS.items():
        for alpha in ALPHAS:
            all_rows = [
                record
                for record in records
                if record["dataset_group"] == group
                and float(record["alpha"]) == alpha
            ]
            if len(all_rows) != expected_count:
                raise ValueError(f"{group} alpha={alpha} count mismatch")
            for subset_name, rows in (
                ("all", all_rows),
                ("baseline_success", [row for row in all_rows if row["baseline_success"]]),
            ):
                if subset_name == "baseline_success" and group == "selection20" and len(rows) != 20:
                    raise ValueError("selection baseline-success subset must contain 20")
                if subset_name == "baseline_success" and group == "holdout48" and len(rows) != 21:
                    raise ValueError("holdout baseline-success subset must contain 21")
                distributions = {
                    field: _distribution([float(row[field]) for row in rows])
                    for field in LOGIT_DISTRIBUTION_FIELDS
                }
                distributions["absolute_margin_change"] = _distribution(
                    [abs(float(row["margin_change"])) for row in rows]
                )
                conditions.append(
                    {
                        "dataset_group": group,
                        "subset": subset_name,
                        "alpha": alpha,
                        "prompt_count": len(rows),
                        "argmax_changed_count": sum(not row["argmax_equal"] for row in rows),
                        "argmax_changed_dataset_indices": [
                            int(row["dataset_index"]) for row in rows if not row["argmax_equal"]
                        ],
                        **distributions,
                    }
                )
    return {
        "kl_direction": "D_KL(P_baseline || P_intervention)",
        "probability_dtype": "torch.float32",
        "conditions": conditions,
    }


def percentile_rank(values: Sequence[float], value: float) -> float:
    """Inclusive empirical percentile: 100 * count(x <= value) / n."""

    finite_values = [_finite(item, "percentile value") for item in values]
    value = _finite(value, "percentile target")
    if not finite_values:
        raise ValueError("percentile values must not be empty")
    return 100.0 * sum(item <= value for item in finite_values) / len(finite_values)


def build_index88_context(records: Sequence[Mapping[str, object]]) -> dict:
    selection = [
        row
        for row in records
        if row["dataset_group"] == "selection20" and row["alpha"] == PRIMARY_ALPHA
    ]
    holdout = [
        row
        for row in records
        if row["dataset_group"] == "holdout48" and row["alpha"] == PRIMARY_ALPHA
    ]
    index88 = next((row for row in selection if row["dataset_index"] == 88), None)
    if index88 is None:
        raise ValueError("selection index 88 first-step metric is missing")
    margin = float(index88["baseline_top1_top2_margin"])
    logit_change = float(index88["max_absolute_logit_difference"])
    kl = float(index88["kl_baseline_to_intervention"])
    return {
        "alpha": PRIMARY_ALPHA,
        "percentile_definition": "inclusive empirical: 100*count(x<=value)/n",
        "index88_baseline_margin": margin,
        "index88_baseline_margin_percentile_within_selection20": percentile_rank(
            [row["baseline_top1_top2_margin"] for row in selection], margin
        ),
        "index88_max_absolute_logit_difference": logit_change,
        "index88_logit_difference_percentile_within_selection20": percentile_rank(
            [row["max_absolute_logit_difference"] for row in selection], logit_change
        ),
        "index88_kl": kl,
        "index88_kl_percentile_within_selection20": percentile_rank(
            [row["kl_baseline_to_intervention"] for row in selection], kl
        ),
        "holdout_minimum_baseline_margin": min(
            float(row["baseline_top1_top2_margin"]) for row in holdout
        ),
        "holdout_maximum_logit_difference": max(
            float(row["max_absolute_logit_difference"]) for row in holdout
        ),
        "holdout_logit_difference_at_least_index88_count": sum(
            float(row["max_absolute_logit_difference"]) >= logit_change for row in holdout
        ),
        "holdout_logit_difference_at_least_index88_dataset_indices": [
            int(row["dataset_index"])
            for row in holdout
            if float(row["max_absolute_logit_difference"]) >= logit_change
        ],
        "holdout_baseline_margin_at_most_index88_count": sum(
            float(row["baseline_top1_top2_margin"]) <= margin for row in holdout
        ),
        "holdout_baseline_margin_at_most_index88_dataset_indices": [
            int(row["dataset_index"])
            for row in holdout
            if float(row["baseline_top1_top2_margin"]) <= margin
        ],
    }


def validate_score_record(record: Mapping[str, object]) -> dict:
    required = {"dataset_group", "dataset_index", "baseline_success", "alpha", "layer", "head"} | set(SCORE_FIELDS)
    missing = required - set(record)
    if missing:
        raise ValueError(f"score diagnostic missing fields: {sorted(missing)}")
    output = dict(record)
    for field in SCORE_FIELDS:
        output[field] = _finite(record[field], field)
    if output["current_to_reference_frobenius_distance"] < 0:
        raise ValueError("current-reference distance must not be negative")
    return output


def summarize_score_interventions(records: Sequence[Mapping[str, object]]) -> dict:
    validated = [validate_score_record(record) for record in records]
    groups = []
    for layer, head in EXPECTED_TOP5:
        group_stats = {}
        for group, expected_count in GROUP_COUNTS.items():
            rows = [
                row for row in validated
                if row["dataset_group"] == group
                and (row["layer"], row["head"]) == (layer, head)
            ]
            if len(rows) != expected_count:
                raise ValueError(f"score count mismatch for {group}, {(layer, head)}")
            group_stats[group] = {
                field: {
                    key: value
                    for key, value in _distribution([row[field] for row in rows]).items()
                    if key in {"median", "mean", "p95"}
                }
                for field in SCORE_FIELDS
            }
        differences = {
            field: {
                statistic: group_stats["holdout48"][field][statistic]
                - group_stats["selection20"][field][statistic]
                for statistic in ("median", "mean", "p95")
            }
            for field in SCORE_FIELDS
        }
        groups.append(
            {
                "layer": layer,
                "head": head,
                "selection20": group_stats["selection20"],
                "holdout48": group_stats["holdout48"],
                "holdout_minus_selection": differences,
            }
        )
    return {"alpha": PRIMARY_ALPHA, "heads": groups, "all_finite": True}


def build_all_nonempty_head_subsets(
    expected_heads: Sequence[tuple[int, int]] = EXPECTED_TOP5,
) -> list[dict]:
    heads = tuple(expected_heads)
    if len(heads) != 5 or len(set(heads)) != 5:
        raise ValueError("expected heads must be five unique values")
    output = []
    for size in range(1, len(heads) + 1):
        for subset in combinations(heads, size):
            output.append(
                {
                    "name": "subset_" + "__".join(f"{layer}_{head}" for layer, head in subset),
                    "subset_size": size,
                    "selected_heads": subset,
                }
            )
    if len(output) != 31 or len({item["selected_heads"] for item in output}) != 31:
        raise AssertionError("five heads must produce 31 unique nonempty subsets")
    return output


def head_contribution_conditions() -> list[dict]:
    return build_head_subset_conditions(EXPECTED_TOP5)


def minimal_refusal_subsets(rows: Sequence[Mapping[str, object]]) -> dict:
    refused = [row for row in rows if row.get("generation_failed") is False and row.get("success") is False]
    if not refused:
        return {"minimum_subset_size": None, "minimal_subsets": []}
    minimum = min(int(row["subset_size"]) for row in refused)
    return {
        "minimum_subset_size": minimum,
        "minimal_subsets": [
            {
                "selected_heads": row["selected_heads"],
                "first_token_id": row["first_token_id"],
                "success": row["success"],
                "response_equal_to_full_top5": row["response_equal_to_full_top5"],
            }
            for row in refused
            if int(row["subset_size"]) == minimum
        ],
    }
