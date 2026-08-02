"""CPU-testable helpers for the GCG first-token-only development ablation."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from utils.gcg_wasserstein_reference import (
    TOTAL_COUNT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    validate_split_manifest,
)
from utils.gcg_wasserstein_reference_evaluation import distribution_statistics


SELECTION_COUNT = 20
EXPECTED_SELECTION_DATASET_INDICES = (
    68,
    70,
    72,
    73,
    84,
    87,
    88,
    91,
    96,
    117,
    123,
    127,
    134,
    143,
    163,
    181,
    208,
    215,
    217,
    218,
)
COMPARISON_FIELDS = (
    "timing",
    "alpha",
    "prompt_count",
    "attack_success_count",
    "newly_defended_count",
    "still_attack_success_count",
    "response_equal_count",
    "response_changed_count",
    "first_token_argmax_changed_count",
    "generation_failure_count",
    "empty_response_count",
    "total_intervention_count",
    "later_forward_intervention_count",
)


def load_selection_development_records(
    *,
    split_manifest_path: str | Path,
    jailbreak_input: str | Path,
    baseline_results: str | Path,
) -> tuple[list[dict], dict]:
    """Load the fixed selection 20 with explicit dataset/row/hash validation."""

    split_manifest_path = Path(split_manifest_path)
    jailbreak_input = Path(jailbreak_input)
    baseline_results = Path(baseline_results)
    manifest = read_json(split_manifest_path)
    validate_split_manifest(manifest)
    if sha256_file(jailbreak_input) != manifest["jailbreak_input_sha256"]:
        raise ValueError("jailbreak input hash does not match split manifest")
    if sha256_file(baseline_results) != manifest["baseline_results_sha256"]:
        raise ValueError("baseline result hash does not match split manifest")
    source_rows = read_jsonl(jailbreak_input)
    baseline_rows = read_jsonl(baseline_results)
    if len(source_rows) != TOTAL_COUNT or len(baseline_rows) != TOTAL_COUNT:
        raise ValueError("selection inputs must contain exactly 160 rows")

    records = []
    for split_record in manifest["records"]:
        if split_record["split"] != "selection":
            continue
        row_position = int(split_record["row_position"])
        dataset_index = int(split_record["dataset_index"])
        source = source_rows[row_position]
        baseline = baseline_rows[row_position]
        if source.get("index") != dataset_index:
            raise ValueError("jailbreak row/dataset index mismatch")
        prompt = source.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("selection prompt is missing")
        if sha256_text(prompt) != split_record["jailbreak_prompt_sha256"]:
            raise ValueError("selection prompt hash mismatch")
        if baseline.get("source_index") != dataset_index:
            raise ValueError("baseline row/dataset index mismatch")
        if baseline.get("prompt") != prompt:
            raise ValueError("baseline prompt mismatch")
        if baseline.get("success") is not True:
            raise ValueError("selection baseline must be attack success")
        if baseline.get("generation_failed") is not False:
            raise ValueError("selection baseline contains a generation failure")
        records.append(
            {
                "prompt_id": split_record["prompt_id"],
                "row_position": row_position,
                "dataset_index": dataset_index,
                "split": "selection",
                "prompt": prompt,
                "jailbreak_prompt_sha256": split_record[
                    "jailbreak_prompt_sha256"
                ],
                "baseline_response": baseline["response"],
                "baseline_success": True,
                "baseline_generation_failed": False,
                "baseline_response_token_count": baseline.get(
                    "response_length_tokens"
                ),
            }
        )
    indices = tuple(record["dataset_index"] for record in records)
    if len(records) != SELECTION_COUNT:
        raise ValueError("selection development record count mismatch")
    if indices != EXPECTED_SELECTION_DATASET_INDICES:
        raise ValueError(f"selection dataset indices mismatch: {indices}")
    if len(indices) != len(set(indices)):
        raise ValueError("selection dataset indices contain duplicates")
    return records, manifest


def validate_first_token_response_invariant(
    *,
    response_equal: bool,
    first_different_token_position: int | None,
) -> None:
    if response_equal:
        if first_different_token_position is not None:
            raise ValueError("equal responses cannot have a differing token")
        return
    if first_different_token_position != 0:
        raise ValueError(
            "first-token-only response changed after token 0; "
            "later-forward intervention is suspected"
        )


def summarize_development_condition(
    rows: Sequence[Mapping[str, object]],
    *,
    timing: str,
    alpha: float,
    expected_count: int = SELECTION_COUNT,
) -> dict:
    if len(rows) != expected_count:
        raise ValueError(f"condition row count mismatch: {len(rows)}")
    indices = [int(row["dataset_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("condition rows contain duplicate dataset indices")
    failures = [row for row in rows if bool(row["generation_failed"])]
    changed = [row for row in rows if not bool(row["response_equal"])]
    argmax_changed = [
        row for row in rows if not bool(row["first_token_argmax_equal"])
    ]
    newly_defended = [
        row
        for row in rows
        if not row["generation_failed"] and not bool(row["success"])
    ]
    still_success = [
        row
        for row in rows
        if not row["generation_failed"] and bool(row["success"])
    ]
    if timing == "first_token_only":
        for row in rows:
            validate_first_token_response_invariant(
                response_equal=bool(row["response_equal"]),
                first_different_token_position=row[
                    "first_different_token_position"
                ],
            )
        if {row["dataset_index"] for row in changed} != {
            row["dataset_index"] for row in argmax_changed
        }:
            raise ValueError(
                "first-token-only response changes and first-token argmax changes differ"
            )
    return {
        "timing": timing,
        "alpha": float(alpha),
        "prompt_count": len(rows),
        "attack_success_count": len(still_success),
        "newly_defended_count": len(newly_defended),
        "still_attack_success_count": len(still_success),
        "newly_defended_dataset_indices": [
            int(row["dataset_index"]) for row in newly_defended
        ],
        "still_attack_success_dataset_indices": [
            int(row["dataset_index"]) for row in still_success
        ],
        "response_equal_count": len(rows) - len(changed),
        "response_changed_count": len(changed),
        "response_changed_dataset_indices": [
            int(row["dataset_index"]) for row in changed
        ],
        "first_token_argmax_changed_count": len(argmax_changed),
        "first_token_argmax_changed_dataset_indices": [
            int(row["dataset_index"]) for row in argmax_changed
        ],
        "response_change_matches_first_token_argmax_change": (
            {row["dataset_index"] for row in changed}
            == {row["dataset_index"] for row in argmax_changed}
        ),
        "generation_failure_count": len(failures),
        "generation_failure_dataset_indices": [
            int(row["dataset_index"]) for row in failures
        ],
        "empty_response_count": sum(
            row.get("empty_response") is True for row in rows
        ),
        "nan_or_inf_count": sum(
            not bool(row["diagnostics"]["all_scores_finite"]) for row in rows
        ),
        "total_intervention_count": sum(
            int(row["diagnostics"]["total_intervention_count"])
            for row in rows
        ),
        "later_forward_intervention_count": sum(
            int(row["diagnostics"]["later_forward_intervention_count"])
            for row in rows
        ),
    }


def summarize_first_token_logits(
    rows: Sequence[Mapping[str, object]],
    *,
    alphas: Sequence[float],
) -> dict:
    output = []
    for alpha in alphas:
        selected = [row for row in rows if float(row["alpha"]) == float(alpha)]
        if len(selected) != SELECTION_COUNT:
            raise ValueError(f"first-token logit count mismatch for alpha={alpha}")
        metrics = [row["metrics"] for row in selected]
        output.append(
            {
                "alpha": float(alpha),
                "prompt_count": len(selected),
                "argmax_changed_count": sum(
                    not bool(metric["argmax_equal"]) for metric in metrics
                ),
                "argmax_changed_dataset_indices": [
                    int(row["dataset_index"])
                    for row in selected
                    if not bool(row["metrics"]["argmax_equal"])
                ],
                "max_absolute_logit_difference": distribution_statistics(
                    [
                        float(metric["max_absolute_logit_difference"])
                        for metric in metrics
                    ]
                ),
                "mean_absolute_logit_difference": distribution_statistics(
                    [
                        float(metric["mean_absolute_logit_difference"])
                        for metric in metrics
                    ]
                ),
                "kl_baseline_to_intervention": distribution_statistics(
                    [
                        float(metric["kl_baseline_to_intervention"])
                        for metric in metrics
                    ]
                ),
                "margin_change": distribution_statistics(
                    [float(metric["margin_change"]) for metric in metrics]
                ),
                "all_finite": True,
            }
        )
    return {
        "kl_direction": "D_KL(P_baseline || P_intervention)",
        "probability_dtype": "torch.float32",
        "conditions": output,
    }


def write_comparison_table(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite comparison table: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in COMPARISON_FIELDS} for row in rows
        )

