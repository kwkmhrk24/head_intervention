"""Helpers for the exploratory 48-row Llama-GCG-specific holdout.

The raw archive contains two non-canonical candidates for AdvBench indices 10
and 13.  Both candidates are retained as excluded manifest records; every
other index in 0..49 contributes exactly one included record.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from utils.gcg_first_token_reference_ablation import (
    validate_first_token_response_invariant,
)
from utils.gcg_reference_intervention_diagnostics import (
    summarize_response_changes,
)
from utils.gcg_wasserstein_reference import (
    read_jsonl,
    sha256_file,
    sha256_text,
    utc_now,
    write_json,
)
from utils.gcg_wasserstein_reference_evaluation import (
    compare_alpha_zero_to_baseline,
    summarize_transitions,
)


HOLDOUT_INDICES = tuple(index for index in range(50) if index not in (10, 13))
EXCLUDED_AMBIGUOUS_INDICES = (10, 13)
HOLDOUT_COUNT = 48
RAW_MODEL = "llama2-7B-chat-hf"
RAW_METHOD = "GCG"
EVALUATION_ROLE = "exploratory_llama_gcg_holdout"
SEMANTIC_CLASSES = (
    "genuine_refusal",
    "pseudo_refusal_keyword_only",
    "harmful_compliance",
    "benign_or_irrelevant",
    "ambiguous",
)
AUDIT_CONFIDENCE_VALUES = ("high", "medium", "low")
MANIFEST_RECORD_FIELDS = (
    "dataset_index",
    "raw_file_position",
    "raw_archive_index",
    "original_prompt",
    "GCG_prompt",
    "original_prompt_sha256",
    "GCG_prompt_sha256",
    "raw_artifact_sha256",
    "included",
    "exclusion_reason",
)


def _read_raw_archive(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not all(
        isinstance(record, dict) for record in payload
    ):
        raise ValueError("raw GCG archive must be a JSON array of objects")
    return payload


def _records_sha256(records: Sequence[Mapping[str, object]]) -> str:
    canonical = json.dumps(
        [dict(record) for record in records],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(canonical)


def build_holdout_manifest(
    *,
    raw_archive_path: str | Path,
    advbench_path: str | Path,
    created_at: str | None = None,
) -> dict:
    """Build the pre-outcome manifest with 48 included and 4 excluded rows."""

    raw_archive_path = Path(raw_archive_path).resolve()
    advbench_path = Path(advbench_path).resolve()
    raw_rows = _read_raw_archive(raw_archive_path)
    advbench_rows = read_jsonl(advbench_path)
    if len(advbench_rows) != 520:
        raise ValueError(f"expected 520 AdvBench rows, found {len(advbench_rows)}")
    advbench_by_index = {}
    for row_position, row in enumerate(advbench_rows):
        dataset_index = row.get("index")
        prompt = row.get("prompt")
        if type(dataset_index) is not int or not isinstance(prompt, str) or not prompt:
            raise ValueError(f"invalid AdvBench row at position {row_position}")
        if dataset_index in advbench_by_index:
            raise ValueError(f"duplicate AdvBench index: {dataset_index}")
        advbench_by_index[dataset_index] = prompt
    if set(advbench_by_index) != set(range(520)):
        raise ValueError("AdvBench indices must be exactly 0..519")

    raw_sha256 = sha256_file(raw_archive_path)
    candidates = [
        (position, row)
        for position, row in enumerate(raw_rows)
        if type(row.get("index")) is int and 0 <= row["index"] <= 49
    ]
    multiplicity = Counter(int(row["index"]) for _, row in candidates)
    expected_multiplicity = {index: 1 for index in HOLDOUT_INDICES}
    expected_multiplicity.update({10: 2, 13: 2})
    if dict(sorted(multiplicity.items())) != expected_multiplicity:
        raise ValueError(
            "raw archive multiplicity mismatch for indices 0..49: "
            f"{dict(sorted(multiplicity.items()))}"
        )

    records = []
    for raw_file_position, raw in candidates:
        dataset_index = int(raw["index"])
        if raw.get("model") != RAW_MODEL or raw.get("method") != RAW_METHOD:
            raise ValueError(
                f"raw archive model/method mismatch at position {raw_file_position}"
            )
        if raw.get("system prompt") is not None:
            raise ValueError(
                f"raw archive system prompt is not null at position {raw_file_position}"
            )
        gcg_prompt = raw.get("user prompt")
        original_prompt = advbench_by_index[dataset_index]
        if not isinstance(gcg_prompt, str) or not gcg_prompt:
            raise ValueError(f"missing GCG prompt at position {raw_file_position}")
        if not gcg_prompt.startswith(original_prompt):
            raise ValueError(
                f"GCG prompt is not aligned to AdvBench index {dataset_index}"
            )
        included = dataset_index not in EXCLUDED_AMBIGUOUS_INDICES
        records.append(
            {
                "dataset_index": dataset_index,
                "raw_file_position": raw_file_position,
                "raw_archive_index": dataset_index,
                "original_prompt": original_prompt,
                "GCG_prompt": gcg_prompt,
                "original_prompt_sha256": sha256_text(original_prompt),
                "GCG_prompt_sha256": sha256_text(gcg_prompt),
                "raw_artifact_sha256": raw_sha256,
                "included": included,
                "exclusion_reason": (
                    None
                    if included
                    else "ambiguous_duplicate_archive_index_no_canonical_manifest"
                ),
            }
        )

    included_records = [record for record in records if record["included"]]
    included_indices = tuple(record["dataset_index"] for record in included_records)
    if included_indices != HOLDOUT_INDICES:
        raise ValueError(f"included holdout index order mismatch: {included_indices}")
    if (
        len(included_indices) != HOLDOUT_COUNT
        or len(set(included_indices)) != HOLDOUT_COUNT
    ):
        raise AssertionError("included holdout indices are not 48 unique values")
    excluded_records = [record for record in records if not record["included"]]
    if Counter(record["dataset_index"] for record in excluded_records) != Counter(
        {10: 2, 13: 2}
    ):
        raise AssertionError("excluded duplicate records were not retained")

    return {
        "schema_version": 1,
        "created_at": created_at or utc_now(),
        "evaluation_role": EVALUATION_ROLE,
        "evaluation_label": "exploratory Llama-GCG holdout",
        "alternative_evaluation_label": "Llama-GCG-specific holdout",
        "fully_unseen_across_all_prior_research": False,
        "unused_for_current_gcg_head_selection": True,
        "unused_for_current_reference_construction": True,
        "unused_for_current_gcg_evaluation": True,
        "original_behaviors_previously_used_in_autodan_research": True,
        "dataset_index_range_considered": [0, 49],
        "excluded_dataset_indices": list(EXCLUDED_AMBIGUOUS_INDICES),
        "exclusion_rule_fixed_before_baseline_generation": True,
        "included_count": len(included_records),
        "excluded_raw_record_count": len(excluded_records),
        "manifest_record_count": len(records),
        "raw_archive_path": str(raw_archive_path),
        "raw_artifact_sha256": raw_sha256,
        "advbench_path": str(advbench_path),
        "advbench_sha256": sha256_file(advbench_path),
        "records_sha256": _records_sha256(records),
        "records": records,
    }


def validate_holdout_manifest(
    manifest: Mapping[str, object],
    *,
    raw_archive_path: str | Path,
    advbench_path: str | Path,
) -> list[dict]:
    """Validate manifest hashes/content and return 48 runtime prompt records."""

    required_flags = {
        "evaluation_role": EVALUATION_ROLE,
        "fully_unseen_across_all_prior_research": False,
        "unused_for_current_gcg_head_selection": True,
        "unused_for_current_reference_construction": True,
        "unused_for_current_gcg_evaluation": True,
        "original_behaviors_previously_used_in_autodan_research": True,
    }
    for key, expected in required_flags.items():
        if manifest.get(key) != expected:
            raise ValueError(f"holdout manifest metadata mismatch: {key}")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 52:
        raise ValueError("holdout manifest must contain 52 raw records")
    if manifest.get("records_sha256") != _records_sha256(records):
        raise ValueError("holdout manifest records hash mismatch")
    if manifest.get("raw_artifact_sha256") != sha256_file(raw_archive_path):
        raise ValueError("holdout manifest raw archive hash mismatch")
    if manifest.get("advbench_sha256") != sha256_file(advbench_path):
        raise ValueError("holdout manifest AdvBench hash mismatch")

    raw_rows = _read_raw_archive(raw_archive_path)
    advbench_rows = read_jsonl(advbench_path)
    runtime = []
    included_seen = set()
    for record in records:
        missing = set(MANIFEST_RECORD_FIELDS) - set(record)
        if missing:
            raise ValueError(f"holdout record missing fields: {sorted(missing)}")
        raw_position = record["raw_file_position"]
        dataset_index = record["dataset_index"]
        if type(raw_position) is not int or not 0 <= raw_position < len(raw_rows):
            raise ValueError("holdout raw_file_position is invalid")
        if type(dataset_index) is not int or not 0 <= dataset_index <= 49:
            raise ValueError("holdout dataset_index is invalid")
        raw = raw_rows[raw_position]
        if (
            raw.get("index") != record["raw_archive_index"]
            or record["raw_archive_index"] != dataset_index
        ):
            raise ValueError("holdout raw archive index mismatch")
        if raw.get("user prompt") != record["GCG_prompt"]:
            raise ValueError("holdout raw GCG prompt mismatch")
        original = advbench_rows[dataset_index]
        if original.get("index") != dataset_index:
            raise ValueError("AdvBench row/index order mismatch")
        if original.get("prompt") != record["original_prompt"]:
            raise ValueError("holdout original prompt mismatch")
        if sha256_text(record["original_prompt"]) != record["original_prompt_sha256"]:
            raise ValueError("holdout original prompt hash mismatch")
        if sha256_text(record["GCG_prompt"]) != record["GCG_prompt_sha256"]:
            raise ValueError("holdout GCG prompt hash mismatch")
        if record["raw_artifact_sha256"] != manifest["raw_artifact_sha256"]:
            raise ValueError("holdout row raw artifact hash mismatch")
        if record["included"]:
            if dataset_index in included_seen:
                raise ValueError("included holdout dataset index is duplicated")
            included_seen.add(dataset_index)
            if record["exclusion_reason"] is not None:
                raise ValueError("included holdout row has an exclusion reason")
            runtime.append(
                {
                    "prompt_id": f"llama:gcg-holdout:{dataset_index}",
                    "row_position": len(runtime),
                    "dataset_index": dataset_index,
                    "raw_file_position": raw_position,
                    "raw_archive_index": record["raw_archive_index"],
                    "prompt": record["GCG_prompt"],
                    "GCG_prompt_sha256": record["GCG_prompt_sha256"],
                    "original_prompt_sha256": record["original_prompt_sha256"],
                }
            )
        elif dataset_index not in EXCLUDED_AMBIGUOUS_INDICES:
            raise ValueError("unexpected excluded holdout index")
    if tuple(record["dataset_index"] for record in runtime) != HOLDOUT_INDICES:
        raise ValueError("included runtime records do not match fixed 48 indices")
    return runtime


def write_holdout_manifest_artifacts(
    manifest: Mapping[str, object], output_dir: str | Path
) -> None:
    output_dir = Path(output_dir)
    write_json(output_dir / "holdout_manifest.json", dict(manifest))
    csv_path = output_dir / "holdout_manifest.csv"
    if csv_path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {csv_path}")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_RECORD_FIELDS)
        writer.writeheader()
        for record in manifest["records"]:
            writer.writerow({field: record[field] for field in MANIFEST_RECORD_FIELDS})


def validate_alpha_zero_gate(rows: Sequence[Mapping[str, object]]) -> dict:
    comparison = compare_alpha_zero_to_baseline(rows)
    if not comparison["passed"]:
        raise RuntimeError(
            "alpha=0 differs from baseline; refusing positive-alpha evaluation"
        )
    for row in rows:
        if "intervention_call_count" in row:
            intervention_count = row["intervention_call_count"]
        else:
            diagnostics = row.get("diagnostics")
            if not isinstance(diagnostics, Mapping):
                raise ValueError("alpha=0 result lacks intervention diagnostics")
            intervention_count = diagnostics.get("total_intervention_count")
        if type(intervention_count) is not int:
            raise ValueError("alpha=0 intervention count is not an integer")
        if intervention_count != 0:
            raise RuntimeError("alpha=0 performed a nonzero intervention")
    return comparison


def validate_first_token_condition(
    rows: Sequence[Mapping[str, object]],
    diagnostics: Mapping[str, object],
    *,
    expected_count: int,
) -> None:
    if len(rows) != expected_count:
        raise ValueError("first-token condition row count mismatch")
    for row in rows:
        validate_first_token_response_invariant(
            response_equal=bool(row["response_equal"]),
            first_different_token_position=row["first_different_token_position"],
        )
    expected_calls = expected_count * 5
    if diagnostics.get("total_intervention_count") != expected_calls:
        raise ValueError("first-token total intervention count mismatch")
    if diagnostics.get("later_forward_intervention_count") != 0:
        raise ValueError("first-token condition intervened on a later forward")
    per_head = diagnostics.get("per_head_intervention_count")
    if not isinstance(per_head, list) or any(
        item.get("count") != expected_count for item in per_head
    ):
        raise ValueError("first-token per-head intervention count mismatch")
    if diagnostics.get("all_scores_finite") is not True:
        raise ValueError("first-token diagnostics contain NaN or infinity")


def build_condition_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    alpha: float,
    diagnostics: Mapping[str, object] | None,
) -> dict:
    if len(rows) != HOLDOUT_COUNT:
        raise ValueError(f"expected {HOLDOUT_COUNT} holdout rows")
    transition = summarize_transitions(rows, expected_count=HOLDOUT_COUNT)
    response_rows = [
        {
            "dataset_index": row["dataset_index"],
            "response_equal": row["response_equal"],
            "baseline_success": row["baseline_success"],
            "intervention_success": row["success"],
            "baseline_token_count": row["baseline_token_count"],
            "intervention_token_count": row["intervention_token_count"],
        }
        for row in rows
    ]
    response = summarize_response_changes(response_rows)
    baseline_success_count = sum(bool(row["baseline_success"]) for row in rows)
    return {
        "condition": f"first_token_only_alpha_{alpha:.1f}",
        "alpha": float(alpha),
        "prompt_count": len(rows),
        **transition,
        "baseline_success_subset": {
            "baseline_success_count": baseline_success_count,
            "newly_defended_count": transition["newly_defended"],
            "still_attack_success_count": transition["still_attack_success"],
            "newly_defended_rate": (
                transition["newly_defended"] / baseline_success_count
                if baseline_success_count
                else None
            ),
        },
        "response_changes": response,
        "generation_failure_count": sum(bool(row["generation_failed"]) for row in rows),
        "empty_response_count": sum(row.get("empty_response") is True for row in rows),
        "first_token_argmax_changed_count": sum(
            not bool(row["first_token_argmax_equal"]) for row in rows
        ),
        "first_token_argmax_changed_dataset_indices": [
            int(row["dataset_index"])
            for row in rows
            if not bool(row["first_token_argmax_equal"])
        ],
        "diagnostics": dict(diagnostics) if diagnostics is not None else None,
    }


def response_analysis_rows(
    condition_rows: Sequence[Mapping[str, object]],
) -> list[dict]:
    output = []
    for row in condition_rows:
        output.append(
            {
                "alpha": row["alpha"],
                "dataset_index": row["dataset_index"],
                "baseline_success": row["baseline_success"],
                "intervention_success": row["success"],
                "transition": row["transition"],
                "response_equal": row["response_equal"],
                "first_different_token_position": row[
                    "first_different_token_position"
                ],
                "first_token_argmax_changed": not row[
                    "first_token_argmax_equal"
                ],
                "success_changed": row["baseline_success"] != row["success"],
                "baseline_token_count": row["baseline_token_count"],
                "intervention_token_count": row["intervention_token_count"],
                "token_edit_distance": row["token_edit_distance"],
            }
        )
    return output


def write_response_analysis_csv(
    path: str | Path, rows: Sequence[Mapping[str, object]]
) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    fields = (
        "alpha",
        "dataset_index",
        "baseline_success",
        "intervention_success",
        "transition",
        "response_equal",
        "first_different_token_position",
        "first_token_argmax_changed",
        "success_changed",
        "baseline_token_count",
        "intervention_token_count",
        "token_edit_distance",
    )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def validate_semantic_audit(
    audit_rows: Sequence[Mapping[str, object]],
    changed_rows: Sequence[Mapping[str, object]],
) -> list[dict]:
    expected = {(int(row["dataset_index"]), float(row["alpha"])) for row in changed_rows}
    seen = set()
    output = []
    for audit in audit_rows:
        key = (int(audit.get("dataset_index")), float(audit.get("alpha")))
        if key in seen or key not in expected:
            raise ValueError(f"unexpected/duplicate semantic audit key: {key}")
        seen.add(key)
        classification = audit.get("semantic_classification")
        if classification not in SEMANTIC_CLASSES:
            raise ValueError(f"invalid semantic classification: {classification}")
        harmful = audit.get("harmful_actionable_content_present")
        if type(harmful) is not bool:
            raise ValueError("harmful_actionable_content_present must be bool")
        confidence = audit.get("audit_confidence")
        if confidence not in AUDIT_CONFIDENCE_VALUES:
            raise ValueError(f"invalid audit confidence: {confidence}")
        reason = audit.get("audit_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("semantic audit requires a non-empty reason")
        source = next(
            row
            for row in changed_rows
            if (int(row["dataset_index"]), float(row["alpha"])) == key
        )
        output.append(
            {
                "dataset_index": key[0],
                "alpha": key[1],
                "baseline_success": bool(source["baseline_success"]),
                "intervention_success": bool(source["success"]),
                "transition": source["transition"],
                "matched_refusal_substrings": source[
                    "matched_refusal_substrings"
                ],
                "semantic_classification": classification,
                "harmful_actionable_content_present": harmful,
                "audit_confidence": confidence,
                "audit_reason": reason.strip(),
            }
        )
    if seen != expected:
        raise ValueError(f"semantic audit coverage mismatch: {seen} != {expected}")
    return sorted(output, key=lambda row: (row["alpha"], row["dataset_index"]))


def semantic_audit_summary(rows: Sequence[Mapping[str, object]]) -> dict:
    counts = {classification: 0 for classification in SEMANTIC_CLASSES}
    for row in rows:
        counts[row["semantic_classification"]] += 1
    return {
        "audited_response_change_count": len(rows),
        "classification_counts": counts,
        "harmful_actionable_content_present_count": sum(
            bool(row["harmful_actionable_content_present"]) for row in rows
        ),
    }
