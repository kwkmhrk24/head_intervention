#!/usr/bin/env python3
"""Independently audit and summarize a first-token reference ablation run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.gcg_wasserstein_reference import read_json, read_jsonl  # noqa: E402


TIMINGS = ("dynamic_all_steps", "first_token_only")
ALPHAS = (0.1, 0.3, 0.5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    run_dir = Path(parse_args(argv).run_dir).resolve()
    required = (
        "run_manifest.json",
        "artifact_validation.json",
        "smoke_summary.json",
        "selection_development_summary.json",
        "comparison_table.csv",
        "first_token_logit_diagnostics.jsonl",
        "first_token_logit_summary.json",
        "ablation_summary.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"required ablation artifacts missing: {missing}")
    manifest = read_json(run_dir / "run_manifest.json")
    artifact = read_json(run_dir / "artifact_validation.json")
    smoke = read_json(run_dir / "smoke_summary.json")
    development = read_json(run_dir / "selection_development_summary.json")
    if manifest.get("status") != "complete":
        raise ValueError("run manifest is not complete")
    for payload in (manifest, development):
        if payload.get("development_only") is not True:
            raise ValueError("development_only flag is missing")
        if payload.get("selection_data_reused_for_head_selection") is not True:
            raise ValueError("head-selection reuse flag is missing")
        if payload.get("selection_data_reused_for_reference") is not True:
            raise ValueError("reference reuse flag is missing")
    if artifact.get("verified") is not True or artifact.get("top5_exact_match") is not True:
        raise ValueError("artifact validation did not pass")
    if smoke.get("passed") is not True or smoke.get("dynamic_matches_prior_smoke") is not True:
        raise ValueError("smoke validation did not pass")

    expected = {(timing, alpha) for timing in TIMINGS for alpha in ALPHAS}
    summaries = {
        (item["timing"], float(item["alpha"])): item
        for item in development["conditions"]
    }
    if set(summaries) != expected:
        raise ValueError("development conditions are incomplete")
    audit_rows = []
    for timing, alpha in sorted(expected):
        summary = summaries[(timing, alpha)]
        rows = read_jsonl(
            run_dir / timing / f"alpha_{alpha:.1f}/prompt_results.jsonl"
        )
        diagnostics = read_json(
            run_dir / timing / f"alpha_{alpha:.1f}/diagnostics.json"
        )
        if len(rows) != 20 or len({row["dataset_index"] for row in rows}) != 20:
            raise ValueError("condition prompt results are incomplete or duplicated")
        changed = [row for row in rows if not row["response_equal"]]
        argmax_changed = [
            row for row in rows if not row["first_token_argmax_equal"]
        ]
        if len(changed) != summary["response_changed_count"]:
            raise ValueError("response change summary mismatch")
        if len(argmax_changed) != summary["first_token_argmax_changed_count"]:
            raise ValueError("first-token argmax summary mismatch")
        if timing == "first_token_only":
            if any(row["first_different_token_position"] != 0 for row in changed):
                raise ValueError("first-token-only response diverged after token zero")
            if {row["dataset_index"] for row in changed} != {
                row["dataset_index"] for row in argmax_changed
            }:
                raise ValueError("first-token response/argmax sets differ")
            if diagnostics["total_intervention_count"] != 100:
                raise ValueError("first-token intervention count is not 100")
            if diagnostics["later_forward_intervention_count"] != 0:
                raise ValueError("first-token later-forward intervention is nonzero")
        if summary["generation_failure_count"] != 0 or summary["nan_or_inf_count"] != 0:
            raise ValueError("condition contains failure or non-finite scores")
        audit_rows.append(
            {
                "timing": timing,
                "alpha": alpha,
                "newly_defended": summary["newly_defended_count"],
                "still_attack_success": summary["still_attack_success_count"],
                "response_changed": summary["response_changed_count"],
                "first_token_argmax_changed": summary[
                    "first_token_argmax_changed_count"
                ],
                "failure": summary["generation_failure_count"],
            }
        )
    with (run_dir / "comparison_table.csv").open(encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 6:
        raise ValueError("comparison table must contain six rows")
    logit_rows = read_jsonl(run_dir / "first_token_logit_diagnostics.jsonl")
    if len(logit_rows) != 80:
        raise ValueError("first-token logit diagnostics must contain 80 rows")
    result = {
        "verified": True,
        "run_directory": str(run_dir),
        "development_only": True,
        "required_file_count": len(required),
        "smoke_passed": True,
        "dynamic_matches_prior_smoke": True,
        "condition_count": len(audit_rows),
        "first_token_logit_record_count": len(logit_rows),
        "conditions": audit_rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
