#!/usr/bin/env python3
"""Audit and summarize a completed GCG reference-intervention evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.gcg_wasserstein_reference import read_json, read_jsonl  # noqa: E402
from utils.gcg_wasserstein_reference_evaluation import (  # noqa: E402
    EVALUATION_COUNT,
    compare_alpha_zero_to_baseline,
    summarize_transitions,
    validate_result_rows,
)


ALPHAS = (0.0, 0.1, 0.3, 0.5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args(argv)


def audit_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    required = {
        "run_manifest.json",
        "artifact_validation.json",
        "evaluation_manifest.csv",
        "baseline_evaluation_results.jsonl",
        "preflight_summary.json",
        "transition_summary.csv",
        "transition_summary.json",
        "evaluation_summary.json",
    }
    for alpha in ALPHAS:
        required.update(
            {
                f"alpha_{alpha:.1f}/prompt_results.jsonl",
                f"alpha_{alpha:.1f}/diagnostics.json",
                f"alpha_{alpha:.1f}/summary.json",
            }
        )
    missing = [name for name in sorted(required) if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"evaluation run is missing files: {missing}")

    artifact = read_json(run_dir / "artifact_validation.json")
    if artifact.get("verified") is not True or artifact.get(
        "top5_exact_match"
    ) is not True:
        raise ValueError("artifact validation did not pass")
    preflight = read_json(run_dir / "preflight_summary.json")
    if preflight.get("passed") is not True:
        raise ValueError("preflight did not pass")
    evaluation = read_json(run_dir / "evaluation_summary.json")
    if evaluation.get("status") != "complete":
        raise ValueError("evaluation summary is not complete")

    baseline = read_jsonl(run_dir / "baseline_evaluation_results.jsonl")
    if len(baseline) != EVALUATION_COUNT:
        raise ValueError("baseline evaluation result count mismatch")
    condition_payloads = []
    alpha_zero = None
    for alpha in ALPHAS:
        rows = read_jsonl(run_dir / f"alpha_{alpha:.1f}/prompt_results.jsonl")
        validate_result_rows(rows, baseline)
        stored_summary = read_json(run_dir / f"alpha_{alpha:.1f}/summary.json")
        diagnostics = read_json(run_dir / f"alpha_{alpha:.1f}/diagnostics.json")
        if stored_summary.get("generation_failure_count") != 0:
            raise ValueError(f"alpha={alpha} contains generation failures")
        recomputed = summarize_transitions(rows, expected_count=EVALUATION_COUNT)
        for key in (
            "newly_defended",
            "still_attack_success",
            "newly_worsened",
            "still_defended",
            "attack_success_count",
            "asr",
            "net_defended",
        ):
            if stored_summary.get(key) != recomputed[key]:
                raise ValueError(f"alpha={alpha} stored {key} mismatch")
        if diagnostics.get("all_finite") is not True:
            raise ValueError(f"alpha={alpha} diagnostics are non-finite")
        if diagnostics.get("nonzero_invariant_record_count") != 0:
            raise ValueError(f"alpha={alpha} has nonzero invariant changes")
        condition_payloads.append(
            {
                "alpha": alpha,
                "summary": stored_summary,
                "diagnostics": diagnostics,
            }
        )
        if alpha == 0.0:
            alpha_zero = compare_alpha_zero_to_baseline(rows)
    if not alpha_zero["passed"]:
        raise ValueError("alpha=0 baseline equivalence mismatch")

    transition = read_json(run_dir / "transition_summary.json")
    return {
        "verified": True,
        "required_file_count": len(required),
        "artifact_validation": {
            "verified": artifact["verified"],
            "top5_exact_match": artifact["top5_exact_match"],
            "selection_manifest_sha256": artifact[
                "selection_manifest_sha256"
            ],
            "score_artifact_sha256": artifact["score_artifact_sha256"],
            "reference_mean_shape": artifact["reference_mean_shape"],
            "reference_mean_dtype": artifact["reference_mean_dtype"],
            "reference_mean_finite": artifact["reference_mean_finite"],
        },
        "preflight": preflight,
        "alpha_zero_equivalence": alpha_zero,
        "conditions": condition_payloads,
        "transition_summary": transition,
        "failures": {
            "generation_failure_count": evaluation[
                "total_generation_failure_count"
            ],
            "empty_response_count": evaluation["total_empty_response_count"],
            "short_sequence_skip_count": evaluation[
                "total_short_sequence_skip_count"
            ],
            "all_diagnostics_finite": evaluation["all_diagnostics_finite"],
        },
    }


def main(argv=None):
    args = parse_args(argv)
    result = audit_run(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
