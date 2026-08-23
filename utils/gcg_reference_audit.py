"""CPU audit/reporting helpers for completed GCG reference experiment runs."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from utils.gcg_wasserstein_reference import (
    read_json,
    read_jsonl,
    validate_reference_payload,
    verify_artifact_directory,
)
from utils.gcg_wasserstein_reference_evaluation import (
    EVALUATION_COUNT,
    compare_alpha_zero_to_baseline,
    summarize_transitions,
    validate_result_rows,
)


EVALUATION_ALPHAS = (0.0, 0.1, 0.3, 0.5)
FIRST_TOKEN_ALPHAS = (0.1, 0.3, 0.5)
FIRST_TOKEN_TIMINGS = ("dynamic_all_steps", "first_token_only")


def verify_artifacts(run_dir: str | Path) -> dict:
    """Verify hashes, cross-links, tensor shapes, and Top-5 metadata."""

    return verify_artifact_directory(run_dir)


def summarize_artifacts(run_dir: str | Path, *, top_k: int = 20) -> dict:
    """Summarize ranking, reference tensors, split counts, and refusal audit."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    run_dir = Path(run_dir)
    verification = verify_artifact_directory(run_dir)
    with (run_dir / "head_wasserstein_ranking.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        ranking = list(csv.DictReader(handle))
    reference = torch.load(
        run_dir / "original_reference.pt", map_location="cpu", weights_only=True
    )
    validate_reference_payload(reference)
    mean, std, sem = reference["mean"], reference["std"], reference["sem"]
    split = read_json(run_dir / "split_manifest.json")
    refusal_path = run_dir / "selection_original_refusal_summary.json"
    refusal = read_json(refusal_path) if refusal_path.is_file() else None
    return {
        "verification": verification,
        "top_ranking": [
            {
                "rank": int(row["rank"]),
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                **{
                    field: float(row[field])
                    for field in ("mean", "median", "std", "sem", "min", "max")
                },
            }
            for row in ranking[:top_k]
        ],
        "top5_heads": [
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                **{field: float(row[field]) for field in ("mean", "median", "std")},
            }
            for row in ranking[:5]
        ],
        "reference": {
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
            "sem_shape": list(sem.shape),
            "dtype": str(mean.dtype),
            "finite": all(
                bool(torch.isfinite(value).all().item())
                for value in (mean, std, sem)
            ),
            "mean_tensor_mean": float(mean.mean().item()),
            "mean_tensor_std": float(mean.std(correction=1).item()),
            "mean_tensor_min": float(mean.min().item()),
            "mean_tensor_max": float(mean.max().item()),
            "std_tensor_mean": float(std.mean().item()),
            "std_tensor_min": float(std.min().item()),
            "std_tensor_max": float(std.max().item()),
        },
        "split": {
            key: split[key]
            for key in (
                "selection_count",
                "evaluation_count",
                "baseline_success_count_selection",
                "baseline_success_count_evaluation",
                "baseline_asr_evaluation",
            )
        },
        "original_refusal_diagnostic": (
            {
                key: refusal[key]
                for key in (
                    "selection_count",
                    "refusal_count",
                    "non_refusal_count",
                    "generation_failure_count",
                    "refusal_dataset_indices",
                    "non_refusal_dataset_indices",
                )
            }
            if refusal is not None
            else "not run"
        ),
    }


def audit_evaluation(run_dir: str | Path) -> dict:
    """Independently recompute a completed reference-intervention evaluation."""

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
    for alpha in EVALUATION_ALPHAS:
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
    if artifact.get("verified") is not True or artifact.get("top5_exact_match") is not True:
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
    conditions = []
    alpha_zero = None
    for alpha in EVALUATION_ALPHAS:
        rows = read_jsonl(run_dir / f"alpha_{alpha:.1f}/prompt_results.jsonl")
        validate_result_rows(rows, baseline)
        stored = read_json(run_dir / f"alpha_{alpha:.1f}/summary.json")
        diagnostics = read_json(run_dir / f"alpha_{alpha:.1f}/diagnostics.json")
        if stored.get("generation_failure_count") != 0:
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
            if stored.get(key) != recomputed[key]:
                raise ValueError(f"alpha={alpha} stored {key} mismatch")
        if diagnostics.get("all_finite") is not True:
            raise ValueError(f"alpha={alpha} diagnostics are non-finite")
        if diagnostics.get("nonzero_invariant_record_count") != 0:
            raise ValueError(f"alpha={alpha} has nonzero invariant changes")
        conditions.append({"alpha": alpha, "summary": stored, "diagnostics": diagnostics})
        if alpha == 0.0:
            alpha_zero = compare_alpha_zero_to_baseline(rows)
    if not alpha_zero or not alpha_zero["passed"]:
        raise ValueError("alpha=0 baseline equivalence mismatch")
    return {
        "verified": True,
        "required_file_count": len(required),
        "artifact_validation": {
            key: artifact[key]
            for key in (
                "verified",
                "top5_exact_match",
                "selection_manifest_sha256",
                "score_artifact_sha256",
                "reference_mean_shape",
                "reference_mean_dtype",
                "reference_mean_finite",
            )
        },
        "preflight": preflight,
        "alpha_zero_equivalence": alpha_zero,
        "conditions": conditions,
        "transition_summary": read_json(run_dir / "transition_summary.json"),
        "failures": {
            "generation_failure_count": evaluation["total_generation_failure_count"],
            "empty_response_count": evaluation["total_empty_response_count"],
            "short_sequence_skip_count": evaluation["total_short_sequence_skip_count"],
            "all_diagnostics_finite": evaluation["all_diagnostics_finite"],
        },
    }


def audit_first_token(run_dir: str | Path) -> dict:
    """Independently audit a completed first-token reference ablation."""

    run_dir = Path(run_dir)
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
    expected = {
        (timing, alpha)
        for timing in FIRST_TOKEN_TIMINGS
        for alpha in FIRST_TOKEN_ALPHAS
    }
    summaries = {
        (item["timing"], float(item["alpha"])): item
        for item in development["conditions"]
    }
    if set(summaries) != expected:
        raise ValueError("development conditions are incomplete")
    audit_rows = []
    for timing, alpha in sorted(expected):
        summary = summaries[(timing, alpha)]
        rows = read_jsonl(run_dir / timing / f"alpha_{alpha:.1f}/prompt_results.jsonl")
        diagnostics = read_json(run_dir / timing / f"alpha_{alpha:.1f}/diagnostics.json")
        if len(rows) != 20 or len({row["dataset_index"] for row in rows}) != 20:
            raise ValueError("condition prompt results are incomplete or duplicated")
        changed = [row for row in rows if not row["response_equal"]]
        argmax_changed = [row for row in rows if not row["first_token_argmax_equal"]]
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
                "first_token_argmax_changed": summary["first_token_argmax_changed_count"],
                "failure": summary["generation_failure_count"],
            }
        )
    with (run_dir / "comparison_table.csv").open(encoding="utf-8") as handle:
        if len(list(csv.DictReader(handle))) != 6:
            raise ValueError("comparison table must contain six rows")
    logit_rows = read_jsonl(run_dir / "first_token_logit_diagnostics.jsonl")
    if len(logit_rows) != 80:
        raise ValueError("first-token logit diagnostics must contain 80 rows")
    return {
        "verified": True,
        "run_directory": str(run_dir.resolve()),
        "development_only": True,
        "required_file_count": len(required),
        "smoke_passed": True,
        "dynamic_matches_prior_smoke": True,
        "condition_count": len(audit_rows),
        "first_token_logit_record_count": len(logit_rows),
        "conditions": audit_rows,
    }
