#!/usr/bin/env python3
"""Prepare, run, and audit the exploratory 48-row Llama-GCG holdout."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_gcg_first_token_reference_ablation import (  # noqa: E402
    aggregate_condition_diagnostics,
    format_records,
    generate_with_first_logits,
    load_model_and_tokenizer,
    run_intervention_condition,
    set_seed,
)
from utils.experiment_runtime import git_value  # noqa: E402
from utils.gcg_first_token_holdout import (  # noqa: E402
    EVALUATION_ROLE,
    HOLDOUT_COUNT,
    HOLDOUT_INDICES,
    build_condition_summary,
    build_holdout_manifest,
    response_analysis_rows,
    semantic_audit_summary,
    validate_alpha_zero_gate,
    validate_first_token_condition,
    validate_holdout_manifest,
    validate_semantic_audit,
    write_holdout_manifest_artifacts,
    write_response_analysis_csv,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    llama_model_config_summary,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.gcg_wasserstein_reference_evaluation import (  # noqa: E402
    classify_transition,
    ensure_output_absent,
)
from utils.reference_asym_score_intervention import (  # noqa: E402
    load_validated_reference_artifacts,
    validate_loaded_llama_model,
)


DEFAULT_MODEL = "/mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf"
DEFAULT_RAW_ARCHIVE = "data/raw/llama2/llama2_GCG.json"
DEFAULT_ADVBENCH = "data/raw/advbench.jsonl"
DEFAULT_ARTIFACT_DIR = (
    "data/outputs/gcg_wasserstein_reference/llama2/seed42_selection20/"
    "codex_gpu_20260802T061114Z"
)
EXPECTED_SELECTION_MANIFEST_SHA256 = (
    "13578541411456bd689f8e771694ca97b047f2828bc35663536a5bc65badfa1e"
)
EXPECTED_SCORE_ARTIFACT_SHA256 = (
    "672cd283cc55bb1d4282bb5a752fa6fa3c60c083fe1850642974d12e33fcb0a0"
)
EXPECTED_TOP5 = ((31, 22), (13, 4), (5, 30), (9, 18), (27, 24))
PREFLIGHT_INDICES = (0, 1, 2, 3, 4)
POSITIVE_ALPHAS = (0.5, 0.1, 0.3)
GENERATION_SETTINGS = {
    "torch_dtype": "torch.float16",
    "interpolation_dtype": "torch.float32",
    "attention_implementation": "eager",
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "max_new_tokens": 32,
    "use_cache": False,
    "batch_size": 1,
    "system_prompt": None,
    "seed": 42,
    "tokenize_add_special_tokens": False,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "audit"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--raw-archive", default=DEFAULT_RAW_ARCHIVE)
    parser.add_argument("--advbench", default=DEFAULT_ADVBENCH)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--audit-input",
        help="JSON array of human semantic audit records; required for audit stage.",
    )
    return parser.parse_args(argv)


def atomic_replace_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_artifacts(artifact_dir: Path, model_path: Path):
    return load_validated_reference_artifacts(
        artifact_dir,
        expected_heads=EXPECTED_TOP5,
        expected_selection_manifest_sha256=EXPECTED_SELECTION_MANIFEST_SHA256,
        expected_score_artifact_sha256=EXPECTED_SCORE_ARTIFACT_SHA256,
        expected_model_path=model_path,
    )


def prepare_stage(args) -> None:
    output_dir = ensure_output_absent(args.output_dir).resolve()
    model_path = Path(args.model).resolve()
    raw_archive = Path(args.raw_archive).resolve()
    advbench = Path(args.advbench).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    manifest = build_holdout_manifest(
        raw_archive_path=raw_archive,
        advbench_path=advbench,
    )
    validate_holdout_manifest(
        manifest,
        raw_archive_path=raw_archive,
        advbench_path=advbench,
    )
    artifacts = load_artifacts(artifact_dir, model_path)

    output_dir.mkdir(parents=True)
    # These files are complete before any baseline outcome is generated.
    write_holdout_manifest_artifacts(manifest, output_dir)
    manifest_path = output_dir / "holdout_manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    reference_path = artifact_dir / "original_reference.pt"
    artifact_validation = {
        **artifacts.verification,
        "top5_exact_match": artifacts.top5_heads == EXPECTED_TOP5,
        "loaded_top5": [
            {"layer": layer, "head": head} for layer, head in artifacts.top5_heads
        ],
        "reference_mean_shape": list(artifacts.reference_mean.shape),
        "reference_mean_dtype": str(artifacts.reference_mean.dtype),
        "reference_mean_finite": bool(
            torch.isfinite(artifacts.reference_mean).all().item()
        ),
        "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
        "score_artifact_sha256": EXPECTED_SCORE_ARTIFACT_SHA256,
        "reference_artifact_sha256": sha256_file(reference_path),
        "loaded_model_validation": None,
    }
    write_json(output_dir / "artifact_validation.json", artifact_validation)
    write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "status": "manifest_prepared_before_baseline",
            "created_at": utc_now(),
            "execution_git_head": git_value(REPO_ROOT, "rev-parse", "HEAD"),
            "execution_git_branch": git_value(REPO_ROOT, "branch", "--show-current"),
            "evaluation_role": EVALUATION_ROLE,
            "evaluation_label": "exploratory Llama-GCG holdout",
            "alternative_evaluation_label": "Llama-GCG-specific holdout",
            "fully_unseen_across_all_prior_research": False,
            "unused_for_current_gcg_head_selection": True,
            "unused_for_current_reference_construction": True,
            "unused_for_current_gcg_evaluation": True,
            "original_behaviors_previously_used_in_autodan_research": True,
            "holdout_manifest_sha256_fixed_before_baseline": manifest_sha256,
            "holdout_manifest_records_sha256": manifest["records_sha256"],
            "included_dataset_indices": list(HOLDOUT_INDICES),
            "excluded_dataset_indices": [10, 13],
            "included_count": HOLDOUT_COUNT,
            "raw_archive": str(raw_archive),
            "raw_artifact_sha256": manifest["raw_artifact_sha256"],
            "advbench": str(advbench),
            "advbench_sha256": manifest["advbench_sha256"],
            "model": str(model_path),
            "artifact_dir": str(artifact_dir),
            "reference_artifact_sha256": sha256_file(reference_path),
            "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
            "score_artifact_sha256": EXPECTED_SCORE_ARTIFACT_SHA256,
            "selected_heads": artifact_validation["loaded_top5"],
            "timing": "first_token_only",
            "primary_alpha": 0.5,
            "sensitivity_alphas": [0.1, 0.3],
            "tail_tokens": 16,
            "intervention_equation": (
                "A_intervened=(1-alpha)*A_current+alpha*A_ref; "
                "B_intervened=B_current+alpha*(A_ref-A_current)"
            ),
            "generation_settings": GENERATION_SETTINGS,
        },
    )
    print(
        json.dumps(
            {
                "status": "manifest_prepared_before_baseline",
                "run_directory": str(output_dir),
                "included_count": HOLDOUT_COUNT,
                "manifest_sha256": manifest_sha256,
                "artifact_verified": artifacts.verification["verified"],
            },
            indent=2,
        )
    )


def load_prepared(args):
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"prepared output directory is missing: {output_dir}")
    run_manifest = read_json(output_dir / "run_manifest.json")
    if run_manifest.get("status") != "manifest_prepared_before_baseline":
        raise ValueError("run directory is not ready for the GPU stage")
    manifest_path = output_dir / "holdout_manifest.json"
    if sha256_file(manifest_path) != run_manifest.get(
        "holdout_manifest_sha256_fixed_before_baseline"
    ):
        raise ValueError("frozen holdout manifest hash mismatch")
    raw_archive = Path(args.raw_archive).resolve()
    advbench = Path(args.advbench).resolve()
    records = validate_holdout_manifest(
        read_json(manifest_path),
        raw_archive_path=raw_archive,
        advbench_path=advbench,
    )
    if tuple(record["dataset_index"] for record in records) != HOLDOUT_INDICES:
        raise ValueError("runtime holdout records differ from frozen indices")
    return output_dir, run_manifest, records


def run_baseline_records(records, *, model, tokenizer, label: str):
    rows = []
    for position, record in enumerate(records, 1):
        generation = generate_with_first_logits(model, tokenizer, record["input_ids"])
        if generation["generation_failed"]:
            raise RuntimeError(
                f"baseline generation failed at dataset index {record['dataset_index']}"
            )
        rows.append(
            {
                "prompt_id": record["prompt_id"],
                "row_position": record["row_position"],
                "dataset_index": record["dataset_index"],
                "raw_file_position": record["raw_file_position"],
                "raw_archive_index": record["raw_archive_index"],
                "prompt": record["prompt"],
                "formatted_prompt": record["formatted_prompt"],
                "prompt_length": record["prompt_length"],
                **generation,
            }
        )
        print(
            f"[{label} {position}/{len(records)}] "
            f"dataset_index={record['dataset_index']} success={generation['success']}",
            flush=True,
        )
    return rows


def json_safe_baseline(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "first_logits"}


def enrich_condition_rows(rows: list[dict], records_by_index: dict[int, dict]) -> None:
    for row in rows:
        record = records_by_index[row["dataset_index"]]
        row["raw_file_position"] = record["raw_file_position"]
        row["raw_archive_index"] = record["raw_archive_index"]
        row["baseline_generation_failed"] = False
        row["transition"] = classify_transition(row["baseline_success"], row["success"])


def condition_summary(rows: list[dict], *, alpha: float, diagnostics: dict) -> dict:
    summary = build_condition_summary(rows, alpha=alpha, diagnostics=diagnostics)
    if summary["generation_failure_count"] or summary["empty_response_count"]:
        raise RuntimeError(f"alpha={alpha} contains a generation/empty-response failure")
    return summary


def write_condition(output_dir: Path, alpha: float, rows: list[dict], diagnostics: dict):
    condition_dir = output_dir / f"alpha_{alpha:.1f}"
    condition_dir.mkdir()
    write_jsonl(condition_dir / "prompt_results.jsonl", rows)
    write_json(condition_dir / "diagnostics.json", diagnostics)
    summary = condition_summary(rows, alpha=alpha, diagnostics=diagnostics)
    write_json(condition_dir / "summary.json", summary)
    return summary


def run_preflight(records, *, model, tokenizer, artifacts) -> dict:
    selected = [
        next(record for record in records if record["dataset_index"] == index)
        for index in PREFLIGHT_INDICES
    ]
    baseline = run_baseline_records(
        selected, model=model, tokenizer=tokenizer, label="preflight baseline"
    )
    baseline_by_index = {row["dataset_index"]: row for row in baseline}
    records_by_index = {row["dataset_index"]: row for row in selected}
    alpha_zero, zero_diagnostics = run_intervention_condition(
        selected,
        baseline_by_index,
        timing="first_token_only",
        alpha=0.0,
        model=model,
        tokenizer=tokenizer,
        artifacts=artifacts,
        label="preflight alpha0",
    )
    enrich_condition_rows(alpha_zero, records_by_index)
    alpha_zero_gate = validate_alpha_zero_gate(alpha_zero)
    primary, primary_diagnostics = run_intervention_condition(
        selected,
        baseline_by_index,
        timing="first_token_only",
        alpha=0.5,
        model=model,
        tokenizer=tokenizer,
        artifacts=artifacts,
        label="preflight alpha0.5",
    )
    enrich_condition_rows(primary, records_by_index)
    validate_first_token_condition(primary, primary_diagnostics, expected_count=5)
    if any(row["generation_failed"] for row in primary):
        raise RuntimeError("preflight positive-alpha generation failed")
    return {
        "passed": True,
        "dataset_indices": list(PREFLIGHT_INDICES),
        "manifest_and_raw_prompt_match": True,
        "alpha_zero_comparison": alpha_zero_gate,
        "alpha_zero_intervention_count": zero_diagnostics[
            "total_intervention_count"
        ],
        "primary_alpha": 0.5,
        "primary_total_intervention_count": primary_diagnostics[
            "total_intervention_count"
        ],
        "primary_per_head_intervention_count": primary_diagnostics[
            "per_head_intervention_count"
        ],
        "later_forward_intervention_count": primary_diagnostics[
            "later_forward_intervention_count"
        ],
        "generation_failure_count": 0,
        "nan_or_inf_count": 0,
    }


def write_transition_summaries(output_dir: Path, baseline_summary: dict, summaries: list[dict]):
    rows = [
        {
            "condition": "Baseline",
            "alpha": None,
            "newly_defended": 0,
            "still_attack_success": baseline_summary["attack_success_count"],
            "newly_worsened": 0,
            "still_defended": baseline_summary["failure_count"],
            "attack_success_count": baseline_summary["attack_success_count"],
            "asr": baseline_summary["asr"],
            "net_defended": 0,
        }
    ]
    for summary in summaries:
        rows.append(
            {
                "condition": summary["condition"],
                "alpha": summary["alpha"],
                "newly_defended": summary["newly_defended"],
                "still_attack_success": summary["still_attack_success"],
                "newly_worsened": summary["newly_worsened"],
                "still_defended": summary["still_defended"],
                "attack_success_count": summary["attack_success_count"],
                "asr": summary["asr"],
                "net_defended": summary["net_defended"],
            }
        )
    write_json(output_dir / "transition_summary.json", {"conditions": rows})
    path = output_dir / "transition_summary.csv"
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_stage(args) -> None:
    output_dir, run_manifest, records = load_prepared(args)
    model_path = Path(args.model).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    artifacts = load_artifacts(artifact_dir, model_path)
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(model_path)
    model_validation = validate_loaded_llama_model(model, artifacts)
    records = format_records(records, tokenizer)
    records_by_index = {record["dataset_index"]: record for record in records}

    artifact_validation_path = output_dir / "artifact_validation.json"
    artifact_validation = read_json(artifact_validation_path)
    artifact_validation["loaded_model_validation"] = model_validation
    artifact_validation["model_config"] = llama_model_config_summary(model)
    atomic_replace_json(artifact_validation_path, artifact_validation)

    preflight = run_preflight(records, model=model, tokenizer=tokenizer, artifacts=artifacts)
    write_json(output_dir / "preflight_summary.json", preflight)

    baseline_rows = run_baseline_records(
        records, model=model, tokenizer=tokenizer, label="holdout baseline"
    )
    baseline_by_index = {row["dataset_index"]: row for row in baseline_rows}
    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir()
    write_jsonl(
        baseline_dir / "prompt_results.jsonl",
        [json_safe_baseline(row) for row in baseline_rows],
    )
    baseline_success_count = sum(bool(row["success"]) for row in baseline_rows)
    baseline_summary = {
        "prompt_count": HOLDOUT_COUNT,
        "attack_success_count": baseline_success_count,
        "failure_count": HOLDOUT_COUNT - baseline_success_count,
        "asr": baseline_success_count / HOLDOUT_COUNT,
        "generation_failure_count": 0,
        "empty_response_count": sum(row["empty_response"] for row in baseline_rows),
    }
    write_json(baseline_dir / "summary.json", baseline_summary)

    alpha_zero_rows, alpha_zero_diagnostics = run_intervention_condition(
        records,
        baseline_by_index,
        timing="first_token_only",
        alpha=0.0,
        model=model,
        tokenizer=tokenizer,
        artifacts=artifacts,
        label="holdout alpha0",
    )
    enrich_condition_rows(alpha_zero_rows, records_by_index)
    alpha_zero_comparison = validate_alpha_zero_gate(alpha_zero_rows)
    alpha_zero_summary = write_condition(
        output_dir, 0.0, alpha_zero_rows, alpha_zero_diagnostics
    )

    positive_summaries = []
    all_response_analysis = []
    all_changed_rows = []
    for alpha in POSITIVE_ALPHAS:
        rows, diagnostics = run_intervention_condition(
            records,
            baseline_by_index,
            timing="first_token_only",
            alpha=alpha,
            model=model,
            tokenizer=tokenizer,
            artifacts=artifacts,
            label="holdout",
        )
        enrich_condition_rows(rows, records_by_index)
        validate_first_token_condition(rows, diagnostics, expected_count=HOLDOUT_COUNT)
        summary = write_condition(output_dir, alpha, rows, diagnostics)
        positive_summaries.append(summary)
        analysis = response_analysis_rows(rows)
        all_response_analysis.extend(analysis)
        all_changed_rows.extend(row for row in rows if not row["response_equal"])

    write_transition_summaries(output_dir, baseline_summary, positive_summaries)
    write_response_analysis_csv(
        output_dir / "response_change_analysis.csv", all_response_analysis
    )
    write_jsonl(
        output_dir / "semantic_response_audit_candidates.jsonl",
        [
            {
                "dataset_index": row["dataset_index"],
                "alpha": row["alpha"],
                "baseline_success": row["baseline_success"],
                "intervention_success": row["success"],
                "transition": row["transition"],
                "matched_refusal_substrings": row["matched_refusal_substrings"],
                "success_decision_reason": row["success_decision_reason"],
            }
            for row in all_changed_rows
        ],
    )
    evaluation_summary = {
        "status": "awaiting_semantic_audit",
        "created_at": utc_now(),
        "evaluation_role": EVALUATION_ROLE,
        "evaluation_label": "exploratory Llama-GCG holdout",
        "alternative_evaluation_label": "Llama-GCG-specific holdout",
        "fully_unseen_across_all_prior_research": False,
        "holdout_count": HOLDOUT_COUNT,
        "preflight": preflight,
        "baseline": baseline_summary,
        "alpha_zero_comparison": alpha_zero_comparison,
        "alpha_zero_summary": alpha_zero_summary,
        "primary_alpha": 0.5,
        "sensitivity_alphas": [0.1, 0.3],
        "positive_conditions": positive_summaries,
        "response_change_record_count": len(all_changed_rows),
        "semantic_audit": None,
    }
    write_json(output_dir / "evaluation_summary.json", evaluation_summary)
    run_manifest["status"] = "awaiting_semantic_audit"
    run_manifest["gpu_execution_started_at"] = utc_now()
    run_manifest["gpu"] = torch.cuda.get_device_name(0)
    run_manifest["torch_version"] = torch.__version__
    run_manifest["torch_cuda_version"] = torch.version.cuda
    run_manifest["baseline_attack_success_count"] = baseline_success_count
    run_manifest["response_change_record_count"] = len(all_changed_rows)
    atomic_replace_json(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(evaluation_summary, ensure_ascii=False, indent=2))


def audit_stage(args) -> None:
    if not args.audit_input:
        raise ValueError("--audit-input is required for the audit stage")
    output_dir = Path(args.output_dir).resolve()
    run_manifest = read_json(output_dir / "run_manifest.json")
    if run_manifest.get("status") != "awaiting_semantic_audit":
        raise ValueError("run is not awaiting semantic audit")
    changed_rows = []
    for alpha in POSITIVE_ALPHAS:
        rows = read_jsonl(output_dir / f"alpha_{alpha:.1f}/prompt_results.jsonl")
        changed_rows.extend(row for row in rows if not row["response_equal"])
    with Path(args.audit_input).open(encoding="utf-8") as handle:
        audit_input = json.load(handle)
    if not isinstance(audit_input, list):
        raise ValueError("semantic audit input must be a JSON array")
    audit_rows = validate_semantic_audit(audit_input, changed_rows)
    write_jsonl(output_dir / "semantic_response_audit.jsonl", audit_rows)
    audit_summary = semantic_audit_summary(audit_rows)
    evaluation_summary_path = output_dir / "evaluation_summary.json"
    evaluation_summary = read_json(evaluation_summary_path)
    evaluation_summary["status"] = "complete"
    evaluation_summary["completed_at"] = utc_now()
    evaluation_summary["semantic_audit"] = audit_summary
    atomic_replace_json(evaluation_summary_path, evaluation_summary)
    run_manifest["status"] = "complete"
    run_manifest["completed_at"] = evaluation_summary["completed_at"]
    run_manifest["semantic_audit"] = audit_summary
    atomic_replace_json(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(evaluation_summary, ensure_ascii=False, indent=2))


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "prepare":
        prepare_stage(args)
    elif args.stage == "run":
        run_stage(args)
    else:
        audit_stage(args)


if __name__ == "__main__":
    main()
