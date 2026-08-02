#!/usr/bin/env python3
"""Diagnose selection/holdout first-token effects without changing the method."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_gcg_first_token_reference_ablation import (  # noqa: E402
    format_records,
    generate_with_first_logits,
    load_model_and_tokenizer,
)
from utils.asym_score_intervention import (  # noqa: E402
    temporarily_disable_default_dynamic_cache,
)
from utils.gcg_first_token_generalization_diagnostics import (  # noqa: E402
    ALPHAS,
    EXPECTED_TOP5,
    PRIMARY_ALPHA,
    build_all_nonempty_head_subsets,
    build_index88_context,
    head_contribution_conditions,
    load_group_logit_records,
    minimal_refusal_subsets,
    summarize_group_logits,
    summarize_score_interventions,
)
from utils.gcg_first_token_holdout import validate_holdout_manifest  # noqa: E402
from utils.gcg_first_token_reference_ablation import (  # noqa: E402
    load_selection_development_records,
)
from utils.gcg_reference_intervention_diagnostics import (  # noqa: E402
    compare_token_sequences,
    compute_logit_step_metrics,
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
from utils.gcg_wasserstein_reference_evaluation import ensure_output_absent  # noqa: E402
from utils.reference_asym_score_intervention import (  # noqa: E402
    load_validated_reference_artifacts,
    patch_llama_reference_attention,
    validate_loaded_llama_model,
)


DEFAULT_MODEL = "/mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf"
DEFAULT_SELECTION_RUN = (
    "data/outputs/gcg_wasserstein_reference_intervention/llama2/"
    "seed42_selection20/first_token_ablation_20260802T084341Z"
)
DEFAULT_HOLDOUT_RUN = (
    "data/outputs/gcg_wasserstein_reference_intervention/llama2/"
    "first_token_holdout48/evaluation_20260802T095741Z"
)
DEFAULT_ARTIFACT_DIR = (
    "data/outputs/gcg_wasserstein_reference/llama2/seed42_selection20/"
    "codex_gpu_20260802T061114Z"
)
DEFAULT_JAILBREAK_INPUT = "data/derived/llama2/gcg/jailbreak_prompts.jsonl"
DEFAULT_BASELINE_RESULTS = (
    "data/outputs/multidataset_asym_score_intervention/"
    "beta05_full_20260722_041303/llama_gcg/no_intervention/"
    "prompt_results.jsonl"
)
DEFAULT_RAW_ARCHIVE = "data/raw/llama2/llama2_GCG.json"
DEFAULT_ADVBENCH = "data/raw/advbench.jsonl"
EXPECTED_SELECTION_MANIFEST_SHA256 = (
    "13578541411456bd689f8e771694ca97b047f2828bc35663536a5bc65badfa1e"
)
EXPECTED_SCORE_ARTIFACT_SHA256 = (
    "672cd283cc55bb1d4282bb5a752fa6fa3c60c083fe1850642974d12e33fcb0a0"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--selection-run", default=DEFAULT_SELECTION_RUN)
    parser.add_argument("--holdout-run", default=DEFAULT_HOLDOUT_RUN)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--raw-archive", default=DEFAULT_RAW_ARCHIVE)
    parser.add_argument("--advbench", default=DEFAULT_ADVBENCH)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_prompt_groups(args, tokenizer):
    artifact_dir = Path(args.artifact_dir).resolve()
    selection, _ = load_selection_development_records(
        split_manifest_path=artifact_dir / "split_manifest.json",
        jailbreak_input=Path(args.jailbreak_input).resolve(),
        baseline_results=Path(args.baseline_results).resolve(),
    )
    holdout_run = Path(args.holdout_run).resolve()
    holdout = validate_holdout_manifest(
        read_json(holdout_run / "holdout_manifest.json"),
        raw_archive_path=Path(args.raw_archive).resolve(),
        advbench_path=Path(args.advbench).resolve(),
    )
    holdout_baseline = {
        int(row["dataset_index"]): row
        for row in read_jsonl(holdout_run / "baseline/prompt_results.jsonl")
    }
    for record in holdout:
        baseline = holdout_baseline[record["dataset_index"]]
        record.update(
            baseline_success=bool(baseline["success"]),
            baseline_response=baseline["response"],
            baseline_generation_failed=bool(baseline["generation_failed"]),
        )
    return {
        "selection20": format_records(selection, tokenizer),
        "holdout48": format_records(holdout, tokenizer),
    }


def capture_score_distances(groups, *, model, artifacts):
    rows = []
    for group, records in groups.items():
        with patch_llama_reference_attention(
            model,
            artifacts=artifacts,
            alpha=PRIMARY_ALPHA,
            intervention_timing="first_token_only",
        ) as controller:
            for position, record in enumerate(records, 1):
                controller.clear()
                controller.set_prompt_context(
                    prompt_length=record["prompt_length"],
                    intervention_timing="first_token_only",
                )
                with torch.inference_mode(), temporarily_disable_default_dynamic_cache(model):
                    output = model(input_ids=record["input_ids"].to(model.device), use_cache=False)
                if not bool(torch.isfinite(output.logits).all().item()):
                    raise RuntimeError("first-forward logits contain NaN or infinity")
                summary = controller.summary()
                metrics = controller.head_metric_records()
                if len(metrics) != 5 or summary["total_intervention_count"] != 5:
                    raise RuntimeError("first-forward score diagnostic did not apply five heads")
                if summary["later_forward_intervention_count"] != 0:
                    raise RuntimeError("score diagnostic intervened after first forward")
                for metric in metrics:
                    rows.append(
                        {
                            "dataset_group": group,
                            "dataset_index": record["dataset_index"],
                            "baseline_success": bool(record["baseline_success"]),
                            "alpha": PRIMARY_ALPHA,
                            "layer": metric["layer"],
                            "head": metric["head"],
                            "current_asym_frobenius_norm": metric[
                                "current_asym_frobenius_norm"
                            ],
                            "reference_asym_frobenius_norm": metric[
                                "reference_asym_frobenius_norm"
                            ],
                            "intervened_asym_frobenius_norm": metric[
                                "intervened_asym_frobenius_norm"
                            ],
                            "current_to_reference_frobenius_distance": metric[
                                "current_to_reference_frobenius_distance"
                            ],
                            "current_to_reference_mean_absolute_distance": metric[
                                "current_to_reference_mean_absolute_distance"
                            ],
                            "mean_absolute_delta": metric["mean_absolute_delta"],
                            "max_absolute_delta": metric["max_absolute_delta"],
                            "intervened_to_current_norm_ratio": metric[
                                "intervened_to_current_norm_ratio"
                            ],
                            "mean_absolute_delta_float32": metric[
                                "mean_absolute_delta_float32"
                            ],
                            "max_absolute_delta_float32": metric[
                                "max_absolute_delta_float32"
                            ],
                            "prompt_length": record["prompt_length"],
                            "block_start": summary["block_start"],
                            "block_end": summary["block_end"],
                            "finite": metric["finite"],
                        }
                    )
                print(
                    f"[score {group} {position}/{len(records)}] "
                    f"dataset_index={record['dataset_index']}",
                    flush=True,
                )
    return rows


def run_subset_condition(
    *,
    condition: dict,
    record: dict,
    baseline: dict,
    full_top5: dict | None,
    model,
    tokenizer,
    artifacts,
) -> dict:
    subset = tuple(condition["selected_heads"])
    subset_artifacts = replace(artifacts, top5_heads=subset)
    with patch_llama_reference_attention(
        model,
        artifacts=subset_artifacts,
        alpha=PRIMARY_ALPHA,
        intervention_timing="first_token_only",
    ) as controller:
        controller.clear()
        controller.set_prompt_context(
            prompt_length=record["prompt_length"],
            intervention_timing="first_token_only",
        )
        generation = generate_with_first_logits(
            model, tokenizer, record["input_ids"]
        )
        diagnostics = controller.summary()
    if generation["generation_failed"]:
        raise RuntimeError(f"index 88 generation failed for {condition['name']}")
    baseline_comparison = compare_token_sequences(
        baseline["response_token_ids"], generation["response_token_ids"]
    )
    metrics = compute_logit_step_metrics(
        baseline["first_logits"], generation["first_logits"], step=0
    )
    full_equal = (
        None
        if full_top5 is None
        else generation["response_token_ids"] == full_top5["response_token_ids"]
    )
    if diagnostics["total_intervention_count"] != len(subset):
        raise RuntimeError("subset head intervention count mismatch")
    if diagnostics["later_forward_intervention_count"] != 0:
        raise RuntimeError("subset condition intervened after first forward")
    return {
        "condition": condition["name"],
        "subset_size": len(subset),
        "selected_heads": [
            {"layer": layer, "head": head} for layer, head in subset
        ],
        "omitted_head": (
            {
                "layer": condition["omitted_head"][0],
                "head": condition["omitted_head"][1],
            }
            if "omitted_head" in condition
            else None
        ),
        "dataset_index": 88,
        "alpha": PRIMARY_ALPHA,
        "first_token_id": generation["response_token_ids"][0],
        "first_token_equal_to_baseline": metrics["argmax_equal"],
        "response": generation["response"],
        "response_token_ids": generation["response_token_ids"],
        "success": generation["success"],
        "matched_refusal_substrings": generation["matched_refusal_substrings"],
        "success_decision_reason": generation["success_decision_reason"],
        "response_equal_to_baseline": baseline_comparison["response_equal"],
        "response_equal_to_full_top5": full_equal,
        "first_step_max_logit_difference": metrics[
            "max_absolute_logit_difference"
        ],
        "first_step_KL": metrics["kl_baseline_to_intervention"],
        "baseline_top1_margin": metrics["baseline_top1_top2_margin"],
        "intervention_top1_margin": metrics["intervention_top1_top2_margin"],
        "generation_failed": False,
        "generation_error": None,
        "intervention_call_count": diagnostics["total_intervention_count"],
        "later_forward_intervention_count": diagnostics[
            "later_forward_intervention_count"
        ],
        "all_scores_finite": diagnostics["all_scores_finite"],
    }


def diagnose_index88(*, selection_records, model, tokenizer, artifacts, selection_run):
    record = next(row for row in selection_records if row["dataset_index"] == 88)
    baseline = generate_with_first_logits(model, tokenizer, record["input_ids"])
    if baseline["generation_failed"]:
        raise RuntimeError("index 88 baseline reproduction failed")
    saved_baseline = next(
        row
        for row in read_jsonl(Path(selection_run) / "baseline_selection_results.jsonl")
        if row["dataset_index"] == 88
    )
    if baseline["response"] != saved_baseline["response"] or baseline["success"] != saved_baseline["success"]:
        raise RuntimeError("index 88 baseline does not reproduce saved result")

    conditions = head_contribution_conditions()
    full_condition = conditions[0]
    full_row = run_subset_condition(
        condition=full_condition,
        record=record,
        baseline=baseline,
        full_top5=None,
        model=model,
        tokenizer=tokenizer,
        artifacts=artifacts,
    )
    full_row["response_equal_to_full_top5"] = True
    saved_full = next(
        row
        for row in read_jsonl(
            Path(selection_run) / "first_token_only/alpha_0.5/prompt_results.jsonl"
        )
        if row["dataset_index"] == 88
    )
    if full_row["response"] != saved_full["response"] or full_row["success"] != saved_full["success"]:
        raise RuntimeError("index 88 full Top 5 does not reproduce saved refusal")
    rows = [full_row]
    for condition in conditions[1:]:
        row = run_subset_condition(
            condition=condition,
            record=record,
            baseline=baseline,
            full_top5=full_row,
            model=model,
            tokenizer=tokenizer,
            artifacts=artifacts,
        )
        rows.append(row)
        print(
            f"[index88 {condition['name']}] success={row['success']} "
            f"first_token={row['first_token_id']}",
            flush=True,
        )

    singles = [row for row in rows if row["condition"].startswith("single_")]
    leave_one_out = [row for row in rows if row["condition"].startswith("without_")]
    single_refusals = [row for row in singles if row["success"] is False]
    exhaustive = not single_refusals
    subset_rows = []
    if exhaustive:
        by_subset = {
            tuple((item["layer"], item["head"]) for item in row["selected_heads"]): row
            for row in rows
        }
        for condition in build_all_nonempty_head_subsets():
            subset = tuple(condition["selected_heads"])
            row = by_subset.get(subset)
            if row is None:
                row = run_subset_condition(
                    condition=condition,
                    record=record,
                    baseline=baseline,
                    full_top5=full_row,
                    model=model,
                    tokenizer=tokenizer,
                    artifacts=artifacts,
                )
            subset_rows.append(row)
    else:
        subset_rows = singles + [full_row]
    minimal = minimal_refusal_subsets(subset_rows)
    summary = {
        "dataset_index": 88,
        "alpha": PRIMARY_ALPHA,
        "baseline_success": baseline["success"],
        "full_top5_success": full_row["success"],
        "full_top5_saved_response_exact_match": True,
        "single_heads_causing_refusal": [
            row["selected_heads"][0] for row in single_refusals
        ],
        "heads_whose_omission_removes_refusal": [
            row["omitted_head"] for row in leave_one_out if row["success"] is True
        ],
        "multiple_head_interaction_required": not single_refusals,
        "exhaustive_31_subsets_executed": exhaustive,
        **minimal,
        "generation_failure_count": 0,
        "all_scores_finite": all(row["all_scores_finite"] for row in rows),
    }
    subset_payload = {
        "executed": exhaustive,
        "reason": (
            "no single head caused refusal; evaluated all 31 subsets"
            if exhaustive
            else "single-head refusal found; exhaustive search not required"
        ),
        "evaluated_subset_count": len(subset_rows),
        "subsets": [
            {
                key: row[key]
                for key in (
                    "subset_size",
                    "selected_heads",
                    "first_token_id",
                    "success",
                    "response_equal_to_full_top5",
                )
            }
            for row in subset_rows
        ],
        **minimal,
    }
    return rows, summary, subset_payload


def main(argv=None):
    args = parse_args(argv)
    output_dir = ensure_output_absent(args.output_dir).resolve()
    model_path = Path(args.model).resolve()
    selection_run = Path(args.selection_run).resolve()
    holdout_run = Path(args.holdout_run).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()

    logit_records = load_group_logit_records(
        selection_run=selection_run,
        holdout_run=holdout_run,
    )
    logit_summary = summarize_group_logits(logit_records)
    index88_context = build_index88_context(logit_records)
    artifacts = load_validated_reference_artifacts(
        artifact_dir,
        expected_heads=EXPECTED_TOP5,
        expected_selection_manifest_sha256=EXPECTED_SELECTION_MANIFEST_SHA256,
        expected_score_artifact_sha256=EXPECTED_SCORE_ARTIFACT_SHA256,
        expected_model_path=model_path,
    )
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "group_logit_diagnostics.jsonl", logit_records)
    write_json(
        output_dir / "group_logit_summary.json",
        {**logit_summary, "index88_context": index88_context},
    )

    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(model_path)
    model_validation = validate_loaded_llama_model(model, artifacts)
    groups = load_prompt_groups(args, tokenizer)
    score_rows = capture_score_distances(groups, model=model, artifacts=artifacts)
    score_summary = summarize_score_interventions(score_rows)
    write_jsonl(output_dir / "group_score_intervention_diagnostics.jsonl", score_rows)
    write_json(output_dir / "group_score_intervention_summary.json", score_summary)

    head_rows, head_summary, subset_payload = diagnose_index88(
        selection_records=groups["selection20"],
        model=model,
        tokenizer=tokenizer,
        artifacts=artifacts,
        selection_run=selection_run,
    )
    write_jsonl(output_dir / "index88_head_contribution.jsonl", head_rows)
    write_json(output_dir / "index88_head_contribution_summary.json", head_summary)
    write_json(output_dir / "index88_minimal_subsets.json", subset_payload)

    run_manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": utc_now(),
        "execution_git_head": git_value("rev-parse", "HEAD"),
        "execution_git_branch": git_value("branch", "--show-current"),
        "model": str(model_path),
        "model_config": llama_model_config_summary(model),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "selection_run": str(selection_run),
        "selection_run_manifest_sha256": sha256_file(selection_run / "run_manifest.json"),
        "holdout_run": str(holdout_run),
        "holdout_run_manifest_sha256": sha256_file(holdout_run / "run_manifest.json"),
        "artifact_dir": str(artifact_dir),
        "reference_artifact_sha256": sha256_file(artifact_dir / "original_reference.pt"),
        "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
        "score_artifact_sha256": EXPECTED_SCORE_ARTIFACT_SHA256,
        "selected_heads": [
            {"layer": layer, "head": head} for layer, head in artifacts.top5_heads
        ],
        "timing": "first_token_only",
        "alphas": list(ALPHAS),
        "primary_alpha": PRIMARY_ALPHA,
        "tail_tokens": 16,
        "score_distance_domain": "full 16x16 antisymmetric matrix",
        "model_validation": model_validation,
        "artifact_verified": artifacts.verification["verified"],
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    diagnosis_summary = {
        "status": "complete",
        "created_at": utc_now(),
        "run_directory": str(output_dir),
        "group_logit_summary": logit_summary,
        "index88_context": index88_context,
        "group_score_intervention_summary": score_summary,
        "index88_head_contribution_summary": head_summary,
        "generation_failure_count": head_summary["generation_failure_count"],
        "nan_or_inf_count": 0,
    }
    write_json(output_dir / "diagnosis_summary.json", diagnosis_summary)
    print(json.dumps(diagnosis_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
