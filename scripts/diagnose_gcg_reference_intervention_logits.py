#!/usr/bin/env python3
"""Run fixed-prompt logit and head-contribution diagnostics on GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.asym_score_intervention import (  # noqa: E402
    temporarily_disable_default_dynamic_cache,
)
from utils.evaluate_utils import check_success_with_details  # noqa: E402
from utils.experiment_runtime import git_head, seed_everything as set_seed  # noqa: E402
from utils.gcg_reference_intervention_diagnostics import (  # noqa: E402
    aggregate_logit_steps,
    build_head_subset_conditions,
    compare_token_sequences,
    compute_logit_step_metrics,
    validate_logit_diagnostic_record,
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
    load_evaluation_records,
)
from utils.prompt_utils import apply_template  # noqa: E402
from utils.reference_asym_score_intervention import (  # noqa: E402
    load_validated_reference_artifacts,
    patch_llama_reference_attention,
    validate_loaded_llama_model,
)


DEFAULT_MODEL = "/mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf"
DEFAULT_ARTIFACT_DIR = (
    "data/outputs/gcg_wasserstein_reference/llama2/seed42_selection20/"
    "codex_gpu_20260802T061114Z"
)
DEFAULT_EVALUATION_RUN = (
    "data/outputs/gcg_wasserstein_reference_intervention/llama2/"
    "seed42_selection20/evaluation_20260802T071032Z"
)
DEFAULT_JAILBREAK_INPUT = "data/derived/llama2/gcg/jailbreak_prompts.jsonl"
DEFAULT_BASELINE_RESULTS = (
    "data/outputs/multidataset_asym_score_intervention/"
    "beta05_full_20260722_041303/llama_gcg/no_intervention/"
    "prompt_results.jsonl"
)
EXPECTED_SELECTION_MANIFEST_SHA256 = (
    "13578541411456bd689f8e771694ca97b047f2828bc35663536a5bc65badfa1e"
)
EXPECTED_SCORE_ARTIFACT_SHA256 = (
    "672cd283cc55bb1d4282bb5a752fa6fa3c60c083fe1850642974d12e33fcb0a0"
)
EXPECTED_TOP5 = ((31, 22), (13, 4), (5, 30), (9, 18), (27, 24))
DIAGNOSTIC_DATASET_INDICES = (182, 62, 69, 74, 60, 61, 64)
ALPHAS = (0.0, 0.1, 0.3, 0.5)
MAX_NEW_TOKENS = 32
GPU_OUTPUTS = (
    "index182_head_contribution.jsonl",
    "index182_head_contribution_summary.json",
    "teacher_forced_logit_diagnostics.jsonl",
    "teacher_forced_logit_summary.json",
    "free_running_logit_diagnostics.jsonl",
    "free_running_logit_summary.json",
    "diagnosis_summary.json",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--evaluation-run", default=DEFAULT_EVALUATION_RUN)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args(argv)


def finalize_run_manifest(path: Path, *, diagnosis_summary: dict) -> None:
    """Atomically transition this run's validated manifest to complete."""

    payload = read_json(path)
    if payload.get("status") != "response_analysis_complete_gpu_pending":
        raise ValueError("run manifest is not in the expected pending state")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary manifest: {temporary}")
    payload.update(
        {
            "status": "complete",
            "gpu_completed_at": diagnosis_summary["created_at"],
            "gpu_execution_git_head": diagnosis_summary["execution_git_head"],
            "gpu": diagnosis_summary["gpu"],
            "torch_version": diagnosis_summary["torch_version"],
            "torch_cuda_version": diagnosis_summary["torch_cuda_version"],
            "diagnostic_dataset_indices": diagnosis_summary[
                "diagnostic_dataset_indices"
            ],
            "artifact_verified": diagnosis_summary["artifact_verified"],
            "generation_failure_count": diagnosis_summary[
                "generation_failure_count"
            ],
            "nan_or_inf_count": diagnosis_summary["nan_or_inf_count"],
        }
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_model_and_tokenizer(model_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to run logit diagnosis on CPU")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def format_record(record: dict, tokenizer) -> dict:
    formatted_prompt = apply_template(
        [record["prompt"]],
        system_prompt=None,
        tokenizer=tokenizer,
        model_config={"model_name": "Llama"},
    )[0]
    input_ids = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    return {**record, "formatted_prompt": formatted_prompt, "input_ids": input_ids}


def eos_token_ids(model) -> set[int]:
    value = model.generation_config.eos_token_id
    if value is None:
        value = model.config.eos_token_id
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def manual_greedy_generate(model, tokenizer, input_ids: torch.Tensor) -> dict:
    """Greedy generation matching the audited no-cache generation settings."""

    sequence = input_ids.to(model.device)
    generated: list[int] = []
    scores: list[torch.Tensor] = []
    stops = eos_token_ids(model)
    with torch.inference_mode(), temporarily_disable_default_dynamic_cache(model):
        for _ in range(MAX_NEW_TOKENS):
            output = model(input_ids=sequence, use_cache=False)
            logits = output.logits[0, -1].float()
            if not bool(torch.isfinite(logits).all().item()):
                raise RuntimeError("generation logits contain NaN or infinity")
            token_id = int(torch.argmax(logits).item())
            scores.append(logits.detach())
            generated.append(token_id)
            next_token = torch.tensor(
                [[token_id]],
                dtype=sequence.dtype,
                device=sequence.device,
            )
            sequence = torch.cat((sequence, next_token), dim=-1)
            if token_id in stops:
                break
    response_ids = torch.tensor(generated, dtype=torch.long)
    response = tokenizer.decode(response_ids)
    success, checker = check_success_with_details(response)
    return {
        "response": response,
        "response_token_ids": generated,
        "response_token_count": len(generated),
        "success": bool(success),
        "matched_refusal_substrings": checker["matched_refusal_substrings"],
        "success_decision_reason": checker["decision_reason"],
        "generation_failed": False,
        "generation_error": None,
        "scores": scores,
    }


def teacher_forced_condition(
    *,
    record: dict,
    baseline_generation: dict,
    alpha: float,
    model,
    artifacts,
) -> dict:
    baseline_token_ids = baseline_generation["response_token_ids"]
    baseline_scores = baseline_generation["scores"]
    input_ids = record["input_ids"].to(model.device)
    prefix = input_ids
    steps = []
    intervention_argmax_ids = []
    with patch_llama_reference_attention(
        model,
        artifacts=artifacts,
        alpha=alpha,
    ) as controller:
        controller.clear()
        with torch.inference_mode(), temporarily_disable_default_dynamic_cache(model):
            for step, baseline_token_id in enumerate(baseline_token_ids):
                intervention_logits = model(
                    input_ids=prefix,
                    use_cache=False,
                ).logits[0, -1].float()
                metric = compute_logit_step_metrics(
                    baseline_scores[step],
                    intervention_logits,
                    step=step,
                )
                metric["baseline_generated_token_id"] = baseline_token_id
                metric["intervention_argmax_token_id_on_baseline_prefix"] = metric[
                    "intervention_top1_token_id"
                ]
                steps.append(metric)
                intervention_argmax_ids.append(metric["intervention_top1_token_id"])
                next_baseline = torch.tensor(
                    [[baseline_token_id]],
                    dtype=prefix.dtype,
                    device=prefix.device,
                )
                prefix = torch.cat((prefix, next_baseline), dim=-1)
        controller_summary = controller.summary()
    result = {
        "diagnostic_mode": "teacher_forced_common_prefix",
        "row_position": record["row_position"],
        "dataset_index": record["dataset_index"],
        "baseline_success": record["baseline_success"],
        "alpha": alpha,
        "selected_heads": [
            {"layer": layer, "head": head} for layer, head in artifacts.top5_heads
        ],
        "baseline_response_token_ids": baseline_token_ids,
        "intervention_response_token_ids": intervention_argmax_ids,
        "steps": steps,
        "summary": aggregate_logit_steps(steps),
        "intervention_call_count": controller_summary["intervention_call_count"],
        "patched_attention_call_count": controller_summary[
            "attention_forward_call_count"
        ],
        "short_sequence_skip_count": controller_summary[
            "short_sequence_skip_count"
        ],
        "all_scores_finite": controller_summary["all_scores_finite"],
    }
    validate_logit_diagnostic_record(
        result,
        mode="teacher_forced_common_prefix",
    )
    return result


def free_running_condition(
    *,
    record: dict,
    baseline_generation: dict,
    alpha: float,
    model,
    tokenizer,
    artifacts,
) -> dict:
    with patch_llama_reference_attention(
        model,
        artifacts=artifacts,
        alpha=alpha,
    ) as controller:
        controller.clear()
        intervention = manual_greedy_generate(
            model,
            tokenizer,
            record["input_ids"],
        )
        controller_summary = controller.summary()
    paired_count = min(
        len(baseline_generation["scores"]),
        len(intervention["scores"]),
    )
    if paired_count == 0:
        raise RuntimeError("free-running diagnostic produced no paired steps")
    steps = [
        compute_logit_step_metrics(
            baseline_generation["scores"][step],
            intervention["scores"][step],
            step=step,
        )
        for step in range(paired_count)
    ]
    comparison = compare_token_sequences(
        baseline_generation["response_token_ids"],
        intervention["response_token_ids"],
    )
    result = {
        "diagnostic_mode": "free_running",
        "row_position": record["row_position"],
        "dataset_index": record["dataset_index"],
        "baseline_success": record["baseline_success"],
        "intervention_success": intervention["success"],
        "success_changed": bool(record["baseline_success"])
        != bool(intervention["success"]),
        "alpha": alpha,
        "selected_heads": [
            {"layer": layer, "head": head} for layer, head in artifacts.top5_heads
        ],
        "baseline_response": baseline_generation["response"],
        "intervention_response": intervention["response"],
        "baseline_response_token_ids": baseline_generation["response_token_ids"],
        "intervention_response_token_ids": intervention["response_token_ids"],
        **comparison,
        "paired_generation_step_count": paired_count,
        "steps": steps,
        "summary": aggregate_logit_steps(steps),
        "matched_refusal_substrings": intervention["matched_refusal_substrings"],
        "success_decision_reason": intervention["success_decision_reason"],
        "generation_failed": False,
        "generation_error": None,
        "intervention_call_count": controller_summary["intervention_call_count"],
        "patched_attention_call_count": controller_summary[
            "attention_forward_call_count"
        ],
        "short_sequence_skip_count": controller_summary[
            "short_sequence_skip_count"
        ],
        "all_scores_finite": controller_summary["all_scores_finite"],
    }
    validate_logit_diagnostic_record(result, mode="free_running")
    return result


def compact_logit_summary(records: list[dict], *, mode: str) -> dict:
    if len(records) != len(DIAGNOSTIC_DATASET_INDICES) * len(ALPHAS):
        raise ValueError("logit diagnostic record count mismatch")
    if any(record["diagnostic_mode"] != mode for record in records):
        raise ValueError("logit diagnostic modes are mixed")
    return {
        "diagnostic_mode": mode,
        "prompt_count": len(DIAGNOSTIC_DATASET_INDICES),
        "condition_count": len(records),
        "dataset_indices": list(DIAGNOSTIC_DATASET_INDICES),
        "alphas": list(ALPHAS),
        "kl_direction": "D_KL(P_baseline || P_intervention)",
        "all_finite": all(record["summary"]["all_finite"] for record in records),
        "total_argmax_change_steps": sum(
            record["summary"]["argmax_change_step_count"] for record in records
        ),
        "records": [
            {
                "dataset_index": record["dataset_index"],
                "alpha": record["alpha"],
                **record["summary"],
                **(
                    {
                        "response_equal": record["response_equal"],
                        "first_different_token_position": record[
                            "first_different_token_position"
                        ],
                        "baseline_success": record["baseline_success"],
                        "intervention_success": record["intervention_success"],
                        "success_changed": record["success_changed"],
                    }
                    if mode == "free_running"
                    else {}
                ),
            }
            for record in records
        ],
    }


def run_head_contribution(
    *,
    record: dict,
    baseline_generation: dict,
    full_top5_free_record: dict,
    model,
    tokenizer,
    artifacts,
    saved_full_alpha_row: dict,
) -> tuple[list[dict], dict]:
    conditions = build_head_subset_conditions(EXPECTED_TOP5)
    full_top5_token_ids = full_top5_free_record["intervention_response_token_ids"]
    rows = []
    for condition in conditions:
        subset_artifacts = replace(
            artifacts,
            top5_heads=condition["selected_heads"],
        )
        with patch_llama_reference_attention(
            model,
            artifacts=subset_artifacts,
            alpha=0.5,
        ) as controller:
            controller.clear()
            generation = manual_greedy_generate(
                model,
                tokenizer,
                record["input_ids"],
            )
            controller_summary = controller.summary()
        baseline_comparison = compare_token_sequences(
            baseline_generation["response_token_ids"],
            generation["response_token_ids"],
        )
        full_comparison = compare_token_sequences(
            full_top5_token_ids,
            generation["response_token_ids"],
        )
        row = {
            "condition": condition["name"],
            "dataset_index": record["dataset_index"],
            "alpha": 0.5,
            "selected_heads": [
                {"layer": layer, "head": head}
                for layer, head in condition["selected_heads"]
            ],
            "omitted_head": (
                {"layer": condition["omitted_head"][0], "head": condition["omitted_head"][1]}
                if "omitted_head" in condition
                else None
            ),
            "response": generation["response"],
            "response_token_ids": generation["response_token_ids"],
            "success": generation["success"],
            "matched_refusal_substrings": generation[
                "matched_refusal_substrings"
            ],
            "success_decision_reason": generation["success_decision_reason"],
            "response_equal_to_baseline": baseline_comparison["response_equal"],
            "response_equal_to_full_top5": full_comparison["response_equal"],
            "first_different_token_position": baseline_comparison[
                "first_different_token_position"
            ],
            "generation_failed": False,
            "generation_error": None,
            "intervention_call_count": controller_summary[
                "intervention_call_count"
            ],
            "patched_attention_call_count": controller_summary[
                "attention_forward_call_count"
            ],
            "short_sequence_skip_count": controller_summary[
                "short_sequence_skip_count"
            ],
            "all_scores_finite": controller_summary["all_scores_finite"],
        }
        if condition["name"] == "full_top5":
            if row["response"] != saved_full_alpha_row["response"] or bool(
                row["success"]
            ) != bool(saved_full_alpha_row["success"]):
                raise RuntimeError(
                    "index 182 full_top5 does not reproduce saved alpha=0.5 result"
                )
        rows.append(row)
        print(
            f"[head contribution] {condition['name']} success={row['success']} "
            f"first_diff={row['first_different_token_position']}",
            flush=True,
        )
    single_rows = [row for row in rows if row["condition"].startswith("single_")]
    without_rows = [row for row in rows if row["condition"].startswith("without_")]
    summary = {
        "dataset_index": 182,
        "alpha": 0.5,
        "baseline_success": record["baseline_success"],
        "full_top5_success": rows[0]["success"],
        "full_top5_saved_response_exact_match": rows[0]["response"]
        == saved_full_alpha_row["response"],
        "single_heads_causing_attack_success": [
            row["selected_heads"][0] for row in single_rows if row["success"]
        ],
        "omitted_heads_restoring_defense": [
            row["omitted_head"] for row in without_rows if not row["success"]
        ],
        "requires_multiple_heads": bool(rows[0]["success"])
        and not any(row["success"] for row in single_rows),
        "generation_failure_count": sum(row["generation_failed"] for row in rows),
        "short_sequence_skip_count": sum(
            row["short_sequence_skip_count"] for row in rows
        ),
        "all_scores_finite": all(row["all_scores_finite"] for row in rows),
        "conditions": [
            {
                key: row[key]
                for key in (
                    "condition",
                    "selected_heads",
                    "omitted_head",
                    "success",
                    "matched_refusal_substrings",
                    "success_decision_reason",
                    "response_equal_to_baseline",
                    "response_equal_to_full_top5",
                    "first_different_token_position",
                    "generation_failed",
                    "intervention_call_count",
                )
            }
            for row in rows
        ],
    }
    return rows, summary


def main(argv=None):
    args = parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    model_path = Path(args.model).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    evaluation_run = Path(args.evaluation_run).resolve()
    jailbreak_input = Path(args.jailbreak_input).resolve()
    baseline_results = Path(args.baseline_results).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError("response-analysis run directory does not exist")
    for required in (
        "run_manifest.json",
        "response_change_summary.json",
        "diagnostics_index182.json",
        "diagnostics_index182.md",
    ):
        if not (run_dir / required).is_file():
            raise FileNotFoundError(f"required response analysis missing: {required}")
    for name in GPU_OUTPUTS:
        if (run_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite GPU diagnostic: {name}")

    manifest = read_json(run_dir / "run_manifest.json")
    if manifest.get("status") != "response_analysis_complete_gpu_pending":
        raise ValueError("run directory is not awaiting GPU diagnosis")
    if Path(manifest["source_evaluation_run"]).resolve() != evaluation_run:
        raise ValueError("source evaluation run mismatch")

    artifacts = load_validated_reference_artifacts(
        artifact_dir,
        expected_heads=EXPECTED_TOP5,
        expected_selection_manifest_sha256=EXPECTED_SELECTION_MANIFEST_SHA256,
        expected_score_artifact_sha256=EXPECTED_SCORE_ARTIFACT_SHA256,
        expected_model_path=model_path,
    )
    evaluation_records, split_manifest = load_evaluation_records(
        split_manifest_path=artifact_dir / "split_manifest.json",
        jailbreak_input=jailbreak_input,
        baseline_results=baseline_results,
    )
    by_index = {record["dataset_index"]: record for record in evaluation_records}
    if not all(index in by_index for index in DIAGNOSTIC_DATASET_INDICES):
        raise ValueError("one or more diagnostic indices are outside evaluation")
    if any(by_index[index]["split"] != "evaluation" for index in DIAGNOSTIC_DATASET_INDICES):
        raise ValueError("diagnostic set contains selection data")

    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(model_path)
    model_validation = validate_loaded_llama_model(model, artifacts)
    records = [format_record(by_index[index], tokenizer) for index in DIAGNOSTIC_DATASET_INDICES]
    saved_by_alpha = {
        alpha: {
            row["dataset_index"]: row
            for row in read_jsonl(
                evaluation_run / f"alpha_{alpha:.1f}/prompt_results.jsonl"
            )
        }
        for alpha in ALPHAS
    }

    baseline_generations = {}
    teacher_records = []
    for prompt_position, record in enumerate(records, 1):
        baseline = manual_greedy_generate(model, tokenizer, record["input_ids"])
        saved_zero = saved_by_alpha[0.0][record["dataset_index"]]
        if baseline["response"] != saved_zero["response"] or baseline["success"] != bool(
            saved_zero["success"]
        ):
            raise RuntimeError(
                f"normal greedy baseline mismatch at dataset index {record['dataset_index']}"
            )
        baseline_generations[record["dataset_index"]] = baseline
        for alpha in ALPHAS:
            result = teacher_forced_condition(
                record=record,
                baseline_generation=baseline,
                alpha=alpha,
                model=model,
                artifacts=artifacts,
            )
            if alpha == 0.0 and result["summary"]["max_logit_difference_max"] != 0.0:
                raise RuntimeError("teacher-forced alpha=0 is not logit-exact")
            teacher_records.append(result)
            print(
                f"[teacher {prompt_position}/7] index={record['dataset_index']} "
                f"alpha={alpha:.1f} argmax_changes="
                f"{result['summary']['argmax_change_step_count']}",
                flush=True,
            )
    teacher_summary = compact_logit_summary(
        teacher_records,
        mode="teacher_forced_common_prefix",
    )
    write_jsonl(run_dir / "teacher_forced_logit_diagnostics.jsonl", teacher_records)
    write_json(run_dir / "teacher_forced_logit_summary.json", teacher_summary)

    free_records = []
    for prompt_position, record in enumerate(records, 1):
        baseline = baseline_generations[record["dataset_index"]]
        for alpha in ALPHAS:
            result = free_running_condition(
                record=record,
                baseline_generation=baseline,
                alpha=alpha,
                model=model,
                tokenizer=tokenizer,
                artifacts=artifacts,
            )
            saved = saved_by_alpha[alpha][record["dataset_index"]]
            result["saved_evaluation_response_equal"] = result[
                "intervention_response"
            ] == saved["response"]
            result["saved_evaluation_success_equal"] = result[
                "intervention_success"
            ] == bool(saved["success"])
            if not result["saved_evaluation_response_equal"] or not result[
                "saved_evaluation_success_equal"
            ]:
                raise RuntimeError(
                    f"free-running result differs from saved evaluation: "
                    f"index={record['dataset_index']} alpha={alpha}"
                )
            if alpha == 0.0:
                if not result["response_equal"] or result["summary"][
                    "max_logit_difference_max"
                ] != 0.0:
                    raise RuntimeError("free-running alpha=0 is not exactly equivalent")
            free_records.append(result)
            print(
                f"[free {prompt_position}/7] index={record['dataset_index']} "
                f"alpha={alpha:.1f} equal={result['response_equal']} "
                f"first_diff={result['first_different_token_position']}",
                flush=True,
            )
    free_summary = compact_logit_summary(free_records, mode="free_running")
    free_summary["all_saved_evaluation_responses_reproduced"] = all(
        record["saved_evaluation_response_equal"] for record in free_records
    )
    free_summary["all_saved_evaluation_success_reproduced"] = all(
        record["saved_evaluation_success_equal"] for record in free_records
    )
    write_jsonl(run_dir / "free_running_logit_diagnostics.jsonl", free_records)
    write_json(run_dir / "free_running_logit_summary.json", free_summary)

    index182_record = next(record for record in records if record["dataset_index"] == 182)
    index182_full_free = next(
        record
        for record in free_records
        if record["dataset_index"] == 182 and record["alpha"] == 0.5
    )
    contribution_rows, contribution_summary = run_head_contribution(
        record=index182_record,
        baseline_generation=baseline_generations[182],
        full_top5_free_record=index182_full_free,
        model=model,
        tokenizer=tokenizer,
        artifacts=artifacts,
        saved_full_alpha_row=saved_by_alpha[0.5][182],
    )
    write_jsonl(run_dir / "index182_head_contribution.jsonl", contribution_rows)
    write_json(
        run_dir / "index182_head_contribution_summary.json",
        contribution_summary,
    )

    diagnosis_summary = {
        "status": "complete",
        "created_at": utc_now(),
        "execution_git_head": git_head(REPO_ROOT),
        "run_directory": str(run_dir),
        "source_evaluation_run": str(evaluation_run),
        "source_evaluation_summary_sha256": sha256_file(
            evaluation_run / "evaluation_summary.json"
        ),
        "artifact_verified": artifacts.verification["verified"],
        "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
        "score_artifact_sha256": EXPECTED_SCORE_ARTIFACT_SHA256,
        "reference_artifact_sha256": sha256_file(
            artifact_dir / "original_reference.pt"
        ),
        "top5_heads": [
            {"layer": layer, "head": head} for layer, head in artifacts.top5_heads
        ],
        "model": str(model_path),
        "model_config": llama_model_config_summary(model),
        "model_validation": model_validation,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "generation_settings": {
            "torch_dtype": "torch.float16",
            "attention_implementation": "eager",
            "use_cache": False,
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": 1,
            "seed": 42,
            "system_prompt": None,
            "tokenize_add_special_tokens": False,
        },
        "diagnostic_dataset_indices": list(DIAGNOSTIC_DATASET_INDICES),
        "all_diagnostic_indices_are_evaluation": True,
        "selection_count": split_manifest["selection_count"],
        "kl_direction": "D_KL(P_baseline || P_intervention)",
        "logit_probability_dtype": "torch.float32",
        "teacher_forced": teacher_summary,
        "free_running": free_summary,
        "index182_head_contribution": contribution_summary,
        "generation_failure_count": 0,
        "nan_or_inf_count": 0,
    }
    write_json(run_dir / "diagnosis_summary.json", diagnosis_summary)
    finalize_run_manifest(
        run_dir / "run_manifest.json",
        diagnosis_summary=diagnosis_summary,
    )
    print(json.dumps(diagnosis_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
