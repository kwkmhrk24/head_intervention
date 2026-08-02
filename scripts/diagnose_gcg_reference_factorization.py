#!/usr/bin/env python3
"""Factorize fixed-Head and global-reference effects with paired oracles."""

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
from utils.gcg_first_token_holdout import validate_holdout_manifest  # noqa: E402
from utils.gcg_first_token_reference_ablation import (  # noqa: E402
    load_selection_development_records,
    validate_first_token_response_invariant,
)
from utils.gcg_reference_factorization_diagnostics import (  # noqa: E402
    ALPHA,
    EXPECTED_TOP5,
    GROUP_COUNTS,
    build_paired_original_records,
    classify_factorization_causes,
    factorization_conditions,
    reference_for_condition,
    summarize_condition_logits,
    summarize_condition_results,
    summarize_reference_distances,
    validate_condition_definitions,
    validate_paired_original_values,
)
from utils.gcg_reference_intervention_diagnostics import (  # noqa: E402
    compare_token_sequences,
    compute_logit_step_metrics,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    TAIL_TOKENS,
    extract_prompt_scores,
    llama_model_config_summary,
    read_json,
    read_jsonl,
    save_torch_artifact,
    sha256_file,
    strict_lower_indices,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.gcg_wasserstein_reference_evaluation import (  # noqa: E402
    classify_transition,
    ensure_output_absent,
)
from utils.reference_asym_score_intervention import (  # noqa: E402
    EXPECTED_SCORE_CAPTURE_POSITION,
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
DEFAULT_SELECTION_ORIGINAL = "data/derived/llama2/gcg/original_prompts.jsonl"
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
SCORE_CALCULATION_DTYPE = (
    "torch.float16 Q/K matmul and scaling; S_asym computed in torch.float16"
)
GENERATION_SETTINGS = {
    "torch_dtype": "torch.float16",
    "interpolation_dtype": "torch.float32",
    "attention_implementation": "eager",
    "use_cache": False,
    "batch_size": 1,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "max_new_tokens": 32,
    "seed": 42,
    "system_prompt": None,
    "tokenize_add_special_tokens": False,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--selection-run", default=DEFAULT_SELECTION_RUN)
    parser.add_argument("--holdout-run", default=DEFAULT_HOLDOUT_RUN)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--selection-original", default=DEFAULT_SELECTION_ORIGINAL)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--raw-archive", default=DEFAULT_RAW_ARCHIVE)
    parser.add_argument("--advbench", default=DEFAULT_ADVBENCH)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_group_inputs(args):
    artifact_dir = Path(args.artifact_dir).resolve()
    selection_run = Path(args.selection_run).resolve()
    holdout_run = Path(args.holdout_run).resolve()
    selection, selection_manifest = load_selection_development_records(
        split_manifest_path=artifact_dir / "split_manifest.json",
        jailbreak_input=Path(args.jailbreak_input).resolve(),
        baseline_results=Path(args.baseline_results).resolve(),
    )
    selection_baseline_rows = read_jsonl(
        selection_run / "baseline_selection_results.jsonl"
    )
    selection_baseline = {
        int(row["dataset_index"]): row for row in selection_baseline_rows
    }
    if len(selection_baseline) != GROUP_COUNTS["selection20"]:
        raise ValueError("selection saved baseline count mismatch")
    for record in selection:
        saved = selection_baseline[record["dataset_index"]]
        if (
            saved["prompt"] != record["prompt"]
            or saved["response"] != record["baseline_response"]
            or saved["success"] is not record["baseline_success"]
        ):
            raise ValueError("selection saved baseline alignment mismatch")
        record["baseline_response_token_ids"] = saved["response_token_ids"]
        record["baseline_generation_failed"] = saved["generation_failed"]

    holdout_manifest = read_json(holdout_run / "holdout_manifest.json")
    holdout = validate_holdout_manifest(
        holdout_manifest,
        raw_archive_path=Path(args.raw_archive).resolve(),
        advbench_path=Path(args.advbench).resolve(),
    )
    holdout_baseline_rows = read_jsonl(holdout_run / "baseline/prompt_results.jsonl")
    holdout_baseline = {
        int(row["dataset_index"]): row for row in holdout_baseline_rows
    }
    if len(holdout_baseline) != GROUP_COUNTS["holdout48"]:
        raise ValueError("holdout saved baseline count mismatch")
    for record in holdout:
        saved = holdout_baseline[record["dataset_index"]]
        if saved["prompt"] != record["prompt"]:
            raise ValueError("holdout saved baseline prompt mismatch")
        record.update(
            baseline_response=saved["response"],
            baseline_response_token_ids=saved["response_token_ids"],
            baseline_success=bool(saved["success"]),
            baseline_generation_failed=bool(saved["generation_failed"]),
            baseline_response_token_count=saved["response_token_count"],
        )

    groups = {"selection20": selection, "holdout48": holdout}
    paired_records = build_paired_original_records(
        groups,
        selection_original_rows=read_jsonl(args.selection_original),
        advbench_rows=read_jsonl(args.advbench),
        selection_split_manifest=selection_manifest,
        holdout_manifest=holdout_manifest,
    )
    saved_global = {
        "selection20": {
            int(row["dataset_index"]): row
            for row in read_jsonl(
                selection_run
                / "first_token_only/alpha_0.5/prompt_results.jsonl"
            )
        },
        "holdout48": {
            int(row["dataset_index"]): row
            for row in read_jsonl(holdout_run / "alpha_0.5/prompt_results.jsonl")
        },
    }
    for group, count in GROUP_COUNTS.items():
        if len(saved_global[group]) != count:
            raise ValueError(f"{group} saved global-mean result count mismatch")
    return groups, paired_records, selection_manifest, holdout_manifest, saved_global


def _first_step_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(), temporarily_disable_default_dynamic_cache(model):
        output = model(
            input_ids=input_ids.to(model.device),
            use_cache=False,
            return_dict=True,
        )
    logits = output.logits[0, -1].float().detach().cpu()
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("baseline first-step logits contain NaN or infinity")
    return logits


def capture_baseline_logits(groups, *, model, saved_global):
    output = {}
    for group, records in groups.items():
        output[group] = {}
        for position, record in enumerate(records, 1):
            logits = _first_step_logits(model, record["input_ids"])
            top1 = int(torch.argmax(logits).item())
            if top1 != int(record["baseline_response_token_ids"][0]):
                raise RuntimeError(
                    f"baseline first token mismatch at {group}/{record['dataset_index']}"
                )
            prior_metric = saved_global[group][record["dataset_index"]][
                "first_token_logit_metrics"
            ]
            if top1 != int(prior_metric["baseline_top1_token_id"]):
                raise RuntimeError("baseline top-1 differs from saved diagnostic")
            margin = float(torch.topk(logits, k=2).values[0] - torch.topk(logits, k=2).values[1])
            if margin != float(prior_metric["baseline_top1_top2_margin"]):
                raise RuntimeError("baseline margin differs from saved diagnostic")
            output[group][record["dataset_index"]] = logits
            print(
                f"[baseline-logit {group} {position}/{len(records)}] "
                f"index={record['dataset_index']}",
                flush=True,
            )
    return output


def extract_paired_references(
    paired_records,
    *,
    model,
    tokenizer,
) -> tuple[torch.Tensor, list[dict]]:
    values = torch.empty(
        (len(paired_records), NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD),
        dtype=torch.float32,
        device="cpu",
    )
    manifest_records = []
    for position, record in enumerate(paired_records, 1):
        score, sequence_length = extract_prompt_scores(
            model=model,
            tokenizer=tokenizer,
            prompt=record["original_prompt"],
            row_position=record["row_position"],
            dataset_index=record["dataset_index"],
            tail_tokens=TAIL_TOKENS,
        )
        values[record["paired_position"]].copy_(score)
        manifest_records.append(
            {
                key: record[key]
                for key in (
                    "paired_position",
                    "dataset_group",
                    "group_position",
                    "row_position",
                    "dataset_index",
                    "prompt_id",
                    "original_prompt_sha256",
                )
            }
            | {"original_sequence_length": sequence_length}
        )
        print(
            f"[paired-original {position}/{len(paired_records)}] "
            f"group={record['dataset_group']} index={record['dataset_index']} "
            f"length={sequence_length}",
            flush=True,
        )
    validate_paired_original_values(values, manifest_records)
    return values.contiguous(), manifest_records


def _validate_prompt_diagnostics(summary: dict, expected_heads: tuple) -> None:
    if summary["first_forward_detected"] is not True:
        raise RuntimeError("first forward was not detected")
    if summary["first_forward_intervened"] is not True:
        raise RuntimeError("first forward was not intervened")
    if summary["later_forward_intervention_count"] != 0:
        raise RuntimeError("factorization condition intervened after first forward")
    actual = {
        (item["layer"], item["head"]): item["count"]
        for item in summary["per_head_intervention_count"]
    }
    if actual != {head: 1 for head in expected_heads}:
        raise RuntimeError(f"per-head intervention count mismatch: {actual}")
    if summary["total_intervention_count"] != len(expected_heads):
        raise RuntimeError("total intervention count mismatch")
    if summary["short_sequence_skip_count"] != 0:
        raise RuntimeError("factorization prompt was shorter than tail window")
    if not summary["all_scores_finite"]:
        raise RuntimeError("factorization condition contains nonfinite scores")
    for field in (
        "max_diagonal_change",
        "max_unselected_head_change",
        "max_outside_tail_change",
    ):
        if float(summary[field]) != 0.0:
            raise RuntimeError(f"factorization structural invariant failed: {field}")


def run_condition_prompt(
    *,
    group: str,
    condition: dict,
    record: dict,
    baseline_logits: torch.Tensor,
    paired_reference: torch.Tensor,
    model,
    tokenizer,
    artifacts,
    saved_global: dict,
) -> tuple[dict, list[dict]]:
    reference = reference_for_condition(
        condition,
        global_reference=artifacts.reference_mean,
        paired_reference=paired_reference,
    )
    heads = tuple(condition["heads"])
    prompt_artifacts = replace(
        artifacts,
        top5_heads=heads,
        reference_mean=reference.contiguous(),
    )
    with patch_llama_reference_attention(
        model,
        artifacts=prompt_artifacts,
        alpha=ALPHA,
        intervention_timing="first_token_only",
    ) as controller:
        controller.set_prompt_context(
            prompt_length=record["prompt_length"],
            intervention_timing="first_token_only",
        )
        generation = generate_with_first_logits(model, tokenizer, record["input_ids"])
        diagnostics = controller.summary()
        head_metrics = controller.head_metric_records()
    if generation["generation_failed"]:
        raise RuntimeError(
            f"generation failed at {group}/{condition['name']}/"
            f"{record['dataset_index']}: {generation['generation_error']}"
        )
    _validate_prompt_diagnostics(diagnostics, heads)
    comparison = compare_token_sequences(
        record["baseline_response_token_ids"], generation["response_token_ids"]
    )
    validate_first_token_response_invariant(
        response_equal=comparison["response_equal"],
        first_different_token_position=comparison["first_different_token_position"],
    )
    logit_metrics = compute_logit_step_metrics(
        baseline_logits,
        generation["first_logits"],
        step=0,
    )
    if comparison["response_equal"] != logit_metrics["argmax_equal"]:
        raise RuntimeError("response/first-token argmax invariance mismatch")
    transition = classify_transition(record["baseline_success"], generation["success"])
    prior_match = None
    if condition["name"] == "top5_global_mean":
        prior = saved_global[record["dataset_index"]]
        prior_match = {
            "response_equal": generation["response"] == prior["response"],
            "success_equal": generation["success"] == prior["success"],
            "generation_failure_equal": (
                generation["generation_failed"] == prior["generation_failed"]
            ),
        }
        if not all(prior_match.values()):
            raise RuntimeError(
                f"current global condition mismatch at {group}/{record['dataset_index']}"
            )
    row = {
        "dataset_group": group,
        "condition": condition["name"],
        "dataset_index": record["dataset_index"],
        "row_position": record["row_position"],
        "prompt_id": record["prompt_id"],
        "prompt": record["prompt"],
        "formatted_prompt": record["formatted_prompt"],
        "prompt_length": record["prompt_length"],
        "alpha": ALPHA,
        "intervention_timing": "first_token_only",
        "selected_heads": [
            {"layer": layer, "head": head} for layer, head in heads
        ],
        "reference_kind": condition["reference_kind"],
        "oracle_reference": condition["oracle_reference"],
        "deployable_defense": condition["deployable_defense"],
        "requires_paired_original_prompt": condition[
            "requires_paired_original_prompt"
        ],
        "baseline_response": record["baseline_response"],
        "baseline_response_token_ids": record["baseline_response_token_ids"],
        "baseline_success": record["baseline_success"],
        **{
            key: value
            for key, value in generation.items()
            if key != "first_logits"
        },
        **comparison,
        "first_token_argmax_equal": logit_metrics["argmax_equal"],
        "first_token_logit_metrics": logit_metrics,
        "transition": transition,
        "diagnostics": diagnostics,
        "matches_existing_global_artifact": prior_match,
    }
    metric_rows = [
        {
            "dataset_group": group,
            "condition": condition["name"],
            "dataset_index": record["dataset_index"],
            "reference_kind": condition["reference_kind"],
            **{
                key: metric[key]
                for key in (
                    "layer",
                    "head",
                    "current_asym_frobenius_norm",
                    "reference_asym_frobenius_norm",
                    "intervened_asym_frobenius_norm",
                    "current_to_reference_frobenius_distance",
                    "current_to_reference_mean_absolute_distance",
                    "mean_absolute_delta",
                    "max_absolute_delta",
                    "intervened_to_current_norm_ratio",
                )
            },
        }
        for metric in head_metrics
    ]
    return row, metric_rows


def main(argv=None):
    args = parse_args(argv)
    output_dir = ensure_output_absent(args.output_dir).resolve()
    model_path = Path(args.model).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    selection_run = Path(args.selection_run).resolve()
    holdout_run = Path(args.holdout_run).resolve()
    conditions = factorization_conditions()
    validate_condition_definitions(conditions)

    artifacts = load_validated_reference_artifacts(
        artifact_dir,
        expected_heads=EXPECTED_TOP5,
        expected_selection_manifest_sha256=EXPECTED_SELECTION_MANIFEST_SHA256,
        expected_score_artifact_sha256=EXPECTED_SCORE_ARTIFACT_SHA256,
        expected_model_path=model_path,
    )
    groups, paired_records, selection_manifest, holdout_manifest, saved_global = (
        _load_group_inputs(args)
    )

    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(model_path)
    model_validation = validate_loaded_llama_model(model, artifacts)
    groups = {
        group: format_records(records, tokenizer)
        for group, records in groups.items()
    }

    output_dir.mkdir(parents=True)
    run_manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at": utc_now(),
        "execution_git_head": git_value("rev-parse", "HEAD"),
        "execution_git_branch": git_value("branch", "--show-current"),
        "scope": "mechanistic/oracle factorization; not a new defense evaluation",
        "final_test": False,
        "independent_evaluation": False,
        "dataset_groups": list(GROUP_COUNTS),
        "group_counts": GROUP_COUNTS,
        "selection_run": str(selection_run),
        "selection_run_manifest_sha256": sha256_file(
            selection_run / "run_manifest.json"
        ),
        "holdout_run": str(holdout_run),
        "holdout_run_manifest_sha256": sha256_file(holdout_run / "run_manifest.json"),
        "artifact_dir": str(artifact_dir),
        "reference_artifact_sha256": sha256_file(
            artifact_dir / "original_reference.pt"
        ),
        "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
        "score_artifact_sha256": EXPECTED_SCORE_ARTIFACT_SHA256,
        "model": str(model_path),
        "model_config": llama_model_config_summary(model),
        "model_validation": model_validation,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "generation_settings": GENERATION_SETTINGS,
        "timing": "first_token_only",
        "alpha": ALPHA,
        "tail_tokens": TAIL_TOKENS,
        "conditions": [
            {
                **condition,
                "heads": [
                    {"layer": layer, "head": head}
                    for layer, head in condition["heads"]
                ],
            }
            for condition in conditions
        ],
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    baseline_logits = capture_baseline_logits(
        groups, model=model, saved_global=saved_global
    )
    paired_values, paired_manifest_records = extract_paired_references(
        paired_records,
        model=model,
        tokenizer=tokenizer,
    )
    lower = strict_lower_indices(TAIL_TOKENS, device="cpu")
    paired_payload = {
        "values": paired_values,
        "axes": ["paired_prompt", "layer", "head", "strict_lower_position"],
        "dataset_groups": [
            record["dataset_group"] for record in paired_manifest_records
        ],
        "dataset_indices": [
            record["dataset_index"] for record in paired_manifest_records
        ],
        "tril_rows": lower[0].to(torch.int64),
        "tril_cols": lower[1].to(torch.int64),
        "metadata": {
            "reference_name": "paired original reference",
            "alternative_name": "paired non-jailbreak oracle reference",
            "oracle_reference": True,
            "deployable_defense": False,
            "requires_paired_original_prompt": True,
            "tail_tokens": TAIL_TOKENS,
            "stored_value_dtype": "torch.float32",
            "torch_dtype": "torch.float16",
            "attention_implementation": "eager",
            "score_capture_position": EXPECTED_SCORE_CAPTURE_POSITION,
            "score_calculation_dtype": SCORE_CALCULATION_DTYPE,
            "strict_lower_position_order": (
                "torch.tril_indices(16, 16, offset=-1)"
            ),
        },
    }
    paired_path = output_dir / "paired_original_references.pt"
    save_torch_artifact(paired_path, paired_payload)
    paired_manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        **paired_payload["metadata"],
        "model": str(model_path),
        "model_config": llama_model_config_summary(model),
        "value_shape": list(paired_values.shape),
        "prompt_reference_shape": list(paired_values.shape[1:]),
        "finite": bool(torch.isfinite(paired_values).all().item()),
        "paired_original_reference_sha256": sha256_file(paired_path),
        "selection_original_input": str(Path(args.selection_original).resolve()),
        "selection_original_input_sha256": sha256_file(args.selection_original),
        "advbench_input": str(Path(args.advbench).resolve()),
        "advbench_input_sha256": sha256_file(args.advbench),
        "selection_split_manifest_sha256": sha256_file(
            artifact_dir / "split_manifest.json"
        ),
        "holdout_manifest_sha256": sha256_file(
            holdout_run / "holdout_manifest.json"
        ),
        "records": paired_manifest_records,
    }
    write_json(output_dir / "paired_original_reference_manifest.json", paired_manifest)

    paired_by_key = {
        (record["dataset_group"], record["dataset_index"]): paired_values[
            record["paired_position"]
        ]
        for record in paired_manifest_records
    }
    condition_rows = []
    reference_rows = []
    for group, records in groups.items():
        for condition in conditions:
            for position, record in enumerate(records, 1):
                row, metrics = run_condition_prompt(
                    group=group,
                    condition=condition,
                    record=record,
                    baseline_logits=baseline_logits[group][record["dataset_index"]],
                    paired_reference=paired_by_key[(group, record["dataset_index"])],
                    model=model,
                    tokenizer=tokenizer,
                    artifacts=artifacts,
                    saved_global=saved_global[group],
                )
                condition_rows.append(row)
                reference_rows.extend(metrics)
                print(
                    f"[condition {group}/{condition['name']} "
                    f"{position}/{len(records)}] index={record['dataset_index']} "
                    f"success={row['success']} changed={not row['response_equal']}",
                    flush=True,
                )

    condition_summary = summarize_condition_results(condition_rows)
    logit_summary = summarize_condition_logits(condition_rows)
    reference_summary = summarize_reference_distances(reference_rows)
    cause_classification = classify_factorization_causes(condition_summary)

    write_jsonl(output_dir / "condition_results.jsonl", condition_rows)
    write_json(output_dir / "condition_summary.json", condition_summary)
    write_json(output_dir / "group_logit_summary.json", logit_summary)
    write_json(output_dir / "reference_distance_summary.json", reference_summary)
    diagnosis_summary = {
        "status": "complete",
        "created_at": utc_now(),
        "run_directory": str(output_dir),
        "scope": "mechanistic/oracle factorization; not a new defense evaluation",
        "final_test": False,
        "independent_evaluation": False,
        "paired_reference_validation": {
            "shape": list(paired_values.shape),
            "dtype": str(paired_values.dtype),
            "finite": bool(torch.isfinite(paired_values).all().item()),
            "short_sequence_count": sum(
                record["original_sequence_length"] < TAIL_TOKENS
                for record in paired_manifest_records
            ),
        },
        "global_condition_existing_artifact_mismatch_count": sum(
            not all(row["matches_existing_global_artifact"].values())
            for row in condition_rows
            if row["condition"] == "top5_global_mean"
        ),
        "condition_summary": condition_summary,
        "cause_classification": cause_classification,
        "generation_failure_count": 0,
        "empty_response_count": sum(bool(row["empty_response"]) for row in condition_rows),
        "nan_or_inf_count": 0,
    }
    write_json(output_dir / "diagnosis_summary.json", diagnosis_summary)

    run_manifest["status"] = "complete"
    run_manifest["completed_at"] = diagnosis_summary["created_at"]
    run_manifest["paired_original_reference_sha256"] = sha256_file(paired_path)
    run_manifest["global_condition_existing_artifact_mismatch_count"] = 0
    # run_manifest is the one intentional in-run status transition.
    temporary_manifest = output_dir / ".run_manifest.json.tmp"
    with temporary_manifest.open("x", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_manifest.replace(output_dir / "run_manifest.json")
    print(json.dumps(diagnosis_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
