#!/usr/bin/env python3
"""Run the nine fixed dynamic full-prompt Asym Stage 1 conditions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_gcg_dynamic_asym_selection import tensor_sha256  # noqa: E402
from scripts.evaluate_gcg_first_token_reference_ablation import (  # noqa: E402
    format_records,
    generate_with_first_logits,
    load_model_and_tokenizer,
    run_baseline,
)
from utils.artifact_io import (  # noqa: E402
    ensure_output_directory_absent,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.experiment_runtime import git_state, git_value, seed_everything  # noqa: E402
from utils.gcg_dynamic_asym_intervention import (  # noqa: E402
    load_dynamic_reference,
)
from utils.gcg_dynamic_asym_selection_diagnostics import (  # noqa: E402
    add_count_summaries,
    concentration_summary,
    patch_llama_abs_z_collection,
    prompt_selected_count_summary,
    zero_count_summary,
)
from utils.gcg_dynamic_asym_stage1 import (  # noqa: E402
    CAP_RATE,
    build_stage1_selection_plan,
    patch_llama_planned_dynamic_asymmetry_attention,
    stage1_conditions,
    validate_stage1_conditions,
)
from utils.gcg_first_token_holdout import validate_holdout_manifest  # noqa: E402
from utils.gcg_first_token_reference_ablation import (  # noqa: E402
    load_selection_development_records,
    validate_first_token_response_invariant,
)
from utils.gcg_reference_intervention_diagnostics import (  # noqa: E402
    compare_token_sequences,
    compute_logit_step_metrics,
)
from utils.gcg_reference_factorization_diagnostics import (  # noqa: E402
    BASELINE_SUCCESS_COUNTS,
    GROUP_COUNTS,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    NUM_HEADS,
    NUM_LAYERS,
    llama_model_config_summary,
)
from utils.gcg_wasserstein_reference_evaluation import (  # noqa: E402
    classify_transition,
    summarize_transitions,
)


DEFAULT_MODEL = "/mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf"
DEFAULT_REFERENCE = (
    "data/outputs/gcg_dynamic_asym_reference/llama2/selection20_original/"
    "reference_20260916T094521Z/"
    "selection20_original_full_prompt_asym_reference.pt"
)
EXPECTED_REFERENCE_SHA256 = (
    "ba8f3dd92fea60ccd3cb2b9b880e9c99eaabc8ee6f6d54527673ad85092f19f2"
)
DEFAULT_ARTIFACT_DIR = (
    "data/outputs/gcg_wasserstein_reference/llama2/seed42_selection20/"
    "codex_gpu_20260802T061114Z"
)
DEFAULT_SELECTION_RUN = (
    "data/outputs/gcg_wasserstein_reference_intervention/llama2/"
    "seed42_selection20/first_token_ablation_20260802T084341Z"
)
DEFAULT_HOLDOUT_RUN = (
    "data/outputs/gcg_wasserstein_reference_intervention/llama2/"
    "first_token_holdout48/evaluation_20260802T095741Z"
)
DEFAULT_JAILBREAK_INPUT = "data/derived/llama2/gcg/jailbreak_prompts.jsonl"
DEFAULT_BASELINE_RESULTS = (
    "data/outputs/multidataset_asym_score_intervention/"
    "beta05_full_20260722_041303/llama_gcg/no_intervention/"
    "prompt_results.jsonl"
)
DEFAULT_RAW_ARCHIVE = "data/raw/llama2/llama2_GCG.json"
DEFAULT_ADVBENCH = "data/raw/advbench.jsonl"
EXPECTED_INPUT_SHA256 = {
    "selection_manifest": "13578541411456bd689f8e771694ca97b047f2828bc35663536a5bc65badfa1e",
    "jailbreak_input": "cfafe04323de0bbd1c7e995d6e9e4a4aa74b4e218c499625c460249e612fc588",
    "baseline_results": "7299531491c2fb2a636a96a3ddb7d0e2de5c7d0d70a1504a625a0bab1b98de2a",
    "selection_baseline": "d190439af1dec9a0f90fd152f20df227dd472ab249bf39888450071cd72822d0",
    "holdout_manifest": "373acc745f24f807b7356524d97a515c869356bd58d5ed3cb63936e206311bc9",
    "holdout_baseline": "6f425a508f95b69d8c3b4c111aff1cfd080652eb1bac406534118bef137306d5",
}
ALPHA = 1.0
SEED = 42
TOTAL_DEVELOPMENT_COUNT = sum(GROUP_COUNTS.values())
BASELINE_SUCCESS_TOTAL = sum(BASELINE_SUCCESS_COUNTS.values())
BASELINE_FILE = "baseline_results.jsonl"
PLANS_FILE = "selection_plans.jsonl"
RESULTS_FILE = "condition_results.jsonl"
SUMMARY_FILE = "comparison_summary.json"
RUN_MANIFEST_FILE = "run_manifest.json"
ARTIFACT_MANIFEST_FILE = "artifact_manifest.json"
SOURCE_CODE_PATHS = (
    "scripts/evaluate_gcg_dynamic_asym_stage1.py",
    "utils/gcg_dynamic_asym_stage1.py",
    "utils/gcg_dynamic_asym_intervention.py",
    "utils/gcg_dynamic_asym_selection_diagnostics.py",
    "utils/reference_asym_score_intervention.py",
    "scripts/evaluate_gcg_first_token_reference_ablation.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("cpu-preflight", "run"), default="cpu-preflight"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--selection-run", default=DEFAULT_SELECTION_RUN)
    parser.add_argument("--holdout-run", default=DEFAULT_HOLDOUT_RUN)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--raw-archive", default=DEFAULT_RAW_ARCHIVE)
    parser.add_argument("--advbench", default=DEFAULT_ADVBENCH)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--sandbox-exempt-gpu",
        action="store_true",
        help="Record that GPU execution required sandbox-exempt permissions.",
    )
    return parser.parse_args(argv)


def validate_request(args) -> Path | None:
    if args.stage == "cpu-preflight":
        if args.output_dir is not None:
            raise ValueError("cpu-preflight does not write an output directory")
        if args.sandbox_exempt_gpu:
            raise ValueError("cpu-preflight must not claim GPU execution")
        return None
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    return ensure_output_directory_absent(Path(args.output_dir).resolve())


def _require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return observed


def runtime_provenance() -> dict:
    git_head, git_branch = git_state(REPO_ROOT)
    return {
        "git_head": git_head,
        "git_branch": git_branch,
        "dirty_status": git_value(REPO_ROOT, "status", "--porcelain").splitlines(),
        "source_code_sha256": {
            relative_path: sha256_file(REPO_ROOT / relative_path)
            for relative_path in SOURCE_CODE_PATHS
        },
    }


def load_development_inputs(args) -> dict:
    artifact_dir = Path(args.artifact_dir).resolve()
    selection_run = Path(args.selection_run).resolve()
    holdout_run = Path(args.holdout_run).resolve()
    jailbreak_input = Path(args.jailbreak_input).resolve()
    baseline_results = Path(args.baseline_results).resolve()
    raw_archive = Path(args.raw_archive).resolve()
    advbench = Path(args.advbench).resolve()
    selection_manifest_path = artifact_dir / "split_manifest.json"
    selection_baseline_path = selection_run / "baseline_selection_results.jsonl"
    holdout_manifest_path = holdout_run / "holdout_manifest.json"
    holdout_baseline_path = holdout_run / "baseline/prompt_results.jsonl"

    fixed_hashes = {
        "selection_manifest": _require_sha(
            selection_manifest_path,
            EXPECTED_INPUT_SHA256["selection_manifest"],
            "selection manifest",
        ),
        "jailbreak_input": _require_sha(
            jailbreak_input,
            EXPECTED_INPUT_SHA256["jailbreak_input"],
            "jailbreak input",
        ),
        "baseline_results": _require_sha(
            baseline_results,
            EXPECTED_INPUT_SHA256["baseline_results"],
            "baseline results",
        ),
        "selection_baseline": _require_sha(
            selection_baseline_path,
            EXPECTED_INPUT_SHA256["selection_baseline"],
            "selection saved baseline",
        ),
        "holdout_manifest": _require_sha(
            holdout_manifest_path,
            EXPECTED_INPUT_SHA256["holdout_manifest"],
            "holdout manifest",
        ),
        "holdout_baseline": _require_sha(
            holdout_baseline_path,
            EXPECTED_INPUT_SHA256["holdout_baseline"],
            "holdout saved baseline",
        ),
    }

    selection, selection_manifest = load_selection_development_records(
        split_manifest_path=selection_manifest_path,
        jailbreak_input=jailbreak_input,
        baseline_results=baseline_results,
    )
    selection_saved_rows = read_jsonl(selection_baseline_path)
    selection_saved = {
        int(row["dataset_index"]): row for row in selection_saved_rows
    }
    if len(selection_saved) != GROUP_COUNTS["selection20"]:
        raise ValueError("selection saved baseline count mismatch")
    for record in selection:
        saved = selection_saved.get(record["dataset_index"])
        if saved is None or (
            saved["prompt"] != record["prompt"]
            or saved["response"] != record["baseline_response"]
            or saved["success"] is not record["baseline_success"]
            or saved["generation_failed"] is not False
        ):
            raise ValueError("selection saved baseline alignment mismatch")
        record.update(
            baseline_response_token_ids=saved["response_token_ids"],
            saved_formatted_prompt=saved["formatted_prompt"],
            saved_prompt_length=saved["prompt_length"],
        )

    holdout_manifest = read_json(holdout_manifest_path)
    holdout = validate_holdout_manifest(
        holdout_manifest,
        raw_archive_path=raw_archive,
        advbench_path=advbench,
    )
    holdout_saved_rows = read_jsonl(holdout_baseline_path)
    holdout_saved = {int(row["dataset_index"]): row for row in holdout_saved_rows}
    if len(holdout_saved) != GROUP_COUNTS["holdout48"]:
        raise ValueError("holdout saved baseline count mismatch")
    for record in holdout:
        saved = holdout_saved.get(record["dataset_index"])
        if saved is None or saved["prompt"] != record["prompt"]:
            raise ValueError("holdout saved baseline alignment mismatch")
        if saved["generation_failed"] is not False:
            raise ValueError("holdout saved baseline contains generation failure")
        record.update(
            baseline_response=saved["response"],
            baseline_response_token_ids=saved["response_token_ids"],
            baseline_success=bool(saved["success"]),
            baseline_generation_failed=False,
            baseline_response_token_count=saved["response_token_count"],
            saved_formatted_prompt=saved["formatted_prompt"],
            saved_prompt_length=saved["prompt_length"],
        )

    groups = {"selection20": selection, "holdout48": holdout}
    for group, expected_count in GROUP_COUNTS.items():
        records = groups[group]
        if len(records) != expected_count:
            raise ValueError(f"{group} count mismatch")
        success_count = sum(bool(row["baseline_success"]) for row in records)
        if success_count != BASELINE_SUCCESS_COUNTS[group]:
            raise ValueError(f"{group} baseline success count mismatch")
    return {
        "groups": groups,
        "selection_manifest": selection_manifest,
        "holdout_manifest": holdout_manifest,
        "paths": {
            "selection_manifest": selection_manifest_path,
            "jailbreak_input": jailbreak_input,
            "baseline_results": baseline_results,
            "selection_baseline": selection_baseline_path,
            "holdout_manifest": holdout_manifest_path,
            "holdout_baseline": holdout_baseline_path,
            "raw_archive": raw_archive,
            "advbench": advbench,
        },
        "fixed_hashes": fixed_hashes,
    }


def validate_and_load_reference(path: str | Path):
    reference_path = Path(path).resolve()
    _require_sha(reference_path, EXPECTED_REFERENCE_SHA256, "dynamic reference")
    reference = load_dynamic_reference(reference_path)
    if reference.metadata.get("jailbreak_reference_data_used") is not False:
        raise ValueError("dynamic reference indicates jailbreak data use")
    if reference.metadata.get("tau") is not None:
        raise ValueError("dynamic reference unexpectedly fixes tau")
    return reference_path, reference


def preflight_summary(inputs: dict, reference_path: Path, conditions) -> dict:
    return {
        "diagnostic_role": "dynamic_full_prompt_asym_stage1_development_tuning",
        "development_only": True,
        "final_test": False,
        "independent100_used": False,
        "autodan_used": False,
        "dataset_groups": list(GROUP_COUNTS),
        "group_counts": GROUP_COUNTS,
        "total_prompt_count": TOTAL_DEVELOPMENT_COUNT,
        "baseline_success_counts": {
            **BASELINE_SUCCESS_COUNTS,
            "all68": BASELINE_SUCCESS_TOTAL,
        },
        "reference": str(reference_path),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "alpha": ALPHA,
        "layer0_included": True,
        "cap_rate": CAP_RATE,
        "position_scope": "full_prompt_strict_lower_i_gt_j",
        "component": "A_asym=(S-S^T)/2",
        "intervention": "pure_asym_pairwise",
        "intervention_timing": "first_prefill_only",
        "selection_score_source": "one unmodified prefill per prompt",
        "conditions": list(conditions),
        "stage2_conditions_run": False,
        "fixed_input_sha256": inputs["fixed_hashes"],
    }


def validate_runtime_records(groups: dict, tokenizer) -> dict:
    formatted = {
        group: format_records(records, tokenizer) for group, records in groups.items()
    }
    for records in formatted.values():
        for record in records:
            if (
                record["formatted_prompt"] != record["saved_formatted_prompt"]
                or record["prompt_length"] != record["saved_prompt_length"]
            ):
                raise RuntimeError("runtime tokenization differs from saved baseline")
    return formatted


def collect_selection_plans(groups, *, model, reference, conditions):
    plans = {}
    serializable_rows = []
    with patch_llama_abs_z_collection(
        model, layer_count=NUM_LAYERS, head_count=NUM_HEADS
    ) as collector:
        for group, records in groups.items():
            for position, record in enumerate(records, 1):
                collector.clear()
                collector.set_prompt_context(
                    prompt_length=record["prompt_length"],
                    excluded_selection_position=record["row_position"],
                    mu=reference.mu,
                    sigma=reference.sigma,
                )
                with torch.inference_mode():
                    model.model(
                        input_ids=record["input_ids"].to(model.device),
                        use_cache=False,
                        return_dict=True,
                    )
                abs_z, runtime = collector.result()
                elements_per_head = runtime["strict_lower_elements_per_head"]
                condition_plans = {}
                for condition in conditions:
                    plan = build_stage1_selection_plan(
                        abs_z,
                        condition=condition,
                        layer_count=NUM_LAYERS,
                        head_count=NUM_HEADS,
                        elements_per_head=elements_per_head,
                    )
                    plan["selected_indices_sha256"] = tensor_sha256(
                        plan["selected_indices"]
                    )
                    condition_plans[condition["name"]] = plan
                    serializable_rows.append(
                        {
                            "dataset_group": group,
                            "dataset_index": record["dataset_index"],
                            "row_position": record["row_position"],
                            "prompt_id": record["prompt_id"],
                            "prompt_length": record["prompt_length"],
                            "condition": condition,
                            **{
                                key: value
                                for key, value in plan.items()
                                if key not in {"condition", "selected_indices"}
                            },
                            "runtime": runtime,
                        }
                    )
                plans[(group, record["dataset_index"])] = condition_plans
                print(
                    f"[selection-plan {group} {position}/{len(records)}] "
                    f"index={record['dataset_index']} L={record['prompt_length']} "
                    + " ".join(
                        f"{condition['name']}="
                        f"{condition_plans[condition['name']]['selected_count']}"
                        for condition in conditions
                    ),
                    flush=True,
                )
    return plans, serializable_rows


def serialize_baseline_rows(baseline_rows) -> list[dict]:
    output = []
    for group, rows in baseline_rows.items():
        for row in rows:
            output.append(
                {
                    "dataset_group": group,
                    "dataset_index": row["dataset_index"],
                    "row_position": row["row_position"],
                    "prompt_id": row["prompt_id"],
                    "jailbreak_prompt_sha256": row.get("jailbreak_prompt_sha256"),
                    "prompt_length": row["prompt_length"],
                    "response": row["response"],
                    "response_token_ids": row["response_token_ids"],
                    "response_token_count": row["response_token_count"],
                    "success": row["success"],
                    "generation_failed": row["generation_failed"],
                    "empty_response": row["empty_response"],
                    "matches_saved_response": row["response"]
                    == row["baseline_response"],
                    "matches_saved_success": row["success"]
                    is row["baseline_success"],
                }
            )
    if len(output) != TOTAL_DEVELOPMENT_COUNT:
        raise AssertionError("serialized baseline count mismatch")
    return output


def run_stage1_conditions(
    groups,
    baseline_rows,
    plans,
    *,
    model,
    tokenizer,
    reference,
    conditions,
):
    baseline_by_key = {
        (group, row["dataset_index"]): row
        for group, rows in baseline_rows.items()
        for row in rows
    }
    output = []
    for condition in conditions:
        with patch_llama_planned_dynamic_asymmetry_attention(
            model,
            reference=reference,
            condition=condition,
            alpha=ALPHA,
        ) as controller:
            for group, records in groups.items():
                for position, record in enumerate(records, 1):
                    plan = plans[(group, record["dataset_index"])][condition["name"]]
                    controller.clear()
                    controller.set_prompt_plan(
                        prompt_length=record["prompt_length"], plan=plan
                    )
                    generation = generate_with_first_logits(
                        model, tokenizer, record["input_ids"]
                    )
                    if generation["generation_failed"]:
                        raise RuntimeError(
                            f"generation failed at {condition['name']}/{group}/"
                            f"{record['dataset_index']}: {generation['generation_error']}"
                        )
                    controller.assert_complete_prefill()
                    diagnostics = controller.summary()
                    baseline = baseline_by_key[(group, record["dataset_index"])]
                    comparison = compare_token_sequences(
                        baseline["response_token_ids"],
                        generation["response_token_ids"],
                    )
                    validate_first_token_response_invariant(
                        response_equal=comparison["response_equal"],
                        first_different_token_position=comparison[
                            "first_different_token_position"
                        ],
                    )
                    logit_metrics = compute_logit_step_metrics(
                        baseline["first_logits"], generation["first_logits"], step=0
                    )
                    if comparison["response_equal"] != logit_metrics["argmax_equal"]:
                        raise RuntimeError(
                            "response and first-token argmax change disagree"
                        )
                    transition = classify_transition(
                        bool(baseline["success"]), bool(generation["success"])
                    )
                    output.append(
                        {
                            "condition": condition["name"],
                            "condition_definition": condition,
                            "dataset_group": group,
                            "dataset_index": record["dataset_index"],
                            "row_position": record["row_position"],
                            "prompt_id": record["prompt_id"],
                            "jailbreak_prompt_sha256": record.get(
                                "jailbreak_prompt_sha256"
                            ),
                            "prompt_length": record["prompt_length"],
                            "alpha": ALPHA,
                            "layer0_included": True,
                            "baseline_response": baseline["response"],
                            "baseline_response_token_ids": baseline[
                                "response_token_ids"
                            ],
                            "baseline_success": bool(baseline["success"]),
                            **{
                                key: value
                                for key, value in generation.items()
                                if key != "first_logits"
                            },
                            **comparison,
                            "first_token_logit_metrics": logit_metrics,
                            "transition": transition,
                            "selected_count": plan["selected_count"],
                            "selected_fraction": plan["selected_fraction"],
                            "cap_count": plan["cap_count"],
                            "cap_activated": plan["cap_activated"],
                            "cap_excluded_count": plan["cap_excluded_count"],
                            "diagnostics": diagnostics,
                        }
                    )
                    print(
                        f"[condition {condition['name']} {group} "
                        f"{position}/{len(records)}] index={record['dataset_index']} "
                        f"success={generation['success']} transition={transition} "
                        f"selected={plan['selected_count']}",
                        flush=True,
                    )
    if len(output) != len(conditions) * TOTAL_DEVELOPMENT_COUNT:
        raise AssertionError("Stage 1 condition result count mismatch")
    return output


def summarize_condition_group(rows, plan_rows, *, group: str) -> dict:
    expected_count = (
        TOTAL_DEVELOPMENT_COUNT if group == "all68" else GROUP_COUNTS[group]
    )
    selected_rows = (
        list(rows)
        if group == "all68"
        else [row for row in rows if row["dataset_group"] == group]
    )
    selected_plans = (
        list(plan_rows)
        if group == "all68"
        else [row for row in plan_rows if row["dataset_group"] == group]
    )
    if len(selected_rows) != expected_count or len(selected_plans) != expected_count:
        raise ValueError(f"{group} Stage 1 summary count mismatch")
    transitions = summarize_transitions(selected_rows, expected_count=expected_count)
    baseline_success_count = sum(bool(row["baseline_success"]) for row in selected_rows)
    baseline_asr = baseline_success_count / expected_count
    counts = [int(row["selected_count"]) for row in selected_plans]
    candidate_total = sum(int(row["candidate_count"]) for row in selected_plans)
    count_summary = prompt_selected_count_summary(counts)
    aggregate = zero_count_summary(NUM_LAYERS, NUM_HEADS)
    for row in selected_plans:
        aggregate = add_count_summaries(aggregate, row)
    layer_head_flat = [value for row in aggregate["layer_head_counts"] for value in row]
    applied_total = sum(
        int(row["diagnostics"]["selected_element_intervention_count"])
        for row in selected_rows
    )
    planned_total = sum(counts)
    if applied_total != planned_total:
        raise AssertionError("planned/applied selected total mismatch")
    return {
        "dataset_group": group,
        "prompt_count": expected_count,
        "baseline_success_count": baseline_success_count,
        "baseline_asr": baseline_asr,
        "successful_count": transitions["attack_success_count"],
        "asr": transitions["asr"],
        "asr_difference_from_baseline": transitions["asr"] - baseline_asr,
        "improvement_count": transitions["newly_defended"],
        "worsening_count": transitions["newly_worsened"],
        "still_success_count": transitions["still_attack_success"],
        "still_failure_count": transitions["still_defended"],
        "transition_indices": transitions["indices"],
        "response_changed_count": sum(
            not bool(row["response_equal"]) for row in selected_rows
        ),
        "response_equal_count": sum(
            bool(row["response_equal"]) for row in selected_rows
        ),
        "selected_count": planned_total,
        "selected_count_per_prompt": count_summary,
        "candidate_count": candidate_total,
        "selected_fraction": planned_total / candidate_total,
        "cap_activated_prompt_count": sum(
            bool(row["cap_activated"]) for row in selected_plans
        ),
        "cap_excluded_count": sum(
            int(row["cap_excluded_count"]) for row in selected_plans
        ),
        **aggregate,
        "head_concentration": concentration_summary(
            aggregate["head_counts"], labels=tuple(range(NUM_HEADS))
        ),
        "layer_head_concentration": concentration_summary(
            layer_head_flat,
            labels=tuple(
                (layer, head)
                for layer in range(NUM_LAYERS)
                for head in range(NUM_HEADS)
            ),
        ),
        "generation_failure_count": sum(
            bool(row["generation_failed"]) for row in selected_rows
        ),
        "empty_response_count": sum(
            bool(row["empty_response"]) for row in selected_rows
        ),
        "all_scores_finite": all(
            bool(row["diagnostics"]["all_scores_finite"])
            for row in selected_rows
        ),
        "sigma_floor_head_count": max(
            int(row["diagnostics"]["sigma_floor_head_count"])
            for row in selected_rows
        ),
        "later_forward_intervention_count": sum(
            int(row["diagnostics"]["later_forward_intervention_count"])
            for row in selected_rows
        ),
        "max_symmetric_component_error_float32": max(
            float(row["diagnostics"]["max_symmetric_component_error_float32"])
            for row in selected_rows
        ),
        "max_symmetric_component_error_after_cast": max(
            float(row["diagnostics"]["max_symmetric_component_error_after_cast"])
            for row in selected_rows
        ),
        "max_unselected_element_change": max(
            float(row["diagnostics"]["max_unselected_element_change"])
            for row in selected_rows
        ),
        "max_diagonal_change_float32": max(
            float(row["diagnostics"]["max_diagonal_change_float32"])
            for row in selected_rows
        ),
    }


def build_comparison_summary(condition_rows, plan_rows, conditions) -> dict:
    summaries = []
    for condition in conditions:
        current_rows = [
            row for row in condition_rows if row["condition"] == condition["name"]
        ]
        current_plans = [
            row
            for row in plan_rows
            if row["condition"]["name"] == condition["name"]
        ]
        if len(current_rows) != TOTAL_DEVELOPMENT_COUNT:
            raise ValueError("Stage 1 condition row count mismatch")
        summaries.append(
            {
                "condition": condition,
                "groups": {
                    group: summarize_condition_group(
                        current_rows, current_plans, group=group
                    )
                    for group in ("all68", "selection20", "holdout48")
                },
            }
        )
    return {
        "schema_version": 1,
        "experiment_role": "development_stage1_tuning",
        "final_selection_rule_chosen": False,
        "stage2_conditions_run": False,
        "baseline": {
            "all68": {
                "prompt_count": TOTAL_DEVELOPMENT_COUNT,
                "successful_count": BASELINE_SUCCESS_TOTAL,
                "asr": BASELINE_SUCCESS_TOTAL / TOTAL_DEVELOPMENT_COUNT,
            },
            **{
                group: {
                    "prompt_count": GROUP_COUNTS[group],
                    "successful_count": BASELINE_SUCCESS_COUNTS[group],
                    "asr": BASELINE_SUCCESS_COUNTS[group] / GROUP_COUNTS[group],
                }
                for group in GROUP_COUNTS
            },
        },
        "conditions": summaries,
    }


def main(argv=None):
    args = parse_args(argv)
    output_dir = validate_request(args)
    conditions = stage1_conditions()
    validate_stage1_conditions(conditions)
    inputs = load_development_inputs(args)
    reference_path, reference = validate_and_load_reference(args.reference)
    preflight = preflight_summary(inputs, reference_path, conditions)
    preflight["stage"] = args.stage
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.stage == "cpu-preflight":
        print("CPU preflight complete; no files created.")
        return

    provenance = runtime_provenance()
    seed_everything(SEED)
    model, tokenizer = load_model_and_tokenizer(Path(args.model).resolve())
    groups = validate_runtime_records(inputs["groups"], tokenizer)
    baseline_rows = {
        group: run_baseline(
            records,
            model=model,
            tokenizer=tokenizer,
            label=f"Stage1 baseline {group}",
        )
        for group, records in groups.items()
    }
    plans, plan_rows = collect_selection_plans(
        groups,
        model=model,
        reference=reference,
        conditions=conditions,
    )
    condition_rows = run_stage1_conditions(
        groups,
        baseline_rows,
        plans,
        model=model,
        tokenizer=tokenizer,
        reference=reference,
        conditions=conditions,
    )
    summary = build_comparison_summary(condition_rows, plan_rows, conditions)
    created_at = utc_now()
    summary.update(
        {
            "created_at": created_at,
            "reference": str(reference_path),
            "reference_sha256": EXPECTED_REFERENCE_SHA256,
            "scientific_conditions": preflight,
        }
    )

    ensure_output_directory_absent(output_dir)
    output_dir.mkdir(parents=True)
    baseline_path = output_dir / BASELINE_FILE
    plans_path = output_dir / PLANS_FILE
    results_path = output_dir / RESULTS_FILE
    summary_path = output_dir / SUMMARY_FILE
    write_jsonl(baseline_path, serialize_baseline_rows(baseline_rows))
    write_jsonl(plans_path, plan_rows)
    write_jsonl(results_path, condition_rows)
    write_json(summary_path, summary)
    run_manifest = {
        **preflight,
        "schema_version": 1,
        "status": "complete",
        "created_at": created_at,
        **provenance,
        "sandbox_exempt_gpu": args.sandbox_exempt_gpu,
        "model": str(Path(args.model).resolve()),
        "model_config": llama_model_config_summary(model),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "seed": SEED,
        "generation_settings": {
            "torch_dtype": "torch.float16",
            "attention_implementation": "eager",
            "use_cache": False,
            "batch_size": 1,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": 32,
            "system_prompt": None,
            "tokenize_add_special_tokens": False,
        },
        "files": {
            BASELINE_FILE: sha256_file(baseline_path),
            PLANS_FILE: sha256_file(plans_path),
            RESULTS_FILE: sha256_file(results_path),
            SUMMARY_FILE: sha256_file(summary_path),
        },
        "raw_per_element_tensor_saved": False,
    }
    run_manifest_path = output_dir / RUN_MANIFEST_FILE
    write_json(run_manifest_path, run_manifest)
    artifact_manifest = {
        "schema_version": 1,
        "created_at": created_at,
        "files": [
            {"path": path.name, "sha256": sha256_file(path)}
            for path in (
                baseline_path,
                plans_path,
                results_path,
                summary_path,
                run_manifest_path,
            )
        ],
    }
    artifact_manifest_path = output_dir / ARTIFACT_MANIFEST_FILE
    write_json(artifact_manifest_path, artifact_manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "comparison_summary_sha256": sha256_file(summary_path),
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
                "baseline": summary["baseline"],
                "conditions": [
                    {
                        "condition": row["condition"]["name"],
                        "groups": {
                            group: {
                                key: value
                                for key, value in values.items()
                                if key
                                in {
                                    "successful_count",
                                    "asr",
                                    "asr_difference_from_baseline",
                                    "improvement_count",
                                    "worsening_count",
                                    "response_changed_count",
                                    "selected_count_per_prompt",
                                    "selected_fraction",
                                    "cap_activated_prompt_count",
                                    "cap_excluded_count",
                                }
                            }
                            for group, values in row["groups"].items()
                        },
                    }
                    for row in summary["conditions"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
