#!/usr/bin/env python3
"""Run the twelve new dynamic full-prompt Asym Stage 2 conditions."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_gcg_dynamic_asym_selection import tensor_sha256  # noqa: E402
from scripts.evaluate_gcg_dynamic_asym_stage1 import (  # noqa: E402
    DEFAULT_ADVBENCH,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BASELINE_RESULTS,
    DEFAULT_HOLDOUT_RUN,
    DEFAULT_JAILBREAK_INPUT,
    DEFAULT_MODEL,
    DEFAULT_RAW_ARCHIVE,
    DEFAULT_REFERENCE,
    DEFAULT_SELECTION_RUN,
    EXPECTED_REFERENCE_SHA256,
    load_development_inputs,
    summarize_condition_group,
    validate_and_load_reference,
    validate_runtime_records,
)
from scripts.evaluate_gcg_first_token_reference_ablation import (  # noqa: E402
    generate_with_first_logits,
    load_model_and_tokenizer,
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
from utils.gcg_dynamic_asym_selection_diagnostics import (  # noqa: E402
    concentration_summary,
    patch_llama_abs_z_collection,
)
from utils.gcg_dynamic_asym_stage2 import (  # noqa: E402
    STAGE2_REUSED_VARIANT,
    build_stage2_selection_plan,
    patch_llama_stage2_dynamic_asymmetry_attention,
    stage2_new_conditions,
    stage2_selection_conditions,
    validate_stage2_conditions,
)
from utils.gcg_first_token_reference_ablation import (  # noqa: E402
    validate_first_token_response_invariant,
)
from utils.gcg_reference_intervention_diagnostics import (  # noqa: E402
    compare_token_sequences,
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
from utils.gcg_wasserstein_reference_evaluation import classify_transition  # noqa: E402


DEFAULT_STAGE1_DIR = (
    "data/outputs/gcg_dynamic_asym_stage1/llama2/development68/"
    "stage1_20260916T130504Z"
)
EXPECTED_STAGE1_SUMMARY_SHA256 = (
    "980ebd466ef3015f48b8b502f130b958e692403cc9ec18caa3d5863a45c999ac"
)
EXPECTED_STAGE1_ARTIFACT_MANIFEST_SHA256 = (
    "e85a03690e70dd90b1935d08da71f0e8d874df848ce80b94024454a09a498e96"
)
TOTAL_DEVELOPMENT_COUNT = sum(GROUP_COUNTS.values())
SEED = 42
ALLOWED_STAGE1_FILES = {
    "baseline_results.jsonl",
    "selection_plans.jsonl",
    "condition_results.jsonl",
    "comparison_summary.json",
    "run_manifest.json",
}
BASELINE_REUSE_FILE = "stage1_baseline_reuse.jsonl"
PLANS_FILE = "selection_plans.jsonl"
RESULTS_FILE = "condition_results.jsonl"
SUMMARY_FILE = "comparison_summary.json"
ANALYSIS_FILE = "comparison_analysis.json"
RUN_MANIFEST_FILE = "run_manifest.json"
ARTIFACT_MANIFEST_FILE = "artifact_manifest.json"
SOURCE_CODE_PATHS = (
    "scripts/evaluate_gcg_dynamic_asym_stage2.py",
    "utils/gcg_dynamic_asym_stage2.py",
    "utils/gcg_dynamic_asym_stage1.py",
    "utils/gcg_dynamic_asym_intervention.py",
    "utils/gcg_dynamic_asym_selection_diagnostics.py",
    "utils/reference_asym_score_intervention.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("cpu-preflight", "run"), default="cpu-preflight"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--stage1-dir", default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--selection-run", default=DEFAULT_SELECTION_RUN)
    parser.add_argument("--holdout-run", default=DEFAULT_HOLDOUT_RUN)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--raw-archive", default=DEFAULT_RAW_ARCHIVE)
    parser.add_argument("--advbench", default=DEFAULT_ADVBENCH)
    parser.add_argument("--output-dir")
    parser.add_argument("--sandbox-exempt-gpu", action="store_true")
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


def _condition_name(selection_name: str, variant_name: str) -> str:
    return f"{selection_name}__{variant_name}"


def _prompt_key(row: dict) -> str:
    return f"{row['dataset_group']}:{int(row['dataset_index'])}"


def runtime_provenance() -> dict:
    git_head, git_branch = git_state(REPO_ROOT)
    return {
        "git_head": git_head,
        "git_branch": git_branch,
        "dirty_status": git_value(REPO_ROOT, "status", "--porcelain").splitlines(),
        "source_code_sha256": {
            path: sha256_file(REPO_ROOT / path) for path in SOURCE_CODE_PATHS
        },
    }


def load_and_validate_stage1(path: str | Path) -> dict:
    root = Path(path).resolve()
    summary_path = root / SUMMARY_FILE
    artifact_manifest_path = root / ARTIFACT_MANIFEST_FILE
    if sha256_file(summary_path) != EXPECTED_STAGE1_SUMMARY_SHA256:
        raise ValueError("Stage 1 comparison summary SHA-256 mismatch")
    if sha256_file(artifact_manifest_path) != EXPECTED_STAGE1_ARTIFACT_MANIFEST_SHA256:
        raise ValueError("Stage 1 artifact manifest SHA-256 mismatch")
    artifact_manifest = read_json(artifact_manifest_path)
    entries = artifact_manifest.get("files")
    if not isinstance(entries, list) or {row.get("path") for row in entries} != ALLOWED_STAGE1_FILES:
        raise ValueError("Stage 1 artifact manifest file set mismatch")
    for row in entries:
        if sha256_file(root / row["path"]) != row["sha256"]:
            raise ValueError(f"Stage 1 artifact hash mismatch: {row['path']}")

    run_manifest = read_json(root / RUN_MANIFEST_FILE)
    summary = read_json(summary_path)
    if (
        run_manifest.get("status") != "complete"
        or run_manifest.get("reference_sha256") != EXPECTED_REFERENCE_SHA256
        or run_manifest.get("independent100_used") is not False
        or run_manifest.get("autodan_used") is not False
        or run_manifest.get("stage2_conditions_run") is not False
        or run_manifest.get("alpha") != 1.0
        or run_manifest.get("layer0_included") is not True
    ):
        raise ValueError("Stage 1 run manifest scientific conditions mismatch")
    if (
        summary.get("experiment_role") != "development_stage1_tuning"
        or summary.get("stage2_conditions_run") is not False
        or summary.get("final_selection_rule_chosen") is not False
    ):
        raise ValueError("Stage 1 comparison summary role mismatch")

    baseline_rows = read_jsonl(root / "baseline_results.jsonl")
    plan_rows = read_jsonl(root / PLANS_FILE)
    result_rows = read_jsonl(root / RESULTS_FILE)
    if len(baseline_rows) != TOTAL_DEVELOPMENT_COUNT:
        raise ValueError("Stage 1 baseline row count mismatch")
    if len(plan_rows) != 9 * TOTAL_DEVELOPMENT_COUNT:
        raise ValueError("Stage 1 plan row count mismatch")
    if len(result_rows) != 9 * TOTAL_DEVELOPMENT_COUNT:
        raise ValueError("Stage 1 result row count mismatch")

    selected_names = {row["name"] for row in stage2_selection_conditions()}
    selected_plans = [
        row for row in plan_rows if row["condition"]["name"] in selected_names
    ]
    selected_results = [
        row for row in result_rows if row["condition"] in selected_names
    ]
    if len(selected_plans) != 4 * TOTAL_DEVELOPMENT_COUNT:
        raise ValueError("Stage 1 selected plan subset count mismatch")
    if len(selected_results) != 4 * TOTAL_DEVELOPMENT_COUNT:
        raise ValueError("Stage 1 selected result subset count mismatch")
    if any(
        row.get("alpha") != 1.0 or row.get("layer0_included") is not True
        for row in selected_results
    ):
        raise ValueError("Stage 1 reused results are not alpha1/Layer0 included")

    summary_by_name = {
        row["condition"]["name"]: row for row in summary["conditions"]
    }
    if not selected_names <= summary_by_name.keys():
        raise ValueError("Stage 1 summary is missing selected conditions")
    return {
        "root": root,
        "summary": summary,
        "summary_by_name": summary_by_name,
        "baseline_rows": baseline_rows,
        "plan_rows": selected_plans,
        "result_rows": selected_results,
        "run_manifest": run_manifest,
        "artifact_manifest_sha256": EXPECTED_STAGE1_ARTIFACT_MANIFEST_SHA256,
    }


def validate_stage1_input_alignment(groups: dict, stage1: dict) -> None:
    baseline_by_key = {_prompt_key(row): row for row in stage1["baseline_rows"]}
    if len(baseline_by_key) != TOTAL_DEVELOPMENT_COUNT:
        raise ValueError("Stage 1 baseline prompt keys are not unique")
    for group, records in groups.items():
        for record in records:
            key = f"{group}:{record['dataset_index']}"
            baseline = baseline_by_key.get(key)
            if baseline is None or (
                baseline["prompt_id"] != record["prompt_id"]
                or baseline["prompt_length"] != record["saved_prompt_length"]
                or baseline["response"] != record["baseline_response"]
                or baseline["success"] is not record["baseline_success"]
                or baseline["generation_failed"] is not False
            ):
                raise ValueError("Stage 1 baseline/input alignment mismatch")


def preflight_summary(inputs: dict, stage1: dict, reference_path: Path) -> dict:
    conditions = stage2_new_conditions()
    return {
        "experiment_role": "dynamic_full_prompt_asym_stage2_development_tuning",
        "development_only": True,
        "final_test": False,
        "independent100_used": False,
        "autodan_used": False,
        "group_counts": GROUP_COUNTS,
        "total_prompt_count": TOTAL_DEVELOPMENT_COUNT,
        "reference": str(reference_path),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "stage1_artifact": str(stage1["root"]),
        "stage1_comparison_summary_sha256": EXPECTED_STAGE1_SUMMARY_SHA256,
        "stage1_artifact_manifest_sha256": stage1["artifact_manifest_sha256"],
        "stage1_reused_variant": STAGE2_REUSED_VARIANT,
        "stage1_reused_gpu_execution": False,
        "new_gpu_condition_count": len(conditions),
        "new_gpu_conditions": list(conditions),
        "selection_settings": list(stage2_selection_conditions()),
        "position_scope": "full_prompt_strict_lower_i_gt_j",
        "component": "A_asym=(S-S^T)/2",
        "intervention": "pure_asym_pairwise",
        "intervention_timing": "first_prefill_only",
        "cap_rate": 0.005,
        "cap_denominator": "original_all_layer_candidate_count_C",
        "layer0_excluded_candidate_definition": "layers_1_through_31",
        "proportional_n_denominator_when_layer0_excluded": (
            "eligible_layers_1_through_31_candidate_count"
        ),
        "layer0_excluded_top_up": False,
        "fixed_input_sha256": inputs["fixed_hashes"],
    }


def _stage1_plan_lookup(stage1: dict) -> dict:
    return {
        (
            row["dataset_group"],
            int(row["dataset_index"]),
            row["condition"]["name"],
        ): row
        for row in stage1["plan_rows"]
    }


def collect_selection_plans(groups, *, model, reference, stage1):
    selections = stage2_selection_conditions()
    old_plans = _stage1_plan_lookup(stage1)
    plans = {}
    serialized = []
    differences = []
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
                prompt_plans = {}
                for selection in selections:
                    pair = {}
                    for layer0_included in (True, False):
                        plan = build_stage2_selection_plan(
                            abs_z,
                            selection=selection,
                            layer_count=NUM_LAYERS,
                            head_count=NUM_HEADS,
                            elements_per_head=elements_per_head,
                            layer0_included=layer0_included,
                        )
                        plan["selected_indices_sha256"] = tensor_sha256(
                            plan["selected_indices"]
                        )
                        if layer0_included:
                            old = old_plans.get(
                                (group, record["dataset_index"], selection["name"])
                            )
                            checked = (
                                "selected_count",
                                "cap_count",
                                "cap_activated",
                                "cap_excluded_count",
                                "layer_counts",
                                "head_counts",
                                "layer_head_counts",
                                "selected_indices_sha256",
                            )
                            if old is None or any(old[key] != plan[key] for key in checked):
                                raise RuntimeError(
                                    "recomputed Stage 1 included plan differs from artifact"
                                )
                        pair[layer0_included] = plan
                        prompt_plans[(selection["name"], layer0_included)] = plan
                        serialized.append(
                            {
                                "dataset_group": group,
                                "dataset_index": record["dataset_index"],
                                "row_position": record["row_position"],
                                "prompt_id": record["prompt_id"],
                                "prompt_length": record["prompt_length"],
                                "selection": selection,
                                **{
                                    key: value
                                    for key, value in plan.items()
                                    if key not in {"condition", "selected_indices"}
                                },
                                "runtime": runtime,
                            }
                        )
                    included = pair[True]
                    excluded = pair[False]
                    included_set = set(included["selected_indices"].tolist())
                    excluded_set = set(excluded["selected_indices"].tolist())
                    union = included_set | excluded_set
                    differences.append(
                        {
                            "dataset_group": group,
                            "dataset_index": record["dataset_index"],
                            "prompt_id": record["prompt_id"],
                            "selection": selection["name"],
                            "included_selected_count": included["selected_count"],
                            "excluded_selected_count": excluded["selected_count"],
                            "excluded_minus_included": (
                                excluded["selected_count"] - included["selected_count"]
                            ),
                            "included_layer0_selected_count": included["layer_counts"][0],
                            "overlap_count": len(included_set & excluded_set),
                            "jaccard": len(included_set & excluded_set) / len(union),
                            "included_only_count": len(included_set - excluded_set),
                            "excluded_only_count": len(excluded_set - included_set),
                        }
                    )
                plans[(group, record["dataset_index"])] = prompt_plans
                print(
                    f"[selection-plan {group} {position}/{len(records)}] "
                    f"index={record['dataset_index']} L={record['prompt_length']}",
                    flush=True,
                )
    if len(serialized) != 8 * TOTAL_DEVELOPMENT_COUNT:
        raise AssertionError("Stage 2 selection plan count mismatch")
    return plans, serialized, differences


def run_new_conditions(groups, plans, *, model, tokenizer, reference, stage1):
    baseline_by_key = {_prompt_key(row): row for row in stage1["baseline_rows"]}
    output = []
    for condition in stage2_new_conditions():
        selection = condition["selection"]
        with patch_llama_stage2_dynamic_asymmetry_attention(
            model,
            reference=reference,
            selection=selection,
            alpha=condition["alpha"],
            layer0_included=condition["layer0_included"],
        ) as controller:
            for group, records in groups.items():
                for position, record in enumerate(records, 1):
                    plan = plans[(group, record["dataset_index"])][
                        (selection["name"], condition["layer0_included"])
                    ]
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
                    baseline = baseline_by_key[f"{group}:{record['dataset_index']}"]
                    comparison = compare_token_sequences(
                        baseline["response_token_ids"], generation["response_token_ids"]
                    )
                    validate_first_token_response_invariant(
                        response_equal=comparison["response_equal"],
                        first_different_token_position=comparison[
                            "first_different_token_position"
                        ],
                    )
                    transition = classify_transition(
                        bool(baseline["success"]), bool(generation["success"])
                    )
                    output.append(
                        {
                            "condition": condition["name"],
                            "selection": selection,
                            "alpha": condition["alpha"],
                            "layer0_included": condition["layer0_included"],
                            "dataset_group": group,
                            "dataset_index": record["dataset_index"],
                            "row_position": record["row_position"],
                            "prompt_id": record["prompt_id"],
                            "jailbreak_prompt_sha256": record.get(
                                "jailbreak_prompt_sha256"
                            ),
                            "prompt_length": record["prompt_length"],
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
    if len(output) != 12 * TOTAL_DEVELOPMENT_COUNT:
        raise AssertionError("Stage 2 result count mismatch")
    return output


def _group_rows(rows: list[dict], group: str) -> list[dict]:
    if group == "all68":
        return list(rows)
    return [row for row in rows if row["dataset_group"] == group]


def _improvement_set(rows: list[dict], group: str) -> set[str]:
    return {
        _prompt_key(row)
        for row in _group_rows(rows, group)
        if row["transition"] == "newly_defended"
    }


def _set_comparison(left: set[str], right: set[str]) -> dict:
    union = left | right
    return {
        "left_count": len(left),
        "right_count": len(right),
        "overlap_count": len(left & right),
        "jaccard": len(left & right) / len(union) if union else 1.0,
        "left_only": sorted(left - right),
        "right_only": sorted(right - left),
        "overlap": sorted(left & right),
    }


def _integer_summary(values: list[int]) -> dict:
    if not values or any(type(value) is not int for value in values):
        raise ValueError("integer summary requires a non-empty integer list")
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "prompt_count": len(values),
        "mean": float(tensor.mean().item()),
        "median": float(torch.quantile(tensor, 0.5).item()),
        "min": int(tensor.min().item()),
        "max": int(tensor.max().item()),
        "std": float(tensor.std(correction=0).item()),
        "std_definition": "population_ddof_0",
    }


def _outcome_comparison(left_rows: list[dict], right_rows: list[dict], group: str) -> dict:
    left = {_prompt_key(row): row for row in _group_rows(left_rows, group)}
    right = {_prompt_key(row): row for row in _group_rows(right_rows, group)}
    if left.keys() != right.keys():
        raise ValueError("condition outcome prompt sets differ")
    changed = [key for key in sorted(left) if left[key]["success"] != right[key]["success"]]
    left_success_count = sum(bool(row["success"]) for row in left.values())
    right_success_count = sum(bool(row["success"]) for row in right.values())
    left_improvements = _improvement_set(left_rows, group)
    right_improvements = _improvement_set(right_rows, group)
    return {
        "prompt_count": len(left),
        "left_success_count": left_success_count,
        "right_success_count": right_success_count,
        "left_asr": left_success_count / len(left),
        "right_asr": right_success_count / len(right),
        "right_minus_left_asr": (
            right_success_count / len(right) - left_success_count / len(left)
        ),
        "left_improvement_count": len(left_improvements),
        "right_improvement_count": len(right_improvements),
        "right_minus_left_improvement_count": (
            len(right_improvements) - len(left_improvements)
        ),
        "judgment_changed_count": len(changed),
        "judgment_changed_prompts": [
            {
                "prompt_key": key,
                "prompt_id": left[key]["prompt_id"],
                "dataset_group": left[key]["dataset_group"],
                "dataset_index": left[key]["dataset_index"],
                "left_success": bool(left[key]["success"]),
                "right_success": bool(right[key]["success"]),
                "baseline_success": bool(left[key]["baseline_success"]),
            }
            for key in changed
        ],
        "left_success_to_right_failure": sum(
            bool(left[key]["success"]) and not bool(right[key]["success"])
            for key in changed
        ),
        "left_failure_to_right_success": sum(
            not bool(left[key]["success"]) and bool(right[key]["success"])
            for key in changed
        ),
        "improvement_sets": _set_comparison(
            left_improvements, right_improvements
        ),
    }


def _add_layer_concentration(group_summary: dict) -> dict:
    result = dict(group_summary)
    result["layer_concentration"] = concentration_summary(
        result["layer_counts"], labels=tuple(range(NUM_LAYERS))
    )
    return result


def build_condition_summaries(new_rows, plan_rows, stage1) -> tuple[list[dict], dict]:
    summaries = []
    rows_by_condition = {}
    stage1_rows_by_selection = {
        selection["name"]: [
            row
            for row in stage1["result_rows"]
            if row["condition"] == selection["name"]
        ]
        for selection in stage2_selection_conditions()
    }
    for selection in stage2_selection_conditions():
        reused_name = _condition_name(selection["name"], STAGE2_REUSED_VARIANT["name"])
        reused_rows = stage1_rows_by_selection[selection["name"]]
        rows_by_condition[reused_name] = reused_rows
        summaries.append(
            {
                "condition": reused_name,
                "selection": selection,
                **STAGE2_REUSED_VARIANT,
                "source": "reused_stage1_artifact_no_new_gpu_execution",
                "groups": {
                    group: _add_layer_concentration(values)
                    for group, values in stage1["summary_by_name"][selection["name"]][
                        "groups"
                    ].items()
                },
            }
        )
        for condition in stage2_new_conditions():
            if condition["selection"]["name"] != selection["name"]:
                continue
            current_rows = [row for row in new_rows if row["condition"] == condition["name"]]
            current_plans = [
                row
                for row in plan_rows
                if row["selection"]["name"] == selection["name"]
                and row["layer0_included"] is condition["layer0_included"]
            ]
            rows_by_condition[condition["name"]] = current_rows
            summaries.append(
                {
                    "condition": condition["name"],
                    "selection": selection,
                    "alpha": condition["alpha"],
                    "layer0_included": condition["layer0_included"],
                    "reused_from_stage1": False,
                    "source": "new_stage2_gpu_execution",
                    "groups": {
                        group: _add_layer_concentration(
                            summarize_condition_group(
                                current_rows, current_plans, group=group
                            )
                        )
                        for group in ("all68", "selection20", "holdout48")
                    },
                }
            )
    if len(summaries) != 16 or len(rows_by_condition) != 16:
        raise AssertionError("Stage 2 combined condition count mismatch")
    return summaries, rows_by_condition


def summarize_plan_differences(rows: list[dict]) -> list[dict]:
    output = []
    for selection in stage2_selection_conditions():
        for group in ("all68", "selection20", "holdout48"):
            current = [
                row
                for row in rows
                if row["selection"] == selection["name"]
                and (group == "all68" or row["dataset_group"] == group)
            ]
            deltas = [row["excluded_minus_included"] for row in current]
            output.append(
                {
                    "selection": selection["name"],
                    "dataset_group": group,
                    "prompt_count": len(current),
                    "included_selected_count": sum(
                        row["included_selected_count"] for row in current
                    ),
                    "excluded_selected_count": sum(
                        row["excluded_selected_count"] for row in current
                    ),
                    "excluded_minus_included_per_prompt": _integer_summary(deltas),
                    "included_layer0_selected_count": sum(
                        row["included_layer0_selected_count"] for row in current
                    ),
                    "mean_plan_jaccard": sum(row["jaccard"] for row in current)
                    / len(current),
                    "prompt_details": current,
                }
            )
    return output


def build_comparison_analysis(rows_by_condition: dict) -> dict:
    groups = ("all68", "selection20", "holdout48")
    pairwise = []
    names = sorted(rows_by_condition)
    for left_name, right_name in itertools.combinations(names, 2):
        pairwise.append(
            {
                "left": left_name,
                "right": right_name,
                "groups": {
                    group: _set_comparison(
                        _improvement_set(rows_by_condition[left_name], group),
                        _improvement_set(rows_by_condition[right_name], group),
                    )
                    for group in groups
                },
            }
        )

    focused = []
    l0h15_changes = []
    for selection in stage2_selection_conditions():
        stem = selection["name"]
        names_by_variant = {
            "a1_in": _condition_name(stem, "alpha_1p0_layer0_included"),
            "a05_in": _condition_name(stem, "alpha_0p5_layer0_included"),
            "a1_ex": _condition_name(stem, "alpha_1p0_layer0_excluded"),
            "a05_ex": _condition_name(stem, "alpha_0p5_layer0_excluded"),
        }
        comparisons = (
            ("alpha_1p0_to_0p5_layer0_included", "a1_in", "a05_in"),
            ("alpha_1p0_to_0p5_layer0_excluded", "a1_ex", "a05_ex"),
            ("layer0_included_to_excluded_alpha_1p0", "a1_in", "a1_ex"),
            ("layer0_included_to_excluded_alpha_0p5", "a05_in", "a05_ex"),
        )
        for label, left_key, right_key in comparisons:
            left_name = names_by_variant[left_key]
            right_name = names_by_variant[right_key]
            group_results = {
                group: _outcome_comparison(
                    rows_by_condition[left_name], rows_by_condition[right_name], group
                )
                for group in groups
            }
            focused.append(
                {
                    "selection": stem,
                    "comparison": label,
                    "left": left_name,
                    "right": right_name,
                    "groups": group_results,
                }
            )
            if label.startswith("layer0_included_to_excluded"):
                l0h15_changes.append(
                    {
                        "selection": stem,
                        "alpha": 1.0 if "1p0" in label else 0.5,
                        "scope_note": (
                            "Entire Layer 0 was excluded; L0H15 was removed as part of "
                            "that intervention, so this is not an isolated L0H15 ablation."
                        ),
                        **group_results["all68"],
                    }
                )
    improvement_prompts = {
        name: {
            group: sorted(_improvement_set(rows, group)) for group in groups
        }
        for name, rows in rows_by_condition.items()
    }
    return {
        "improvement_prompts": improvement_prompts,
        "all_condition_pairwise_improvement_overlap": pairwise,
        "focused_alpha_and_layer0_comparisons": focused,
        "layer0_exclusion_including_l0h15_judgment_changes": l0h15_changes,
    }


def main(argv=None):
    args = parse_args(argv)
    output_dir = validate_request(args)
    conditions = stage2_new_conditions()
    validate_stage2_conditions(conditions)
    inputs = load_development_inputs(args)
    stage1 = load_and_validate_stage1(args.stage1_dir)
    validate_stage1_input_alignment(inputs["groups"], stage1)
    reference_path, reference = validate_and_load_reference(args.reference)
    preflight = preflight_summary(inputs, stage1, reference_path)
    preflight["stage"] = args.stage
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.stage == "cpu-preflight":
        print("CPU preflight complete; no files created.")
        return

    provenance = runtime_provenance()
    seed_everything(SEED)
    model, tokenizer = load_model_and_tokenizer(Path(args.model).resolve())
    groups = validate_runtime_records(inputs["groups"], tokenizer)
    plans, plan_rows, plan_differences = collect_selection_plans(
        groups, model=model, reference=reference, stage1=stage1
    )
    condition_rows = run_new_conditions(
        groups,
        plans,
        model=model,
        tokenizer=tokenizer,
        reference=reference,
        stage1=stage1,
    )
    condition_summaries, rows_by_condition = build_condition_summaries(
        condition_rows, plan_rows, stage1
    )
    analysis = build_comparison_analysis(rows_by_condition)
    analysis["layer0_selection_plan_differences"] = summarize_plan_differences(
        plan_differences
    )
    created_at = utc_now()
    summary = {
        "schema_version": 1,
        "created_at": created_at,
        "experiment_role": "development_stage2_tuning",
        "final_selection_rule_chosen": False,
        "independent100_used": False,
        "autodan_used": False,
        "baseline": stage1["summary"]["baseline"],
        "scientific_conditions": preflight,
        "condition_count": len(condition_summaries),
        "new_gpu_condition_count": len(conditions),
        "reused_stage1_condition_count": 4,
        "conditions": condition_summaries,
    }

    ensure_output_directory_absent(output_dir)
    output_dir.mkdir(parents=True)
    baseline_path = output_dir / BASELINE_REUSE_FILE
    plans_path = output_dir / PLANS_FILE
    results_path = output_dir / RESULTS_FILE
    summary_path = output_dir / SUMMARY_FILE
    analysis_path = output_dir / ANALYSIS_FILE
    write_jsonl(baseline_path, stage1["baseline_rows"])
    write_jsonl(plans_path, plan_rows)
    write_jsonl(results_path, condition_rows)
    write_json(summary_path, summary)
    write_json(analysis_path, analysis)
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
            "max_new_tokens": 32,
            "baseline_generation_reused_from_stage1": True,
        },
        "files": {
            path.name: sha256_file(path)
            for path in (baseline_path, plans_path, results_path, summary_path, analysis_path)
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
                analysis_path,
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
                "comparison_analysis_sha256": sha256_file(analysis_path),
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
                "new_gpu_condition_count": len(conditions),
                "reused_stage1_condition_count": 4,
                "conditions": [
                    {
                        "condition": row["condition"],
                        "source": row["source"],
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
                    for row in condition_summaries
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
