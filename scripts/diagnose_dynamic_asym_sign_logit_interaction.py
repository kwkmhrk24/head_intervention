#!/usr/bin/env python3
"""Measure first-token logit interaction between frozen absolute sign subsets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_gcg_dynamic_asym_independent100 import (  # noqa: E402
    ALPHA,
    LAYER0_INCLUDED,
    fixed_selection,
    parse_args as parse_evaluation_args,
    validate_frozen_inputs,
)
from scripts.evaluate_gcg_dynamic_asym_stage1 import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_REFERENCE,
    validate_and_load_reference,
)
from scripts.evaluate_gcg_first_token_reference_ablation import (  # noqa: E402
    format_records,
    load_model_and_tokenizer,
)
from scripts.evaluate_gcg_position_element_topk_intervention import (  # noqa: E402
    serialized_input_ids,
)
from scripts.run_autodan_position_element_selection_ablation import (  # noqa: E402
    first_forward_logits,
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
    patch_llama_abs_z_collection,
)
from utils.gcg_dynamic_asym_stage2 import (  # noqa: E402
    build_stage2_selection_plan,
    patch_llama_stage2_dynamic_asymmetry_attention,
    restrict_absolute_plan_to_intervention_sign,
)
from utils.gcg_wasserstein_reference import NUM_HEADS, NUM_LAYERS  # noqa: E402
from utils.harmbench_gcg_independent_test import (  # noqa: E402
    load_independent_evaluation_records,
    validate_formatted_evaluation_identity,
)


SEED = 42
TARGET_INDICES = (290, 313, 254, 249)
SOURCE_RUNS = {
    "all": Path(
        "data/outputs/gcg_dynamic_asym_signed_direction/llama2/gcg_ref/"
        "absolute_0p1/evaluation_20260927T152212Z"
    ),
    "positive": Path(
        "data/outputs/gcg_dynamic_asym_absolute_sign_ablation/llama2/gcg_ref/"
        "absolute_positive_only_0p1/evaluation_20260927T154921Z"
    ),
    "negative": Path(
        "data/outputs/gcg_dynamic_asym_absolute_sign_ablation/llama2/gcg_ref/"
        "absolute_negative_only_0p1/evaluation_20260927T154921Z"
    ),
}
EXPECTED_SOURCE_MANIFEST_SHA256 = {
    "all": "2d160934bd84fec066f75bcf84cf7964bc0ccb9be441dc860164e5264f5dd600",
    "positive": "8d8cc651cb5693ca13d74b427725dda2b55fb9940316b4557059c42631bb0a91",
    "negative": "14d00cb90d79c03fc7eb8aa0c5b0bb1030f535c98f99e7ef314d75e1ada71acc",
}
EXPECTED_SUCCESS = {
    "all": {290: False, 313: False, 254: False, 249: True},
    "positive": {290: True, 313: True, 254: True, 249: False},
    "negative": {290: True, 313: True, 254: False, 249: True},
}
SOURCE_PATHS = (
    "scripts/diagnose_dynamic_asym_sign_logit_interaction.py",
    "scripts/evaluate_gcg_dynamic_asym_independent100.py",
    "scripts/run_autodan_position_element_selection_ablation.py",
    "utils/gcg_dynamic_asym_stage2.py",
    "utils/gcg_dynamic_asym_intervention.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu-preflight", "run"), default="cpu-preflight")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    evaluation = parse_evaluation_args([])
    parser.add_argument("--independent-config", default=evaluation.independent_config)
    parser.add_argument(
        "--independent-input-manifest", default=evaluation.independent_input_manifest
    )
    parser.add_argument(
        "--independent-attack-run-dir", default=evaluation.independent_attack_run_dir
    )
    parser.add_argument("--independent-adapter", default=evaluation.independent_adapter)
    parser.add_argument("--reused-baseline", default=evaluation.reused_baseline)
    parser.add_argument("--output-dir")
    parser.add_argument("--sandbox-exempt-gpu", action="store_true")
    return parser.parse_args(argv)


def validate_request(args) -> Path | None:
    if args.stage == "cpu-preflight":
        if args.output_dir is not None or args.sandbox_exempt_gpu:
            raise ValueError("cpu-preflight cannot write output or claim GPU execution")
        return None
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    return ensure_output_directory_absent(Path(args.output_dir).resolve())


def evaluation_args(args):
    return parse_evaluation_args(
        [
            "--dataset",
            "independent100",
            "--model",
            args.model,
            "--reference",
            args.reference,
            "--independent-config",
            args.independent_config,
            "--independent-input-manifest",
            args.independent_input_manifest,
            "--independent-attack-run-dir",
            args.independent_attack_run_dir,
            "--independent-adapter",
            args.independent_adapter,
            "--reused-baseline",
            args.reused_baseline,
        ]
    )


def load_source_rows() -> tuple[dict[str, dict[int, dict]], dict[str, str]]:
    loaded = {}
    hashes = {}
    for target, relative_directory in SOURCE_RUNS.items():
        directory = (REPO_ROOT / relative_directory).resolve()
        manifest_path = directory / "artifact_manifest.json"
        observed = sha256_file(manifest_path)
        if observed != EXPECTED_SOURCE_MANIFEST_SHA256[target]:
            raise ValueError(f"{target} source artifact manifest SHA-256 mismatch")
        manifest = read_json(manifest_path)
        for item in manifest["files"]:
            if sha256_file(directory / item["path"]) != item["sha256"]:
                raise ValueError(f"{target} source artifact file SHA-256 mismatch")
        run_manifest = read_json(directory / "run_manifest.json")
        conditions = run_manifest["scientific_conditions"]
        if (
            conditions["dataset"] != "independent100"
            or conditions["component"] != "antisymmetric"
            or conditions["selection_direction"] != "absolute"
            or conditions["reference_source"] != "gcg-selection20-original"
            or conditions["selection"] != fixed_selection()
            or conditions["alpha"] != ALPHA
            or conditions["layer0_included"] is not LAYER0_INCLUDED
            or conditions["timing"] != "first_prefill_only"
            or conditions["cap_rate"] != 0.005
            or conditions.get("intervention_target_sign", "all") != target
        ):
            raise ValueError(f"{target} source scientific conditions mismatch")
        rows = {
            int(row["dataset_index"]): row
            for row in read_jsonl(directory / "prompt_results.jsonl")
            if int(row["dataset_index"]) in TARGET_INDICES
        }
        if set(rows) != set(TARGET_INDICES):
            raise ValueError(f"{target} source target prompt set mismatch")
        if any(bool(rows[index]["success"]) != EXPECTED_SUCCESS[target][index] for index in TARGET_INDICES):
            raise ValueError(f"{target} source known outcome mismatch")
        loaded[target] = rows
        hashes[target] = observed
    return loaded, hashes


def preflight(args) -> dict:
    provenance, baseline = validate_frozen_inputs(evaluation_args(args))
    baseline_by_index = {int(row["dataset_index"]): row for row in baseline}
    if set(TARGET_INDICES) - set(baseline_by_index):
        raise ValueError("target prompts are missing from the frozen baseline")
    if any(not bool(baseline_by_index[index]["success"]) for index in TARGET_INDICES):
        raise ValueError("all four target prompts must be baseline successes")
    source_rows, source_hashes = load_source_rows()
    return {
        "provenance": provenance,
        "baseline_by_index": baseline_by_index,
        "source_rows": source_rows,
        "source_manifest_sha256": source_hashes,
    }


def logit_condition_metrics(logits: torch.Tensor, baseline_top1: int) -> dict:
    values = logits.double().cpu()
    if values.ndim != 1 or not bool(torch.isfinite(values).all().item()):
        raise ValueError("logits must be a finite rank-1 tensor")
    top = torch.topk(values, 2)
    top1 = int(top.indices[0])
    return {
        "top1_token_id": top1,
        "top1_logit": float(top.values[0]),
        "top2_token_id": int(top.indices[1]),
        "top2_logit": float(top.values[1]),
        "top1_top2_margin": float(top.values[0] - top.values[1]),
        "argmax_changed_from_baseline": top1 != baseline_top1,
    }


def interaction_metrics(baseline, positive, negative, absolute_all) -> dict:
    values = [tensor.double().cpu() for tensor in (baseline, positive, negative, absolute_all)]
    if any(tensor.ndim != 1 or tensor.shape != values[0].shape for tensor in values):
        raise ValueError("logit vectors must have matching rank-1 shapes")
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in values):
        raise ValueError("logit vectors must be finite")
    delta_positive = values[1] - values[0]
    delta_negative = values[2] - values[0]
    delta_all = values[3] - values[0]
    delta_add = delta_positive + delta_negative
    interaction = delta_all - delta_add
    norm_all = float(torch.linalg.vector_norm(delta_all))
    result = {
        "delta_positive_l2": float(torch.linalg.vector_norm(delta_positive)),
        "delta_negative_l2": float(torch.linalg.vector_norm(delta_negative)),
        "delta_all_l2": norm_all,
        "delta_add_l2": float(torch.linalg.vector_norm(delta_add)),
        "interaction_l2": float(torch.linalg.vector_norm(interaction)),
        "interaction_max_absolute": float(interaction.abs().max()),
        "interaction_over_delta_all": (
            float(torch.linalg.vector_norm(interaction)) / norm_all if norm_all else None
        ),
    }
    if not all(value is None or math.isfinite(value) for value in result.values()):
        raise ValueError("interaction metrics contain NaN or infinity")
    return result, values[0] + delta_add


def add_token_text(metrics: dict, tokenizer) -> dict:
    enriched = dict(metrics)
    for rank in ("top1", "top2"):
        token_id = metrics[f"{rank}_token_id"]
        enriched[f"{rank}_token"] = tokenizer.convert_ids_to_tokens(token_id)
        enriched[f"{rank}_decoded"] = tokenizer.decode([token_id])
    return enriched


def intervention_logits(model, input_ids, reference, selection, plan) -> tuple[torch.Tensor, dict]:
    with patch_llama_stage2_dynamic_asymmetry_attention(
        model,
        reference=reference,
        selection=selection,
        alpha=ALPHA,
        layer0_included=LAYER0_INCLUDED,
        component="antisymmetric",
    ) as controller:
        controller.set_prompt_plan(prompt_length=input_ids.shape[1], plan=plan)
        logits = first_forward_logits(model, input_ids)
        controller.assert_complete_prefill()
        runtime = controller.summary()
    if (
        runtime["all_scores_finite"] is not True
        or runtime["later_forward_intervention_count"] != 0
        or runtime["selected_element_intervention_count"] != plan["selected_count"]
        or runtime["max_unselected_element_change"] != 0.0
        or runtime["max_diagonal_change_float32"] != 0.0
        or plan["selected_count"] > plan["cap_count"]
    ):
        raise RuntimeError("intervention runtime invariant failed")
    return logits, runtime


def diagnose_prompt(record, baseline_row, source_rows, *, model, tokenizer, reference) -> dict:
    input_ids = record["input_ids"]
    baseline_logits = first_forward_logits(model, input_ids)
    baseline_top1 = int(torch.argmax(baseline_logits))
    if baseline_top1 != int(baseline_row["response_token_ids"][0]):
        raise RuntimeError("baseline first-token reproduction mismatch")
    with patch_llama_abs_z_collection(
        model,
        layer_count=NUM_LAYERS,
        head_count=NUM_HEADS,
        component="antisymmetric",
    ) as collector:
        collector.set_prompt_context(
            prompt_length=record["prompt_length"],
            excluded_selection_position=0,
            mu=reference.mu,
            sigma=reference.sigma,
        )
        with torch.inference_mode():
            model.model(input_ids=input_ids.to(model.device), use_cache=False, return_dict=True)
        signed_z, collection = collector.result(signed=True)
    selection = fixed_selection()
    absolute_plan = build_stage2_selection_plan(
        signed_z.abs(),
        signed_z=signed_z,
        selection_direction="absolute",
        selection=selection,
        layer_count=NUM_LAYERS,
        head_count=NUM_HEADS,
        elements_per_head=collection["strict_lower_elements_per_head"],
        layer0_included=LAYER0_INCLUDED,
    )
    plans = {
        target: restrict_absolute_plan_to_intervention_sign(
            absolute_plan,
            signed_z,
            intervention_target_sign=target,
            layer_count=NUM_LAYERS,
            head_count=NUM_HEADS,
            elements_per_head=collection["strict_lower_elements_per_head"],
        )
        for target in ("positive", "negative", "all")
    }
    index = int(record["dataset_index"])
    for target, plan in plans.items():
        saved = source_rows[target][index]
        if (
            int(saved["selected_count"]) != plan["selected_count"]
            or int(saved.get("positive_selected_count", plan["positive_selected_count"]))
            != plan["positive_selected_count"]
            or int(saved.get("negative_selected_count", plan["negative_selected_count"]))
            != plan["negative_selected_count"]
        ):
            raise RuntimeError(f"{target} reconstructed selection count mismatch")
    logits = {"baseline": baseline_logits}
    runtime = {}
    for target in ("positive", "negative", "all"):
        logits[target], runtime[target] = intervention_logits(
            model, input_ids, reference, selection, plans[target]
        )
        expected_token = int(source_rows[target][index]["response_token_ids"][0])
        if int(torch.argmax(logits[target])) != expected_token:
            raise RuntimeError(f"{target} first-token reproduction mismatch")
    interaction, additive_logits = interaction_metrics(
        logits["baseline"], logits["positive"], logits["negative"], logits["all"]
    )
    logits["additive_prediction"] = additive_logits
    conditions = {
        condition: add_token_text(
            logit_condition_metrics(values, baseline_top1), tokenizer
        )
        for condition, values in logits.items()
    }
    add_changed = conditions["additive_prediction"]["argmax_changed_from_baseline"]
    all_changed = conditions["all"]["argmax_changed_from_baseline"]
    if add_changed and all_changed and (
        conditions["additive_prediction"]["top1_token_id"]
        == conditions["all"]["top1_token_id"]
    ):
        decision_case = "A_additive_and_actual_change_to_same_top1"
    elif not add_changed and all_changed:
        decision_case = "B_only_actual_all_changes_top1"
    elif add_changed and not all_changed:
        decision_case = "C_only_additive_prediction_changes_top1"
    else:
        decision_case = "other_or_unclear"
    return {
        "dataset_index": index,
        "prompt_id": record["prompt_id"],
        "prompt_length": record["prompt_length"],
        "selection": {
            "absolute_top_n_count": plans["all"]["selected_count"],
            "positive_count": plans["positive"]["selected_count"],
            "negative_count": plans["negative"]["selected_count"],
            "cap_count": plans["all"]["cap_count"],
            "cap_activated": plans["all"]["cap_activated"],
        },
        "conditions": conditions,
        "logit_change": interaction,
        "decision_case": decision_case,
        "runtime": {
            target: {
                "selected_element_intervention_count": values[
                    "selected_element_intervention_count"
                ],
                "later_forward_intervention_count": values[
                    "later_forward_intervention_count"
                ],
                "all_scores_finite": values["all_scores_finite"],
            }
            for target, values in runtime.items()
        },
        "runtime_invariants_passed": True,
    }


def run(args, output_dir: Path, frozen: dict) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA is required; refusing CPU model execution")
    seed_everything(SEED)
    model, tokenizer = load_model_and_tokenizer(Path(args.model).resolve())
    _, reference = validate_and_load_reference(args.reference)
    records = load_independent_evaluation_records(
        config_path=args.independent_config,
        input_manifest_path=args.independent_input_manifest,
        attack_run_dir=args.independent_attack_run_dir,
        adapter_path=args.independent_adapter,
    )
    records = format_records(records, tokenizer)
    validate_formatted_evaluation_identity(records)
    records_by_index = {int(row["dataset_index"]): row for row in records}
    rows = []
    for position, index in enumerate(TARGET_INDICES, 1):
        record = records_by_index[index]
        baseline = frozen["baseline_by_index"][index]
        if serialized_input_ids(record["input_ids"]) != baseline["input_ids"]:
            raise RuntimeError("formatted input differs from frozen baseline")
        rows.append(
            diagnose_prompt(
                record,
                baseline,
                frozen["source_rows"],
                model=model,
                tokenizer=tokenizer,
                reference=reference,
            )
        )
        print(f"[first-token interaction {position}/4] index={index}", flush=True)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "prompt_count": len(rows),
        "dataset_indices": list(TARGET_INDICES),
        "generation_used": False,
        "first_token_logits_only": True,
        "numerical_or_runtime_failure_count": 0,
        "decision_cases": {str(row["dataset_index"]): row["decision_case"] for row in rows},
    }
    output_dir.mkdir(parents=True)
    rows_path = output_dir / "per_prompt.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(rows_path, rows)
    write_json(summary_path, summary)
    head, branch = git_state(REPO_ROOT)
    run_manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": utc_now(),
        "git_head": head,
        "git_branch": branch,
        "dirty_status": git_value(REPO_ROOT, "status", "--porcelain").splitlines(),
        "sandbox_exempt_gpu": args.sandbox_exempt_gpu,
        "gpu": torch.cuda.get_device_name(0),
        "scientific_conditions": {
            "dataset": "GCG independent100 selected four prompts",
            "dataset_indices": list(TARGET_INDICES),
            "component": "antisymmetric",
            "selection": fixed_selection(),
            "selection_direction": "absolute_then_fixed_sign_subsets",
            "alpha": ALPHA,
            "layer0_included": LAYER0_INCLUDED,
            "position_scope": "full_prompt_strict_lower_i_gt_j",
            "timing": "first_prefill_only",
            "cap_rate": 0.005,
            "generation_used": False,
        },
        "provenance": frozen["provenance"],
        "source_artifact_manifest_sha256": frozen["source_manifest_sha256"],
        "source_code_sha256": {path: sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS},
        "files": {
            rows_path.name: sha256_file(rows_path),
            summary_path.name: sha256_file(summary_path),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, run_manifest)
    artifact_path = output_dir / "artifact_manifest.json"
    write_json(
        artifact_path,
        {
            "schema_version": 1,
            "files": [
                {"path": path.name, "sha256": sha256_file(path)}
                for path in (rows_path, summary_path, manifest_path)
            ],
        },
    )
    return {
        "output_dir": str(output_dir),
        "artifact_manifest_sha256": sha256_file(artifact_path),
        "summary": summary,
    }


def main(argv=None):
    args = parse_args(argv)
    output_dir = validate_request(args)
    frozen = preflight(args)
    payload = {
        "status": "cpu_preflight_passed",
        "dataset_indices": list(TARGET_INDICES),
        "selection": fixed_selection(),
        "alpha": ALPHA,
        "layer0_included": LAYER0_INCLUDED,
        "source_artifact_manifest_sha256": frozen["source_manifest_sha256"],
        "generation_used": False,
        "planned_gpu_first_forward_count": len(TARGET_INDICES) * 4,
        "planned_selection_collection_prefill_count": len(TARGET_INDICES),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.stage == "run":
        print(json.dumps(run(args, output_dir, frozen), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
