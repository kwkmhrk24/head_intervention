#!/usr/bin/env python3
"""Diagnose fixed dynamic Asym score-to-first-logit propagation on small samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_gcg_autodan_logit_propagation import (  # noqa: E402
    decision_metrics,
    distribution,
)
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
    format_autodan_records,
    load_autodan_transfer_records,
)
from scripts.evaluate_gcg_pure_symmetric_autodan_transfer import OFFICIAL_FORMAT  # noqa: E402
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
)
from utils.gcg_wasserstein_reference import NUM_HEADS, NUM_LAYERS  # noqa: E402
from utils.harmbench_gcg_independent_test import (  # noqa: E402
    load_independent_evaluation_records,
    validate_formatted_evaluation_identity,
)


GCG_RUN = REPO_ROOT / (
    "data/outputs/gcg_dynamic_asym_independent100/llama2/"
    "final_proportional_0p1_alpha1_layer0_excluded/evaluation_20260916T140449Z"
)
AUTODAN_RUN = REPO_ROOT / (
    "data/outputs/gcg_dynamic_asym_autodan_transfer/llama2/official_fastchat_34062e9/"
    "final_proportional_0p1_alpha1_layer0_excluded/evaluation_20260916T142538Z"
)
EXPECTED_MANIFEST_SHA256 = {
    "GCG": "1195264fb053a76bc427149741314061cfdcae0a5a176f537ffdd9a2961fc19a",
    "AutoDAN": "893f50497b62867067047ea1ac4873b09daf43134ba478efdbde2f39b25d655e",
}
SAMPLE_COUNTS = {"GCG": 5, "AutoDAN": 10}
METRICS = (
    "selected_score_change_mean_abs",
    "selected_score_change_rms",
    "selected_score_change_l2",
    "selected_score_change_max_abs",
    "max_absolute_logit_difference",
    "logit_l2",
    "kl_baseline_to_intervention",
    "baseline_margin",
    "intervention_residual_margin",
    "boundary_progress",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu-preflight", "run"), default="cpu-preflight")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir")
    parser.add_argument("--sandbox-exempt-gpu", action="store_true")
    return parser.parse_args(argv)


def evenly_spaced_by_length(rows: list[dict], count: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (int(row["prompt_length"]), int(row["dataset_index"])))
    if count < 2 or len(ordered) < count:
        raise ValueError("invalid deterministic sample size")
    positions = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    if len(set(positions)) != count:
        raise AssertionError("deterministic sample positions are not unique")
    return [ordered[position] for position in positions]


def verify_run(label: str, directory: Path) -> list[dict]:
    manifest_path = directory / "artifact_manifest.json"
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256[label]:
        raise ValueError(f"{label} source artifact manifest SHA-256 mismatch")
    manifest = read_json(manifest_path)
    for item in manifest["files"]:
        if sha256_file(directory / item["path"]) != item["sha256"]:
            raise ValueError(f"{label} source artifact hash mismatch")
    run_manifest = read_json(directory / "run_manifest.json")
    conditions = run_manifest["scientific_conditions"]
    if (
        conditions["selection"] != fixed_selection()
        or conditions["alpha"] != ALPHA
        or conditions["layer0_included"] is not LAYER0_INCLUDED
        or conditions["cap_rate"] != 0.005
        or conditions["timing"] != "first_prefill_only"
    ):
        raise ValueError(f"{label} source scientific conditions mismatch")
    return read_jsonl(directory / "prompt_results.jsonl")


def preflight(args) -> dict:
    frozen = {}
    for dataset, label in (("independent100", "GCG"), ("autodan130", "AutoDAN")):
        evaluation_args = parse_evaluation_args(["--dataset", dataset])
        provenance, baseline = validate_frozen_inputs(evaluation_args)
        results = verify_run(label, GCG_RUN if label == "GCG" else AUTODAN_RUN)
        if len(results) != len(baseline):
            raise ValueError(f"{label} result/baseline count mismatch")
        eligible = (
            [row for row in results if not row["response_equal"]]
            if label == "GCG"
            else results
        )
        selected = evenly_spaced_by_length(eligible, SAMPLE_COUNTS[label])
        frozen[label] = {
            "args": evaluation_args,
            "provenance": provenance,
            "baseline": baseline,
            "selected_indices": [int(row["dataset_index"]) for row in selected],
            "selection_basis": (
                "response-changed positive controls, evenly spaced by prompt-length rank"
                if label == "GCG"
                else "all prompts, evenly spaced by prompt-length rank"
            ),
            "selected_prompt_lengths": [int(row["prompt_length"]) for row in selected],
            "results_by_index": {int(row["dataset_index"]): row for row in results},
        }
    return frozen


def formatted_records(label: str, frozen: dict, tokenizer) -> dict[int, dict]:
    args = frozen["args"]
    if label == "AutoDAN":
        raw, _ = load_autodan_transfer_records(Path(args.autodan_input))
        records = format_autodan_records(raw, tokenizer, prompt_format=OFFICIAL_FORMAT)
    else:
        records = load_independent_evaluation_records(
            config_path=args.independent_config,
            input_manifest_path=args.independent_input_manifest,
            attack_run_dir=args.independent_attack_run_dir,
            adapter_path=args.independent_adapter,
        )
        records = format_records(records, tokenizer)
        validate_formatted_evaluation_identity(records)
    return {int(row["dataset_index"]): row for row in records}


def diagnose_prompt(label, record, baseline, saved, *, model, reference) -> dict:
    ids = record["input_ids"]
    baseline_logits = first_forward_logits(model, ids)
    if int(torch.argmax(baseline_logits)) != int(baseline["response_token_ids"][0]):
        raise RuntimeError("baseline first-token reproduction mismatch")
    with patch_llama_abs_z_collection(model, layer_count=NUM_LAYERS, head_count=NUM_HEADS) as collector:
        collector.set_prompt_context(
            prompt_length=record["prompt_length"],
            excluded_selection_position=0,
            mu=reference.mu,
            sigma=reference.sigma,
        )
        with torch.inference_mode():
            model.model(input_ids=ids.to(model.device), use_cache=False, return_dict=True)
        abs_z, collection = collector.result()
    plan = build_stage2_selection_plan(
        abs_z,
        selection=fixed_selection(),
        layer_count=NUM_LAYERS,
        head_count=NUM_HEADS,
        elements_per_head=collection["strict_lower_elements_per_head"],
        layer0_included=LAYER0_INCLUDED,
    )
    with patch_llama_stage2_dynamic_asymmetry_attention(
        model,
        reference=reference,
        selection=fixed_selection(),
        alpha=ALPHA,
        layer0_included=LAYER0_INCLUDED,
        capture_score_change_metrics=True,
    ) as controller:
        controller.set_prompt_plan(prompt_length=record["prompt_length"], plan=plan)
        intervention_logits = first_forward_logits(model, ids)
        controller.assert_complete_prefill()
        runtime = controller.summary()
    metrics = decision_metrics(baseline_logits, intervention_logits)
    if (
        metrics["intervention_top1_token_id"] != int(saved["response_token_ids"][0])
        or metrics["argmax_changed"] is not (not saved["response_equal"])
        or runtime["later_forward_intervention_count"] != 0
        or runtime["selected_element_intervention_count"] != plan["selected_count"]
    ):
        raise RuntimeError("saved-result or runtime invariant mismatch")
    score = runtime["selected_raw_score_change"]
    return {
        "dataset": label,
        "dataset_index": record["dataset_index"],
        "prompt_id": record["prompt_id"],
        "prompt_length": record["prompt_length"],
        "selected_count": plan["selected_count"],
        "selected_fraction": plan["selected_fraction"],
        "baseline_success": baseline["success"],
        "response_changed": not saved["response_equal"],
        "selected_score_change_mean_abs": score["mean_absolute"],
        "selected_score_change_rms": score["rms"],
        "selected_score_change_l2": score["l2"],
        "selected_score_change_max_abs": score["max_absolute"],
        **metrics,
        "runtime_invariants_passed": True,
    }


def run(args, frozen) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA is required")
    output_dir = ensure_output_directory_absent(Path(args.output_dir).resolve())
    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(Path(args.model).resolve())
    _, reference = validate_and_load_reference(args.reference)
    rows = []
    for label in ("AutoDAN", "GCG"):
        records = formatted_records(label, frozen[label], tokenizer)
        baseline = {int(row["dataset_index"]): row for row in frozen[label]["baseline"]}
        for position, index in enumerate(frozen[label]["selected_indices"], 1):
            row = diagnose_prompt(
                label,
                records[index],
                baseline[index],
                frozen[label]["results_by_index"][index],
                model=model,
                reference=reference,
            )
            rows.append(row)
            print(f"[{label} diagnostic {position}/{SAMPLE_COUNTS[label]}] index={index}", flush=True)
    summary = {}
    for label in ("AutoDAN", "GCG"):
        subset = [row for row in rows if row["dataset"] == label]
        summary[label] = {
            "prompt_count": len(subset),
            "selection_basis": frozen[label]["selection_basis"],
            "dataset_indices": frozen[label]["selected_indices"],
            "prompt_lengths": frozen[label]["selected_prompt_lengths"],
            "argmax_changed_count": sum(row["argmax_changed"] for row in subset),
            "metrics": {
                metric: distribution([row[metric] for row in subset])
                for metric in METRICS
            },
            "numerical_or_runtime_invariant_failure_count": 0,
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
        "scientific_conditions_unchanged": True,
        "raw_per_element_values_saved": False,
        "source_artifact_manifests": EXPECTED_MANIFEST_SHA256,
        "source_code_sha256": {
            path: sha256_file(REPO_ROOT / path)
            for path in (
                "scripts/diagnose_dynamic_asym_score_logit_propagation.py",
                "scripts/diagnose_gcg_autodan_logit_propagation.py",
                "scripts/evaluate_gcg_dynamic_asym_independent100.py",
                "utils/gcg_dynamic_asym_intervention.py",
                "utils/gcg_dynamic_asym_stage2.py",
            )
        },
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
        "summary_sha256": sha256_file(summary_path),
        "artifact_manifest_sha256": sha256_file(artifact_path),
        "summary": summary,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "run" and not args.output_dir:
        raise ValueError("--output-dir is required for run")
    if args.stage == "cpu-preflight" and (args.output_dir or args.sandbox_exempt_gpu):
        raise ValueError("cpu-preflight cannot write output or claim GPU")
    frozen = preflight(args)
    payload = {
        "status": "cpu_preflight_passed",
        "fixed_selection": fixed_selection(),
        "alpha": ALPHA,
        "layer0_included": LAYER0_INCLUDED,
        "samples": {
            label: {
                "count": SAMPLE_COUNTS[label],
                "indices": frozen[label]["selected_indices"],
                "lengths": frozen[label]["selected_prompt_lengths"],
                "basis": frozen[label]["selection_basis"],
            }
            for label in ("AutoDAN", "GCG")
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.stage == "run":
        print(json.dumps(run(args, frozen), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
