#!/usr/bin/env python3
"""Evaluate the frozen dynamic Asym finalist on frozen GCG independent100."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_gcg_dynamic_asym_stage1 import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_REFERENCE,
    EXPECTED_REFERENCE_SHA256,
    validate_and_load_reference,
)
from scripts.evaluate_gcg_first_token_reference_ablation import (  # noqa: E402
    format_records,
    generate_with_first_logits,
    load_model_and_tokenizer,
)
from scripts.evaluate_gcg_position_element_topk_intervention import (  # noqa: E402
    AUTODAN_OFFICIAL_REPOSITORY_COMMIT,
    DEFAULT_AUTODAN_INPUT,
    DEFAULT_INDEPENDENT_ADAPTER,
    DEFAULT_INDEPENDENT_CONFIG,
    DEFAULT_INDEPENDENT_INPUT_MANIFEST,
    format_autodan_records,
    load_autodan_transfer_records,
    serialized_input_ids,
)
from scripts.evaluate_gcg_pure_symmetric_autodan_transfer import (  # noqa: E402
    BASELINE_SUCCESS_COUNT as AUTODAN_BASELINE_SUCCESS_COUNT,
    DEFAULT_OFFICIAL_BASELINE_DIR,
    EXPECTED_OFFICIAL_BASELINE_HASHES,
    MODEL_REPOSITORY_HEAD,
    OFFICIAL_FORMAT,
    normalize_canonical_baseline_rows,
    verify_artifact_manifest,
    verify_fixed_hashes,
)
from scripts.evaluate_gcg_symmetric_position_element_topk_intervention import (  # noqa: E402
    DEFAULT_ATTACK_RUN,
    DEFAULT_REUSED_BASELINE,
    EXPECTED_REUSED_BASELINE_SHA256,
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
from utils.gcg_dynamic_asym_intervention import load_dynamic_reference  # noqa: E402
from utils.gcg_dynamic_asym_stage2 import (  # noqa: E402
    ABSOLUTE_INTERVENTION_TARGETS,
    build_stage2_selection_plan,
    patch_llama_stage2_dynamic_asymmetry_attention,
    restrict_absolute_plan_to_intervention_sign,
    stage2_selection_conditions,
)
from utils.gcg_first_token_reference_ablation import (  # noqa: E402
    validate_first_token_response_invariant,
)
from utils.gcg_reference_intervention_diagnostics import compare_token_sequences  # noqa: E402
from utils.gcg_wasserstein_reference import NUM_HEADS, NUM_LAYERS  # noqa: E402
from utils.gcg_wasserstein_reference_evaluation import classify_transition  # noqa: E402
from utils.harmbench_gcg_independent_test import (  # noqa: E402
    load_independent_evaluation_records,
    validate_fixed_config,
    validate_formatted_evaluation_identity,
    validate_harmbench_revision,
    validate_input_manifest,
    validate_model_revision,
    verify_all_configured_sources,
)
from scripts.run_autodan_position_element_selection_ablation import (  # noqa: E402
    EXPECTED_SELECTION_INDICES as AUTODAN_SELECTION_INDICES,
)


SEED = 42
DEFAULT_PROPORTIONAL_RATE = 0.001
ALLOWED_PROPORTIONAL_RATES = (0.001, 0.005)
ALPHA = 1.0
LAYER0_INCLUDED = False
EXPECTED_BASELINE_SUCCESS_COUNT = 53
AUTODAN_SELECTION_MANIFEST = (
    "data/outputs/gcg_position_element_topk/autodan_selection20/"
    "official_fastchat_34062e9_20260914T163205Z/selection_manifest.json"
)
EXPECTED_AUTODAN_SELECTION_MANIFEST_SHA256 = (
    "494c6beba4f6367ff353c0716ae986734b7bcbdeb823a342fc78ba8fb9741318"
)
SOURCE_PATHS = (
    "scripts/evaluate_gcg_dynamic_asym_independent100.py",
    "scripts/evaluate_gcg_pure_symmetric_autodan_transfer.py",
    "scripts/evaluate_gcg_position_element_topk_intervention.py",
    "utils/gcg_dynamic_asym_stage2.py",
    "utils/gcg_dynamic_asym_stage1.py",
    "utils/gcg_dynamic_asym_intervention.py",
)


def fixed_selection(rate: float = DEFAULT_PROPORTIONAL_RATE) -> dict:
    if rate not in ALLOWED_PROPORTIONAL_RATES:
        raise ValueError("proportional rate must be 0.001 or 0.005")
    name = f"proportional_{rate * 100:.1f}pct".replace(".", "p")
    expected = {
        "name": name,
        "family": "proportional_top_n",
        "top_n": None,
        "rate": rate,
        "rate_numerator": int(rate * 1000),
        "tau": None,
    }
    if rate == DEFAULT_PROPORTIONAL_RATE:
        matches = [
            row for row in stage2_selection_conditions() if row["name"] == name
        ]
        if len(matches) != 1 or matches[0] != expected:
            raise RuntimeError("frozen independent100 selection condition drifted")
    return expected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu-preflight", "run"), default="cpu-preflight")
    parser.add_argument("--dataset", choices=("independent100", "autodan130"), default="independent100")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--proportional-rate",
        type=float,
        choices=ALLOWED_PROPORTIONAL_RATES,
        default=DEFAULT_PROPORTIONAL_RATE,
    )
    parser.add_argument(
        "--component",
        choices=("antisymmetric", "symmetric"),
        default="antisymmetric",
    )
    parser.add_argument(
        "--selection-direction",
        choices=("absolute", "positive", "negative"),
        default="absolute",
    )
    parser.add_argument(
        "--absolute-intervention-target",
        choices=ABSOLUTE_INTERVENTION_TARGETS,
        default="all",
    )
    parser.add_argument(
        "--reference-source",
        choices=("gcg-selection20-original", "autodan-selection20-official"),
        default="gcg-selection20-original",
    )
    parser.add_argument("--independent-config", default=DEFAULT_INDEPENDENT_CONFIG)
    parser.add_argument("--independent-input-manifest", default=DEFAULT_INDEPENDENT_INPUT_MANIFEST)
    parser.add_argument("--independent-attack-run-dir", default=DEFAULT_ATTACK_RUN)
    parser.add_argument("--independent-adapter", default=DEFAULT_INDEPENDENT_ADAPTER)
    parser.add_argument("--reused-baseline", default=DEFAULT_REUSED_BASELINE)
    parser.add_argument("--autodan-input", default=DEFAULT_AUTODAN_INPUT)
    parser.add_argument("--official-baseline-dir", default=DEFAULT_OFFICIAL_BASELINE_DIR)
    parser.add_argument(
        "--autodan-split", choices=("full130", "evaluation110"), default="full130"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--sandbox-exempt-gpu", action="store_true")
    return parser.parse_args(argv)


def validate_request(args) -> Path | None:
    if (
        args.absolute_intervention_target != "all"
        and args.selection_direction != "absolute"
    ):
        raise ValueError(
            "sign-restricted absolute intervention requires --selection-direction absolute"
        )
    if (
        args.absolute_intervention_target != "all"
        and args.component != "antisymmetric"
    ):
        raise ValueError("sign-restricted absolute intervention is fixed to antisymmetric")
    if args.stage == "cpu-preflight":
        if args.output_dir is not None or args.sandbox_exempt_gpu:
            raise ValueError("cpu-preflight cannot write output or claim GPU execution")
        return None
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    return ensure_output_directory_absent(Path(args.output_dir).resolve())


def load_requested_reference(args):
    if args.reference_source == "gcg-selection20-original":
        if args.component == "antisymmetric":
            reference_path, reference = validate_and_load_reference(args.reference)
        else:
            reference_path = Path(args.reference).resolve()
            reference = load_dynamic_reference(reference_path)
        expected = (
            "A_asym=(S-S^T)/2"
            if args.component == "antisymmetric"
            else "A_sym=(S+S^T)/2"
        )
        if reference.metadata.get("component") != expected:
            raise ValueError("GCG dynamic reference component mismatch")
        if (
            reference.metadata.get("reference_condition")
            != "selection20_original_only"
            or reference.metadata.get("jailbreak_reference_data_used") is not False
        ):
            raise ValueError("GCG dynamic reference provenance mismatch")
        return reference_path, reference
    reference_path = Path(args.reference).resolve()
    reference = load_dynamic_reference(reference_path)
    metadata = reference.metadata
    expected_component = (
        "A_asym=(S-S^T)/2"
        if args.component == "antisymmetric"
        else "A_sym=(S+S^T)/2"
    )
    if (
        metadata.get("reference_condition")
        != "autodan_selection20_official_jailbreak_only"
        or metadata.get("selection_dataset_indices") != AUTODAN_SELECTION_INDICES
        or metadata.get("selection_manifest_sha256")
        != EXPECTED_AUTODAN_SELECTION_MANIFEST_SHA256
        or metadata.get("prompt_format") != OFFICIAL_FORMAT
        or metadata.get("component") != expected_component
    ):
        raise ValueError("AutoDAN dynamic reference provenance mismatch")
    return reference_path, reference


def autodan_evaluation_indices() -> list[int]:
    path = Path(AUTODAN_SELECTION_MANIFEST).resolve()
    if sha256_file(path) != EXPECTED_AUTODAN_SELECTION_MANIFEST_SHA256:
        raise ValueError("AutoDAN Selection20 manifest SHA-256 mismatch")
    manifest = read_json(path)
    if (
        manifest.get("selection_indices") != AUTODAN_SELECTION_INDICES
        or manifest.get("evaluation_count") != 110
        or manifest.get("format") != OFFICIAL_FORMAT
    ):
        raise ValueError("AutoDAN evaluation110 split mismatch")
    return [int(value) for value in manifest["evaluation_indices"]]


def restrict_autodan_baseline(args, baseline: list[dict]) -> list[dict]:
    if args.dataset != "autodan130" or args.autodan_split == "full130":
        return baseline
    indices = set(autodan_evaluation_indices())
    filtered = [row for row in baseline if int(row["dataset_index"]) in indices]
    if len(filtered) != 110 or sum(bool(row["success"]) for row in filtered) != 29:
        raise ValueError("AutoDAN evaluation110 baseline mismatch")
    return filtered


def validate_frozen_inputs(args) -> tuple[dict, list[dict]]:
    reference_path, reference = load_requested_reference(args)
    reference_sha256 = sha256_file(reference_path)
    if (
        args.reference_source == "gcg-selection20-original"
        and args.component == "antisymmetric"
        and reference_sha256 != EXPECTED_REFERENCE_SHA256
    ):
        raise ValueError("GCG reference SHA-256 mismatch")
    if args.dataset == "autodan130":
        raw_records, raw_provenance = load_autodan_transfer_records(
            Path(args.autodan_input)
        )
        baseline_dir = Path(args.official_baseline_dir).resolve()
        baseline_hashes = verify_fixed_hashes(
            baseline_dir, EXPECTED_OFFICIAL_BASELINE_HASHES
        )
        verify_artifact_manifest(baseline_dir)
        manifest = read_json(baseline_dir / "run_manifest.json")
        summary = read_json(baseline_dir / "summary.json")
        if (
            manifest.get("status") != "complete"
            or manifest.get("autodan_repository_commit")
            != AUTODAN_OFFICIAL_REPOSITORY_COMMIT
            or manifest.get("model_repository_head") != MODEL_REPOSITORY_HEAD
            or manifest.get("index_10_unreplaced_placeholder_retained") is not True
            or summary.get("condition_b_success_count")
            != AUTODAN_BASELINE_SUCCESS_COUNT
            or summary.get("generation_failure_count") != 0
        ):
            raise ValueError("official AutoDAN baseline conditions mismatch")
        baseline = normalize_canonical_baseline_rows(
            read_jsonl(baseline_dir / "paired_results.jsonl"), raw_records
        )
        return {
            "reference": str(reference_path),
            "reference_sha256": reference_sha256,
            "reference_source": args.reference_source,
            "reference_metadata": reference.metadata,
            "dataset": "autodan130_official_format",
            "raw_provenance": raw_provenance,
            "official_format": OFFICIAL_FORMAT,
            "official_baseline_dir": str(baseline_dir),
            "official_baseline_hashes": baseline_hashes,
        }, baseline

    config_path = Path(args.independent_config).resolve()
    manifest_path = Path(args.independent_input_manifest).resolve()
    attack_run = Path(args.independent_attack_run_dir).resolve()
    adapter_path = Path(args.independent_adapter).resolve()
    config = read_json(config_path)
    validate_fixed_config(config)
    source_hashes = verify_all_configured_sources(config)
    harmbench_revision = validate_harmbench_revision(config)
    model_revision = validate_model_revision(config)
    input_validation = validate_input_manifest(read_json(manifest_path), config)
    final_manifest = read_json(attack_run / "final_manifest.json")
    for key, expected in (
        ("status", "complete_validated"),
        ("complete_count", 100),
        ("missing_count", 0),
        ("duplicate_count", 0),
        ("processing_failure_history_count", 0),
    ):
        if final_manifest.get(key) != expected:
            raise ValueError(f"independent100 final manifest {key} mismatch")
    baseline_path = Path(args.reused_baseline).resolve()
    if sha256_file(baseline_path) != EXPECTED_REUSED_BASELINE_SHA256:
        raise ValueError("reused independent100 baseline SHA-256 mismatch")
    baseline = read_jsonl(baseline_path)
    indices = [int(row["dataset_index"]) for row in baseline]
    if len(baseline) != 100 or len(set(indices)) != 100 or set(indices) != set(range(220, 320)):
        raise ValueError("reused baseline is not the complete independent100 set")
    if sum(bool(row["success"]) for row in baseline) != EXPECTED_BASELINE_SUCCESS_COUNT:
        raise ValueError("reused baseline success count mismatch")
    provenance = {
        "reference": str(reference_path),
        "reference_sha256": reference_sha256,
        "reference_source": args.reference_source,
        "reference_metadata": reference.metadata,
        "independent_config": str(config_path),
        "independent_config_sha256": sha256_file(config_path),
        "independent_input_manifest": str(manifest_path),
        "independent_input_manifest_sha256": sha256_file(manifest_path),
        "independent_attack_run_dir": str(attack_run),
        "independent_attack_final_manifest_sha256": sha256_file(attack_run / "final_manifest.json"),
        "independent_adapter": str(adapter_path),
        "independent_adapter_sha256": sha256_file(adapter_path),
        "reused_baseline": str(baseline_path),
        "reused_baseline_sha256": EXPECTED_REUSED_BASELINE_SHA256,
        "input_count": len(input_validation),
        "source_hashes": source_hashes,
        "harmbench_revision": harmbench_revision,
        "model_revision": model_revision,
    }
    return provenance, baseline


def summarize(rows: list[dict], baseline: list[dict]) -> dict:
    counts = [int(row["selected_count"]) for row in rows]
    nominal_counts = [int(row["nominal_selected_count"]) for row in rows]
    fractions = [float(row["selected_fraction"]) for row in rows]
    selected_z_rows = [row["selected_z"] for row in rows if row["selected_z"]]
    selected_z_count = sum(int(row["count"]) for row in selected_z_rows)
    selected_z_summary = (
        {
            "count": selected_z_count,
            "min": min(float(row["min"]) for row in selected_z_rows),
            "mean": sum(float(row["mean"]) * int(row["count"]) for row in selected_z_rows)
            / selected_z_count,
            "max": max(float(row["max"]) for row in selected_z_rows),
        }
        if selected_z_count
        else None
    )
    absolute_counts = [
        int(row["absolute_top_n_count"])
        for row in rows
        if row.get("absolute_top_n_count") is not None
    ]
    absolute_positive_counts = [
        int(row["absolute_positive_selected_count"])
        for row in rows
        if row.get("absolute_positive_selected_count") is not None
    ]
    absolute_negative_counts = [
        int(row["absolute_negative_selected_count"])
        for row in rows
        if row.get("absolute_negative_selected_count") is not None
    ]
    return {
        "prompt_count": len(rows),
        "baseline_successful_count": sum(bool(row["success"]) for row in baseline),
        "baseline_asr": sum(bool(row["success"]) for row in baseline) / len(baseline),
        "intervention_successful_count": sum(bool(row["success"]) for row in rows),
        "intervention_asr": sum(bool(row["success"]) for row in rows) / len(rows),
        "improvement_count": sum(row["transition"] == "newly_defended" for row in rows),
        "worsening_count": sum(row["transition"] == "newly_worsened" for row in rows),
        "response_changed_count": sum(not row["response_equal"] for row in rows),
        "nominal_selected_count_mean": statistics.fmean(nominal_counts),
        "selected_count_mean": statistics.fmean(counts),
        "selected_fraction_mean": statistics.fmean(fractions),
        "sign_eligible_candidate_count_mean": statistics.fmean(
            int(row["sign_eligible_candidate_count"]) for row in rows
        ),
        "selection_shortfall_prompt_count": sum(
            int(row["selection_shortfall_count"]) > 0 for row in rows
        ),
        "selection_shortfall_count_total": sum(
            int(row["selection_shortfall_count"]) for row in rows
        ),
        "positive_selected_count": sum(
            int(row["positive_selected_count"]) for row in rows
        ),
        "negative_selected_count": sum(
            int(row["negative_selected_count"]) for row in rows
        ),
        "selected_z": selected_z_summary,
        "absolute_top_n_count_mean": (
            statistics.fmean(absolute_counts) if absolute_counts else None
        ),
        "absolute_positive_selected_count_mean": (
            statistics.fmean(absolute_positive_counts)
            if absolute_positive_counts
            else None
        ),
        "absolute_negative_selected_count_mean": (
            statistics.fmean(absolute_negative_counts)
            if absolute_negative_counts
            else None
        ),
        "cap_activated_prompt_count": sum(bool(row["cap_activated"]) for row in rows),
        "cap_violation_count": sum(row["selected_count"] > row["cap_count"] for row in rows),
        "generation_failure_count": 0,
        "nonfinite_count": 0,
        "decode_intervention_count": sum(
            int(row["diagnostics"]["later_forward_intervention_count"]) for row in rows
        ),
    }


def run(args, output_dir: Path, provenance: dict, baseline: list[dict]) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA is required; refusing CPU model execution")
    seed_everything(SEED)
    model, tokenizer = load_model_and_tokenizer(Path(args.model).resolve())
    _, reference = load_requested_reference(args)
    if args.dataset == "autodan130":
        raw_records, _ = load_autodan_transfer_records(Path(args.autodan_input))
        records = format_autodan_records(
            raw_records, tokenizer, prompt_format=OFFICIAL_FORMAT
        )
    else:
        records = load_independent_evaluation_records(
            config_path=args.independent_config,
            input_manifest_path=args.independent_input_manifest,
            attack_run_dir=args.independent_attack_run_dir,
            adapter_path=args.independent_adapter,
        )
        records = format_records(records, tokenizer)
        validate_formatted_evaluation_identity(records)
    baseline = restrict_autodan_baseline(args, baseline)
    if args.dataset == "autodan130" and args.autodan_split == "evaluation110":
        evaluation_indices = set(autodan_evaluation_indices())
        records = [
            row for row in records if int(row["dataset_index"]) in evaluation_indices
        ]
        if len(records) != 110:
            raise ValueError("AutoDAN evaluation110 formatted record count mismatch")
    baseline_by_index = {int(row["dataset_index"]): row for row in baseline}
    selection = fixed_selection(args.proportional_rate)
    rows = []
    for position, record in enumerate(records, 1):
        base = baseline_by_index[int(record["dataset_index"])]
        if (
            serialized_input_ids(record["input_ids"]) != base["input_ids"]
            or (
                args.dataset == "autodan130"
                and record["formatted_prompt"] != base["formatted_prompt"]
            )
        ):
            raise RuntimeError(f"{args.dataset} input differs from reused baseline")
        with patch_llama_abs_z_collection(
            model,
            layer_count=NUM_LAYERS,
            head_count=NUM_HEADS,
            component=args.component,
        ) as collector:
            collector.set_prompt_context(
                prompt_length=record["prompt_length"],
                excluded_selection_position=0,
                mu=reference.mu,
                sigma=reference.sigma,
            )
            with torch.inference_mode():
                model.model(
                    input_ids=record["input_ids"].to(model.device),
                    use_cache=False,
                    return_dict=True,
                )
            z_scores, collection = collector.result(signed=True)
            abs_z = z_scores.abs()
        plan = build_stage2_selection_plan(
            abs_z,
            selection=selection,
            layer_count=NUM_LAYERS,
            head_count=NUM_HEADS,
            elements_per_head=collection["strict_lower_elements_per_head"],
            layer0_included=LAYER0_INCLUDED,
            signed_z=z_scores,
            selection_direction=args.selection_direction,
        )
        if args.selection_direction == "absolute":
            plan = restrict_absolute_plan_to_intervention_sign(
                plan,
                z_scores,
                intervention_target_sign=args.absolute_intervention_target,
                layer_count=NUM_LAYERS,
                head_count=NUM_HEADS,
                elements_per_head=collection["strict_lower_elements_per_head"],
            )
        with patch_llama_stage2_dynamic_asymmetry_attention(
            model,
            reference=reference,
            selection=selection,
            alpha=ALPHA,
            layer0_included=LAYER0_INCLUDED,
            component=args.component,
        ) as controller:
            controller.set_prompt_plan(prompt_length=record["prompt_length"], plan=plan)
            generated = generate_with_first_logits(model, tokenizer, record["input_ids"])
            if generated["generation_failed"]:
                raise RuntimeError(f"generation failed at dataset_index={record['dataset_index']}")
            controller.assert_complete_prefill()
            diagnostics = controller.summary()
        comparison = compare_token_sequences(
            base["response_token_ids"], generated["response_token_ids"]
        )
        validate_first_token_response_invariant(
            response_equal=comparison["response_equal"],
            first_different_token_position=comparison["first_different_token_position"],
        )
        if (
            collection["all_finite"] is not True
            or diagnostics["all_scores_finite"] is not True
            or diagnostics["later_forward_intervention_count"] != 0
            or diagnostics["selected_element_intervention_count"] != plan["selected_count"]
            or diagnostics["max_unselected_element_change"] != 0.0
            or diagnostics["max_diagonal_change_float32"] != 0.0
            or plan["selected_count"] > plan["cap_count"]
        ):
            raise RuntimeError("runtime numerical, timing, or cap invariant failed")
        rows.append(
            {
                "dataset_index": record["dataset_index"],
                "prompt_id": record["prompt_id"],
                "prompt_length": record["prompt_length"],
                "baseline_response": base["response"],
                "baseline_success": bool(base["success"]),
                **{key: value for key, value in generated.items() if key != "first_logits"},
                **comparison,
                "transition": classify_transition(bool(base["success"]), bool(generated["success"])),
                "selection_direction": plan["selection_direction"],
                "intervention_target_sign": plan.get(
                    "intervention_target_sign", "all"
                ),
                "absolute_top_n_count": plan.get("absolute_top_n_count"),
                "absolute_positive_selected_count": plan.get(
                    "absolute_positive_selected_count"
                ),
                "absolute_negative_selected_count": plan.get(
                    "absolute_negative_selected_count"
                ),
                "requested_count": plan["requested_count"],
                "nominal_selected_count": plan["requested_or_threshold_eligible_count"],
                "sign_eligible_candidate_count": plan[
                    "sign_eligible_candidate_count"
                ],
                "selection_shortfall_count": plan["selection_shortfall_count"],
                "selected_count": plan["selected_count"],
                "selected_fraction": plan["selected_fraction"],
                "selected_z": plan["selected_z"],
                "positive_selected_count": plan["positive_selected_count"],
                "negative_selected_count": plan["negative_selected_count"],
                "cap_count": plan["cap_count"],
                "cap_activated": plan["cap_activated"],
                "diagnostics": diagnostics,
            }
        )
        print(f"[{args.dataset} {position}/{len(records)}] index={record['dataset_index']} success={generated['success']} selected={plan['selected_count']}", flush=True)

    summary = {
        "schema_version": 1,
        "experiment_role": f"fixed_dynamic_{args.component}_{args.dataset}_evaluation",
        "component": args.component,
        "selection_direction": args.selection_direction,
        "intervention_target_sign": args.absolute_intervention_target,
        "selection": selection,
        "alpha": ALPHA,
        "layer0_included": LAYER0_INCLUDED,
        "cap_rate": 0.005,
        **summarize(rows, baseline),
    }
    ensure_output_directory_absent(output_dir)
    output_dir.mkdir(parents=True)
    results_path = output_dir / "prompt_results.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(results_path, rows)
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
        "scientific_conditions": {
            "dataset": args.dataset,
            "component": args.component,
            "selection_direction": args.selection_direction,
            "intervention_target_sign": args.absolute_intervention_target,
            "autodan_split": args.autodan_split if args.dataset == "autodan130" else None,
            "reference_source": args.reference_source,
            "selection": selection,
            "alpha": ALPHA,
            "layer0_included": LAYER0_INCLUDED,
            "position_scope": "full_prompt_strict_lower_i_gt_j",
            "intervention": f"pure_{args.component}_pairwise",
            "timing": "first_prefill_only",
            "cap_rate": 0.005,
            "development68_used": False,
            "independent100_used": args.dataset == "independent100",
            "autodan_used": args.dataset == "autodan130",
        },
        "provenance": provenance,
        "source_code_sha256": {path: sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS},
        "gpu": torch.cuda.get_device_name(0),
        "files": {
            results_path.name: sha256_file(results_path),
            summary_path.name: sha256_file(summary_path),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, run_manifest)
    artifact_manifest = {
        "schema_version": 1,
        "files": [
            {"path": path.name, "sha256": sha256_file(path)}
            for path in (results_path, summary_path, manifest_path)
        ],
    }
    artifact_path = output_dir / "artifact_manifest.json"
    write_json(artifact_path, artifact_manifest)
    return {
        "output_dir": str(output_dir),
        "summary_sha256": sha256_file(summary_path),
        "artifact_manifest_sha256": sha256_file(artifact_path),
        "summary": summary,
    }


def main(argv=None):
    args = parse_args(argv)
    output_dir = validate_request(args)
    provenance, baseline = validate_frozen_inputs(args)
    baseline = restrict_autodan_baseline(args, baseline)
    payload = {
        "status": "cpu_preflight_passed",
        "selection": fixed_selection(args.proportional_rate),
        "component": args.component,
        "selection_direction": args.selection_direction,
        "intervention_target_sign": args.absolute_intervention_target,
        "alpha": ALPHA,
        "layer0_included": LAYER0_INCLUDED,
        "cap_rate": 0.005,
        "dataset": args.dataset,
        "autodan_split": args.autodan_split if args.dataset == "autodan130" else None,
        "reference_source": args.reference_source,
        "prompt_count": len(baseline),
        "baseline_successful_count": sum(bool(row["success"]) for row in baseline),
        "reference_sha256": provenance["reference_sha256"],
        "development68_used": False,
        "independent100_used": args.dataset == "independent100",
        "autodan_used": args.dataset == "autodan130",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.stage == "cpu-preflight":
        return
    print(json.dumps(run(args, output_dir, provenance, baseline), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
