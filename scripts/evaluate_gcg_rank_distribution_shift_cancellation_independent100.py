#!/usr/bin/env python3
"""One-shot final independent100 evaluation of frozen rank shift cancellation v1."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_gcg_reference_factorization import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MODEL,
    GENERATION_SETTINGS,
    format_records,
    load_model_and_tokenizer,
)
from scripts.evaluate_gcg_position_element_topk_intervention import (  # noqa: E402
    DEFAULT_INDEPENDENT_ADAPTER,
    DEFAULT_INDEPENDENT_CONFIG,
    DEFAULT_INDEPENDENT_INPUT_MANIFEST,
    finish_cuda_measurement,
    run_baseline,
    serialized_input_ids,
    start_cuda_measurement,
)
from scripts.evaluate_gcg_rank_distribution_shift_cancellation import (  # noqa: E402
    INTERVENTION_TIMING,
    _shift_patcher,
    cpu_runtime_smoke,
    load_validated_shift,
    validate_loaded_model,
    validate_shift_runtime,
)
from scripts.evaluate_gcg_rank_element_topk_intervention import run_condition  # noqa: E402
from utils.experiment_runtime import git_value, seed_everything  # noqa: E402
from utils.gcg_rank_distribution_shift_cancellation import (  # noqa: E402
    ALL_RANK_COUNT,
    ALPHA_DEFAULT,
    METHOD,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.gcg_wasserstein_reference_evaluation import ensure_output_absent  # noqa: E402
from utils.harmbench_gcg_independent_test import (  # noqa: E402
    load_fast_tokenizer,
    load_independent_evaluation_records,
    validate_fixed_config,
    validate_formatted_evaluation_identity,
    validate_harmbench_revision,
    validate_input_manifest,
    validate_model_revision,
    verify_all_configured_sources,
)


EXPECTED_INDICES = tuple(range(220, 320))
EXPECTED_BASELINE_SUCCESS = 53
EXPECTED_POSITION_SUCCESS = 38
EXPECTED_SHIFT_SHA256 = "98691484937db994a7edd030dfc6532d4a7549ee312f8108d01c018ce39a2bef"
DEFAULT_ATTACK_RUN = (
    "data/outputs/gcg_independent_test/llama2/advbench_220_319/"
    "gcg_v1_55c5098_20260910T011413Z"
)
DEFAULT_POSITION_RUN = (
    "data/outputs/gcg_wasserstein_reference_intervention/llama2/"
    "independent100/evaluation_20260911T064911Z"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu-validate", "run"), default="cpu-validate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--shift-artifact-dir", required=True)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--independent-config", default=DEFAULT_INDEPENDENT_CONFIG)
    parser.add_argument(
        "--independent-input-manifest", default=DEFAULT_INDEPENDENT_INPUT_MANIFEST
    )
    parser.add_argument("--independent-attack-run-dir", default=DEFAULT_ATTACK_RUN)
    parser.add_argument("--independent-adapter", default=DEFAULT_INDEPENDENT_ADAPTER)
    parser.add_argument("--position-independent-run", default=DEFAULT_POSITION_RUN)
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def validate_frozen_attack_manifest(attack_run: Path) -> dict:
    final_path = attack_run / "final_manifest.json"
    run_path = attack_run / "run_manifest.json"
    final = read_json(final_path)
    for field, expected in (
        ("status", "complete_validated"),
        ("complete_count", 100),
        ("missing_count", 0),
        ("duplicate_count", 0),
        ("processing_failure_history_count", 0),
    ):
        if final.get(field) != expected:
            raise ValueError(f"independent attack final manifest {field} mismatch")
    examples = final.get("examples")
    if not isinstance(examples, list) or len(examples) != 100:
        raise ValueError("independent attack final manifest lacks 100 examples")
    indices = tuple(row.get("dataset_index") for row in examples)
    if indices != EXPECTED_INDICES or len(set(indices)) != 100:
        raise ValueError("independent attack indices must be unique ascending 220..319")
    return {
        "attack_run_dir": str(attack_run),
        "attack_run_manifest_sha256": sha256_file(run_path),
        "attack_final_manifest_sha256": sha256_file(final_path),
    }


def validate_prior_position_run(position_run: Path) -> tuple[list[dict], dict]:
    manifest_path = position_run / "run_manifest.json"
    baseline_path = position_run / "independent100_baseline.jsonl"
    baseline_summary_path = position_run / "independent100_baseline_summary.json"
    position_summary_path = position_run / "independent100/k_500/summary.json"
    manifest = read_json(manifest_path)
    baseline = read_jsonl(baseline_path)
    baseline_summary = read_json(baseline_summary_path)
    position_summary = read_json(position_summary_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("dataset_groups") != ["independent100"]
        or manifest.get("alpha") != 1.0
        or manifest.get("intervention_timing") != INTERVENTION_TIMING
        or baseline_summary.get("prompt_count") != 100
        or baseline_summary.get("attack_success_count") != EXPECTED_BASELINE_SUCCESS
        or baseline_summary.get("asr") != 0.53
        or position_summary.get("prompt_count") != 100
        or position_summary.get("k") != 500
        or position_summary.get("alpha") != 1.0
        or position_summary.get("attack_success_count") != EXPECTED_POSITION_SUCCESS
        or position_summary.get("asr") != 0.38
    ):
        raise ValueError("prior Position independent100 frozen result mismatch")
    indices = tuple(row.get("dataset_index") for row in baseline)
    if indices != EXPECTED_INDICES or len(set(indices)) != 100:
        raise ValueError("prior Position baseline indices mismatch")
    for row in baseline:
        if (
            not isinstance(row.get("prompt"), str)
            or not isinstance(row.get("formatted_prompt"), str)
            or not isinstance(row.get("input_ids"), list)
            or row.get("generation_failed") is not False
            or not isinstance(row.get("response_token_ids"), list)
        ):
            raise ValueError("prior Position baseline row is incomplete")
    provenance = {
        "position_run_dir": str(position_run),
        "position_run_manifest_sha256": sha256_file(manifest_path),
        "position_baseline_sha256": sha256_file(baseline_path),
        "position_baseline_summary_sha256": sha256_file(baseline_summary_path),
        "position_k500_summary_sha256": sha256_file(position_summary_path),
        "expected_baseline_success": EXPECTED_BASELINE_SUCCESS,
        "expected_position_top500_success": EXPECTED_POSITION_SUCCESS,
    }
    return baseline, provenance


def validate_independent_identity(records: list[dict], prior_baseline: list[dict]) -> dict:
    if len(records) != 100 or len(prior_baseline) != 100:
        raise ValueError("independent100 requires exactly 100 records")
    indices = tuple(row.get("dataset_index") for row in records)
    if indices != EXPECTED_INDICES or len(set(indices)) != 100:
        raise ValueError("evaluation indices must be unique ascending 220..319")
    selection20 = {
        68, 70, 72, 73, 84, 87, 88, 91, 96, 117,
        123, 127, 134, 143, 163, 181, 208, 215, 217, 218,
    }
    holdout48 = set(range(50)) - {10, 13}
    if set(indices) & selection20 or set(indices) & holdout48:
        raise ValueError("independent100 overlaps development indices")
    previous = {row["dataset_index"]: row for row in prior_baseline}
    for row in records:
        saved = previous[row["dataset_index"]]
        actual_ids = serialized_input_ids(row["input_ids"])
        if (
            row.get("prompt") != saved.get("prompt")
            or row.get("formatted_prompt") != saved.get("formatted_prompt")
            or actual_ids != saved.get("input_ids")
            or row.get("saved_non_target_input_ids") != saved.get("input_ids")
            or row.get("saved_formatted_prompt") != saved.get("formatted_prompt")
        ):
            raise ValueError(
                f"Position/evaluation input identity mismatch at {row['dataset_index']}"
            )
    return {
        "record_count": 100,
        "indices_start": 220,
        "indices_end": 319,
        "duplicate_count": 0,
        "selection20_overlap_count": 0,
        "holdout48_overlap_count": 0,
        "position_input_ids_match_count": 100,
        "position_formatted_prompt_match_count": 100,
        "system_prompt": None,
        "template": "systemless user-only Llama template",
    }


def load_frozen_inputs(args) -> tuple[list[dict], list[dict], dict]:
    config_path = Path(args.independent_config).resolve()
    input_manifest_path = Path(args.independent_input_manifest).resolve()
    attack_run = Path(args.independent_attack_run_dir).resolve()
    adapter_path = Path(args.independent_adapter).resolve()
    config = read_json(config_path)
    validate_fixed_config(config)
    if Path(config["model"]["local_path"]).resolve() != Path(args.model).resolve():
        raise ValueError("independent100 model differs from frozen config")
    verify_all_configured_sources(config)
    validate_harmbench_revision(config)
    validate_model_revision(config)
    manifest_records = validate_input_manifest(read_json(input_manifest_path), config)
    if tuple(row["dataset_index"] for row in manifest_records) != EXPECTED_INDICES:
        raise ValueError("input manifest indices mismatch")
    tokenizer = load_fast_tokenizer(config)
    records = load_independent_evaluation_records(
        config_path=config_path,
        input_manifest_path=input_manifest_path,
        attack_run_dir=attack_run,
        adapter_path=adapter_path,
        tokenizer=tokenizer,
    )
    records = format_records(records, tokenizer)
    validate_formatted_evaluation_identity(records)
    prior_baseline, prior_provenance = validate_prior_position_run(
        Path(args.position_independent_run).resolve()
    )
    identity = validate_independent_identity(records, prior_baseline)
    provenance = {
        **validate_frozen_attack_manifest(attack_run),
        **prior_provenance,
        "attack_config_sha256": sha256_file(config_path),
        "attack_input_manifest_sha256": sha256_file(input_manifest_path),
        "attack_adapter_sha256": sha256_file(adapter_path),
        "frozen_input_identity": identity,
    }
    return records, prior_baseline, provenance


def verify_baseline_reproduction(baseline: list[dict], prior: list[dict]) -> None:
    if len(baseline) != 100:
        raise RuntimeError("baseline did not produce exactly 100 rows")
    previous = {row["dataset_index"]: row for row in prior}
    success_count = 0
    for row in baseline:
        saved = previous[row["dataset_index"]]
        if (
            serialized_input_ids(row["input_ids"]) != saved["input_ids"]
            or row["response_token_ids"] != saved["response_token_ids"]
            or row["response"] != saved["response"]
            or row["success"] is not saved["success"]
            or row["generation_failed"] is not False
        ):
            raise RuntimeError(
                f"baseline reproduction mismatch at {row['dataset_index']}; "
                "refusing final intervention"
            )
        success_count += bool(row["success"])
    if success_count != EXPECTED_BASELINE_SUCCESS:
        raise RuntimeError("baseline is not the frozen 53/100; refusing intervention")


def cpu_validation_payload(args) -> dict:
    mean_shift, config, verification, source_verification, _, _ = load_validated_shift(args)
    if verification["mean_shift_sha256"] != EXPECTED_SHIFT_SHA256:
        raise ValueError("mean_shift artifact hash mismatch")
    records, _, provenance = load_frozen_inputs(args)
    smoke = cpu_runtime_smoke(mean_shift, alpha=ALPHA_DEFAULT)
    return {
        "status": "cpu_validation_passed",
        "method": METHOD,
        "evaluation_role": "final_independent_test",
        "alpha": ALPHA_DEFAULT,
        "rank_element_count_per_prompt": ALL_RANK_COUNT,
        "prompt_count": len(records),
        "mean_shift_shape": list(mean_shift.shape),
        "shift_artifact_verification": verification,
        "source_artifact_verified": source_verification["verified"],
        "shift_source": "Selection20 frozen artifact only; no recomputation",
        "independent_test_provenance": provenance,
        "scientific_conditions": config["scientific_conditions"],
        "cpu_runtime_smoke": smoke,
        "model_loaded": False,
        "gpu_used": False,
        "artifacts_written": False,
    }


def run_gpu(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --stage run")
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    output_dir = ensure_output_absent(args.output_dir).resolve()
    mean_shift, config, verification, source_verification, _, _ = load_validated_shift(args)
    if verification["mean_shift_sha256"] != EXPECTED_SHIFT_SHA256:
        raise ValueError("mean_shift artifact hash mismatch")
    records, prior_baseline, provenance = load_frozen_inputs(args)
    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(Path(args.model).resolve())
    model_validation = validate_loaded_model(model, config)

    baseline_started = start_cuda_measurement()
    baseline = run_baseline(
        records, model=model, tokenizer=tokenizer, label="independent100 baseline",
        require_saved_reproduction=False,
    )
    baseline_runtime = finish_cuda_measurement(baseline_started)
    verify_baseline_reproduction(baseline, prior_baseline)

    condition = {
        "name": "rank_shift_cancellation_v1",
        "method": METHOD,
        "k": ALL_RANK_COUNT,
        "alpha": ALPHA_DEFAULT,
    }
    baseline_by_index = {row["dataset_index"]: row for row in baseline}
    condition_started = start_cuda_measurement()
    rows, summary = run_condition(
        records,
        baseline_by_index,
        model=model,
        tokenizer=tokenizer,
        selection=mean_shift,
        condition=condition,
        label="independent100",
        patcher=_shift_patcher,
        runtime_validator=validate_shift_runtime,
    )
    summary.update(finish_cuda_measurement(condition_started))
    if summary["prompt_count"] != 100 or summary["input_ids_match_count"] != 100:
        raise RuntimeError("independent100 input/runtime count mismatch")
    if summary["selected_rank_intervention_count"] != 100 * ALL_RANK_COUNT:
        raise RuntimeError("independent100 all-rank intervention count mismatch")
    if summary["generation_failure_count"] != 0:
        raise RuntimeError("independent100 contains generation failures")

    output_dir.mkdir(parents=True)
    baseline_path = output_dir / "independent100_baseline.jsonl"
    baseline_summary_path = output_dir / "independent100_baseline_summary.json"
    prompt_path = output_dir / "independent100_prompt_results.jsonl"
    summary_path = output_dir / "independent100_summary.json"
    evaluation_path = output_dir / "evaluation_summary.json"
    manifest_path = output_dir / "run_manifest.json"
    write_jsonl(baseline_path, [
        {key: serialized_input_ids(value) if key == "input_ids" else value
         for key, value in row.items() if key != "first_logits"}
        for row in baseline
    ])
    baseline_success = sum(bool(row["success"]) for row in baseline)
    baseline_summary = {
        "dataset_group": "independent100",
        "prompt_count": 100,
        "attack_success_count": baseline_success,
        "asr": baseline_success / 100,
        "generation_failure_count": 0,
        "input_ids_saved_count": 100,
        "exact_prior_position_baseline_reproduction": True,
        **baseline_runtime,
    }
    write_json(baseline_summary_path, baseline_summary)
    write_jsonl(prompt_path, rows)
    prompt_with_crossing = sum(
        row["diagnostics"]["selected_value_crossing_count"] > 0 for row in rows
    )
    summary = dict(
        summary,
        prompt_with_crossing_count=prompt_with_crossing,
        runtime_invariants_passed=True,
    )
    write_json(summary_path, summary)
    result = {
        "status": "complete",
        "baseline": baseline_summary,
        "condition": summary,
    }
    write_json(evaluation_path, result)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": utc_now(),
        "execution_git_head": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "execution_git_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "execution_git_status_porcelain": git_value(
            REPO_ROOT, "status", "--porcelain"
        ).splitlines(),
        "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
        "shift_implementation_source_sha256": sha256_file(
            REPO_ROOT / "utils/gcg_rank_distribution_shift_cancellation.py"
        ),
        "rank_runtime_source_sha256": sha256_file(
            REPO_ROOT / "utils/gcg_rank_element_topk_intervention.py"
        ),
        "model": str(Path(args.model).resolve()),
        "model_validation": model_validation,
        "gpu": torch.cuda.get_device_name(0),
        "shift_artifact_verification": verification,
        "source_artifact_verification": source_verification,
        "shift_source": "Selection20 frozen mean_shift artifact only",
        "shift_recomputed": False,
        "evaluation_role": "final_independent_test",
        "dataset": "AdvBench indices 220..319",
        "prompt_count": 100,
        "condition": condition,
        "intervention_timing": INTERVENTION_TIMING,
        "generation_settings": GENERATION_SETTINGS,
        "independent_test_provenance": provenance,
        "prior_position_baseline_reproduced": True,
        "prior_position_baseline_success": EXPECTED_BASELINE_SUCCESS,
        "prior_position_top500_success": EXPECTED_POSITION_SUCCESS,
        "development_data_used_for_evaluation": False,
    }
    write_json(manifest_path, manifest)
    # The run manifest names the artifact-manifest hash, so exclude it from
    # that manifest to avoid a circular self-hash dependency.
    artifacts = [baseline_path, baseline_summary_path, prompt_path, summary_path,
                 evaluation_path]
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    write_json(artifact_manifest_path, {
        "schema_version": 1,
        "created_at": utc_now(),
        "artifacts": [
            {"relative_path": path.name, "sha256": sha256_file(path)}
            for path in artifacts
        ],
    })
    saved = read_json(manifest_path)
    saved["artifact_manifest_sha256"] = sha256_file(artifact_manifest_path)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(saved, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, manifest_path)
    return dict(result, output_dir=str(output_dir))


def main(argv=None):
    args = parse_args(argv)
    result = cpu_validation_payload(args) if args.stage == "cpu-validate" else run_gpu(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
