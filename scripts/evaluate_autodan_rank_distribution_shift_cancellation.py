#!/usr/bin/env python3
"""Evaluate frozen GCG rank-shift cancellation on AutoDAN evaluation110."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import transformers


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_gcg_reference_factorization import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MODEL,
    GENERATION_SETTINGS,
    load_model_and_tokenizer,
)
from scripts.evaluate_autodan_asym_score_intervention import (  # noqa: E402
    align_evaluation_records,
    format_records as format_autodan_records,
)
from scripts.evaluate_autodan_baseline import read_records  # noqa: E402
from scripts.evaluate_autodan_head_sets import (  # noqa: E402
    EXPECTED_EVALUATION_SIZE,
    EXPECTED_SELECTION_INDICES,
    validate_fixed_experiment_manifest,
)
from scripts.evaluate_gcg_position_element_topk_intervention import (  # noqa: E402
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
from utils.evaluate_utils import REFUSAL_SUBSTRINGS  # noqa: E402
from utils.experiment_runtime import git_value, seed_everything  # noqa: E402
from utils.gcg_rank_distribution_shift_cancellation import (  # noqa: E402
    ALL_RANK_COUNT,
    ALPHA_DEFAULT,
    METHOD,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    read_json,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.gcg_wasserstein_reference_evaluation import ensure_output_absent  # noqa: E402


EXPECTED_SHIFT_SHA256 = (
    "98691484937db994a7edd030dfc6532d4a7549ee312f8108d01c018ce39a2bef"
)
EXPECTED_BASELINE_SUCCESS = 13
EXPECTED_BASELINE_ASR = EXPECTED_BASELINE_SUCCESS / EXPECTED_EVALUATION_SIZE
DEFAULT_INPUT = (
    "data/derived/llama2/autodan/holdout_seed42/"
    "evaluation_jailbreak_prompts.jsonl"
)
DEFAULT_SPLIT_MANIFEST = (
    "data/derived/llama2/autodan/holdout_seed42/split_manifest.json"
)
DEFAULT_SOURCE_LABELS = "data/raw/llama2/llama2_AutoDAN.json"
DEFAULT_PRIOR_NO_CACHE_RUN = (
    "data/outputs/llama2/autodan/holdout_seed42/"
    "asym_score_ratio_headsets_full_beta05_10/no_intervention"
)
DEFAULT_HISTORICAL_BASELINE = (
    "data/outputs/llama2/autodan/baseline_alignment/"
    "no_intervention_prompt_results.jsonl"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu-validate", "run"), default="cpu-validate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--shift-artifact-dir", required=True)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--input-file", default=DEFAULT_INPUT)
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--source-labels", default=DEFAULT_SOURCE_LABELS)
    parser.add_argument("--prior-no-cache-run", default=DEFAULT_PRIOR_NO_CACHE_RUN)
    parser.add_argument("--historical-baseline", default=DEFAULT_HISTORICAL_BASELINE)
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def validate_prior_no_cache_baseline(
    rows: list[dict], summary: dict, evaluation_indices: list[int]
) -> None:
    if len(rows) != EXPECTED_EVALUATION_SIZE:
        raise ValueError("prior AutoDAN no-cache baseline must contain 110 rows")
    indices = [row.get("source_index") for row in rows]
    if indices != evaluation_indices or len(set(indices)) != len(indices):
        raise ValueError("prior AutoDAN baseline identity/order mismatch")
    for row in rows:
        if (
            row.get("condition") != "no_intervention"
            or row.get("use_cache") is not False
            or row.get("generation_failed") is not False
            or type(row.get("success")) is not bool
            or not isinstance(row.get("prompt"), str)
            or not isinstance(row.get("formatted_prompt"), str)
            or not isinstance(row.get("response"), str)
            or row.get("response_length_tokens") != 32
        ):
            raise ValueError("prior AutoDAN no-cache baseline row mismatch")
    metadata = summary.get("metadata", {})
    generation = metadata.get("generation_config", {})
    expected_generation = {
        "max_new_tokens": 32,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "use_cache": False,
        "seed": 42,
        "system_prompt": None,
        "batch_size": 1,
    }
    if (
        summary.get("condition") != "no_intervention"
        or summary.get("use_cache") is not False
        or summary.get("total_count") != EXPECTED_EVALUATION_SIZE
        or summary.get("evaluated_count") != EXPECTED_EVALUATION_SIZE
        or summary.get("success_count") != EXPECTED_BASELINE_SUCCESS
        or summary.get("asr") != EXPECTED_BASELINE_ASR
        or summary.get("generation_failure_count") != 0
        or generation != expected_generation
        or metadata.get("refusal_substring_count") != len(REFUSAL_SUBSTRINGS)
        or metadata.get("attention_implementation") != "eager"
        or metadata.get("transformers_version") != "4.42.3"
    ):
        raise ValueError("prior AutoDAN no-cache baseline summary mismatch")


def _validate_historical_baseline(
    rows: list[dict], evaluation_indices: list[int], prior_rows: list[dict]
) -> dict:
    by_index = {}
    for row in rows:
        index = row.get("source_index")
        if type(index) is not int or index in by_index:
            raise ValueError("historical AutoDAN baseline identity mismatch")
        by_index[index] = row
    if len(rows) != 130 or set(by_index) != set(range(130)):
        raise ValueError("historical AutoDAN baseline must cover all 130 prompts")
    prior_by_index = {row["source_index"]: row for row in prior_rows}
    success_matches = 0
    response_matches = 0
    for index in evaluation_indices:
        historical = by_index[index]
        prior = prior_by_index[index]
        if (
            historical.get("prompt") != prior["prompt"]
            or historical.get("formatted_prompt") != prior["formatted_prompt"]
            or historical.get("generation_failed") is not False
            or type(historical.get("success")) is not bool
        ):
            raise ValueError("historical/prior AutoDAN baseline alignment mismatch")
        success_matches += historical["success"] is prior["success"]
        response_matches += historical.get("response") == prior.get("response")
    if success_matches != EXPECTED_EVALUATION_SIZE:
        raise ValueError("historical/prior AutoDAN baseline success mismatch")
    return {
        "evaluated_count": EXPECTED_EVALUATION_SIZE,
        "success_match_count": success_matches,
        "response_exact_match_count": response_matches,
    }


def format_rank_records(
    records: list[dict], tokenizer, prior_rows: list[dict]
) -> list[dict]:
    formatted = format_autodan_records(
        records,
        tokenizer,
        {"model_name": "Llama"},
    )
    prior_by_index = {row["source_index"]: row for row in prior_rows}
    output = []
    for row_position, record in enumerate(formatted):
        index = record["source_index"]
        prior = prior_by_index[index]
        if (
            record["prompt"] != prior["prompt"]
            or record["formatted_prompt"] != prior["formatted_prompt"]
        ):
            raise ValueError(f"AutoDAN prior prompt mismatch at source_index {index}")
        input_ids = tokenizer(
            record["formatted_prompt"],
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        if input_ids.shape[0] != 1 or input_ids.shape[1] < 16:
            raise ValueError(f"invalid AutoDAN input IDs at source_index {index}")
        output.append(
            {
                **record,
                "dataset_index": index,
                "row_position": row_position,
                "prompt_length": int(input_ids.shape[1]),
                "input_ids": input_ids,
            }
        )
    return output


def load_autodan_evaluation(args, tokenizer) -> tuple[list[dict], list[dict], dict]:
    input_path = Path(args.input_file).resolve()
    split_path = Path(args.split_manifest).resolve()
    source_path = Path(args.source_labels).resolve()
    prior_dir = Path(args.prior_no_cache_run).resolve()
    historical_path = Path(args.historical_baseline).resolve()
    manifest = read_json(split_path)
    validate_fixed_experiment_manifest(manifest)
    if set(manifest["selection_indices"]) != set(EXPECTED_SELECTION_INDICES):
        raise ValueError("AutoDAN selection20 identity mismatch")
    evaluation_indices = manifest["evaluation_indices"]
    if set(evaluation_indices) & set(manifest["selection_indices"]):
        raise ValueError("AutoDAN selection20 leaked into evaluation110")
    input_rows = read_records(input_path)
    source_rows = read_records(source_path)
    aligned = align_evaluation_records(
        input_rows,
        manifest,
        source_rows,
        str(input_path),
        str(source_path),
    )
    prior_path = prior_dir / "prompt_results.jsonl"
    prior_summary_path = prior_dir / "summary.json"
    prior_rows = read_records(prior_path)
    prior_summary = read_json(prior_summary_path)
    validate_prior_no_cache_baseline(prior_rows, prior_summary, evaluation_indices)
    historical_rows = read_records(historical_path)
    historical_comparison = _validate_historical_baseline(
        historical_rows, evaluation_indices, prior_rows
    )
    rank_records = format_rank_records(aligned, tokenizer, prior_rows)
    if [row["dataset_index"] for row in rank_records] != evaluation_indices:
        raise ValueError("formatted AutoDAN evaluation identity mismatch")
    provenance = {
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "evaluation_input": str(input_path),
        "evaluation_input_sha256": sha256_file(input_path),
        "source_labels": str(source_path),
        "source_labels_sha256": sha256_file(source_path),
        "prior_no_cache_baseline": str(prior_path),
        "prior_no_cache_baseline_sha256": sha256_file(prior_path),
        "prior_no_cache_summary_sha256": sha256_file(prior_summary_path),
        "historical_baseline": str(historical_path),
        "historical_baseline_sha256": sha256_file(historical_path),
        "evaluation_prompt_count": len(rank_records),
        "evaluation_indices": evaluation_indices,
        "duplicate_count": len(evaluation_indices) - len(set(evaluation_indices)),
        "selection_overlap_count": len(
            set(evaluation_indices) & set(manifest["selection_indices"])
        ),
        "prior_no_cache_success_count": EXPECTED_BASELINE_SUCCESS,
        "historical_comparison": historical_comparison,
        "system_prompt": None,
        "template": "systemless user-only Llama chat template",
        "autodan_selection_used_for_shift": False,
        "autodan_evaluation_used_for_shift": False,
    }
    return rank_records, prior_rows, provenance


def verify_baseline_reproduction(
    baseline: list[dict], prior_rows: list[dict], *, expected_count=EXPECTED_EVALUATION_SIZE
) -> None:
    if len(baseline) != expected_count:
        raise RuntimeError("AutoDAN baseline count mismatch")
    prior_by_index = {row["source_index"]: row for row in prior_rows}
    success_count = 0
    for row in baseline:
        index = row["dataset_index"]
        prior = prior_by_index.get(index)
        if prior is None or (
            row["prompt"] != prior["prompt"]
            or row["formatted_prompt"] != prior["formatted_prompt"]
            or row["response"] != prior["response"]
            or row["success"] is not prior["success"]
            or row["response_token_count"] != prior["response_length_tokens"]
            or row["generation_failed"] is not False
        ):
            raise RuntimeError(
                f"AutoDAN no-cache baseline reproduction mismatch at {index}; "
                "refusing intervention"
            )
        success_count += bool(row["success"])
    if expected_count == EXPECTED_EVALUATION_SIZE and success_count != EXPECTED_BASELINE_SUCCESS:
        raise RuntimeError("AutoDAN baseline is not the frozen 13/110")


def _load_cpu_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def cpu_validation_payload(args) -> dict:
    mean_shift, config, verification, source_verification, _, _ = load_validated_shift(args)
    if verification["mean_shift_sha256"] != EXPECTED_SHIFT_SHA256:
        raise ValueError("mean_shift artifact hash mismatch")
    tokenizer = _load_cpu_tokenizer(Path(args.model).resolve())
    records, _, provenance = load_autodan_evaluation(args, tokenizer)
    smoke = cpu_runtime_smoke(mean_shift, alpha=ALPHA_DEFAULT)
    return {
        "status": "cpu_validation_passed",
        "method": METHOD,
        "evaluation_role": "cross_attack_generalization",
        "source_attack": "GCG Selection20 only",
        "target_attack": "AutoDAN evaluation110",
        "alpha": ALPHA_DEFAULT,
        "rank_element_count_per_prompt": ALL_RANK_COUNT,
        "prompt_count": len(records),
        "mean_shift_shape": list(mean_shift.shape),
        "shift_artifact_verification": verification,
        "source_artifact_verified": source_verification["verified"],
        "shift_recomputed": False,
        "autodan_used_for_shift_or_tuning": False,
        "gcg_independent100_evaluation_artifact_read": False,
        "autodan_provenance": provenance,
        "scientific_conditions": config["scientific_conditions"],
        "cpu_runtime_smoke": smoke,
        "model_loaded": False,
        "gpu_used": False,
        "artifacts_written": False,
    }


def _baseline_json_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            key: serialized_input_ids(value) if key == "input_ids" else value
            for key, value in row.items()
            if key != "first_logits"
        }
        for row in rows
    ]


def run_gpu(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --stage run")
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    output_dir = ensure_output_absent(args.output_dir).resolve()
    mean_shift, config, verification, source_verification, _, _ = load_validated_shift(args)
    if verification["mean_shift_sha256"] != EXPECTED_SHIFT_SHA256:
        raise ValueError("mean_shift artifact hash mismatch")
    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(Path(args.model).resolve())
    model_validation = validate_loaded_model(model, config)
    records, prior_rows, provenance = load_autodan_evaluation(args, tokenizer)

    baseline_started = start_cuda_measurement()
    baseline = run_baseline(
        records,
        model=model,
        tokenizer=tokenizer,
        label="AutoDAN evaluation110 baseline",
        require_saved_reproduction=False,
    )
    baseline_runtime = finish_cuda_measurement(baseline_started)
    verify_baseline_reproduction(baseline, prior_rows)

    output_dir.mkdir(parents=True)
    baseline_path = output_dir / "autodan_evaluation110_baseline.jsonl"
    baseline_summary_path = output_dir / "autodan_evaluation110_baseline_summary.json"
    write_jsonl(baseline_path, _baseline_json_rows(baseline))
    baseline_summary = {
        "dataset_group": "autodan_evaluation110",
        "prompt_count": EXPECTED_EVALUATION_SIZE,
        "attack_success_count": EXPECTED_BASELINE_SUCCESS,
        "asr": EXPECTED_BASELINE_ASR,
        "generation_failure_count": 0,
        "input_ids_saved_count": EXPECTED_EVALUATION_SIZE,
        "exact_prior_no_cache_baseline_reproduction": True,
        **baseline_runtime,
    }
    write_json(baseline_summary_path, baseline_summary)

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
        label="autodan_evaluation110",
        patcher=_shift_patcher,
        runtime_validator=validate_shift_runtime,
    )
    summary.update(finish_cuda_measurement(condition_started))
    summary["prompt_with_crossing_count"] = sum(
        row["diagnostics"]["selected_value_crossing_count"] > 0 for row in rows
    )
    summary["runtime_invariants_passed"] = True
    if (
        summary["prompt_count"] != EXPECTED_EVALUATION_SIZE
        or summary["input_ids_match_count"] != EXPECTED_EVALUATION_SIZE
        or summary["selected_rank_intervention_count"]
        != EXPECTED_EVALUATION_SIZE * ALL_RANK_COUNT
        or summary["generation_failure_count"] != 0
    ):
        raise RuntimeError("AutoDAN evaluation runtime invariant mismatch")

    prompt_path = output_dir / "autodan_evaluation110_prompt_results.jsonl"
    summary_path = output_dir / "autodan_evaluation110_summary.json"
    evaluation_path = output_dir / "evaluation_summary.json"
    manifest_path = output_dir / "run_manifest.json"
    write_jsonl(prompt_path, rows)
    write_json(summary_path, summary)
    result = {"status": "complete", "baseline": baseline_summary, "condition": summary}
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
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "shift_artifact_verification": verification,
        "source_artifact_verification": source_verification,
        "shift_source": "GCG Selection20 frozen mean_shift artifact only",
        "shift_recomputed": False,
        "autodan_used_for_shift_or_tuning": False,
        "evaluation_role": "cross_attack_generalization",
        "dataset": "AutoDAN fixed evaluation110",
        "prompt_count": EXPECTED_EVALUATION_SIZE,
        "condition": condition,
        "intervention_timing": INTERVENTION_TIMING,
        "generation_settings": GENERATION_SETTINGS,
        "refusal_substring_count": len(REFUSAL_SUBSTRINGS),
        "autodan_provenance": provenance,
        "prior_no_cache_baseline_reproduced": True,
        "prior_no_cache_baseline_success": EXPECTED_BASELINE_SUCCESS,
    }
    write_json(manifest_path, manifest)
    artifacts = [
        baseline_path,
        baseline_summary_path,
        prompt_path,
        summary_path,
        evaluation_path,
    ]
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    write_json(
        artifact_manifest_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "artifacts": [
                {"relative_path": path.name, "sha256": sha256_file(path)}
                for path in artifacts
            ],
        },
    )
    saved = read_json(manifest_path)
    saved["artifact_manifest_sha256"] = sha256_file(artifact_manifest_path)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(saved, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, manifest_path)
    return {**result, "output_dir": str(output_dir)}


def main(argv=None):
    args = parse_args(argv)
    result = cpu_validation_payload(args) if args.stage == "cpu-validate" else run_gpu(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
