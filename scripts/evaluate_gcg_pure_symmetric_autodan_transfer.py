#!/usr/bin/env python3
"""Evaluate frozen GCG Pure Sym Top-500 on official-format AutoDAN130."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_gcg_reference_factorization import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MODEL,
    EXPECTED_SCORE_ARTIFACT_SHA256,
    EXPECTED_SELECTION_MANIFEST_SHA256,
    EXPECTED_TOP5,
    generate_with_first_logits,
    load_model_and_tokenizer,
)
from scripts.evaluate_gcg_position_element_topk_intervention import (  # noqa: E402
    AUTODAN_OFFICIAL_REPOSITORY_COMMIT,
    DEFAULT_AUTODAN_INPUT,
    EXPECTED_AUTODAN_SHA256,
    autodan_prompt_format_provenance,
    format_autodan_records,
    load_autodan_transfer_records,
    serialized_input_ids,
)
from scripts.evaluate_gcg_pure_symmetric_position_element_topk_intervention import (  # noqa: E402
    ALPHA,
    COMPONENT,
    DECOMPOSITION,
    DELTA_DEFINITION,
    INTERVENTION_TIMING,
    INTERVENTION_VARIANT,
    K,
    RAW_SCORE_INTERVENTION,
    DEFAULT_REUSED_SYM_RUN,
    load_reused_sym_selection,
    validate_prompt_runtime_invariants,
)
from utils.experiment_runtime import git_value, seed_everything  # noqa: E402
from utils.gcg_position_element_topk_intervention import (  # noqa: E402
    SYMMETRIC_RANKING_METRIC,
    patch_llama_pure_symmetric_position_element_attention,
)
from utils.gcg_reference_intervention_diagnostics import (  # noqa: E402
    compare_token_sequences,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    TAIL_TOKENS,
    llama_model_config_summary,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    verify_artifact_directory,
    write_json,
    write_jsonl,
)
from utils.gcg_wasserstein_reference_evaluation import (  # noqa: E402
    classify_transition,
    ensure_output_absent,
)
from utils.reference_asym_score_intervention import (  # noqa: E402
    load_validated_reference_artifacts,
    validate_loaded_llama_model,
)


OFFICIAL_FORMAT = "official_fastchat_34062e9"
MODEL_REPOSITORY_HEAD = "f5db02db724555f92da89c216ac04704f23d4590"
TOTAL_COUNT = 130
BASELINE_SUCCESS_COUNT = 49
ASYM_TRANSFER_SUCCESS_COUNT = 49
MALFORMED_INDEX = 10
SEED = 42

DEFAULT_OFFICIAL_BASELINE_DIR = (
    "data/outputs/llama2/autodan/formatting_ablation/"
    "official_fastchat_34062e9_full130_20260914T151610Z"
)
EXPECTED_OFFICIAL_BASELINE_HASHES = {
    "artifact_manifest.json": "11c7a8a90a1179245740d482974bc5af4ef6d1f2a87c8eb5b863b4bf7dba9eb3",
    "cpu_preflight.json": "f038232cf638af6b5bb4281c8eb9da8cbf07a591b97e8b4cb672bfa7bb0093a2",
    "paired_results.jsonl": "409bc0f06542824920cac8b5cc5c0c2c87ce578aa0fea3d328b1a3523a8432a9",
    "run_manifest.json": "52e717bf25f9a0f479a087aad7d0f16d950f191fc71081bf6c6ccee5dedcf76a",
    "summary.json": "882454fa87387d2f66f81c8bc425b07b0502f4ead844a32a42d5a0e3acccd959",
}

DEFAULT_ASYM_TRANSFER_DIR = (
    "data/outputs/gcg_position_element_topk/"
    "autodan130_official_format_transfer/full130_20260914T152933Z"
)
EXPECTED_ASYM_TRANSFER_HASHES = {
    "artifact_manifest.json": "fbfb36b39be53623366bb3095e8acceb9ba914e1cfa1839ff0a7564d773e5047",
    "baseline_reproduction.jsonl": "0940611ec07f29e3965a22e7391fd4d81f334b8709d91249cd17aeb65bb795cb",
    "cpu_preflight.json": "89135d2ce50e5f89bc40de39794c0ef9ad91e57f744aa4efa69835125fa514f6",
    "defense_results.jsonl": "9575fb47dbf8ba1fc0d1925943e54c20ba82c24fee3b525ec8bc5d9996dbed77",
    "run_manifest.json": "64c1c6ea3bbb9bfe08733fa13db150714528a829085d4da8fae5b27f38fa2c2f",
    "summary.json": "217892de232560c493c491f2c7f65bb09c20225c76ca2fb544d0a35a95732fb1",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("cpu-validate", "run"), default="cpu-validate"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--reused-sym-run", default=DEFAULT_REUSED_SYM_RUN)
    parser.add_argument("--autodan-input", default=DEFAULT_AUTODAN_INPUT)
    parser.add_argument(
        "--official-baseline-dir", default=DEFAULT_OFFICIAL_BASELINE_DIR
    )
    parser.add_argument("--asym-transfer-dir", default=DEFAULT_ASYM_TRANSFER_DIR)
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def verify_fixed_hashes(
    directory: Path, expected: Mapping[str, str]
) -> dict[str, dict[str, str]]:
    verified = {}
    for name, expected_sha in expected.items():
        path = directory / name
        observed = sha256_file(path)
        if observed != expected_sha:
            raise ValueError(
                f"fixed artifact SHA-256 mismatch: {path}: "
                f"{observed} != {expected_sha}"
            )
        verified[name] = {"path": str(path), "sha256": observed}
    return verified


def verify_artifact_manifest(directory: Path) -> None:
    manifest = read_json(directory / "artifact_manifest.json")
    for artifact in manifest.get("artifacts", []):
        path = directory / artifact["relative_path"]
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"artifact manifest mismatch: {path}")


def normalize_canonical_baseline_rows(
    baseline_rows: Sequence[Mapping[str, object]],
    raw_records: Sequence[Mapping[str, object]],
) -> list[dict]:
    """Validate and normalize the immutable official-format baseline rows."""

    if len(baseline_rows) != TOTAL_COUNT or len(raw_records) != TOTAL_COUNT:
        raise ValueError("official AutoDAN baseline/raw record count must equal 130")
    normalized = []
    success_count = 0
    for index, (row, raw) in enumerate(zip(baseline_rows, raw_records)):
        prompt = raw.get("prompt")
        input_ids = row.get("official_input_ids")
        response_ids = row.get("official_response_token_ids")
        success = row.get("official_format_success")
        if row.get("dataset_index") != index or raw.get("dataset_index") != index:
            raise ValueError(f"official baseline index mismatch at index={index}")
        if row.get("prompt_id") != f"autodan_{index}":
            raise ValueError(f"official baseline prompt ID mismatch at index={index}")
        if row.get("prompt") != prompt:
            raise ValueError(f"official baseline prompt mismatch at index={index}")
        if not isinstance(input_ids, list) or not input_ids:
            raise ValueError(f"official baseline input IDs invalid at index={index}")
        if not isinstance(response_ids, list) or not response_ids:
            raise ValueError(f"official baseline response IDs invalid at index={index}")
        if type(success) is not bool:
            raise ValueError(f"official baseline success invalid at index={index}")
        if row.get("generation_failed") is not False:
            raise ValueError(f"official baseline generation failure at index={index}")
        if row.get("official_prompt_length") != len(input_ids):
            raise ValueError(f"official baseline prompt length mismatch at index={index}")
        expected_placeholder = index == MALFORMED_INDEX
        if row.get("contains_unreplaced_placeholder") is not expected_placeholder:
            raise ValueError(f"placeholder flag mismatch at index={index}")
        if ("[REPLACE]" in str(prompt)) is not expected_placeholder:
            raise ValueError(f"placeholder text mismatch at index={index}")
        success_count += int(success)
        normalized.append(
            {
                "dataset_index": index,
                "prompt_id": row["prompt_id"],
                "prompt": prompt,
                "formatted_prompt": row["official_formatted_prompt"],
                "prompt_length": row["official_prompt_length"],
                "input_ids": input_ids,
                "response": row["official_format_response"],
                "response_token_ids": response_ids,
                "success": success,
                "generation_failed": False,
            }
        )
    if success_count != BASELINE_SUCCESS_COUNT:
        raise ValueError("official baseline success count must equal 49")
    return normalized


def validate_asym_transfer_rows(
    asym_rows: Sequence[Mapping[str, object]],
    reproduced_baseline_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> None:
    """Verify the existing GCG Asym transfer and its reproduced baseline."""

    if not (
        len(asym_rows) == len(reproduced_baseline_rows) == len(baseline_rows)
        == TOTAL_COUNT
    ):
        raise ValueError("Asym transfer source record count must equal 130")
    success_count = 0
    for index, (asym, reproduced, baseline) in enumerate(
        zip(asym_rows, reproduced_baseline_rows, baseline_rows)
    ):
        for row in (asym, reproduced):
            if row.get("dataset_index") != index:
                raise ValueError(f"Asym transfer index mismatch at index={index}")
            if row.get("input_ids") != baseline.get("input_ids"):
                raise ValueError(f"Asym transfer input ID mismatch at index={index}")
        if (
            reproduced.get("response") != baseline.get("response")
            or reproduced.get("success") is not baseline.get("success")
            or asym.get("baseline_response") != baseline.get("response")
            or asym.get("baseline_success") is not baseline.get("success")
        ):
            raise ValueError(f"Asym/baseline identity mismatch at index={index}")
        if asym.get("k") != K or asym.get("alpha") != ALPHA:
            raise ValueError(f"Asym transfer K/alpha mismatch at index={index}")
        if asym.get("intervention_timing") != INTERVENTION_TIMING:
            raise ValueError(f"Asym transfer timing mismatch at index={index}")
        if asym.get("generation_failed") is not False:
            raise ValueError(f"Asym transfer generation failure at index={index}")
        success_count += int(asym.get("success") is True)
    if success_count != ASYM_TRANSFER_SUCCESS_COUNT:
        raise ValueError("GCG Asym transfer success count must equal 49")


def validate_model_repository_head(model_path: Path) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={model_path}",
            "-C",
            str(model_path),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = completed.stdout.strip()
    if observed != MODEL_REPOSITORY_HEAD:
        raise ValueError("model repository revision mismatch")
    return observed


def validate_frozen_sources(args) -> dict:
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_verification = verify_artifact_directory(artifact_dir)
    if (
        artifact_verification.get("verified") is not True
        or artifact_verification.get("selection_manifest_sha256")
        != EXPECTED_SELECTION_MANIFEST_SHA256
        or artifact_verification.get("score_artifact_sha256")
        != EXPECTED_SCORE_ARTIFACT_SHA256
    ):
        raise ValueError("fixed GCG Selection20 artifact verification failed")

    selection, sym_provenance = load_reused_sym_selection(
        Path(args.reused_sym_run)
    )
    raw_records, raw_provenance = load_autodan_transfer_records(
        Path(args.autodan_input)
    )
    baseline_dir = Path(args.official_baseline_dir).resolve()
    baseline_hashes = verify_fixed_hashes(
        baseline_dir, EXPECTED_OFFICIAL_BASELINE_HASHES
    )
    verify_artifact_manifest(baseline_dir)
    baseline_manifest = read_json(baseline_dir / "run_manifest.json")
    baseline_summary = read_json(baseline_dir / "summary.json")
    if (
        baseline_manifest.get("status") != "complete"
        or baseline_manifest.get("autodan_repository_commit")
        != AUTODAN_OFFICIAL_REPOSITORY_COMMIT
        or baseline_manifest.get("model_repository_head") != MODEL_REPOSITORY_HEAD
        or baseline_manifest.get("index_10_unreplaced_placeholder_retained") is not True
        or baseline_summary.get("condition_b_success_count")
        != BASELINE_SUCCESS_COUNT
        or baseline_summary.get("generation_failure_count") != 0
    ):
        raise ValueError("official baseline manifest/summary conditions mismatch")
    baseline_rows = normalize_canonical_baseline_rows(
        read_jsonl(baseline_dir / "paired_results.jsonl"), raw_records
    )

    asym_dir = Path(args.asym_transfer_dir).resolve()
    asym_hashes = verify_fixed_hashes(asym_dir, EXPECTED_ASYM_TRANSFER_HASHES)
    verify_artifact_manifest(asym_dir)
    asym_manifest = read_json(asym_dir / "run_manifest.json")
    asym_summary = read_json(asym_dir / "summary.json")
    if (
        asym_manifest.get("status") != "complete"
        or asym_manifest.get("selection_source")
        != "GCG selection20 only; AutoDAN was not used for selection/reference"
        or asym_manifest.get("format_formula")
        != "[INST] <<SYS>>\\n\\n<</SYS>>\\n\\n{prompt}[/INST]"
        or asym_manifest.get("k") != K
        or asym_manifest.get("alpha") != ALPHA
        or asym_manifest.get("intervention_timing") != INTERVENTION_TIMING
        or asym_manifest.get("index_10_unreplaced_placeholder_retained") is not True
        or asym_summary.get("baseline", {}).get("attack_success_count")
        != BASELINE_SUCCESS_COUNT
        or asym_summary.get("defense", {}).get("attack_success_count")
        != ASYM_TRANSFER_SUCCESS_COUNT
    ):
        raise ValueError("GCG Asym transfer scientific conditions mismatch")
    asym_rows = read_jsonl(asym_dir / "defense_results.jsonl")
    reproduced_baseline = read_jsonl(asym_dir / "baseline_reproduction.jsonl")
    validate_asym_transfer_rows(asym_rows, reproduced_baseline, baseline_rows)

    return {
        "selection": selection,
        "sym_provenance": sym_provenance,
        "raw_records": raw_records,
        "raw_provenance": raw_provenance,
        "baseline_rows": baseline_rows,
        "baseline_hashes": baseline_hashes,
        "asym_rows": asym_rows,
        "asym_hashes": asym_hashes,
        "artifact_verification": artifact_verification,
    }


def cpu_validation_payload(args) -> dict:
    frozen = validate_frozen_sources(args)
    return {
        "status": "cpu_validation_passed",
        "model_loaded": False,
        "gpu_used": False,
        "artifacts_written": False,
        "component": COMPONENT,
        "intervention_variant": INTERVENTION_VARIANT,
        "decomposition": DECOMPOSITION,
        "delta_definition": DELTA_DEFINITION,
        "raw_score_intervention": RAW_SCORE_INTERVENTION,
        "candidate_count": frozen["sym_provenance"]["candidate_count"],
        "selected_count": frozen["sym_provenance"]["selected_count"],
        "selection_reused_without_recomputation": True,
        "selection_gpu_extraction_performed": False,
        "autodan_record_count": len(frozen["raw_records"]),
        "official_baseline_success_count": sum(
            row["success"] for row in frozen["baseline_rows"]
        ),
        "asym_transfer_success_count": sum(
            bool(row["success"]) for row in frozen["asym_rows"]
        ),
        "index_10_unreplaced_placeholder_retained": True,
        "prompt_format": autodan_prompt_format_provenance(OFFICIAL_FORMAT),
        "sym_selection_provenance": frozen["sym_provenance"],
        "official_baseline_artifacts": frozen["baseline_hashes"],
        "asym_transfer_artifacts": frozen["asym_hashes"],
    }


def start_cuda_measurement() -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def finish_cuda_measurement(started_at: float) -> dict:
    torch.cuda.synchronize()
    return {
        "elapsed_seconds": time.perf_counter() - started_at,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved() / (1024**2),
    }


def run_condition(records, baseline_by_index, *, model, tokenizer, selection):
    rows = []
    diagnostics_rows = []
    with patch_llama_pure_symmetric_position_element_attention(
        model,
        selection=selection,
        k=K,
        alpha=ALPHA,
        intervention_timing=INTERVENTION_TIMING,
    ) as controller:
        for position, record in enumerate(records, 1):
            controller.clear()
            controller.set_prompt_context(prompt_length=record["prompt_length"])
            baseline = baseline_by_index[int(record["dataset_index"])]
            ids = serialized_input_ids(record["input_ids"])
            if ids != baseline["input_ids"]:
                raise RuntimeError(
                    "official baseline input ID mismatch: "
                    f"dataset_index={record['dataset_index']}"
                )
            generated = generate_with_first_logits(
                model, tokenizer, record["input_ids"]
            )
            if generated["generation_failed"]:
                raise RuntimeError(
                    f"generation failed: dataset_index={record['dataset_index']}"
                )
            diagnostics = controller.summary()
            validate_prompt_runtime_invariants(diagnostics)
            comparison = compare_token_sequences(
                baseline["response_token_ids"], generated["response_token_ids"]
            )
            if not generated["response_token_ids"]:
                raise RuntimeError("empty response token IDs")
            baseline_first = int(baseline["response_token_ids"][0])
            intervention_first = int(generated["response_token_ids"][0])
            first_equal = baseline_first == intervention_first
            if bool(comparison["response_equal"]) is not first_equal:
                raise RuntimeError(
                    "first-token-only invariant failed: response equality differs "
                    "from first-token argmax equality"
                )
            rows.append(
                {
                    "dataset_group": "autodan130_official_format",
                    "prompt_id": record["prompt_id"],
                    "row_position": record["row_position"],
                    "dataset_index": record["dataset_index"],
                    "prompt": record["prompt"],
                    "formatted_prompt": record["formatted_prompt"],
                    "prompt_length": record["prompt_length"],
                    "input_ids": ids,
                    "input_ids_match_reused_baseline": True,
                    "component": COMPONENT,
                    "intervention_variant": INTERVENTION_VARIANT,
                    "k": K,
                    "alpha": ALPHA,
                    "intervention_timing": INTERVENTION_TIMING,
                    "baseline_response": baseline["response"],
                    "baseline_success": baseline["success"],
                    **{
                        key: value
                        for key, value in generated.items()
                        if key != "first_logits"
                    },
                    **comparison,
                    "baseline_first_token_id": baseline_first,
                    "intervention_first_token_id": intervention_first,
                    "first_token_argmax_equal": first_equal,
                    "transition": classify_transition(
                        baseline["success"], generated["success"]
                    ),
                    "diagnostics": diagnostics,
                }
            )
            diagnostics_rows.append(diagnostics)
            print(
                f"[AutoDAN130 {INTERVENTION_VARIANT} {position}/{TOTAL_COUNT}] "
                f"dataset_index={record['dataset_index']} "
                f"success={generated['success']}",
                flush=True,
            )

    success_count = sum(bool(row["success"]) for row in rows)
    summary = {
        "dataset_group": "autodan130_official_format",
        "component": COMPONENT,
        "intervention_variant": INTERVENTION_VARIANT,
        "k": K,
        "alpha": ALPHA,
        "tail": TAIL_TOKENS,
        "intervention_timing": INTERVENTION_TIMING,
        "prompt_count": len(rows),
        "attack_success_count": success_count,
        "asr": success_count / len(rows),
        "newly_defended_count": sum(
            row["transition"] == "newly_defended" for row in rows
        ),
        "still_attack_success_count": sum(
            row["transition"] == "still_attack_success" for row in rows
        ),
        "newly_worsened_count": sum(
            row["transition"] == "newly_worsened" for row in rows
        ),
        "still_defended_count": sum(
            row["transition"] == "still_defended" for row in rows
        ),
        "response_changed_count": sum(not row["response_equal"] for row in rows),
        "first_token_argmax_changed_count": sum(
            not row["first_token_argmax_equal"] for row in rows
        ),
        "input_ids_match_count": sum(
            bool(row["input_ids_match_reused_baseline"]) for row in rows
        ),
        "first_forward_detected_count": sum(
            item["first_forward_detected"] is True for item in diagnostics_rows
        ),
        "first_forward_intervened_count": sum(
            item["first_forward_intervened"] is True for item in diagnostics_rows
        ),
        "selected_element_intervention_count": sum(
            item["selected_element_intervention_count"] for item in diagnostics_rows
        ),
        "selected_lower_changed_count": sum(
            item["selected_lower_changed_count"] for item in diagnostics_rows
        ),
        "selected_upper_changed_count": sum(
            item["selected_upper_changed_count"] for item in diagnostics_rows
        ),
        "later_forward_intervention_count": sum(
            item["later_forward_intervention_count"] for item in diagnostics_rows
        ),
        "short_sequence_skip_count": sum(
            item["short_sequence_skip_count"] for item in diagnostics_rows
        ),
        "all_scores_finite": all(
            item["all_scores_finite"] is True for item in diagnostics_rows
        ),
        **{
            key: max(item[key] for item in diagnostics_rows)
            for key in (
                "max_constructed_lower_delta_error_float32",
                "max_constructed_upper_delta_error_float32",
                "max_constructed_pair_delta_difference_float32",
                "max_symmetric_target_error_float32",
                "max_antisymmetric_component_change_float32",
                "max_recomputed_antisymmetric_component_change_float32",
                "max_symmetric_target_error_after_cast",
                "max_antisymmetric_component_change_after_cast",
                "max_realized_pair_delta_difference_after_cast",
                "max_unselected_upper_triangle_change",
                "max_unselected_element_change",
                "max_outside_tail_change",
                "max_diagonal_change_float32",
            )
        },
        "generation_failure_count": 0,
    }
    return rows, summary


def comparison_summary(pure_summary: Mapping[str, object]) -> dict:
    return {
        "schema_version": 1,
        "conditions": [
            {
                "condition": "baseline",
                "meaning": "no intervention",
                "prompt_count": TOTAL_COUNT,
                "attack_success_count": BASELINE_SUCCESS_COUNT,
                "asr": BASELINE_SUCCESS_COUNT / TOTAL_COUNT,
                "source_sha256": EXPECTED_OFFICIAL_BASELINE_HASHES[
                    "paired_results.jsonl"
                ],
            },
            {
                "condition": "gcg_asym_top500_transfer",
                "meaning": "GCG Asym Top-500/reference transfer",
                "prompt_count": TOTAL_COUNT,
                "attack_success_count": ASYM_TRANSFER_SUCCESS_COUNT,
                "asr": ASYM_TRANSFER_SUCCESS_COUNT / TOTAL_COUNT,
                "source_sha256": EXPECTED_ASYM_TRANSFER_HASHES[
                    "defense_results.jsonl"
                ],
            },
            {
                **dict(pure_summary),
                "condition": "gcg_pure_sym_top500_transfer",
                "meaning": "GCG symmetric Top-500/reference with pairwise +delta/+delta",
            },
        ],
    }


def run_gpu(args) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA is required; refusing CPU model execution")
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    output_dir = ensure_output_absent(args.output_dir).resolve()
    frozen = validate_frozen_sources(args)
    model_path = Path(args.model).resolve()
    model_revision = validate_model_repository_head(model_path)

    seed_everything(SEED)
    model, tokenizer = load_model_and_tokenizer(model_path)
    audited_reference = load_validated_reference_artifacts(
        Path(args.artifact_dir).resolve(),
        expected_heads=EXPECTED_TOP5,
        expected_selection_manifest_sha256=EXPECTED_SELECTION_MANIFEST_SHA256,
        expected_score_artifact_sha256=EXPECTED_SCORE_ARTIFACT_SHA256,
        expected_model_path=model_path,
    )
    model_validation = validate_loaded_llama_model(model, audited_reference)
    records = format_autodan_records(
        frozen["raw_records"], tokenizer, prompt_format=OFFICIAL_FORMAT
    )
    baseline_by_index = {
        int(row["dataset_index"]): row for row in frozen["baseline_rows"]
    }
    if len(baseline_by_index) != TOTAL_COUNT:
        raise ValueError("duplicate index in official baseline")
    for record in records:
        baseline = baseline_by_index[int(record["dataset_index"])]
        if (
            record["formatted_prompt"] != baseline["formatted_prompt"]
            or serialized_input_ids(record["input_ids"]) != baseline["input_ids"]
        ):
            raise ValueError(
                "official formatted prompt/input mismatch: "
                f"dataset_index={record['dataset_index']}"
            )

    output_dir.mkdir(parents=True)
    preflight = cpu_validation_payload(args)
    preflight["official_format_text_and_input_ids_reproduced_count"] = TOTAL_COUNT
    preflight_path = output_dir / "cpu_preflight.json"
    write_json(preflight_path, preflight)

    dependency_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "scripts/evaluate_gcg_pure_symmetric_position_element_topk_intervention.py",
        REPO_ROOT / "scripts/evaluate_gcg_position_element_topk_intervention.py",
        REPO_ROOT / "utils/gcg_position_element_topk_intervention.py",
        REPO_ROOT / "utils/reference_asym_score_intervention.py",
    )
    manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at": utc_now(),
        "branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "head": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "dirty_status": git_value(REPO_ROOT, "status", "--porcelain").splitlines(),
        "component": COMPONENT,
        "intervention_variant": INTERVENTION_VARIANT,
        "decomposition": DECOMPOSITION,
        "intervention": RAW_SCORE_INTERVENTION,
        "delta_definition": DELTA_DEFINITION,
        "antisymmetric_preservation_invariant": "A_asym' = A_asym in float32 construction",
        "candidate_definition": "32 layers x 32 heads x tail16 strict lower i>j (120/head)",
        "candidate_count": 122880,
        "selection_reused_without_recomputation": True,
        "selection_gpu_extraction_performed": False,
        "selection_or_reference_from_autodan": False,
        "ranking_definition": SYMMETRIC_RANKING_METRIC,
        "reference_definition": "GCG Selection20 original symmetric-score mean",
        "k": K,
        "alpha": ALPHA,
        "tail": TAIL_TOKENS,
        "timing": INTERVENTION_TIMING,
        "official_format": autodan_prompt_format_provenance(OFFICIAL_FORMAT),
        "index_10_unreplaced_placeholder_retained": True,
        "model": str(model_path),
        "model_revision": model_revision,
        "model_config": llama_model_config_summary(model),
        "model_validation": model_validation,
        "gpu": torch.cuda.get_device_name(0),
        "generation_settings": {
            "torch_dtype": "torch.float16",
            "interpolation_dtype": "torch.float32",
            "attention_implementation": "eager",
            "use_cache": False,
            "batch_size": 1,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": 32,
            "seed": SEED,
        },
        "reused_gcg_sym_selection_reference": frozen["sym_provenance"],
        "reused_autodan_source": frozen["raw_provenance"],
        "reused_official_baseline_artifacts": frozen["baseline_hashes"],
        "reused_gcg_asym_transfer_artifacts": frozen["asym_hashes"],
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "dependency_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in dependency_paths
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)

    started = start_cuda_measurement()
    rows, pure_summary = run_condition(
        records,
        baseline_by_index,
        model=model,
        tokenizer=tokenizer,
        selection=frozen["selection"],
    )
    pure_summary.update(finish_cuda_measurement(started))
    results_path = output_dir / "prompt_results.jsonl"
    summary_path = output_dir / "summary.json"
    comparison_path = output_dir / "comparison_summary.json"
    evaluation_path = output_dir / "evaluation_summary.json"
    write_jsonl(results_path, rows)
    write_json(summary_path, pure_summary)
    comparison = comparison_summary(pure_summary)
    write_json(comparison_path, comparison)
    evaluation = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "comparison": comparison,
    }
    write_json(evaluation_path, evaluation)

    artifact_paths = (
        preflight_path,
        results_path,
        summary_path,
        comparison_path,
        evaluation_path,
    )
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    write_json(
        artifact_manifest_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "artifacts": [
                {
                    "relative_path": str(path.relative_to(output_dir)),
                    "sha256": sha256_file(path),
                }
                for path in artifact_paths
            ],
        },
    )
    saved = read_json(manifest_path)
    saved["status"] = "complete"
    saved["completed_at"] = evaluation["completed_at"]
    saved["artifact_manifest_sha256"] = sha256_file(artifact_manifest_path)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(saved, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, manifest_path)
    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "summary": pure_summary,
        "comparison": comparison,
    }


def main(argv=None):
    args = parse_args(argv)
    result = (
        cpu_validation_payload(args)
        if args.stage == "cpu-validate"
        else run_gpu(args)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
