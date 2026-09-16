#!/usr/bin/env python3
"""Evaluate pure pairwise Sym Top-500 using frozen, reused Sym selection artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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
    format_records,
    generate_with_first_logits,
    load_model_and_tokenizer,
)
from scripts.evaluate_gcg_symmetric_position_element_topk_intervention import (  # noqa: E402
    DEFAULT_ASYM_RESULTS,
    DEFAULT_ATTACK_RUN,
    DEFAULT_INDEPENDENT_ADAPTER,
    DEFAULT_INDEPENDENT_CONFIG,
    DEFAULT_INDEPENDENT_INPUT_MANIFEST,
    DEFAULT_REUSED_BASELINE,
    EXPECTED_ASYM_RESULTS_SHA256,
    EXPECTED_REUSED_BASELINE_SHA256,
    load_reused_baseline,
    serialized_input_ids,
)
from utils.experiment_runtime import git_value, seed_everything  # noqa: E402
from utils.gcg_position_element_topk_intervention import (  # noqa: E402
    PositionElementSelection,
    SYMMETRIC_RANKING_METRIC,
    patch_llama_pure_symmetric_position_element_attention,
)
from utils.gcg_reference_intervention_diagnostics import compare_token_sequences  # noqa: E402
from utils.gcg_wasserstein_reference import (  # noqa: E402
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
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
from utils.harmbench_gcg_independent_test import (  # noqa: E402
    load_independent_evaluation_records,
    validate_fixed_config,
    validate_formatted_evaluation_identity,
    validate_harmbench_revision,
    validate_input_manifest,
    validate_model_revision,
    verify_all_configured_sources,
)
from utils.reference_asym_score_intervention import (  # noqa: E402
    load_validated_reference_artifacts,
    validate_loaded_llama_model,
)


COMPONENT = "symmetric"
INTERVENTION_VARIANT = "sym_pure_pairwise_delta"
HISTORICAL_SYM_VARIANT = "sym_lower_only_2delta"
K = 500
ALPHA = 1.0
INTERVENTION_TIMING = "first_token_only"
DECOMPOSITION = "(S + S^T) / 2"
DELTA_DEFINITION = "alpha*(reference_sym_ij-current_sym_ij)"
RAW_SCORE_INTERVENTION = "S_ij += delta, S_ji += delta"

DEFAULT_REUSED_SYM_RUN = (
    "data/outputs/gcg_symmetric_position_element_topk/llama2/independent100/"
    "evaluation_20260916T0622Z"
)
EXPECTED_REUSED_SYM_HASHES = {
    "artifact_manifest.json": "145080123b8d6d086c661d407dbde0f233599bbe7d233155930a103ba5f456ba",
    "run_manifest.json": "a5ead28970042210c17d4e089d19683c8d2f517cba8bdb867b2d045fb02b13af",
    "selection_original_symmetric_scores.pt": "3b3bbc129b6215a129f8b5b4b37395b8fdae118b8272a9ea5bff1ffda39dcc06",
    "selection_jailbreak_symmetric_scores.pt": "8070697bed211a36573f391d6b8193b4e7acdcf21fdeeebed2941749d0cbd8d5",
    "selection20_original_symmetric_mean_reference.pt": "80c0685759fafcb407d43cc9e4ac525a236cc647e7331ec1a53a025472bafc5d",
    "symmetric_full_ranking.pt": "57035039efd26a080beefed5af67cc407422068911a01f5d042867ad54afdeb2",
    "symmetric_top1000.jsonl": "f4c0d9cb930da37d019f714cbf1bc85387c11d450697736fcb08193cc14f926f",
    "symmetric_selection_summary.json": "12af23bbfb0734301cf447e3e9339c40609f5d976feddf027fc7476831cce49b",
    "independent100/k_500/prompt_results.jsonl": "f772ed0a343dca2fab2cbf4d7eb0716211e15b287f1da8a4edb09e6ed7dc17d5",
    "independent100/k_500/summary.json": "2537b9a4755fbe5b1d9bd98298c1903f787cc6ceff6098557b165eae44871ac9",
    "evaluation_summary.json": "d92ec3aab7783cc458d517e8ea5e9644aaff500301e1b7cd129eaa6d0a7b96d4",
}

REQUIRED_EXACT_RUNTIME_INVARIANTS = {
    "component": COMPONENT,
    "intervention_variant": INTERVENTION_VARIANT,
    "first_forward_detected": True,
    "first_forward_intervened": True,
    "later_forward_intervention_count": 0,
    "short_sequence_skip_count": 0,
    "all_scores_finite": True,
    "max_constructed_lower_delta_error_float32": 0.0,
    "max_constructed_upper_delta_error_float32": 0.0,
    "max_constructed_pair_delta_difference_float32": 0.0,
    "max_antisymmetric_component_change_float32": 0.0,
    "max_unselected_upper_triangle_change": 0.0,
    "max_unselected_element_change": 0.0,
    "max_outside_tail_change": 0.0,
    "max_diagonal_change_float32": 0.0,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu-validate", "run"), default="cpu-validate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--reused-sym-run", default=DEFAULT_REUSED_SYM_RUN)
    parser.add_argument("--independent-config", default=DEFAULT_INDEPENDENT_CONFIG)
    parser.add_argument("--independent-input-manifest", default=DEFAULT_INDEPENDENT_INPUT_MANIFEST)
    parser.add_argument("--independent-attack-run-dir", default=DEFAULT_ATTACK_RUN)
    parser.add_argument("--independent-adapter", default=DEFAULT_INDEPENDENT_ADAPTER)
    parser.add_argument("--reused-baseline", default=DEFAULT_REUSED_BASELINE)
    parser.add_argument("--asym-results", default=DEFAULT_ASYM_RESULTS)
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def _validate_hashes(root: Path, expected: dict[str, str]) -> dict[str, dict]:
    verified = {}
    for relative, expected_sha in expected.items():
        path = root / relative
        observed = sha256_file(path)
        if observed != expected_sha:
            raise ValueError(
                f"reused Sym artifact SHA-256 mismatch: {relative}: "
                f"{observed} != {expected_sha}"
            )
        verified[relative] = {"path": str(path), "sha256": observed}
    return verified


def load_reused_sym_selection(sym_run: Path) -> tuple[PositionElementSelection, dict]:
    """Load the exact historical Sym Top-500/reference without score extraction."""

    sym_run = sym_run.resolve()
    verified = _validate_hashes(sym_run, EXPECTED_REUSED_SYM_HASHES)
    run_manifest = read_json(sym_run / "run_manifest.json")
    if run_manifest.get("status") != "complete" or run_manifest.get("component") != COMPONENT:
        raise ValueError("historical Sym run manifest identity mismatch")
    old_summary = read_json(sym_run / "independent100/k_500/summary.json")
    if (
        old_summary.get("attack_success_count") != 34
        or old_summary.get("prompt_count") != 100
        or old_summary.get("k") != K
        or old_summary.get("alpha") != ALPHA
    ):
        raise ValueError("historical sym_lower_only_2delta result mismatch")

    selection_summary = read_json(sym_run / "symmetric_selection_summary.json")
    diagnostics = selection_summary.get("selection")
    if not isinstance(diagnostics, dict):
        raise ValueError("reused symmetric selection diagnostics missing")
    if (
        diagnostics.get("component") != COMPONENT
        or diagnostics.get("candidate_element_count") != 122880
        or diagnostics.get("selection_prompt_count") != 20
        or diagnostics.get("tail_tokens") != TAIL_TOKENS
        or diagnostics.get("ranking_metric") != SYMMETRIC_RANKING_METRIC
    ):
        raise ValueError("reused symmetric selection conditions mismatch")
    top500 = selection_summary.get("top500")
    if not isinstance(top500, list) or len(top500) != K:
        raise ValueError("reused Sym Top-500 is missing or has wrong cardinality")
    top1000 = read_jsonl(sym_run / "symmetric_top1000.jsonl")
    if len(top1000) != 1000 or top1000[:K] != top500:
        raise ValueError("reused Sym Top-500/Top-1000 identity mismatch")

    ranking = torch.load(
        sym_run / "symmetric_full_ranking.pt",
        map_location="cpu",
        weights_only=True,
    )
    ranks = ranking.get("rank_by_flat_index")
    scores = ranking.get("ranking_score_by_flat_index")
    mean_delta = ranking.get("mean_delta_by_flat_index")
    total = NUM_LAYERS * NUM_HEADS * ELEMENTS_PER_HEAD
    if not all(
        isinstance(value, torch.Tensor) and tuple(value.shape) == (total,)
        for value in (ranks, scores, mean_delta)
    ):
        raise ValueError("reused full Sym ranking tensor shape mismatch")
    if set(int(value) for value in ranks.tolist()) != set(range(1, total + 1)):
        raise ValueError("reused full Sym ranking is not a permutation")
    for expected_rank, record in enumerate(top1000, 1):
        flat = int(record["flat_index"])
        if int(ranks[flat].item()) != expected_rank:
            raise ValueError("reused Top-1000 rank disagrees with full ranking")
        if float(scores[flat].item()) != float(record["ranking_score"]):
            raise ValueError("reused Top-1000 score disagrees with full ranking")

    reference_payload = torch.load(
        sym_run / "selection20_original_symmetric_mean_reference.pt",
        map_location="cpu",
        weights_only=True,
    )
    reference = reference_payload.get("mean")
    if (
        not isinstance(reference, torch.Tensor)
        or tuple(reference.shape) != (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)
        or reference.dtype != torch.float32
        or reference.device.type != "cpu"
        or not bool(torch.isfinite(reference).all().item())
    ):
        raise ValueError("reused Sym reference tensor invalid")
    metadata = reference_payload.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("component") != COMPONENT
        or metadata.get("selection_manifest_sha256")
        != EXPECTED_SELECTION_MANIFEST_SHA256
    ):
        raise ValueError("reused Sym reference metadata mismatch")

    mask = torch.zeros(
        (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD), dtype=torch.bool
    )
    identities = set()
    for rank, record in enumerate(top500, 1):
        layer = int(record["layer"])
        head = int(record["head"])
        position = int(record["position"])
        identity = (layer, head, position)
        if identity in identities or int(record["rank"]) != rank:
            raise ValueError("duplicate or unordered identity in reused Sym Top-500")
        identities.add(identity)
        mask[identity] = True
    if int(mask.sum().item()) != K:
        raise ValueError("reused Sym Top-500 mask cardinality mismatch")
    selection = PositionElementSelection(
        ranking=tuple(dict(row) for row in top1000),
        masks={K: mask.contiguous()},
        original_mean_reference=reference.contiguous(),
        k_values=(K,),
        component=COMPONENT,
    )
    provenance = {
        "source_run": str(sym_run),
        "artifact_hashes": verified,
        "candidate_count": total,
        "selected_count": int(mask.sum().item()),
        "selection_prompt_count": diagnostics["selection_prompt_count"],
        "ranking_definition": diagnostics["ranking_metric"],
        "reference_definition": selection_summary["reference_definition"],
        "historical_intervention_variant": HISTORICAL_SYM_VARIANT,
        "historical_attack_success_count": old_summary["attack_success_count"],
    }
    return selection, provenance


def validate_prompt_runtime_invariants(diagnostics: dict) -> None:
    if diagnostics.get("selected_element_intervention_count") != K:
        raise RuntimeError("selected_element_intervention_count must equal 500")
    for field in ("selected_lower_changed_count", "selected_upper_changed_count"):
        observed = diagnostics.get(field)
        if type(observed) is not int or not 1 <= observed <= K:
            raise RuntimeError(f"{field} must be in [1, 500]")
    target_error = diagnostics.get("max_symmetric_target_error_float32")
    if type(target_error) not in (int, float) or not 0 <= target_error <= 1e-5:
        raise RuntimeError("max_symmetric_target_error_float32 exceeds tolerance")
    recomputed_asym_error = diagnostics.get(
        "max_recomputed_antisymmetric_component_change_float32"
    )
    if (
        type(recomputed_asym_error) not in (int, float)
        or not 0 <= recomputed_asym_error <= 1e-5
    ):
        raise RuntimeError(
            "max_recomputed_antisymmetric_component_change_float32 exceeds tolerance"
        )
    for field, expected in REQUIRED_EXACT_RUNTIME_INVARIANTS.items():
        observed = diagnostics.get(field)
        if isinstance(expected, bool):
            matches = observed is expected
        elif isinstance(expected, str):
            matches = observed == expected
        else:
            matches = type(observed) in (int, float) and observed == expected
        if not matches:
            raise RuntimeError(
                f"runtime invariant failed: {field}={observed!r}, expected={expected!r}"
            )


def validate_frozen_sources(args) -> dict:
    artifact_dir = Path(args.artifact_dir).resolve()
    verification = verify_artifact_directory(artifact_dir)
    if (
        verification.get("verified") is not True
        or verification.get("selection_manifest_sha256")
        != EXPECTED_SELECTION_MANIFEST_SHA256
        or verification.get("score_artifact_sha256")
        != EXPECTED_SCORE_ARTIFACT_SHA256
    ):
        raise ValueError("fixed Selection20 artifact verification failed")
    selection, sym_provenance = load_reused_sym_selection(
        Path(args.reused_sym_run)
    )
    config = read_json(Path(args.independent_config).resolve())
    validate_fixed_config(config)
    verify_all_configured_sources(config)
    validate_harmbench_revision(config)
    model_revision = validate_model_revision(config)
    input_rows = validate_input_manifest(
        read_json(Path(args.independent_input_manifest).resolve()), config
    )
    baseline, baseline_verification = load_reused_baseline(
        Path(args.reused_baseline).resolve(), Path(args.asym_results).resolve()
    )
    attack_run = Path(args.independent_attack_run_dir).resolve()
    final_manifest = read_json(attack_run / "final_manifest.json")
    for key, expected in (
        ("status", "complete_validated"),
        ("complete_count", 100),
        ("missing_count", 0),
        ("duplicate_count", 0),
        ("processing_failure_history_count", 0),
    ):
        if final_manifest.get(key) != expected:
            raise ValueError(f"frozen attack final manifest {key} mismatch")
    return {
        "selection": selection,
        "sym_provenance": sym_provenance,
        "config": config,
        "model_revision": model_revision,
        "independent_input_count": len(input_rows),
        "baseline_rows": baseline,
        "baseline_verification": baseline_verification,
        "artifact_verification": verification,
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
        "antisymmetric_preservation": "A_asym' = A_asym in float32 construction",
        "candidate_count": frozen["sym_provenance"]["candidate_count"],
        "selected_count": frozen["sym_provenance"]["selected_count"],
        "selection_reused_without_recomputation": True,
        "selection_gpu_extraction_performed": False,
        "sym_selection_provenance": frozen["sym_provenance"],
        "independent_input_count": frozen["independent_input_count"],
        "baseline_verification": frozen["baseline_verification"],
        "model_revision": frozen["model_revision"],
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
                    f"reused baseline input ID mismatch: dataset_index={record['dataset_index']}"
                )
            generated = generate_with_first_logits(model, tokenizer, record["input_ids"])
            if generated["generation_failed"]:
                raise RuntimeError(
                    f"generation failed: dataset_index={record['dataset_index']}"
                )
            diagnostics = controller.summary()
            validate_prompt_runtime_invariants(diagnostics)
            comparison = compare_token_sequences(
                baseline["response_token_ids"], generated["response_token_ids"]
            )
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
                    "dataset_group": "independent100",
                    "prompt_id": record["prompt_id"],
                    "row_position": record["row_position"],
                    "dataset_index": record["dataset_index"],
                    "prompt": record["prompt"],
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
                    **{key: value for key, value in generated.items() if key != "first_logits"},
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
                f"[independent100 sym_pure_pairwise_delta {position}/100] "
                f"dataset_index={record['dataset_index']} success={generated['success']}",
                flush=True,
            )
    summary = {
        "dataset_group": "independent100",
        "component": COMPONENT,
        "intervention_variant": INTERVENTION_VARIANT,
        "k": K,
        "alpha": ALPHA,
        "prompt_count": len(rows),
        "attack_success_count": sum(bool(row["success"]) for row in rows),
        "asr": sum(bool(row["success"]) for row in rows) / len(rows),
        "newly_defended_count": sum(row["transition"] == "newly_defended" for row in rows),
        "still_attack_success_count": sum(row["transition"] == "still_attack_success" for row in rows),
        "newly_worsened_count": sum(row["transition"] == "newly_worsened" for row in rows),
        "still_defended_count": sum(row["transition"] == "still_defended" for row in rows),
        "response_changed_count": sum(not row["response_equal"] for row in rows),
        "first_token_argmax_changed_count": sum(not row["first_token_argmax_equal"] for row in rows),
        "input_ids_match_count": sum(bool(row["input_ids_match_reused_baseline"]) for row in rows),
        "first_forward_detected_count": sum(item["first_forward_detected"] is True for item in diagnostics_rows),
        "first_forward_intervened_count": sum(item["first_forward_intervened"] is True for item in diagnostics_rows),
        "selected_element_intervention_count": sum(item["selected_element_intervention_count"] for item in diagnostics_rows),
        "selected_lower_changed_count": sum(item["selected_lower_changed_count"] for item in diagnostics_rows),
        "selected_upper_changed_count": sum(item["selected_upper_changed_count"] for item in diagnostics_rows),
        "later_forward_intervention_count": sum(item["later_forward_intervention_count"] for item in diagnostics_rows),
        "short_sequence_skip_count": sum(item["short_sequence_skip_count"] for item in diagnostics_rows),
        "all_scores_finite": all(item["all_scores_finite"] is True for item in diagnostics_rows),
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


def _condition_summary(rows, *, condition: str) -> dict:
    return {
        "condition": condition,
        "prompt_count": len(rows),
        "attack_success_count": sum(bool(row["success"]) for row in rows),
        "asr": sum(bool(row["success"]) for row in rows) / len(rows),
        "source_only": True,
    }


def run_gpu(args) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA is required; refusing CPU model execution")
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    output_dir = ensure_output_absent(args.output_dir).resolve()
    frozen = validate_frozen_sources(args)
    config = frozen["config"]
    model_path = Path(args.model).resolve()
    if Path(config["model"]["local_path"]).resolve() != model_path:
        raise ValueError("model path differs from frozen independent100 config")

    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(model_path)
    audited_reference = load_validated_reference_artifacts(
        Path(args.artifact_dir).resolve(),
        expected_heads=EXPECTED_TOP5,
        expected_selection_manifest_sha256=EXPECTED_SELECTION_MANIFEST_SHA256,
        expected_score_artifact_sha256=EXPECTED_SCORE_ARTIFACT_SHA256,
        expected_model_path=model_path,
    )
    model_validation = validate_loaded_llama_model(model, audited_reference)
    independent_records = load_independent_evaluation_records(
        config_path=Path(args.independent_config).resolve(),
        input_manifest_path=Path(args.independent_input_manifest).resolve(),
        attack_run_dir=Path(args.independent_attack_run_dir).resolve(),
        adapter_path=Path(args.independent_adapter).resolve(),
    )
    independent_records = format_records(independent_records, tokenizer)
    validate_formatted_evaluation_identity(independent_records)
    baseline_by_index = {
        int(row["dataset_index"]): row for row in frozen["baseline_rows"]
    }
    if len(baseline_by_index) != 100:
        raise ValueError("duplicate dataset index in reused baseline")
    for record in independent_records:
        baseline = baseline_by_index.get(int(record["dataset_index"]))
        if baseline is None or serialized_input_ids(record["input_ids"]) != baseline["input_ids"]:
            raise ValueError(
                f"formatted independent input mismatch at dataset_index={record['dataset_index']}"
            )

    output_dir.mkdir(parents=True)
    dependency_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "utils/gcg_position_element_topk_intervention.py",
        REPO_ROOT / "utils/reference_asym_score_intervention.py",
        REPO_ROOT / "scripts/diagnose_gcg_reference_factorization.py",
        REPO_ROOT / "scripts/evaluate_gcg_symmetric_position_element_topk_intervention.py",
        Path(args.independent_adapter).resolve(),
    ]
    attack_run = Path(args.independent_attack_run_dir).resolve()
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
        "ranking_definition": SYMMETRIC_RANKING_METRIC,
        "reference_definition": "Selection20 original symmetric-score mean",
        "k": K,
        "alpha": ALPHA,
        "tail": TAIL_TOKENS,
        "timing": INTERVENTION_TIMING,
        "model": str(model_path),
        "model_revision": config["model"]["revision"],
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
            "max_new_tokens": 32,
            "seed": 42,
        },
        "reused_sym_selection_reference": frozen["sym_provenance"],
        "reused_artifacts": {
            "baseline": {
                "path": str(Path(args.reused_baseline).resolve()),
                "sha256": EXPECTED_REUSED_BASELINE_SHA256,
            },
            "asym_top500_results": {
                "path": str(Path(args.asym_results).resolve()),
                "sha256": EXPECTED_ASYM_RESULTS_SHA256,
            },
            "sym_lower_only_2delta_results": frozen["sym_provenance"]["artifact_hashes"][
                "independent100/k_500/prompt_results.jsonl"
            ],
            "independent_attack_run_manifest": {
                "path": str(attack_run / "run_manifest.json"),
                "sha256": sha256_file(attack_run / "run_manifest.json"),
            },
            "independent_attack_final_manifest": {
                "path": str(attack_run / "final_manifest.json"),
                "sha256": sha256_file(attack_run / "final_manifest.json"),
            },
            "independent_input_manifest": {
                "path": str(Path(args.independent_input_manifest).resolve()),
                "sha256": sha256_file(Path(args.independent_input_manifest).resolve()),
            },
        },
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "dependency_sha256": {
            str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path): sha256_file(path)
            for path in dependency_paths
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)

    started = start_cuda_measurement()
    rows, pure_summary = run_condition(
        independent_records,
        baseline_by_index,
        model=model,
        tokenizer=tokenizer,
        selection=frozen["selection"],
    )
    pure_summary.update(finish_cuda_measurement(started))
    condition_dir = output_dir / "independent100" / "sym_pure_pairwise_delta"
    condition_dir.mkdir(parents=True)
    results_path = condition_dir / "prompt_results.jsonl"
    summary_path = condition_dir / "summary.json"
    write_jsonl(results_path, rows)
    write_json(summary_path, pure_summary)

    baseline_summary = {
        "condition": "baseline",
        "meaning": "no intervention",
        "prompt_count": 100,
        "attack_success_count": 53,
        "asr": 0.53,
        "source_sha256": EXPECTED_REUSED_BASELINE_SHA256,
    }
    asym_rows = read_jsonl(Path(args.asym_results).resolve())
    asym_summary = {
        **_condition_summary(asym_rows, condition="asym_top500"),
        "meaning": "existing pure antisymmetric intervention",
        "source_sha256": EXPECTED_ASYM_RESULTS_SHA256,
    }
    old_sym_path = (
        Path(args.reused_sym_run).resolve()
        / "independent100/k_500/prompt_results.jsonl"
    )
    old_sym_rows = read_jsonl(old_sym_path)
    old_sym_summary = {
        **_condition_summary(old_sym_rows, condition=HISTORICAL_SYM_VARIANT),
        "meaning": "historical lower-only 2*delta symmetric-target intervention",
        "component": COMPONENT,
        "intervention_variant": HISTORICAL_SYM_VARIANT,
        "source_sha256": EXPECTED_REUSED_SYM_HASHES[
            "independent100/k_500/prompt_results.jsonl"
        ],
    }
    comparison = {
        "schema_version": 1,
        "conditions": [
            baseline_summary,
            asym_summary,
            old_sym_summary,
            {
                **pure_summary,
                "condition": INTERVENTION_VARIANT,
                "meaning": "pure symmetric pairwise +delta/+delta intervention",
            },
        ],
    }
    comparison_path = output_dir / "independent100_comparison_summary.json"
    write_json(comparison_path, comparison)
    evaluation = {
        "status": "complete",
        "completed_at": utc_now(),
        "comparison": comparison,
    }
    evaluation_path = output_dir / "evaluation_summary.json"
    write_json(evaluation_path, evaluation)
    artifact_paths = [results_path, summary_path, comparison_path, evaluation_path]
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
    return evaluation


def main(argv=None):
    args = parse_args(argv)
    result = cpu_validation_payload(args) if args.stage == "cpu-validate" else run_gpu(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
