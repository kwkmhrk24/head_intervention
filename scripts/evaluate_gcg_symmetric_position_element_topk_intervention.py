#!/usr/bin/env python3
"""Build Sym Top-500 from fixed GCG Selection20 and evaluate frozen independent100."""

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
    DEFAULT_JAILBREAK_INPUT,
    DEFAULT_MODEL,
    DEFAULT_SELECTION_ORIGINAL,
    EXPECTED_SCORE_ARTIFACT_SHA256,
    EXPECTED_SELECTION_MANIFEST_SHA256,
    format_records,
    generate_with_first_logits,
    load_model_and_tokenizer,
)
from scripts.evaluate_gcg_position_element_topk_intervention import (  # noqa: E402
    DEFAULT_INDEPENDENT_ADAPTER,
    DEFAULT_INDEPENDENT_CONFIG,
    DEFAULT_INDEPENDENT_INPUT_MANIFEST,
    serialized_input_ids,
)
from utils.experiment_runtime import git_value, seed_everything  # noqa: E402
from utils.gcg_position_element_topk_intervention import (  # noqa: E402
    SYMMETRIC_RANKING_METRIC,
    build_position_element_selection,
    patch_llama_symmetric_position_element_attention,
    selection_diagnostics,
)
from utils.gcg_reference_intervention_diagnostics import compare_token_sequences  # noqa: E402
from utils.gcg_wasserstein_reference import (  # noqa: E402
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    TAIL_TOKENS,
    extract_selection_scores,
    llama_model_config_summary,
    read_json,
    read_jsonl,
    save_torch_artifact,
    selection_records,
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


K = 500
ALPHA = 1.0
INTERVENTION_TIMING = "first_token_only"
COMPONENT = "symmetric"
DECOMPOSITION = "(S + S^T) / 2"
RAW_SCORE_INTERVENTION = (
    "S_ij' = S_ij + 2*alpha*(reference_sym_ij-current_sym_ij), "
    "for selected tail16 strict-lower i>j only; paired S_ji is unchanged"
)
DEFAULT_ATTACK_RUN = (
    "data/outputs/gcg_independent_test/llama2/advbench_220_319/"
    "gcg_v1_55c5098_20260910T011413Z"
)
DEFAULT_REUSED_BASELINE = (
    "data/outputs/gcg_wasserstein_reference_intervention/llama2/independent100/"
    "evaluation_20260911T064911Z/independent100_baseline.jsonl"
)
DEFAULT_ASYM_RESULTS = (
    "data/outputs/gcg_wasserstein_reference_intervention/llama2/independent100/"
    "evaluation_20260911T064911Z/independent100/k_500/prompt_results.jsonl"
)
EXPECTED_REUSED_BASELINE_SHA256 = (
    "39e7cc6d727a988a25ddbe0829a76d4c1162801589b515b01bf16e0c950fafde"
)
EXPECTED_ASYM_RESULTS_SHA256 = (
    "d857f4faa916ddfa2b2eccfcc5b52de03d7b9e52779d59c276564be885aa58d3"
)

REQUIRED_RUNTIME_INVARIANTS = {
    "component": COMPONENT,
    "first_forward_detected": True,
    "first_forward_intervened": True,
    "later_forward_intervention_count": 0,
    "short_sequence_skip_count": 0,
    "all_scores_finite": True,
    "max_unselected_element_change": 0.0,
    "max_outside_tail_change": 0.0,
    "max_upper_triangle_change": 0.0,
    "max_diagonal_change_float32": 0.0,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu-validate", "run"), default="cpu-validate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--selection-original", default=DEFAULT_SELECTION_ORIGINAL)
    parser.add_argument("--selection-jailbreak", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--independent-config", default=DEFAULT_INDEPENDENT_CONFIG)
    parser.add_argument("--independent-input-manifest", default=DEFAULT_INDEPENDENT_INPUT_MANIFEST)
    parser.add_argument("--independent-attack-run-dir", default=DEFAULT_ATTACK_RUN)
    parser.add_argument("--independent-adapter", default=DEFAULT_INDEPENDENT_ADAPTER)
    parser.add_argument("--reused-baseline", default=DEFAULT_REUSED_BASELINE)
    parser.add_argument("--asym-results", default=DEFAULT_ASYM_RESULTS)
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def _identity(record: dict) -> tuple[int, int, int]:
    return int(record["layer"]), int(record["head"]), int(record["position"])


def structural_comparison(asym_ranking, sym_ranking) -> dict:
    """Compare fixed rankings without using the comparison to alter selection."""

    expected = NUM_LAYERS * NUM_HEADS * ELEMENTS_PER_HEAD
    if len(asym_ranking) != expected or len(sym_ranking) != expected:
        raise ValueError("structural comparison requires both complete rankings")

    overlaps = {}
    for k in (100, 500, 1000):
        asym_ids = {_identity(row) for row in asym_ranking[:k]}
        sym_ids = {_identity(row) for row in sym_ranking[:k]}
        overlap = len(asym_ids & sym_ids)
        overlaps[str(k)] = {
            "overlap_count": overlap,
            "jaccard": overlap / (2 * k - overlap),
        }

    def distribution(rows, keys, domains):
        result = []
        for values in domains:
            if not isinstance(values, tuple):
                values = (values,)
            count = sum(
                all(int(row[key]) == value for key, value in zip(keys, values))
                for row in rows[:500]
            )
            result.append({**dict(zip(keys, values)), "count": count})
        return result

    by_flat_asym = {int(row["flat_index"]): float(row["ranking_score"]) for row in asym_ranking}
    by_flat_sym = {int(row["flat_index"]): float(row["ranking_score"]) for row in sym_ranking}
    if set(by_flat_asym) != set(range(expected)) or set(by_flat_sym) != set(range(expected)):
        raise ValueError("ranking flat-index coverage mismatch")
    asym_scores = torch.tensor([by_flat_asym[index] for index in range(expected)], dtype=torch.float64)
    sym_scores = torch.tensor([by_flat_sym[index] for index in range(expected)], dtype=torch.float64)
    pearson = float(torch.corrcoef(torch.stack((asym_scores, sym_scores)))[0, 1].item())
    if not torch.isfinite(torch.tensor(pearson)):
        raise ValueError("non-finite Pearson correlation")
    return {
        "comparison_scope": "all 122,880 fixed candidate identities",
        "selection_was_not_modified_from_comparison": True,
        "top100_overlap": overlaps["100"]["overlap_count"],
        "top500_overlap": overlaps["500"]["overlap_count"],
        "top500_jaccard": overlaps["500"]["jaccard"],
        "top1000_overlap": overlaps["1000"]["overlap_count"],
        "overlaps": overlaps,
        "pearson_ranking_score_all_candidates": pearson,
        "pearson_definition": "Pearson correlation of abs(mean paired delta) scores aligned by flat_index",
        "rank_correlation": None,
        "rank_correlation_note": "not computed",
        "top500_layer_distribution": {
            "antisymmetric": distribution(asym_ranking, ("layer",), range(NUM_LAYERS)),
            "symmetric": distribution(sym_ranking, ("layer",), range(NUM_LAYERS)),
        },
        "top500_head_distribution": {
            "antisymmetric": distribution(
                asym_ranking,
                ("layer", "head"),
                ((layer, head) for layer in range(NUM_LAYERS) for head in range(NUM_HEADS)),
            ),
            "symmetric": distribution(
                sym_ranking,
                ("layer", "head"),
                ((layer, head) for layer in range(NUM_LAYERS) for head in range(NUM_HEADS)),
            ),
        },
        "top500_lag_distribution": {
            "antisymmetric": distribution(asym_ranking, ("lag",), range(1, TAIL_TOKENS)),
            "symmetric": distribution(sym_ranking, ("lag",), range(1, TAIL_TOKENS)),
        },
    }


def validate_prompt_runtime_invariants(diagnostics: dict) -> None:
    if diagnostics.get("selected_element_intervention_count") != K:
        raise RuntimeError("selected_element_intervention_count must equal 500")
    changed = diagnostics.get("selected_lower_changed_count")
    if type(changed) is not int or not 1 <= changed <= K:
        raise RuntimeError("selected_lower_changed_count must be in [1, 500]")
    target_error = diagnostics.get("max_symmetric_target_error_float32")
    if type(target_error) not in (int, float) or not 0.0 <= target_error <= 1e-5:
        raise RuntimeError(
            "max_symmetric_target_error_float32 exceeds float32 tolerance"
        )
    for field, expected in REQUIRED_RUNTIME_INVARIANTS.items():
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


def load_reused_baseline(baseline_path: Path, asym_path: Path) -> tuple[list[dict], dict]:
    if sha256_file(baseline_path) != EXPECTED_REUSED_BASELINE_SHA256:
        raise ValueError("reused baseline SHA-256 mismatch")
    if sha256_file(asym_path) != EXPECTED_ASYM_RESULTS_SHA256:
        raise ValueError("Asym Top-500 result SHA-256 mismatch")
    baseline = read_jsonl(baseline_path)
    asym = read_jsonl(asym_path)
    if len(baseline) != 100 or len(asym) != 100:
        raise ValueError("reused baseline/asym results must each contain 100 rows")
    asym_by_index = {int(row["dataset_index"]): row for row in asym}
    if len(asym_by_index) != 100:
        raise ValueError("duplicate dataset index in Asym Top-500 results")
    for row in baseline:
        index = int(row["dataset_index"])
        paired = asym_by_index.get(index)
        if paired is None:
            raise ValueError(f"missing Asym row for dataset_index={index}")
        if (
            row["input_ids"] != paired["input_ids"]
            or row["response"] != paired["baseline_response"]
            or row["success"] is not paired["baseline_success"]
        ):
            raise ValueError(f"baseline/Asym provenance mismatch at dataset_index={index}")
    success_count = sum(bool(row["success"]) for row in baseline)
    if success_count != 53:
        raise ValueError(f"reused baseline success count mismatch: {success_count}")
    return baseline, {
        "exact_input_ids_match_asym_count": 100,
        "exact_response_match_asym_count": 100,
        "success_label_match_asym_count": 100,
        "attack_success_count": success_count,
        "prompt_count": 100,
    }


def validate_frozen_sources(args) -> dict:
    artifact_dir = Path(args.artifact_dir).resolve()
    verification = verify_artifact_directory(artifact_dir)
    if verification.get("verified") is not True:
        raise ValueError("Selection20 artifact verification failed")
    if verification["selection_manifest_sha256"] != EXPECTED_SELECTION_MANIFEST_SHA256:
        raise ValueError("Selection20 manifest SHA-256 mismatch")
    if verification["score_artifact_sha256"] != EXPECTED_SCORE_ARTIFACT_SHA256:
        raise ValueError("existing Asym score artifact SHA-256 mismatch")
    split = read_json(artifact_dir / "split_manifest.json")
    if split.get("selection_count") != SELECTION_COUNT:
        raise ValueError("Selection20 count mismatch")
    config = read_json(Path(args.independent_config).resolve())
    validate_fixed_config(config)
    verify_all_configured_sources(config)
    validate_harmbench_revision(config)
    model_revision = validate_model_revision(config)
    inputs = validate_input_manifest(
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
        "candidate_count": NUM_LAYERS * NUM_HEADS * ELEMENTS_PER_HEAD,
        "selected_count": K,
        "selection_count": SELECTION_COUNT,
        "selection_manifest_sha256": verification["selection_manifest_sha256"],
        "asym_score_artifact_sha256": verification["score_artifact_sha256"],
        "independent_input_count": len(inputs),
        "reused_baseline": baseline_verification,
        "model_revision": model_revision,
        "baseline_rows": baseline,
        "split_manifest": split,
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
        "decomposition": DECOMPOSITION,
        "candidate_definition": "32 layers x 32 heads x tail16 strict lower i>j (120/head)",
        "candidate_count": frozen["candidate_count"],
        "selection_count": frozen["selection_count"],
        "selected_count": frozen["selected_count"],
        "ranking": SYMMETRIC_RANKING_METRIC,
        "reference": "Selection20 original symmetric-score mean",
        "raw_score_intervention": RAW_SCORE_INTERVENTION,
        "alpha": ALPHA,
        "tail": TAIL_TOKENS,
        "timing": INTERVENTION_TIMING,
        "independent_input_count": frozen["independent_input_count"],
        "reused_baseline_verification": frozen["reused_baseline"],
        "selection_manifest_sha256": frozen["selection_manifest_sha256"],
        "asym_score_artifact_sha256": frozen["asym_score_artifact_sha256"],
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
    with patch_llama_symmetric_position_element_attention(
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
                f"[independent100 sym k=500 {position}/100] "
                f"dataset_index={record['dataset_index']} success={generated['success']}",
                flush=True,
            )
    summary = {
        "dataset_group": "independent100",
        "component": COMPONENT,
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
        "later_forward_intervention_count": sum(item["later_forward_intervention_count"] for item in diagnostics_rows),
        "short_sequence_skip_count": sum(item["short_sequence_skip_count"] for item in diagnostics_rows),
        "all_scores_finite": all(item["all_scores_finite"] is True for item in diagnostics_rows),
        "max_unselected_element_change": max(item["max_unselected_element_change"] for item in diagnostics_rows),
        "max_outside_tail_change": max(item["max_outside_tail_change"] for item in diagnostics_rows),
        "max_upper_triangle_change": max(item["max_upper_triangle_change"] for item in diagnostics_rows),
        "max_diagonal_change_float32": max(item["max_diagonal_change_float32"] for item in diagnostics_rows),
        "max_symmetric_target_error_float32": max(item["max_symmetric_target_error_float32"] for item in diagnostics_rows),
        "max_symmetric_target_error_after_cast": max(item["max_symmetric_target_error_after_cast"] for item in diagnostics_rows),
        "generation_failure_count": 0,
    }
    return rows, summary


def _ranking_tensor(ranking) -> dict:
    ordered = sorted(ranking, key=lambda row: int(row["flat_index"]))
    return {
        "ranking_score_by_flat_index": torch.tensor(
            [float(row["ranking_score"]) for row in ordered], dtype=torch.float32
        ),
        "mean_delta_by_flat_index": torch.tensor(
            [float(row["mean_delta"]) for row in ordered], dtype=torch.float32
        ),
        "rank_by_flat_index": torch.tensor(
            [int(row["rank"]) for row in ordered], dtype=torch.int32
        ),
    }


def run_gpu(args) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA is required; refusing CPU model execution")
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    output_dir = ensure_output_absent(args.output_dir).resolve()
    frozen = validate_frozen_sources(args)
    model_path = Path(args.model).resolve()
    config = read_json(Path(args.independent_config).resolve())
    if Path(config["model"]["local_path"]).resolve() != model_path:
        raise ValueError("model path differs from frozen independent100 config")

    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(model_path)
    audited_reference = load_validated_reference_artifacts(
        Path(args.artifact_dir).resolve(),
        expected_heads=((31, 22), (13, 4), (5, 30), (9, 18), (27, 24)),
        expected_selection_manifest_sha256=EXPECTED_SELECTION_MANIFEST_SHA256,
        expected_score_artifact_sha256=EXPECTED_SCORE_ARTIFACT_SHA256,
        expected_model_path=model_path,
    )
    model_validation = validate_loaded_llama_model(model, audited_reference)

    original_rows = read_jsonl(Path(args.selection_original).resolve())
    jailbreak_rows = read_jsonl(Path(args.selection_jailbreak).resolve())
    selected = selection_records(frozen["split_manifest"])
    extraction_started = start_cuda_measurement()
    original_sym, jailbreak_sym, sequence_lengths = extract_selection_scores(
        model=model,
        tokenizer=tokenizer,
        original_rows=original_rows,
        jailbreak_rows=jailbreak_rows,
        selected_records=selected,
        progress=print,
        component=COMPONENT,
    )
    extraction_runtime = finish_cuda_measurement(extraction_started)
    selection = build_position_element_selection(
        original_sym,
        jailbreak_sym,
        k_values=(100, 500, 1000),
        component=COMPONENT,
    )
    if len(selection.ranking) != 122880 or int(selection.mask_for_k(K).sum()) != K:
        raise RuntimeError("candidate/selection cardinality invariant failed")

    asym_payloads = [
        torch.load(
            Path(args.artifact_dir).resolve() / name,
            map_location="cpu",
            weights_only=True,
        )["values"]
        for name in ("selection_original_scores.pt", "selection_jailbreak_scores.pt")
    ]
    asym_selection = build_position_element_selection(
        *asym_payloads,
        k_values=(100, 500, 1000),
        component="antisymmetric",
    )
    comparison = structural_comparison(asym_selection.ranking, selection.ranking)

    independent_records = load_independent_evaluation_records(
        config_path=Path(args.independent_config).resolve(),
        input_manifest_path=Path(args.independent_input_manifest).resolve(),
        attack_run_dir=Path(args.independent_attack_run_dir).resolve(),
        adapter_path=Path(args.independent_adapter).resolve(),
    )
    independent_records = format_records(independent_records, tokenizer)
    validate_formatted_evaluation_identity(independent_records)
    baseline_by_index = {int(row["dataset_index"]): row for row in frozen["baseline_rows"]}
    if len(baseline_by_index) != 100:
        raise ValueError("duplicate dataset index in reused baseline")
    for record in independent_records:
        baseline = baseline_by_index.get(int(record["dataset_index"]))
        if baseline is None or serialized_input_ids(record["input_ids"]) != baseline["input_ids"]:
            raise ValueError(
                f"formatted independent input mismatch at dataset_index={record['dataset_index']}"
            )

    output_dir.mkdir(parents=True)
    artifact_paths = []
    score_metadata = {
        "component": COMPONENT,
        "decomposition": DECOMPOSITION,
        "score_capture_position": (
            "post-RoPE, after repeat_kv, immediately after QK^T/sqrt(head_dim), "
            "pre-mask, pre-softmax"
        ),
        "score_calculation_dtype": (
            "torch.float16 Q/K matmul and scaling; S_sym computed in torch.float16"
        ),
        "stored_value_dtype": "torch.float32",
        "tail_tokens": TAIL_TOKENS,
        "candidate_definition": "tail16 strict lower i>j in row-major torch.tril_indices order",
        "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
        "selection_row_positions": frozen["split_manifest"]["selection_row_positions"],
        "selection_dataset_indices": frozen["split_manifest"]["selection_dataset_indices"],
        "sequence_lengths": sequence_lengths,
    }
    for name, condition, values in (
        ("selection_original_symmetric_scores.pt", "original", original_sym),
        ("selection_jailbreak_symmetric_scores.pt", "jailbreak", jailbreak_sym),
    ):
        path = output_dir / name
        save_torch_artifact(path, {"condition": condition, "values": values, "metadata": score_metadata})
        artifact_paths.append(path)
    reference_path = output_dir / "selection20_original_symmetric_mean_reference.pt"
    save_torch_artifact(
        reference_path,
        {"mean": selection.original_mean_reference, "metadata": score_metadata},
    )
    artifact_paths.append(reference_path)
    ranking_path = output_dir / "symmetric_full_ranking.pt"
    save_torch_artifact(
        ranking_path,
        {
            **_ranking_tensor(selection.ranking),
            "metadata": {
                **score_metadata,
                "ranking_definition": SYMMETRIC_RANKING_METRIC,
                "tie_break": "ranking score descending, layer/head/position ascending",
            },
        },
    )
    artifact_paths.append(ranking_path)
    top1000_path = output_dir / "symmetric_top1000.jsonl"
    write_jsonl(top1000_path, [dict(row) for row in selection.ranking[:1000]])
    artifact_paths.append(top1000_path)
    comparison_path = output_dir / "asym_vs_sym_structural_comparison.json"
    write_json(comparison_path, comparison)
    artifact_paths.append(comparison_path)
    selection_summary_path = output_dir / "symmetric_selection_summary.json"
    write_json(
        selection_summary_path,
        {
            "selection": selection_diagnostics(selection),
            "top500": [dict(row) for row in selection.ranking[:500]],
            "reference_definition": "mean over fixed Selection20 original symmetric scores only",
            "ranking_definition": SYMMETRIC_RANKING_METRIC,
            "raw_score_intervention": RAW_SCORE_INTERVENTION,
            "extraction_runtime": extraction_runtime,
        },
    )
    artifact_paths.append(selection_summary_path)

    dependency_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "utils/gcg_position_element_topk_intervention.py",
        REPO_ROOT / "utils/gcg_wasserstein_reference.py",
        REPO_ROOT / "utils/reference_asym_score_intervention.py",
        REPO_ROOT / "scripts/diagnose_gcg_reference_factorization.py",
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
        "decomposition": DECOMPOSITION,
        "candidate_definition": "32 layers x 32 heads x tail16 strict lower i>j (120/head)",
        "candidate_count": 122880,
        "raw_score_intervention": RAW_SCORE_INTERVENTION,
        "selection20_source": str(Path(args.artifact_dir).resolve() / "split_manifest.json"),
        "selection20_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
        "reference_definition": "Selection20 original symmetric-score mean",
        "ranking_definition": SYMMETRIC_RANKING_METRIC,
        "ranking_tie_break": "ranking score descending, layer/head/position ascending",
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
        "independent100_source": str(attack_run),
        "reused_artifacts": {
            "selection_split_manifest": {
                "path": str(Path(args.artifact_dir).resolve() / "split_manifest.json"),
                "sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
            },
            "asym_original_scores": {
                "path": str(Path(args.artifact_dir).resolve() / "selection_original_scores.pt"),
                "sha256": sha256_file(Path(args.artifact_dir).resolve() / "selection_original_scores.pt"),
            },
            "asym_jailbreak_scores": {
                "path": str(Path(args.artifact_dir).resolve() / "selection_jailbreak_scores.pt"),
                "sha256": sha256_file(Path(args.artifact_dir).resolve() / "selection_jailbreak_scores.pt"),
            },
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
            "baseline": {
                "path": str(Path(args.reused_baseline).resolve()),
                "sha256": EXPECTED_REUSED_BASELINE_SHA256,
                "verification": frozen["reused_baseline"],
            },
            "asym_top500_results": {
                "path": str(Path(args.asym_results).resolve()),
                "sha256": EXPECTED_ASYM_RESULTS_SHA256,
            },
        },
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "dependency_sha256": {
            str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path): sha256_file(path)
            for path in dependency_paths
        },
        "selection": selection_diagnostics(selection),
        "top500": [dict(row) for row in selection.ranking[:500]],
        "structural_comparison": comparison,
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)

    condition_started = start_cuda_measurement()
    rows, summary = run_condition(
        independent_records,
        baseline_by_index,
        model=model,
        tokenizer=tokenizer,
        selection=selection,
    )
    summary.update(finish_cuda_measurement(condition_started))
    condition_dir = output_dir / "independent100" / "k_500"
    condition_dir.mkdir(parents=True)
    results_path = condition_dir / "prompt_results.jsonl"
    summary_path = condition_dir / "summary.json"
    write_jsonl(results_path, rows)
    write_json(summary_path, summary)
    artifact_paths.extend((results_path, summary_path))
    evaluation_path = output_dir / "evaluation_summary.json"
    result = {
        "status": "complete",
        "completed_at": utc_now(),
        "reused_baseline": {
            "prompt_count": 100,
            "attack_success_count": 53,
            "asr": 0.53,
            "generation_failure_count": 0,
        },
        "asym_top500": {
            "prompt_count": 100,
            "attack_success_count": 38,
            "asr": 0.38,
            "newly_defended_count": 15,
            "newly_worsened_count": 0,
            "response_changed_count": 23,
            "first_token_argmax_changed_count": 23,
            "source_sha256": EXPECTED_ASYM_RESULTS_SHA256,
        },
        "symmetric_top500": summary,
        "structural_comparison": comparison,
    }
    write_json(evaluation_path, result)
    artifact_paths.append(evaluation_path)
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
    saved["completed_at"] = result["completed_at"]
    saved["artifact_manifest_sha256"] = sha256_file(artifact_manifest_path)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(saved, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, manifest_path)
    return result


def main(argv=None):
    args = parse_args(argv)
    result = cpu_validation_payload(args) if args.stage == "cpu-validate" else run_gpu(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
