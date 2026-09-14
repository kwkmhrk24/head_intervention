#!/usr/bin/env python3
"""Validate or run GCG Rank Element Top-500 v1 intervention."""

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
    DEFAULT_ADVBENCH,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BASELINE_RESULTS,
    DEFAULT_HOLDOUT_RUN,
    DEFAULT_JAILBREAK_INPUT,
    DEFAULT_MODEL,
    DEFAULT_RAW_ARCHIVE,
    DEFAULT_SELECTION_ORIGINAL,
    DEFAULT_SELECTION_RUN,
    EXPECTED_SCORE_ARTIFACT_SHA256,
    EXPECTED_SELECTION_MANIFEST_SHA256,
    EXPECTED_TOP5,
    GENERATION_SETTINGS,
    _load_group_inputs,
    format_records,
    generate_with_first_logits,
    load_model_and_tokenizer,
)
from scripts.evaluate_gcg_position_element_topk_intervention import (  # noqa: E402
    finish_cuda_measurement,
    run_baseline,
    serialized_input_ids,
    start_cuda_measurement,
    validate_first_token_outcome_invariant,
)
from utils.experiment_runtime import git_value, seed_everything  # noqa: E402
from utils.gcg_rank_element_topk_intervention import (  # noqa: E402
    TOP_K,
    RankElementTopKController,
    load_rank_selection_artifact,
    patch_llama_rank_element_attention,
)
from utils.gcg_reference_intervention_diagnostics import (  # noqa: E402
    compare_token_sequences,
    compute_logit_step_metrics,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
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


DATASET_GROUPS = ("selection20", "holdout48")
DEFAULT_DATASET_GROUPS = DATASET_GROUPS
ALPHA = 1.0
INTERVENTION_TIMING = "first_token_only"
REQUIRED_RUNTIME_INVARIANTS = {
    "first_forward_detected": True,
    "first_forward_intervened": True,
    "later_forward_intervention_count": 0,
    "short_sequence_skip_count": 0,
    "all_scores_finite": True,
    "max_unselected_element_change": 0.0,
    "max_outside_tail_change": 0.0,
    "max_diagonal_change_float32": 0.0,
    "max_constructed_delta_antisymmetry_error_float32": 0.0,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("cpu-validate", "run"),
        default="cpu-validate",
        help="cpu-validate is read-only and never loads a model.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--rank-artifact-dir",
        required=True,
        help="Frozen output from prepare_gcg_rank_element_topk.py.",
    )
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--selection-run", default=DEFAULT_SELECTION_RUN)
    parser.add_argument("--holdout-run", default=DEFAULT_HOLDOUT_RUN)
    parser.add_argument("--selection-original", default=DEFAULT_SELECTION_ORIGINAL)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--raw-archive", default=DEFAULT_RAW_ARCHIVE)
    parser.add_argument("--advbench", default=DEFAULT_ADVBENCH)
    parser.add_argument(
        "--dataset-groups",
        nargs="+",
        choices=DATASET_GROUPS,
        default=list(DEFAULT_DATASET_GROUPS),
    )
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def _validate_source_provenance(config: dict, source_dir: Path) -> dict:
    verification = verify_artifact_directory(source_dir)
    expected = {
        "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
        "score_artifact_sha256": EXPECTED_SCORE_ARTIFACT_SHA256,
    }
    for field, value in expected.items():
        if verification.get(field) != value:
            raise ValueError(f"source artifact {field} mismatch")
    if config.get("source_artifact_dir") != str(source_dir):
        raise ValueError("rank artifact source directory mismatch")
    if config.get("source_selection_manifest_sha256") != verification.get(
        "selection_manifest_sha256"
    ):
        raise ValueError("rank/source selection manifest mismatch")
    if config.get("source_score_artifact_sha256") != verification.get(
        "score_artifact_sha256"
    ):
        raise ValueError("rank/source score artifact mismatch")
    for condition, file_name in (
        ("original", "selection_original_scores.pt"),
        ("jailbreak", "selection_jailbreak_scores.pt"),
    ):
        if config.get(f"{condition}_score_sha256") != sha256_file(
            source_dir / file_name
        ):
            raise ValueError(f"rank/source {condition} score hash mismatch")
    return verification


def validate_prompt_runtime_invariants(diagnostics: dict, *, k: int = TOP_K) -> None:
    """Fail fast when a prompt violates the rank runtime contract."""

    if diagnostics.get("selected_rank_intervention_count") != k:
        raise RuntimeError(
            "runtime invariant failed: selected_rank_intervention_count "
            f"!= k ({diagnostics.get('selected_rank_intervention_count')} != {k})"
        )
    for field, expected in REQUIRED_RUNTIME_INVARIANTS.items():
        observed = diagnostics.get(field)
        matches = observed is expected if isinstance(expected, bool) else (
            type(observed) in (int, float) and observed == expected
        )
        if not matches:
            raise RuntimeError(
                f"runtime invariant failed: {field}={observed!r}, "
                f"expected={expected!r}"
            )
    crossing_count = diagnostics.get("selected_value_crossing_count")
    crossing_fraction = diagnostics.get("selected_value_crossing_fraction")
    if (
        type(crossing_count) is not int
        or not 0 <= crossing_count <= k
        or type(crossing_fraction) is not float
        or crossing_fraction != crossing_count / k
    ):
        raise RuntimeError("runtime invariant failed: crossing diagnostics")


def cpu_runtime_smoke(selection) -> dict:
    """Exercise all selected rank mappings without loading a model or GPU."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    controller = RankElementTopKController(
        reference_mean=selection.original_rank_reference,
        rank_mask=selection.mask_for_k(TOP_K),
        alpha=ALPHA,
        intervention_timing=INTERVENTION_TIMING,
    )
    controller.set_prompt_context(prompt_length=20)
    for layer in controller.selected_layers:
        scores = torch.randn(
            (1, 32, 20, 20),
            generator=generator,
            dtype=torch.float32,
        )
        before = scores.clone()
        result = controller.apply(layer, scores)
        if result is scores or torch.equal(result, before):
            raise RuntimeError(f"CPU smoke did not modify selected layer {layer}")
    later = torch.randn(
        (1, 32, 21, 21), generator=generator, dtype=torch.float32
    )
    if controller.apply(controller.selected_layers[0], later) is not later:
        raise RuntimeError("CPU smoke intervened after the first token")
    diagnostics = controller.summary()
    validate_prompt_runtime_invariants(diagnostics)
    return diagnostics


def cpu_validation_payload(args) -> dict:
    rank_artifact_dir = Path(args.rank_artifact_dir).resolve()
    source_dir = Path(args.artifact_dir).resolve()
    selection, config, diagnostics, rank_verification = (
        load_rank_selection_artifact(rank_artifact_dir)
    )
    source_verification = _validate_source_provenance(config, source_dir)
    model_path = Path(args.model).resolve()
    if Path(config.get("model", "")).resolve() != model_path:
        raise ValueError("rank artifact model mismatch")
    smoke = cpu_runtime_smoke(selection)
    top_rows = read_jsonl(
        rank_artifact_dir / "selected_rank_elements.jsonl"
    )[:10]
    return {
        "status": "cpu_validation_passed",
        "model_loaded": False,
        "gpu_used": False,
        "artifacts_written": False,
        "independent100_used": False,
        "rank_artifact_verification": rank_verification,
        "source_artifact_verified": source_verification["verified"],
        "selection": diagnostics,
        "top_rank_elements": top_rows,
        "cpu_runtime_smoke": smoke,
    }


def run_condition(
    records,
    baseline_by_index,
    *,
    model,
    tokenizer,
    selection,
    label: str,
):
    rows = []
    prompt_diagnostics = []
    with patch_llama_rank_element_attention(
        model,
        selection=selection,
        k=TOP_K,
        alpha=ALPHA,
        intervention_timing=INTERVENTION_TIMING,
    ) as controller:
        for position, record in enumerate(records, 1):
            controller.clear()
            controller.set_prompt_context(prompt_length=record["prompt_length"])
            baseline = baseline_by_index[record["dataset_index"]]
            if not torch.equal(record["input_ids"], baseline["input_ids"]):
                raise RuntimeError(
                    "baseline/defense input ID mismatch: "
                    f"dataset_index={record['dataset_index']}"
                )
            generation = generate_with_first_logits(
                model, tokenizer, record["input_ids"]
            )
            if generation["generation_failed"]:
                raise RuntimeError(
                    "rank intervention generation failed: "
                    f"dataset_index={record['dataset_index']}"
                )
            runtime = controller.summary()
            validate_prompt_runtime_invariants(runtime)
            comparison = compare_token_sequences(
                baseline["response_token_ids"], generation["response_token_ids"]
            )
            logit_metrics = compute_logit_step_metrics(
                baseline["first_logits"], generation["first_logits"], step=0
            )
            validate_first_token_outcome_invariant(
                response_equal=bool(comparison["response_equal"]),
                first_token_argmax_equal=bool(logit_metrics["argmax_equal"]),
            )
            rows.append(
                {
                    "dataset_group": label,
                    "prompt_id": record["prompt_id"],
                    "row_position": record["row_position"],
                    "dataset_index": record["dataset_index"],
                    "prompt": record["prompt"],
                    "prompt_length": record["prompt_length"],
                    "input_ids": serialized_input_ids(record["input_ids"]),
                    "input_ids_match_baseline": True,
                    "method": "rank_element_topk_v1",
                    "k": TOP_K,
                    "alpha": ALPHA,
                    "intervention_timing": INTERVENTION_TIMING,
                    "baseline_response": baseline["response"],
                    "baseline_success": baseline["success"],
                    **{
                        key: value
                        for key, value in generation.items()
                        if key != "first_logits"
                    },
                    **comparison,
                    "first_token_argmax_equal": logit_metrics["argmax_equal"],
                    "first_token_logit_metrics": logit_metrics,
                    "transition": classify_transition(
                        bool(baseline["success"]), bool(generation["success"])
                    ),
                    "diagnostics": runtime,
                }
            )
            prompt_diagnostics.append(runtime)
            print(
                f"[{label} rank-top500 {position}/{len(records)}] "
                f"dataset_index={record['dataset_index']} "
                f"success={generation['success']}",
                flush=True,
            )
    selected_interventions = sum(
        item["selected_rank_intervention_count"]
        for item in prompt_diagnostics
    )
    selected_crossings = sum(
        item["selected_value_crossing_count"] for item in prompt_diagnostics
    )
    summary = {
        "dataset_group": label,
        "method": "rank_element_topk_v1",
        "k": TOP_K,
        "alpha": ALPHA,
        "prompt_count": len(rows),
        "attack_success_count": sum(bool(row["success"]) for row in rows),
        "asr": sum(bool(row["success"]) for row in rows) / len(rows),
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
            bool(row["input_ids_match_baseline"]) for row in rows
        ),
        "selected_rank_intervention_count": selected_interventions,
        "selected_value_crossing_count": selected_crossings,
        "selected_value_crossing_fraction": (
            selected_crossings / selected_interventions
        ),
        "adjacent_rank_pair_crossing_count": sum(
            item["adjacent_rank_pair_crossing_count"]
            for item in prompt_diagnostics
        ),
        "later_forward_intervention_count": sum(
            item["later_forward_intervention_count"]
            for item in prompt_diagnostics
        ),
        "max_unselected_element_change": max(
            item["max_unselected_element_change"]
            for item in prompt_diagnostics
        ),
        "max_outside_tail_change": max(
            item["max_outside_tail_change"] for item in prompt_diagnostics
        ),
        "max_constructed_delta_antisymmetry_error_float32": max(
            item["max_constructed_delta_antisymmetry_error_float32"]
            for item in prompt_diagnostics
        ),
        "max_symmetric_component_error_float32": max(
            item["max_symmetric_component_error_float32"]
            for item in prompt_diagnostics
        ),
        "generation_failure_count": 0,
    }
    return rows, summary


def run_gpu(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --stage run; refusing CPU model load")
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    if len(args.dataset_groups) != len(set(args.dataset_groups)):
        raise ValueError("--dataset-groups must not contain duplicates")
    output_dir = ensure_output_absent(args.output_dir).resolve()
    rank_artifact_dir = Path(args.rank_artifact_dir).resolve()
    source_dir = Path(args.artifact_dir).resolve()
    model_path = Path(args.model).resolve()
    selection, config, diagnostics, rank_verification = (
        load_rank_selection_artifact(rank_artifact_dir)
    )
    source_verification = _validate_source_provenance(config, source_dir)
    if Path(config.get("model", "")).resolve() != model_path:
        raise ValueError("rank artifact model mismatch")

    loaded, _, _, _, _ = _load_group_inputs(args)
    groups = {name: loaded[name] for name in args.dataset_groups}
    audited_reference = load_validated_reference_artifacts(
        source_dir,
        expected_heads=EXPECTED_TOP5,
        expected_selection_manifest_sha256=EXPECTED_SELECTION_MANIFEST_SHA256,
        expected_score_artifact_sha256=EXPECTED_SCORE_ARTIFACT_SHA256,
        expected_model_path=model_path,
    )
    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(model_path)
    model_validation = validate_loaded_llama_model(model, audited_reference)
    groups = {
        name: format_records(records, tokenizer) for name, records in groups.items()
    }

    output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at": utc_now(),
        "execution_git_head": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "execution_git_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "execution_git_status_porcelain": git_value(
            REPO_ROOT, "status", "--porcelain"
        ).splitlines(),
        "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_source_sha256": sha256_file(
            REPO_ROOT / "utils/gcg_rank_element_topk_intervention.py"
        ),
        "model": str(model_path),
        "model_config": llama_model_config_summary(model),
        "model_validation": model_validation,
        "gpu": torch.cuda.get_device_name(0),
        "rank_artifact_dir": str(rank_artifact_dir),
        "rank_artifact_verification": rank_verification,
        "source_artifact_dir": str(source_dir),
        "source_artifact_verification": source_verification,
        "selection": diagnostics,
        "top_rank_elements": read_jsonl(
            rank_artifact_dir / "selected_rank_elements.jsonl"
        )[:10],
        "dataset_groups": list(groups),
        "independent100_used": False,
        "k": TOP_K,
        "alpha": ALPHA,
        "intervention_timing": INTERVENTION_TIMING,
        "generation_settings": GENERATION_SETTINGS,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    summaries = []
    baseline_summaries = []
    artifact_paths = []
    for group_name, records in groups.items():
        baseline_started = start_cuda_measurement()
        baseline = run_baseline(
            records,
            model=model,
            tokenizer=tokenizer,
            label=f"{group_name} baseline",
        )
        baseline_runtime = finish_cuda_measurement(baseline_started)
        baseline_by_index = {row["dataset_index"]: row for row in baseline}
        baseline_path = output_dir / f"{group_name}_baseline.jsonl"
        write_jsonl(
            baseline_path,
            [
                {
                    key: serialized_input_ids(value) if key == "input_ids" else value
                    for key, value in row.items()
                    if key != "first_logits"
                }
                for row in baseline
            ],
        )
        artifact_paths.append(baseline_path)
        baseline_summary = {
            "dataset_group": group_name,
            "prompt_count": len(baseline),
            "attack_success_count": sum(
                bool(row["success"]) for row in baseline
            ),
            "asr": sum(bool(row["success"]) for row in baseline) / len(baseline),
            "generation_failure_count": 0,
            "input_ids_saved_count": len(baseline),
            **baseline_runtime,
        }
        baseline_summary_path = (
            output_dir / f"{group_name}_baseline_summary.json"
        )
        write_json(baseline_summary_path, baseline_summary)
        artifact_paths.append(baseline_summary_path)
        baseline_summaries.append(baseline_summary)

        condition_started = start_cuda_measurement()
        rows, summary = run_condition(
            records,
            baseline_by_index,
            model=model,
            tokenizer=tokenizer,
            selection=selection,
            label=group_name,
        )
        summary.update(finish_cuda_measurement(condition_started))
        condition_dir = output_dir / group_name / "k_500"
        condition_dir.mkdir(parents=True)
        prompt_results_path = condition_dir / "prompt_results.jsonl"
        condition_summary_path = condition_dir / "summary.json"
        write_jsonl(prompt_results_path, rows)
        write_json(condition_summary_path, summary)
        artifact_paths.extend([prompt_results_path, condition_summary_path])
        summaries.append(summary)

    result = {
        "status": "complete",
        "completed_at": utc_now(),
        "baselines": baseline_summaries,
        "conditions": summaries,
    }
    evaluation_summary_path = output_dir / "evaluation_summary.json"
    write_json(evaluation_summary_path, result)
    artifact_paths.append(evaluation_summary_path)
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
    manifest_path = output_dir / "run_manifest.json"
    saved_manifest = read_json(manifest_path)
    if saved_manifest.get("status") != "running":
        raise ValueError("run manifest is not in running state")
    saved_manifest["status"] = "complete"
    saved_manifest["completed_at"] = result["completed_at"]
    saved_manifest["artifact_manifest_sha256"] = sha256_file(
        artifact_manifest_path
    )
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"temporary run manifest already exists: {temporary}"
        )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(saved_manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "cpu-validate":
        result = cpu_validation_payload(args)
    else:
        result = run_gpu(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
