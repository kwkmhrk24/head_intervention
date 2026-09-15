#!/usr/bin/env python3
"""Validate or run frozen GCG Rank Element intervention conditions."""

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
    RankElementSelection,
    RankElementTopKController,
    load_rank_selection_artifact,
    patch_llama_rank_element_attention,
)
from utils.gcg_reference_intervention_diagnostics import (  # noqa: E402
    compare_token_sequences,
    compute_logit_step_metrics,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
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
RANK_ALL_ELEMENT_COUNT = NUM_LAYERS * NUM_HEADS * ELEMENTS_PER_HEAD
DEFAULT_CONDITION = "rank_top500_alpha1.0"
EVALUATION_CONDITIONS = {
    DEFAULT_CONDITION: {
        "name": DEFAULT_CONDITION,
        "method": "rank_element_topk_v1",
        "k": TOP_K,
        "alpha": ALPHA,
        "selection_rule": "top_k_by_abs_mean_gcg_minus_original",
        "ranking_used_for_selection": True,
        "output_name": "k_500",
    },
    "rank_all_alpha0.5": {
        "name": "rank_all_alpha0.5",
        "method": "rank_all_alpha0.5",
        "k": RANK_ALL_ELEMENT_COUNT,
        "alpha": 0.5,
        "selection_rule": "all_rank_elements",
        "ranking_used_for_selection": False,
        "output_name": "rank_all_alpha0p5",
    },
}
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
    parser.add_argument(
        "--condition",
        choices=tuple(EVALUATION_CONDITIONS),
        default=DEFAULT_CONDITION,
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


def _condition_for_name(name: str) -> dict:
    """Return an isolated copy of one frozen evaluation condition."""

    try:
        return dict(EVALUATION_CONDITIONS[name])
    except KeyError as error:
        raise ValueError(f"unknown evaluation condition: {name!r}") from error


def _runtime_selection_for_condition(
    source_selection: RankElementSelection,
    condition: dict,
) -> RankElementSelection:
    """Reuse the frozen reference while selecting the condition's rank mask."""

    k = int(condition["k"])
    if k == TOP_K:
        return source_selection
    if k != RANK_ALL_ELEMENT_COUNT:
        raise ValueError(f"unsupported rank element count: {k}")
    reference = source_selection.original_rank_reference
    expected_shape = (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)
    if tuple(reference.shape) != expected_shape:
        raise ValueError("rank reference shape mismatch for all-rank condition")
    mask = torch.ones_like(reference, dtype=torch.bool, device="cpu")
    if int(mask.sum().item()) != RANK_ALL_ELEMENT_COUNT:
        raise AssertionError("all-rank mask cardinality mismatch")
    return RankElementSelection(
        ranking=source_selection.ranking,
        masks={RANK_ALL_ELEMENT_COUNT: mask.contiguous()},
        original_rank_reference=reference,
        k_values=(RANK_ALL_ELEMENT_COUNT,),
    )


def _runtime_selection_diagnostics(
    source_diagnostics: dict,
    condition: dict,
) -> dict:
    """Describe the effective mask separately from the frozen Top-500 artifact."""

    if int(condition["k"]) == TOP_K:
        diagnostics = json.loads(json.dumps(source_diagnostics))
        diagnostics["selection_rule"] = condition["selection_rule"]
        diagnostics["ranking_used_for_selection"] = True
        return diagnostics

    per_rank_count = NUM_LAYERS * NUM_HEADS
    per_layer_count = NUM_HEADS * ELEMENTS_PER_HEAD
    rank_band_counts = [
        {
            "rank_start": start,
            "rank_end": min(start + 11, ELEMENTS_PER_HEAD),
            "selected_rank_element_count": (
                min(start + 11, ELEMENTS_PER_HEAD) - start + 1
            )
            * per_rank_count,
        }
        for start in range(1, ELEMENTS_PER_HEAD + 1, 12)
    ]
    condition_diagnostics = {
        "k": RANK_ALL_ELEMENT_COUNT,
        "selected_rank_element_count": RANK_ALL_ELEMENT_COUNT,
        "selected_head_count": NUM_LAYERS * NUM_HEADS,
        "distinct_within_head_rank_count": ELEMENTS_PER_HEAD,
        "selected_rank_min": 1,
        "selected_rank_max": ELEMENTS_PER_HEAD,
        "selected_rank_mean": (ELEMENTS_PER_HEAD + 1) / 2,
        "selected_rank_median": (ELEMENTS_PER_HEAD + 1) / 2,
        "rank_band_counts": rank_band_counts,
        "within_head_rank_counts": [
            {"rank": rank, "selected_rank_element_count": per_rank_count}
            for rank in range(1, ELEMENTS_PER_HEAD + 1)
        ],
        "layer_selection_counts": [
            {"layer": layer, "selected_rank_element_count": per_layer_count}
            for layer in range(NUM_LAYERS)
        ],
        "head_selection_counts": [
            {
                "layer": layer,
                "head": head,
                "selected_rank_element_count": ELEMENTS_PER_HEAD,
            }
            for layer in range(NUM_LAYERS)
            for head in range(NUM_HEADS)
        ],
    }
    return {
        "method": condition["method"],
        "rank_sort": source_diagnostics["rank_sort"],
        "rank_tie_break": source_diagnostics["rank_tie_break"],
        "rank_numbering": source_diagnostics["rank_numbering"],
        "ranking_metric": source_diagnostics["ranking_metric"],
        "selection_prompt_count": source_diagnostics["selection_prompt_count"],
        "candidate_rank_element_count": RANK_ALL_ELEMENT_COUNT,
        "tail_tokens": source_diagnostics["tail_tokens"],
        "strict_lower_elements_per_head": source_diagnostics[
            "strict_lower_elements_per_head"
        ],
        "selection_rule": condition["selection_rule"],
        "ranking_used_for_selection": False,
        "reference_source": "Selection20 original rank mean from frozen rank artifact",
        "k_values": [RANK_ALL_ELEMENT_COUNT],
        "conditions": [condition_diagnostics],
    }


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


def cpu_runtime_smoke(selection, *, k: int = TOP_K, alpha: float = ALPHA) -> dict:
    """Exercise all selected rank mappings without loading a model or GPU."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    controller = RankElementTopKController(
        reference_mean=selection.original_rank_reference,
        rank_mask=selection.mask_for_k(k),
        alpha=alpha,
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
    validate_prompt_runtime_invariants(diagnostics, k=k)
    return diagnostics


def cpu_validation_payload(args) -> dict:
    rank_artifact_dir = Path(args.rank_artifact_dir).resolve()
    source_dir = Path(args.artifact_dir).resolve()
    source_selection, config, source_diagnostics, rank_verification = (
        load_rank_selection_artifact(rank_artifact_dir)
    )
    source_verification = _validate_source_provenance(config, source_dir)
    model_path = Path(args.model).resolve()
    if Path(config.get("model", "")).resolve() != model_path:
        raise ValueError("rank artifact model mismatch")
    condition = _condition_for_name(args.condition)
    selection = _runtime_selection_for_condition(source_selection, condition)
    diagnostics = _runtime_selection_diagnostics(source_diagnostics, condition)
    smoke = cpu_runtime_smoke(
        selection,
        k=int(condition["k"]),
        alpha=float(condition["alpha"]),
    )
    top_rows = read_jsonl(
        rank_artifact_dir / "selected_rank_elements.jsonl"
    )[:10]
    return {
        "status": "cpu_validation_passed",
        "model_loaded": False,
        "gpu_used": False,
        "artifacts_written": False,
        "independent100_used": False,
        "condition": condition,
        "rank_artifact_verification": rank_verification,
        "source_artifact_verified": source_verification["verified"],
        "selection": diagnostics,
        "source_rank_artifact_selection": source_diagnostics,
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
    condition: dict,
    label: str,
    patcher=patch_llama_rank_element_attention,
    runtime_validator=validate_prompt_runtime_invariants,
):
    k = int(condition["k"])
    alpha = float(condition["alpha"])
    method = str(condition["method"])
    rows = []
    prompt_diagnostics = []
    with patcher(
        model,
        selection=selection,
        k=k,
        alpha=alpha,
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
            runtime_validator(runtime, k=k)
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
                    "condition": condition["name"],
                    "method": method,
                    "k": k,
                    "alpha": alpha,
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
                f"[{label} {condition['name']} {position}/{len(records)}] "
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
        "condition": condition["name"],
        "method": method,
        "k": k,
        "alpha": alpha,
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
            if selected_interventions else 0.0
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
        "max_diagonal_change_float32": max(
            item["max_diagonal_change_float32"] for item in prompt_diagnostics
        ),
        "max_absolute_delta_float32": max(
            item["max_absolute_delta_float32"] for item in prompt_diagnostics
        ),
        "first_forward_detected_count": sum(
            bool(item["first_forward_detected"]) for item in prompt_diagnostics
        ),
        "first_forward_intervened_count": sum(
            bool(item["first_forward_intervened"]) for item in prompt_diagnostics
        ),
        "all_scores_finite": all(
            bool(item["all_scores_finite"]) for item in prompt_diagnostics
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
    source_selection, config, source_diagnostics, rank_verification = (
        load_rank_selection_artifact(rank_artifact_dir)
    )
    source_verification = _validate_source_provenance(config, source_dir)
    if Path(config.get("model", "")).resolve() != model_path:
        raise ValueError("rank artifact model mismatch")
    condition = _condition_for_name(args.condition)
    selection = _runtime_selection_for_condition(source_selection, condition)
    diagnostics = _runtime_selection_diagnostics(source_diagnostics, condition)

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
        "condition": condition,
        "selection": diagnostics,
        "source_rank_artifact_selection": source_diagnostics,
        "top_rank_elements": read_jsonl(
            rank_artifact_dir / "selected_rank_elements.jsonl"
        )[:10],
        "dataset_groups": list(groups),
        "independent100_used": False,
        "k": condition["k"],
        "alpha": condition["alpha"],
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
            condition=condition,
            label=group_name,
        )
        summary.update(finish_cuda_measurement(condition_started))
        condition_dir = output_dir / group_name / str(condition["output_name"])
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
