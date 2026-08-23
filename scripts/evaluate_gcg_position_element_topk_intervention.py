#!/usr/bin/env python3
"""Validate or run GCG Position Element Top-k reference intervention."""

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
    _load_group_inputs,
    format_records,
    generate_with_first_logits,
    load_model_and_tokenizer,
)
from utils.experiment_runtime import git_value, seed_everything  # noqa: E402
from utils.gcg_position_element_topk_intervention import (  # noqa: E402
    TOP_K_VALUES,
    build_position_element_selection,
    patch_llama_position_element_attention,
    selection_diagnostics,
    validate_top_k_values,
)
from utils.gcg_reference_intervention_diagnostics import (  # noqa: E402
    compare_token_sequences,
    compute_logit_step_metrics,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    llama_model_config_summary,
    read_json,
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
    validate_alpha,
    validate_loaded_llama_model,
)


DATASET_GROUPS = ("selection20", "holdout48")
DEFAULT_ALPHA = 0.5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("cpu-validate", "run"),
        default="cpu-validate",
        help="cpu-validate is read-only and never loads a model.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--selection-run", default=DEFAULT_SELECTION_RUN)
    parser.add_argument("--holdout-run", default=DEFAULT_HOLDOUT_RUN)
    parser.add_argument("--selection-original", default=DEFAULT_SELECTION_ORIGINAL)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--raw-archive", default=DEFAULT_RAW_ARCHIVE)
    parser.add_argument("--advbench", default=DEFAULT_ADVBENCH)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(TOP_K_VALUES),
    )
    parser.add_argument("--alpha", type=validate_alpha, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--dataset-groups",
        nargs="+",
        choices=DATASET_GROUPS,
        default=list(DATASET_GROUPS),
    )
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def load_selection(artifact_dir: Path, *, k_values, model_path: Path):
    """Load audited selection20 tensors and construct selection in memory."""

    verification = verify_artifact_directory(artifact_dir)
    if verification.get("verified") is not True:
        raise ValueError("source artifact directory verification failed")
    if verification.get("selection_manifest_sha256") != (
        EXPECTED_SELECTION_MANIFEST_SHA256
    ):
        raise ValueError("selection manifest hash mismatch")
    if verification.get("score_artifact_sha256") != EXPECTED_SCORE_ARTIFACT_SHA256:
        raise ValueError("score artifact hash mismatch")
    paths = {
        "original": artifact_dir / "selection_original_scores.pt",
        "jailbreak": artifact_dir / "selection_jailbreak_scores.pt",
    }
    payloads = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in paths.items()
    }
    if payloads["original"].get("condition") != "original":
        raise ValueError("original score artifact condition mismatch")
    if payloads["jailbreak"].get("condition") != "jailbreak":
        raise ValueError("jailbreak score artifact condition mismatch")
    original_metadata = payloads["original"].get("metadata")
    jailbreak_metadata = payloads["jailbreak"].get("metadata")
    if not isinstance(original_metadata, dict) or original_metadata != jailbreak_metadata:
        raise ValueError("original/jailbreak score metadata mismatch")
    if Path(original_metadata.get("model_path", "")).resolve() != model_path:
        raise ValueError("selection score model path mismatch")
    if original_metadata.get("selection_manifest_sha256") != (
        EXPECTED_SELECTION_MANIFEST_SHA256
    ):
        raise ValueError("selection score split hash mismatch")
    selection = build_position_element_selection(
        payloads["original"].get("values"),
        payloads["jailbreak"].get("values"),
        k_values=validate_top_k_values(k_values),
    )
    return selection, verification, original_metadata, paths


def cpu_validation_payload(args) -> dict:
    artifact_dir = Path(args.artifact_dir).resolve()
    model_path = Path(args.model).resolve()
    selection, verification, metadata, paths = load_selection(
        artifact_dir,
        k_values=args.k_values,
        model_path=model_path,
    )
    return {
        "status": "cpu_validation_passed",
        "model_loaded": False,
        "gpu_used": False,
        "artifacts_written": False,
        "artifact_dir": str(artifact_dir),
        "source_artifact_verified": verification["verified"],
        "selection_manifest_sha256": verification["selection_manifest_sha256"],
        "score_artifact_sha256": verification["score_artifact_sha256"],
        "original_score_sha256": sha256_file(paths["original"]),
        "jailbreak_score_sha256": sha256_file(paths["jailbreak"]),
        "score_metadata": {
            key: metadata[key]
            for key in (
                "tail_tokens",
                "attention_implementation",
                "score_capture_position",
                "stored_value_dtype",
                "selection_row_positions",
                "selection_dataset_indices",
            )
        },
        "reference": {
            "construction": "mean over selection20 original S_asym values only",
            "shape": list(selection.original_mean_reference.shape),
            "dtype": str(selection.original_mean_reference.dtype),
            "device": selection.original_mean_reference.device.type,
            "finite": bool(
                torch.isfinite(selection.original_mean_reference).all().item()
            ),
        },
        "selection": selection_diagnostics(selection),
        "top_elements": [dict(row) for row in selection.ranking[:10]],
    }


def run_baseline(records, *, model, tokenizer, label: str) -> list[dict]:
    rows = []
    for position, record in enumerate(records, 1):
        generation = generate_with_first_logits(model, tokenizer, record["input_ids"])
        if generation["generation_failed"]:
            raise RuntimeError(
                f"baseline generation failed: dataset_index={record['dataset_index']}"
            )
        if (
            generation["response"] != record["baseline_response"]
            or generation["success"] is not record["baseline_success"]
        ):
            raise RuntimeError(
                f"saved baseline reproduction mismatch: "
                f"dataset_index={record['dataset_index']}"
            )
        rows.append({**record, **generation})
        print(
            f"[{label} {position}/{len(records)}] "
            f"dataset_index={record['dataset_index']}",
            flush=True,
        )
    return rows


def run_condition(
    records,
    baseline_by_index,
    *,
    model,
    tokenizer,
    selection,
    k: int,
    alpha: float,
    label: str,
):
    rows = []
    prompt_diagnostics = []
    with patch_llama_position_element_attention(
        model,
        selection=selection,
        k=k,
        alpha=alpha,
        intervention_timing="first_token_only",
    ) as controller:
        for position, record in enumerate(records, 1):
            controller.clear()
            controller.set_prompt_context(prompt_length=record["prompt_length"])
            generation = generate_with_first_logits(
                model, tokenizer, record["input_ids"]
            )
            if generation["generation_failed"]:
                raise RuntimeError(
                    f"intervention generation failed: k={k} "
                    f"dataset_index={record['dataset_index']}"
                )
            diagnostics = controller.summary()
            if diagnostics["selected_element_intervention_count"] != k:
                raise RuntimeError("per-prompt selected element count mismatch")
            if diagnostics["later_forward_intervention_count"] != 0:
                raise RuntimeError("first-token-only intervention occurred later")
            baseline = baseline_by_index[record["dataset_index"]]
            comparison = compare_token_sequences(
                baseline["response_token_ids"], generation["response_token_ids"]
            )
            logit_metrics = compute_logit_step_metrics(
                baseline["first_logits"], generation["first_logits"], step=0
            )
            rows.append(
                {
                    "dataset_group": label,
                    "prompt_id": record["prompt_id"],
                    "row_position": record["row_position"],
                    "dataset_index": record["dataset_index"],
                    "prompt": record["prompt"],
                    "prompt_length": record["prompt_length"],
                    "k": k,
                    "alpha": alpha,
                    "intervention_timing": "first_token_only",
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
                        baseline["success"], generation["success"]
                    ),
                    "diagnostics": diagnostics,
                }
            )
            prompt_diagnostics.append(diagnostics)
            print(
                f"[{label} k={k} {position}/{len(records)}] "
                f"dataset_index={record['dataset_index']} "
                f"success={generation['success']}",
                flush=True,
            )
    summary = {
        "dataset_group": label,
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
        "selected_element_intervention_count": sum(
            item["selected_element_intervention_count"]
            for item in prompt_diagnostics
        ),
        "later_forward_intervention_count": sum(
            item["later_forward_intervention_count"] for item in prompt_diagnostics
        ),
        "max_unselected_element_change": max(
            item["max_unselected_element_change"] for item in prompt_diagnostics
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
    artifact_dir = Path(args.artifact_dir).resolve()
    model_path = Path(args.model).resolve()
    selection, verification, metadata, paths = load_selection(
        artifact_dir,
        k_values=args.k_values,
        model_path=model_path,
    )
    # Reuse the existing strict group alignment/audit path.  Paired-original
    # values returned by it are intentionally not used by this deployable defense.
    groups, _, _, _, _ = _load_group_inputs(args)
    groups = {name: groups[name] for name in args.dataset_groups}

    audited_reference = load_validated_reference_artifacts(
        artifact_dir,
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
        "model": str(model_path),
        "model_config": llama_model_config_summary(model),
        "model_validation": model_validation,
        "gpu": torch.cuda.get_device_name(0),
        "artifact_dir": str(artifact_dir),
        "artifact_verification": verification,
        "source_score_hashes": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "score_metadata": metadata,
        "selection": selection_diagnostics(selection),
        "top_elements_by_k": {
            str(k): [dict(row) for row in selection.topk_records(k)]
            for k in selection.k_values
        },
        "dataset_groups": list(groups),
        "alpha": args.alpha,
        "intervention_timing": "first_token_only",
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
    }
    write_json(output_dir / "run_manifest.json", manifest)

    summaries = []
    for group_name, records in groups.items():
        baseline = run_baseline(
            records,
            model=model,
            tokenizer=tokenizer,
            label=f"{group_name} baseline",
        )
        baseline_by_index = {row["dataset_index"]: row for row in baseline}
        write_jsonl(
            output_dir / f"{group_name}_baseline.jsonl",
            [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"input_ids", "first_logits"}
                }
                for row in baseline
            ],
        )
        for k in selection.k_values:
            rows, summary = run_condition(
                records,
                baseline_by_index,
                model=model,
                tokenizer=tokenizer,
                selection=selection,
                k=k,
                alpha=args.alpha,
                label=group_name,
            )
            condition_dir = output_dir / group_name / f"k_{k}"
            condition_dir.mkdir(parents=True)
            write_jsonl(condition_dir / "prompt_results.jsonl", rows)
            write_json(condition_dir / "summary.json", summary)
            summaries.append(summary)
    result = {
        "status": "complete",
        "completed_at": utc_now(),
        "conditions": summaries,
    }
    write_json(output_dir / "evaluation_summary.json", result)
    manifest_path = output_dir / "run_manifest.json"
    saved_manifest = read_json(manifest_path)
    if saved_manifest.get("status") != "running":
        raise ValueError("run manifest is not in running state")
    saved_manifest["status"] = "complete"
    saved_manifest["completed_at"] = result["completed_at"]
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary run manifest already exists: {temporary}")
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
