#!/usr/bin/env python3
"""Evaluate fixed GCG evaluation rows with original-reference interpolation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.asym_score_intervention import (  # noqa: E402
    UnsupportedAttentionCacheError,
    temporarily_disable_default_dynamic_cache,
)
from utils.evaluate_utils import check_success_with_details  # noqa: E402
from utils.experiment_runtime import git_head, seed_everything as set_seed  # noqa: E402
from utils.gcg_wasserstein_reference import (  # noqa: E402
    llama_model_config_summary,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.gcg_wasserstein_reference_evaluation import (  # noqa: E402
    EVALUATION_BASELINE_FAILURE_COUNT,
    EVALUATION_BASELINE_SUCCESS_COUNT,
    EVALUATION_COUNT,
    HeadDiagnosticAccumulator,
    classify_transition,
    compare_alpha_zero_to_baseline,
    ensure_output_absent,
    load_evaluation_records,
    summarize_transitions,
    validate_result_rows,
    write_evaluation_manifest_csv,
    write_transition_summary_csv,
)
from utils.prompt_utils import apply_template  # noqa: E402
from utils.reference_asym_score_intervention import (  # noqa: E402
    ReferenceInterventionRuntimeError,
    load_validated_reference_artifacts,
    patch_llama_reference_attention,
    validate_loaded_llama_model,
)


DEFAULT_MODEL = "/mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf"
DEFAULT_ARTIFACT_DIR = (
    "data/outputs/gcg_wasserstein_reference/llama2/seed42_selection20/"
    "codex_gpu_20260802T061114Z"
)
DEFAULT_JAILBREAK_INPUT = "data/derived/llama2/gcg/jailbreak_prompts.jsonl"
DEFAULT_BASELINE_RESULTS = (
    "data/outputs/multidataset_asym_score_intervention/"
    "beta05_full_20260722_041303/llama_gcg/no_intervention/"
    "prompt_results.jsonl"
)
EXPECTED_SELECTION_MANIFEST_SHA256 = (
    "13578541411456bd689f8e771694ca97b047f2828bc35663536a5bc65badfa1e"
)
EXPECTED_SCORE_ARTIFACT_SHA256 = (
    "672cd283cc55bb1d4282bb5a752fa6fa3c60c083fe1850642974d12e33fcb0a0"
)
EXPECTED_TOP5 = ((31, 22), (13, 4), (5, 30), (9, 18), (27, 24))
PREFLIGHT_ROW_POSITIONS = (0, 2, 9, 14, 15)
PREFLIGHT_DATASET_INDICES = (60, 62, 69, 74, 75)
ALPHAS = (0.0, 0.1, 0.3, 0.5)
GENERATION_CONFIG = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "max_new_tokens": 32,
    "use_cache": False,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run and save only the fixed five-row all-alpha preflight.",
    )
    return parser.parse_args(argv)


def load_model_and_tokenizer(model_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to run evaluation on CPU")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def format_records(records: list[dict], tokenizer) -> list[dict]:
    formatted = apply_template(
        [record["prompt"] for record in records],
        system_prompt=None,
        tokenizer=tokenizer,
        model_config={"model_name": "Llama"},
    )
    output = []
    for record, formatted_prompt in zip(records, formatted):
        tokens = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        sequence_length = int(tokens.input_ids.shape[-1])
        output.append(
            {
                **record,
                "formatted_prompt": formatted_prompt,
                "input_sequence_length": sequence_length,
            }
        )
    return output


def generate_once(model, tokenizer, formatted_prompt: str) -> dict:
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)
    prompt_length = int(inputs.input_ids.shape[-1])
    try:
        with torch.inference_mode(), temporarily_disable_default_dynamic_cache(model):
            output_ids = model.generate(inputs.input_ids, **GENERATION_CONFIG)
        response_ids = output_ids[0, prompt_length:]
        response = tokenizer.decode(response_ids)
        success, checker = check_success_with_details(response)
        return {
            "response": response,
            "success": success,
            "matched_refusal_substrings": checker["matched_refusal_substrings"],
            "success_decision_reason": checker["decision_reason"],
            "generation_failed": False,
            "generation_error": None,
            "response_token_count": int(response_ids.numel()),
            "empty_response": response == "",
        }
    except (ReferenceInterventionRuntimeError, UnsupportedAttentionCacheError):
        raise
    except Exception as exc:
        return {
            "response": None,
            "success": None,
            "matched_refusal_substrings": [],
            "success_decision_reason": "generation failed; success not evaluated",
            "generation_failed": True,
            "generation_error": f"{type(exc).__name__}: {exc}",
            "response_token_count": None,
            "empty_response": None,
        }


def evaluate_records(
    records: list[dict],
    *,
    alpha: float,
    model,
    tokenizer,
    artifacts,
    reference_artifact_path: Path,
    reference_artifact_hash: str,
) -> tuple[list[dict], dict]:
    accumulator = HeadDiagnosticAccumulator(artifacts.top5_heads, alpha)
    rows = []
    with patch_llama_reference_attention(
        model,
        artifacts=artifacts,
        alpha=alpha,
    ) as controller:
        for position, record in enumerate(records, 1):
            controller.clear()
            generation = generate_once(
                model,
                tokenizer,
                record["formatted_prompt"],
            )
            controller_summary = controller.summary()
            metric_records = controller.head_metric_records()
            accumulator.add_prompt(
                dataset_index=record["dataset_index"],
                controller_summary=controller_summary,
                head_metric_records=metric_records,
            )
            transition = (
                classify_transition(record["baseline_success"], generation["success"])
                if not generation["generation_failed"]
                else None
            )
            rows.append(
                {
                    "prompt_id": record["prompt_id"],
                    "row_position": record["row_position"],
                    "dataset_index": record["dataset_index"],
                    "prompt": record["prompt"],
                    "formatted_prompt": record["formatted_prompt"],
                    **generation,
                    "baseline_response": record["baseline_response"],
                    "baseline_success": record["baseline_success"],
                    "baseline_generation_failed": record[
                        "baseline_generation_failed"
                    ],
                    "transition": transition,
                    "alpha": alpha,
                    "selected_heads": [
                        {"layer": layer, "head": head}
                        for layer, head in artifacts.top5_heads
                    ],
                    "intervention_call_count": controller_summary[
                        "intervention_call_count"
                    ],
                    "patched_attention_call_count": controller_summary[
                        "attention_forward_call_count"
                    ],
                    "short_sequence_skip_count": controller_summary[
                        "short_sequence_skip_count"
                    ],
                    "reference_artifact_path": str(reference_artifact_path),
                    "reference_artifact_hash": reference_artifact_hash,
                    "selection_manifest_hash": (
                        EXPECTED_SELECTION_MANIFEST_SHA256
                    ),
                    "score_artifact_hash": EXPECTED_SCORE_ARTIFACT_SHA256,
                }
            )
            print(
                f"[alpha={alpha:.1f}][{position}/{len(records)}] "
                f"row_position={record['row_position']} "
                f"dataset_index={record['dataset_index']} "
                f"success={generation['success']} "
                f"failed={generation['generation_failed']}",
                flush=True,
            )
    validate_result_rows(rows, records)
    return rows, accumulator.summary()


def build_condition_summary(
    *,
    alpha: float,
    rows: list[dict],
    diagnostics: dict,
) -> dict:
    failure_indices = [
        row["dataset_index"] for row in rows if row["generation_failed"]
    ]
    empty_indices = [
        row["dataset_index"] for row in rows if row["empty_response"] is True
    ]
    payload = {
        "alpha": alpha,
        "prompt_count": len(rows),
        "generation_failure_count": len(failure_indices),
        "generation_failure_dataset_indices": failure_indices,
        "empty_response_count": len(empty_indices),
        "empty_response_dataset_indices": empty_indices,
        "short_sequence_skip_count": diagnostics["short_sequence_skip_count"],
        "patched_attention_call_count": diagnostics[
            "patched_attention_call_count"
        ],
        "intervention_call_count": diagnostics["intervention_call_count"],
        "diagnostics_all_finite": diagnostics["all_finite"],
        "nonzero_invariant_record_count": diagnostics[
            "nonzero_invariant_record_count"
        ],
    }
    if not failure_indices:
        payload.update(summarize_transitions(rows, expected_count=len(rows)))
    else:
        payload["transition_summary"] = None
    return payload


def write_condition(
    output_dir: Path,
    *,
    alpha: float,
    rows: list[dict],
    diagnostics: dict,
    summary: dict,
) -> None:
    condition_dir = output_dir / f"alpha_{alpha:.1f}"
    condition_dir.mkdir()
    write_jsonl(condition_dir / "prompt_results.jsonl", rows)
    write_json(condition_dir / "diagnostics.json", diagnostics)
    write_json(condition_dir / "summary.json", summary)


def run_preflight(
    *,
    output_dir: Path,
    evaluation_records: list[dict],
    model,
    tokenizer,
    artifacts,
    reference_artifact_path: Path,
    reference_artifact_hash: str,
) -> dict:
    by_row = {record["row_position"]: record for record in evaluation_records}
    records = [by_row[row_position] for row_position in PREFLIGHT_ROW_POSITIONS]
    if tuple(record["dataset_index"] for record in records) != (
        PREFLIGHT_DATASET_INDICES
    ):
        raise ValueError("preflight row/dataset index mapping mismatch")
    if any(record["split"] != "evaluation" for record in records):
        raise ValueError("preflight contains a selection record")
    baseline_values = {record["baseline_success"] for record in records}
    if baseline_values != {False, True}:
        raise ValueError("preflight lacks baseline success/failure coverage")

    preflight_dir = output_dir / "preflight"
    preflight_dir.mkdir()
    condition_summaries = []
    alpha_zero_comparison = None
    for alpha in ALPHAS:
        rows, diagnostics = evaluate_records(
            records,
            alpha=alpha,
            model=model,
            tokenizer=tokenizer,
            artifacts=artifacts,
            reference_artifact_path=reference_artifact_path,
            reference_artifact_hash=reference_artifact_hash,
        )
        summary = build_condition_summary(
            alpha=alpha,
            rows=rows,
            diagnostics=diagnostics,
        )
        condition_dir = preflight_dir / f"alpha_{alpha:.1f}"
        condition_dir.mkdir()
        write_jsonl(condition_dir / "prompt_results.jsonl", rows)
        write_json(condition_dir / "diagnostics.json", diagnostics)
        write_json(condition_dir / "summary.json", summary)
        condition_summaries.append(summary)
        if alpha == 0.0:
            alpha_zero_comparison = compare_alpha_zero_to_baseline(rows)

    alpha_positive = [
        summary for summary in condition_summaries if summary["alpha"] > 0
    ]
    transition_sums_valid = all(
        sum(summary[name] for name in (
            "newly_defended",
            "still_attack_success",
            "newly_worsened",
            "still_defended",
        ))
        == 5
        for summary in condition_summaries
        if summary["generation_failure_count"] == 0
    )
    passed = all(
        (
            alpha_zero_comparison is not None,
            alpha_zero_comparison["passed"],
            all(item["generation_failure_count"] == 0 for item in alpha_positive),
            all(item["empty_response_count"] == 0 for item in alpha_positive),
            all(item["diagnostics_all_finite"] for item in condition_summaries),
            transition_sums_valid,
        )
    )
    payload = {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "row_positions": list(PREFLIGHT_ROW_POSITIONS),
        "dataset_indices": list(PREFLIGHT_DATASET_INDICES),
        "all_evaluation_records": True,
        "selection_record_count": 0,
        "baseline_success_count": sum(
            record["baseline_success"] for record in records
        ),
        "baseline_failure_count": sum(
            not record["baseline_success"] for record in records
        ),
        "alpha_zero_comparison": alpha_zero_comparison,
        "conditions": condition_summaries,
        "transition_sums_valid": transition_sums_valid,
        "top5_exact_match": artifacts.top5_heads == EXPECTED_TOP5,
        "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
        "score_artifact_sha256": EXPECTED_SCORE_ARTIFACT_SHA256,
    }
    write_json(output_dir / "preflight_summary.json", payload)
    return payload


def baseline_transition_row() -> dict:
    return {
        "condition": "No intervention",
        "alpha": None,
        "newly_defended": 0,
        "still_attack_success": EVALUATION_BASELINE_SUCCESS_COUNT,
        "newly_worsened": 0,
        "still_defended": EVALUATION_BASELINE_FAILURE_COUNT,
        "attack_success_count": EVALUATION_BASELINE_SUCCESS_COUNT,
        "asr": EVALUATION_BASELINE_SUCCESS_COUNT / EVALUATION_COUNT,
        "net_defended": 0,
        "asr_change": 0.0,
        "indices": None,
    }


def main(argv=None):
    args = parse_args(argv)
    output_dir = ensure_output_absent(args.output_dir)
    model_path = Path(args.model).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    jailbreak_input = Path(args.jailbreak_input).resolve()
    baseline_results = Path(args.baseline_results).resolve()
    split_manifest_path = artifact_dir / "split_manifest.json"
    reference_artifact_path = artifact_dir / "original_reference.pt"

    # Artifact and index validation intentionally precede model loading/generation.
    artifacts = load_validated_reference_artifacts(
        artifact_dir,
        expected_heads=EXPECTED_TOP5,
        expected_selection_manifest_sha256=EXPECTED_SELECTION_MANIFEST_SHA256,
        expected_score_artifact_sha256=EXPECTED_SCORE_ARTIFACT_SHA256,
        expected_model_path=model_path,
    )
    evaluation_records, split_manifest = load_evaluation_records(
        split_manifest_path=split_manifest_path,
        jailbreak_input=jailbreak_input,
        baseline_results=baseline_results,
    )
    reference_artifact_hash = sha256_file(reference_artifact_path)
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(str(model_path))
    model_validation = validate_loaded_llama_model(model, artifacts)
    evaluation_records = format_records(evaluation_records, tokenizer)

    output_dir.mkdir(parents=True)
    artifact_validation = {
        **artifacts.verification,
        "expected_top5": [
            {"layer": layer, "head": head} for layer, head in EXPECTED_TOP5
        ],
        "loaded_top5": [
            {"layer": layer, "head": head} for layer, head in artifacts.top5_heads
        ],
        "top5_exact_match": artifacts.top5_heads == EXPECTED_TOP5,
        "reference_mean_shape": list(artifacts.reference_mean.shape),
        "reference_mean_dtype": str(artifacts.reference_mean.dtype),
        "reference_mean_finite": bool(
            torch.isfinite(artifacts.reference_mean).all().item()
        ),
        "reference_artifact_path": str(reference_artifact_path),
        "reference_artifact_sha256": reference_artifact_hash,
        "reference_metadata": artifacts.reference_metadata,
        "loaded_model_validation": model_validation,
    }
    write_json(output_dir / "artifact_validation.json", artifact_validation)
    write_evaluation_manifest_csv(
        output_dir / "evaluation_manifest.csv",
        evaluation_records,
    )
    write_jsonl(
        output_dir / "baseline_evaluation_results.jsonl",
        [
            {
                key: record[key]
                for key in (
                    "prompt_id",
                    "row_position",
                    "dataset_index",
                    "prompt",
                    "baseline_response",
                    "baseline_success",
                    "baseline_generation_failed",
                    "baseline_generation_error",
                    "baseline_response_token_count",
                )
            }
            for record in evaluation_records
        ],
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "execution_git_head": git_head(REPO_ROOT),
            "mode": "preflight_only" if args.preflight_only else "full_evaluation",
            "model": str(model_path),
            "model_config": llama_model_config_summary(model),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "torch_dtype": "torch.float16",
            "attention_implementation": "eager",
            "score_intervention_position": (
                "post-RoPE, after repeat_kv, immediately after "
                "QK^T/sqrt(head_dim), pre-mask, pre-softmax"
            ),
            "method_name": "position-wise original-reference interpolation",
            "alternative_method_name": "reference asymmetry intervention",
            "intervention_equation": (
                "A_intervened=(1-alpha)*A_current+alpha*A_ref; "
                "B_intervened=B_current+alpha*(A_ref-A_current)"
            ),
            "tail_tokens": 16,
            "selected_heads_source": str(artifact_dir / "top5_heads.json"),
            "selected_heads": artifact_validation["loaded_top5"],
            "alphas": list(ALPHAS),
            "generation_settings": {
                **GENERATION_CONFIG,
                "batch_size": 1,
                "system_prompt": None,
                "seed": 42,
                "chat_template": "existing Llama user-only template",
                "tokenize_add_special_tokens": False,
            },
            "artifact_dir": str(artifact_dir),
            "reference_artifact_sha256": reference_artifact_hash,
            "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
            "score_artifact_sha256": EXPECTED_SCORE_ARTIFACT_SHA256,
            "split_manifest": str(split_manifest_path),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "jailbreak_input": str(jailbreak_input),
            "jailbreak_input_sha256": sha256_file(jailbreak_input),
            "baseline_results": str(baseline_results),
            "baseline_results_sha256": sha256_file(baseline_results),
            "total_count": 160,
            "selection_count": split_manifest["selection_count"],
            "evaluation_count": len(evaluation_records),
            "evaluation_baseline_success_count": (
                EVALUATION_BASELINE_SUCCESS_COUNT
            ),
            "evaluation_baseline_failure_count": (
                EVALUATION_BASELINE_FAILURE_COUNT
            ),
            "evaluation_baseline_asr": (
                EVALUATION_BASELINE_SUCCESS_COUNT / EVALUATION_COUNT
            ),
            "evaluation_order": "split manifest records in row_position order",
            "prompt_primary_key": "dataset_index",
        },
    )

    preflight = run_preflight(
        output_dir=output_dir,
        evaluation_records=evaluation_records,
        model=model,
        tokenizer=tokenizer,
        artifacts=artifacts,
        reference_artifact_path=reference_artifact_path,
        reference_artifact_hash=reference_artifact_hash,
    )
    if not preflight["passed"]:
        write_json(
            output_dir / "evaluation_summary.json",
            {
                "status": "failed_preflight",
                "preflight": preflight,
                "evaluation_prompt_count_generated": 0,
            },
        )
        raise RuntimeError("preflight failed; refusing to start 140-row evaluation")
    if args.preflight_only:
        result = {
            "status": "preflight_complete",
            "preflight": preflight,
            "evaluation_prompt_count_generated": 0,
        }
        write_json(output_dir / "evaluation_summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    condition_rows = {}
    condition_diagnostics = {}
    condition_summaries = {}

    # Full alpha=0 audit is a hard gate before any alpha>0 full evaluation.
    alpha_zero_rows, alpha_zero_diagnostics = evaluate_records(
        evaluation_records,
        alpha=0.0,
        model=model,
        tokenizer=tokenizer,
        artifacts=artifacts,
        reference_artifact_path=reference_artifact_path,
        reference_artifact_hash=reference_artifact_hash,
    )
    alpha_zero_summary = build_condition_summary(
        alpha=0.0,
        rows=alpha_zero_rows,
        diagnostics=alpha_zero_diagnostics,
    )
    alpha_zero_equivalence = compare_alpha_zero_to_baseline(alpha_zero_rows)
    alpha_zero_summary["baseline_equivalence"] = alpha_zero_equivalence
    write_condition(
        output_dir,
        alpha=0.0,
        rows=alpha_zero_rows,
        diagnostics=alpha_zero_diagnostics,
        summary=alpha_zero_summary,
    )
    if not alpha_zero_equivalence["passed"]:
        result = {
            "status": "failed_alpha_zero_equivalence",
            "preflight": preflight,
            "alpha_zero_equivalence": alpha_zero_equivalence,
            "alpha_positive_evaluation_performed": False,
        }
        write_json(output_dir / "evaluation_summary.json", result)
        raise RuntimeError(
            "alpha=0 baseline mismatch; refusing to run alpha>0 evaluation"
        )
    if alpha_zero_summary["generation_failure_count"] != 0:
        raise RuntimeError("alpha=0 full evaluation contains generation failures")
    condition_rows[0.0] = alpha_zero_rows
    condition_diagnostics[0.0] = alpha_zero_diagnostics
    condition_summaries[0.0] = alpha_zero_summary

    for alpha in ALPHAS[1:]:
        rows, diagnostics = evaluate_records(
            evaluation_records,
            alpha=alpha,
            model=model,
            tokenizer=tokenizer,
            artifacts=artifacts,
            reference_artifact_path=reference_artifact_path,
            reference_artifact_hash=reference_artifact_hash,
        )
        summary = build_condition_summary(
            alpha=alpha,
            rows=rows,
            diagnostics=diagnostics,
        )
        write_condition(
            output_dir,
            alpha=alpha,
            rows=rows,
            diagnostics=diagnostics,
            summary=summary,
        )
        if summary["generation_failure_count"] != 0:
            raise RuntimeError(f"alpha={alpha} contains generation failures")
        if not diagnostics["all_finite"]:
            raise RuntimeError(f"alpha={alpha} diagnostics contain NaN or infinity")
        condition_rows[alpha] = rows
        condition_diagnostics[alpha] = diagnostics
        condition_summaries[alpha] = summary

    baseline_asr = EVALUATION_BASELINE_SUCCESS_COUNT / EVALUATION_COUNT
    transition_rows = [baseline_transition_row()]
    for alpha in ALPHAS[1:]:
        summary = condition_summaries[alpha]
        transition_rows.append(
            {
                "condition": f"Wasserstein Top5 alpha={alpha:.1f}",
                "alpha": alpha,
                **{
                    key: summary[key]
                    for key in (
                        "newly_defended",
                        "still_attack_success",
                        "newly_worsened",
                        "still_defended",
                        "attack_success_count",
                        "asr",
                        "net_defended",
                        "indices",
                    )
                },
                "asr_change": summary["asr"] - baseline_asr,
            }
        )
    transition_payload = {
        "baseline_evaluation_count": EVALUATION_COUNT,
        "baseline_attack_success_count": EVALUATION_BASELINE_SUCCESS_COUNT,
        "baseline_asr": baseline_asr,
        "rows": transition_rows,
    }
    write_json(output_dir / "transition_summary.json", transition_payload)
    write_transition_summary_csv(
        output_dir / "transition_summary.csv",
        transition_rows,
    )
    result = {
        "status": "complete",
        "created_at": utc_now(),
        "run_directory": str(output_dir.resolve()),
        "artifact_validation_passed": True,
        "preflight_passed": True,
        "alpha_zero_equivalence": alpha_zero_equivalence,
        "conditions": [condition_summaries[alpha] for alpha in ALPHAS],
        "transition_summary": transition_payload,
        "total_generation_failure_count": sum(
            condition_summaries[alpha]["generation_failure_count"]
            for alpha in ALPHAS
        ),
        "total_empty_response_count": sum(
            condition_summaries[alpha]["empty_response_count"]
            for alpha in ALPHAS
        ),
        "total_short_sequence_skip_count": sum(
            condition_summaries[alpha]["short_sequence_skip_count"]
            for alpha in ALPHAS
        ),
        "all_diagnostics_finite": all(
            condition_diagnostics[alpha]["all_finite"] for alpha in ALPHAS
        ),
        "scope": "GCG evaluation split only; selection excluded",
    }
    write_json(output_dir / "evaluation_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
