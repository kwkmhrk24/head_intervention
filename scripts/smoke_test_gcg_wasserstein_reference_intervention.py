#!/usr/bin/env python3
"""Run the fixed three-prompt GPU smoke test for reference interpolation."""

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
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    utc_now,
    validate_split_manifest,
    write_json,
    write_jsonl,
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
SMOKE_ROW_POSITIONS = (8, 10, 12)
SMOKE_DATASET_INDICES = (68, 70, 72)
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
    return parser.parse_args(argv)


def formatted_inputs(tokenizer, prompt: str):
    formatted = apply_template(
        [prompt],
        system_prompt=None,
        tokenizer=tokenizer,
        model_config={"model_name": "Llama"},
    )[0]
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return formatted, inputs


def generate_once(model, tokenizer, input_ids: torch.Tensor) -> dict:
    prompt_length = int(input_ids.shape[-1])
    try:
        with torch.inference_mode(), temporarily_disable_default_dynamic_cache(model):
            output_ids = model.generate(input_ids, **GENERATION_CONFIG)
        response_ids = output_ids[0, prompt_length:]
        response = tokenizer.decode(response_ids)
        success, checker = check_success_with_details(response)
        return {
            "response": response,
            "response_length_tokens": int(response_ids.numel()),
            "success": success,
            "generation_failed": False,
            "generation_failure_reason": None,
            "empty_response": response == "",
            "matched_refusal_substrings": checker["matched_refusal_substrings"],
            "success_decision_reason": checker["decision_reason"],
        }
    except (ReferenceInterventionRuntimeError, UnsupportedAttentionCacheError):
        raise
    except Exception as exc:
        return {
            "response": None,
            "response_length_tokens": None,
            "success": None,
            "generation_failed": True,
            "generation_failure_reason": f"{type(exc).__name__}: {exc}",
            "empty_response": None,
            "matched_refusal_substrings": [],
            "success_decision_reason": "generation failed; success not evaluated",
        }


def aggregate_prompt_diagnostics(alpha: float, prompts: list[dict]) -> dict:
    summaries = [entry["diagnostics"] for entry in prompts]
    error_fields = (
        "max_constructed_delta_antisymmetry_error_float32",
        "max_delta_antisymmetry_error_float32",
        "max_delta_antisymmetry_error",
        "max_symmetric_component_error_float32",
        "max_symmetric_component_error",
        "max_diagonal_change_float32",
        "max_diagonal_change",
        "max_unselected_head_change",
        "max_outside_tail_change",
        "max_reference_asymmetry_error_float32",
    )
    per_head = []
    for layer, head in EXPECTED_TOP5:
        records = [
            next(
                item
                for item in summary["per_head"]
                if item["layer"] == layer and item["head"] == head
            )
            for summary in summaries
        ]

        def present(key: str) -> list[float]:
            return [float(record[key]) for record in records if record[key] is not None]

        per_head.append(
            {
                "layer": layer,
                "head": head,
                "call_count": sum(int(record["call_count"]) for record in records),
                "current_asym_frobenius_norm_min": min(
                    present("current_asym_frobenius_norm_min"), default=None
                ),
                "current_asym_frobenius_norm_max": max(
                    present("current_asym_frobenius_norm_max"), default=None
                ),
                "reference_asym_frobenius_norm": records[0][
                    "reference_asym_frobenius_norm"
                ],
                "intervened_asym_frobenius_norm_min": min(
                    present("intervened_asym_frobenius_norm_min"), default=None
                ),
                "intervened_asym_frobenius_norm_max": max(
                    present("intervened_asym_frobenius_norm_max"), default=None
                ),
                "intervened_to_current_norm_ratio_min": min(
                    present("intervened_to_current_norm_ratio_min"), default=None
                ),
                "intervened_to_current_norm_ratio_max": max(
                    present("intervened_to_current_norm_ratio_max"), default=None
                ),
                "mean_absolute_delta_max": max(
                    present("mean_absolute_delta_max"), default=0.0
                ),
                "max_absolute_delta": max(
                    present("max_absolute_delta"), default=0.0
                ),
                **{
                    key: max(present(key), default=0.0)
                    for key in error_fields
                },
                "current_min": min(present("current_min"), default=None),
                "current_max": max(present("current_max"), default=None),
                "intervened_min": min(present("intervened_min"), default=None),
                "intervened_max": max(present("intervened_max"), default=None),
            }
        )
    aggregate = {
        "alpha": alpha,
        "selected_heads": [
            {"layer": layer, "head": head} for layer, head in EXPECTED_TOP5
        ],
        "prompt_count": len(prompts),
        "attention_forward_call_count": sum(
            int(summary["attention_forward_call_count"]) for summary in summaries
        ),
        "intervention_call_count": sum(
            int(summary["intervention_call_count"]) for summary in summaries
        ),
        "alpha_zero_bypass_count": sum(
            int(summary["alpha_zero_bypass_count"]) for summary in summaries
        ),
        "short_sequence_skip_count": sum(
            int(summary["short_sequence_skip_count"]) for summary in summaries
        ),
        "all_scores_finite": all(
            bool(summary["all_scores_finite"]) for summary in summaries
        ),
        "per_head": per_head,
    }
    aggregate.update(
        {
            key: max((float(summary[key]) for summary in summaries), default=0.0)
            for key in error_fields
        }
    )
    return aggregate


def load_smoke_records(
    *,
    artifact_dir: Path,
    jailbreak_input: Path,
    baseline_results: Path,
) -> tuple[list[dict], dict]:
    split = read_json(artifact_dir / "split_manifest.json")
    validate_split_manifest(split)
    if sha256_file(artifact_dir / "split_manifest.json") != (
        EXPECTED_SELECTION_MANIFEST_SHA256
    ):
        raise ValueError("fixed selection manifest hash mismatch")
    if sha256_file(jailbreak_input) != split["jailbreak_input_sha256"]:
        raise ValueError("jailbreak input hash mismatch")
    source_rows = read_jsonl(jailbreak_input)
    baseline_rows = read_jsonl(baseline_results)
    if len(source_rows) != 160 or len(baseline_rows) != 160:
        raise ValueError("smoke inputs must contain exactly 160 rows")

    records = []
    for row_position, expected_dataset_index in zip(
        SMOKE_ROW_POSITIONS,
        SMOKE_DATASET_INDICES,
    ):
        split_record = split["records"][row_position]
        source = source_rows[row_position]
        baseline = baseline_rows[row_position]
        dataset_index = split_record["dataset_index"]
        if dataset_index != expected_dataset_index:
            raise ValueError("fixed smoke row/dataset index mismatch")
        if split_record["split"] != "selection":
            raise ValueError("fixed smoke prompt is not in selection")
        if source.get("index") != dataset_index:
            raise ValueError("jailbreak source dataset index mismatch")
        if sha256_text(source.get("prompt", "")) != split_record[
            "jailbreak_prompt_sha256"
        ]:
            raise ValueError("jailbreak prompt hash mismatch")
        if baseline.get("source_index") != dataset_index:
            raise ValueError("baseline dataset index mismatch")
        if baseline.get("prompt") != source["prompt"]:
            raise ValueError("baseline prompt mismatch")
        if baseline.get("success") is not True or baseline.get(
            "generation_failed"
        ) is not False:
            raise ValueError("fixed smoke prompt is not a successful baseline row")
        records.append(
            {
                "row_position": row_position,
                "dataset_index": dataset_index,
                "prompt_id": split_record["prompt_id"],
                "prompt": source["prompt"],
                "jailbreak_prompt_sha256": split_record["jailbreak_prompt_sha256"],
                "existing_baseline_response": baseline["response"],
                "existing_baseline_success": baseline["success"],
                "existing_baseline_generation_failed": baseline[
                    "generation_failed"
                ],
            }
        )
    return records, split


def load_model_and_tokenizer(model_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to run smoke on CPU")
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


def prompt_result_record(record, alpha: float, generation: dict, diagnostics: dict):
    return {
        "row_position": record["row_position"],
        "dataset_index": record["dataset_index"],
        "prompt_id": record["prompt_id"],
        "prompt": record["prompt"],
        "jailbreak_prompt_sha256": record["jailbreak_prompt_sha256"],
        "alpha": alpha,
        "selected_heads": [
            {"layer": layer, "head": head} for layer, head in EXPECTED_TOP5
        ],
        "use_cache": False,
        **generation,
        "score_intervention_call_count": diagnostics[
            "intervention_call_count"
        ],
        "attention_forward_call_count": diagnostics[
            "attention_forward_call_count"
        ],
        "short_sequence_skip_count": diagnostics[
            "short_sequence_skip_count"
        ],
        "max_delta_antisymmetry_error": diagnostics[
            "max_delta_antisymmetry_error"
        ],
        "max_symmetric_component_error": diagnostics[
            "max_symmetric_component_error"
        ],
        "max_unselected_head_change": diagnostics[
            "max_unselected_head_change"
        ],
        "max_outside_tail_change": diagnostics["max_outside_tail_change"],
        "all_scores_finite": diagnostics["all_scores_finite"],
    }


def condition_summary(alpha: float, rows: list[dict], diagnostics: dict) -> dict:
    return {
        "alpha": alpha,
        "prompt_count": len(rows),
        "generation_failure_count": sum(
            bool(row["generation_failed"]) for row in rows
        ),
        "empty_response_count": sum(row["empty_response"] is True for row in rows),
        "attack_success_count": sum(row["success"] is True for row in rows),
        "intervention_call_count": diagnostics["intervention_call_count"],
        "attention_forward_call_count": diagnostics[
            "attention_forward_call_count"
        ],
        "short_sequence_skip_count": diagnostics[
            "short_sequence_skip_count"
        ],
        "all_scores_finite": diagnostics["all_scores_finite"],
    }


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    model_path = Path(args.model).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    jailbreak_input = Path(args.jailbreak_input).resolve()
    baseline_results = Path(args.baseline_results).resolve()

    artifacts = load_validated_reference_artifacts(
        artifact_dir,
        expected_heads=EXPECTED_TOP5,
        expected_selection_manifest_sha256=EXPECTED_SELECTION_MANIFEST_SHA256,
        expected_score_artifact_sha256=EXPECTED_SCORE_ARTIFACT_SHA256,
        expected_model_path=model_path,
    )
    records, _split_manifest = load_smoke_records(
        artifact_dir=artifact_dir,
        jailbreak_input=jailbreak_input,
        baseline_results=baseline_results,
    )
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(str(model_path))
    model_validation = validate_loaded_llama_model(model, artifacts)

    for record in records:
        formatted, inputs = formatted_inputs(tokenizer, record["prompt"])
        record["formatted_prompt"] = formatted
        record["input_ids"] = inputs.input_ids.to(model.device)
        record["input_sequence_length"] = int(inputs.input_ids.shape[-1])
        if record["input_sequence_length"] < 16:
            raise ValueError(
                f"smoke prompt shorter than 16 tokens: {record['dataset_index']}"
            )

    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "artifact_validation.json",
        {
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
            "reference_metadata": artifacts.reference_metadata,
            "loaded_model_validation": model_validation,
        },
    )
    write_json(
        output_dir / "smoke_prompt_manifest.json",
        {
            "row_positions": list(SMOKE_ROW_POSITIONS),
            "dataset_indices": list(SMOKE_DATASET_INDICES),
            "all_selection_members": True,
            "all_existing_baseline_success": True,
            "split_manifest_sha256": sha256_file(
                artifact_dir / "split_manifest.json"
            ),
            "jailbreak_input_sha256": sha256_file(jailbreak_input),
            "baseline_results_sha256": sha256_file(baseline_results),
            "records": [
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"input_ids"}
                }
                for record in records
            ],
        },
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "execution_git_head": git_head(REPO_ROOT),
            "model": str(model_path),
            "model_config": llama_model_config_summary(model),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "torch_dtype": "torch.float16",
            "attention_implementation": "eager",
            "attention_class": type(model.model.layers[0].self_attn).__name__,
            "score_intervention_position": (
                "post-RoPE, after repeat_kv, immediately after "
                "QK^T/sqrt(head_dim), pre-mask, pre-softmax"
            ),
            "intervention_equation": (
                "A_intervened=(1-alpha)*A_current+alpha*A_ref; "
                "B_intervened=B_current+alpha*(A_ref-A_current)"
            ),
            "tail_tokens": 16,
            "selected_heads_source": str(artifact_dir / "top5_heads.json"),
            "selected_heads": [
                {"layer": layer, "head": head} for layer, head in artifacts.top5_heads
            ],
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
            "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
            "score_artifact_sha256": EXPECTED_SCORE_ARTIFACT_SHA256,
            "jailbreak_input": str(jailbreak_input),
            "baseline_results": str(baseline_results),
            "runtime_utility_sha256": sha256_file(
                REPO_ROOT / "utils/reference_asym_score_intervention.py"
            ),
            "smoke_script_sha256": sha256_file(Path(__file__)),
            "scope": "fixed selection smoke only; evaluation split not generated",
        },
    )

    equivalence_rows = []
    alpha_zero_rows = []
    alpha_zero_diagnostics = []
    for record in records:
        input_ids = record["input_ids"]
        with torch.inference_mode():
            normal_logits = model(input_ids=input_ids, use_cache=False).logits
        normal_generation = generate_once(model, tokenizer, input_ids)

        with patch_llama_reference_attention(
            model,
            artifacts=artifacts,
            alpha=0.0,
        ) as controller:
            controller.clear()
            with torch.inference_mode():
                patched_logits = model(input_ids=input_ids, use_cache=False).logits
            forward_diagnostics = controller.summary()
            controller.clear()
            patched_generation = generate_once(model, tokenizer, input_ids)
            generation_diagnostics = controller.summary()

        max_logit_difference = float(
            (patched_logits - normal_logits).abs().amax().float().cpu().item()
        )
        response_equal = (
            patched_generation["response"] == normal_generation["response"]
        )
        success_equal = (
            patched_generation["success"] == normal_generation["success"]
        )
        generation_failure_equal = (
            patched_generation["generation_failed"]
            == normal_generation["generation_failed"]
        )
        existing_response_equal = (
            patched_generation["response"]
            == record["existing_baseline_response"]
        )
        existing_success_equal = (
            patched_generation["success"]
            == record["existing_baseline_success"]
        )
        existing_failure_equal = (
            patched_generation["generation_failed"]
            == record["existing_baseline_generation_failed"]
        )
        passed = all(
            (
                max_logit_difference == 0.0,
                response_equal,
                success_equal,
                generation_failure_equal,
                existing_response_equal,
                existing_success_equal,
                existing_failure_equal,
            )
        )
        equivalence_rows.append(
            {
                "row_position": record["row_position"],
                "dataset_index": record["dataset_index"],
                "max_logit_difference": max_logit_difference,
                "response_equal": response_equal,
                "success_equal": success_equal,
                "generation_failure_equal": generation_failure_equal,
                "existing_baseline_response_equal": existing_response_equal,
                "existing_baseline_success_equal": existing_success_equal,
                "existing_baseline_generation_failure_equal": existing_failure_equal,
                "normal_response": normal_generation["response"],
                "patched_alpha_zero_response": patched_generation["response"],
                "existing_baseline_response": record["existing_baseline_response"],
                "normal_success": normal_generation["success"],
                "patched_alpha_zero_success": patched_generation["success"],
                "existing_baseline_success": record["existing_baseline_success"],
                "forward_diagnostics": forward_diagnostics,
                "generation_diagnostics": generation_diagnostics,
                "passed": passed,
            }
        )
        alpha_zero_rows.append(
            prompt_result_record(
                record,
                0.0,
                patched_generation,
                generation_diagnostics,
            )
        )
        alpha_zero_diagnostics.append(
            {
                "row_position": record["row_position"],
                "dataset_index": record["dataset_index"],
                "diagnostics": generation_diagnostics,
            }
        )

    alpha_zero_aggregate = aggregate_prompt_diagnostics(
        0.0,
        alpha_zero_diagnostics,
    )
    (output_dir / "alpha_0.0").mkdir()
    write_jsonl(output_dir / "alpha_0.0/prompt_results.jsonl", alpha_zero_rows)
    write_json(
        output_dir / "alpha_0.0/diagnostics.json",
        {
            "alpha": 0.0,
            "prompts": alpha_zero_diagnostics,
            "aggregate": alpha_zero_aggregate,
        },
    )
    equivalence_payload = {
        "alpha": 0.0,
        "comparison": (
            "normal eager versus patched eager alpha=0 exact bypass, plus existing baseline"
        ),
        "prompt_count": len(equivalence_rows),
        "passed_count": sum(bool(row["passed"]) for row in equivalence_rows),
        "passed": all(bool(row["passed"]) for row in equivalence_rows),
        "prompts": equivalence_rows,
    }
    write_json(output_dir / "alpha_0_equivalence.json", equivalence_payload)
    if not equivalence_payload["passed"]:
        write_json(
            output_dir / "smoke_summary.json",
            {
                "status": "failed_alpha_zero_equivalence",
                "alpha_0_equivalence": equivalence_payload,
                "evaluation_generation_performed": False,
            },
        )
        raise RuntimeError("alpha=0 equivalence failed; stopping before alpha>0")

    all_condition_rows = {0.0: alpha_zero_rows}
    all_condition_diagnostics = {0.0: alpha_zero_aggregate}
    for alpha in ALPHAS[1:]:
        condition_rows = []
        prompt_diagnostics = []
        with patch_llama_reference_attention(
            model,
            artifacts=artifacts,
            alpha=alpha,
        ) as controller:
            for record in records:
                controller.clear()
                generation = generate_once(
                    model,
                    tokenizer,
                    record["input_ids"],
                )
                diagnostics = controller.summary()
                condition_rows.append(
                    prompt_result_record(
                        record,
                        alpha,
                        generation,
                        diagnostics,
                    )
                )
                prompt_diagnostics.append(
                    {
                        "row_position": record["row_position"],
                        "dataset_index": record["dataset_index"],
                        "diagnostics": diagnostics,
                    }
                )
        aggregate = aggregate_prompt_diagnostics(alpha, prompt_diagnostics)
        condition_dir = output_dir / f"alpha_{alpha:.1f}"
        condition_dir.mkdir()
        write_jsonl(condition_dir / "prompt_results.jsonl", condition_rows)
        write_json(
            condition_dir / "diagnostics.json",
            {
                "alpha": alpha,
                "prompts": prompt_diagnostics,
                "aggregate": aggregate,
            },
        )
        all_condition_rows[alpha] = condition_rows
        all_condition_diagnostics[alpha] = aggregate

    summaries = [
        condition_summary(
            alpha,
            all_condition_rows[alpha],
            all_condition_diagnostics[alpha],
        )
        for alpha in ALPHAS
    ]
    smoke_summary = {
        "status": "complete",
        "created_at": utc_now(),
        "run_directory": str(output_dir.resolve()),
        "artifact_validation_passed": True,
        "alpha_0_equivalence_passed": equivalence_payload["passed"],
        "selected_heads": [
            {"layer": layer, "head": head} for layer, head in artifacts.top5_heads
        ],
        "conditions": summaries,
        "total_generation_failure_count": sum(
            summary["generation_failure_count"] for summary in summaries
        ),
        "total_empty_response_count": sum(
            summary["empty_response_count"] for summary in summaries
        ),
        "all_scores_finite": all(
            summary["all_scores_finite"] for summary in summaries
        ),
        "evaluation_generation_performed": False,
    }
    write_json(output_dir / "smoke_summary.json", smoke_summary)
    print(json.dumps(smoke_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
