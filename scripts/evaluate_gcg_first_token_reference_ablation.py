#!/usr/bin/env python3
"""Compare dynamic and first-token-only reference intervention on selection 20."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.asym_score_intervention import (  # noqa: E402
    temporarily_disable_default_dynamic_cache,
)
from utils.evaluate_utils import check_success_with_details  # noqa: E402
from utils.experiment_runtime import git_head, seed_everything as set_seed  # noqa: E402
from utils.gcg_first_token_reference_ablation import (  # noqa: E402
    EXPECTED_SELECTION_DATASET_INDICES,
    load_selection_development_records,
    summarize_development_condition,
    summarize_first_token_logits,
    validate_first_token_response_invariant,
    write_comparison_table,
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
    write_json,
    write_jsonl,
)
from utils.gcg_wasserstein_reference_evaluation import (  # noqa: E402
    ensure_output_absent,
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
DEFAULT_PRIOR_DYNAMIC_SMOKE = (
    "data/outputs/gcg_wasserstein_reference_intervention/llama2/"
    "seed42_selection20/smoke_20260802T064954Z"
)
EXPECTED_SELECTION_MANIFEST_SHA256 = (
    "13578541411456bd689f8e771694ca97b047f2828bc35663536a5bc65badfa1e"
)
EXPECTED_SCORE_ARTIFACT_SHA256 = (
    "672cd283cc55bb1d4282bb5a752fa6fa3c60c083fe1850642974d12e33fcb0a0"
)
EXPECTED_TOP5 = ((31, 22), (13, 4), (5, 30), (9, 18), (27, 24))
SMOKE_DATASET_INDICES = (68, 70, 72)
TIMINGS = ("dynamic_all_steps", "first_token_only")
ALPHAS = (0.0, 0.1, 0.3, 0.5)
DEVELOPMENT_ALPHAS = (0.1, 0.3, 0.5)
GENERATION_CONFIG = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "max_new_tokens": 32,
    "use_cache": False,
    "return_dict_in_generate": True,
    "output_scores": True,
}
ERROR_FIELDS = (
    "max_constructed_delta_antisymmetry_error_float32",
    "max_delta_antisymmetry_error_float32",
    "max_delta_antisymmetry_error",
    "max_symmetric_component_error_float32",
    "max_symmetric_component_error",
    "max_diagonal_change",
    "max_unselected_head_change",
    "max_outside_tail_change",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--prior-dynamic-smoke", default=DEFAULT_PRIOR_DYNAMIC_SMOKE)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def load_model_and_tokenizer(model_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to run ablation on CPU")
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
    formatted_prompts = apply_template(
        [record["prompt"] for record in records],
        system_prompt=None,
        tokenizer=tokenizer,
        model_config={"model_name": "Llama"},
    )
    output = []
    for record, formatted_prompt in zip(records, formatted_prompts):
        input_ids = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        prompt_length = int(input_ids.shape[-1])
        if prompt_length < 16:
            raise ValueError(
                f"selection prompt is shorter than 16 tokens: "
                f"dataset_index={record['dataset_index']}"
            )
        output.append(
            {
                **record,
                "formatted_prompt": formatted_prompt,
                "prompt_length": prompt_length,
                "input_ids": input_ids,
            }
        )
    return output


def generate_with_first_logits(model, tokenizer, input_ids: torch.Tensor) -> dict:
    prompt_length = int(input_ids.shape[-1])
    try:
        with torch.inference_mode(), temporarily_disable_default_dynamic_cache(model):
            output = model.generate(input_ids.to(model.device), **GENERATION_CONFIG)
        if not output.scores:
            raise RuntimeError("generation returned no first-step logits")
        first_logits = output.scores[0][0].float()
        if not bool(torch.isfinite(first_logits).all().item()):
            raise RuntimeError("first-step logits contain NaN or infinity")
        response_ids = output.sequences[0, prompt_length:]
        if response_ids.numel() == 0:
            raise RuntimeError("generation produced no response token")
        if int(response_ids[0].item()) != int(torch.argmax(first_logits).item()):
            raise RuntimeError(
                "generated first token does not match torch.argmax(first-step scores)"
            )
        response = tokenizer.decode(response_ids)
        success, checker = check_success_with_details(response)
        return {
            "response": response,
            "response_token_ids": [int(value) for value in response_ids.cpu().tolist()],
            "response_token_count": int(response_ids.numel()),
            "success": bool(success),
            "matched_refusal_substrings": checker["matched_refusal_substrings"],
            "success_decision_reason": checker["decision_reason"],
            "generation_failed": False,
            "generation_error": None,
            "empty_response": response == "",
            "first_logits": first_logits.detach().cpu(),
        }
    except ReferenceInterventionRuntimeError:
        raise
    except Exception as exc:
        return {
            "response": None,
            "response_token_ids": None,
            "response_token_count": None,
            "success": None,
            "matched_refusal_substrings": [],
            "success_decision_reason": "generation failed",
            "generation_failed": True,
            "generation_error": f"{type(exc).__name__}: {exc}",
            "empty_response": None,
            "first_logits": None,
        }


def run_baseline(records: list[dict], *, model, tokenizer, label: str) -> list[dict]:
    rows = []
    for position, record in enumerate(records, 1):
        generation = generate_with_first_logits(
            model,
            tokenizer,
            record["input_ids"],
        )
        if generation["generation_failed"]:
            raise RuntimeError(
                f"baseline generation failed at dataset index {record['dataset_index']}"
            )
        if generation["response"] != record["baseline_response"] or generation[
            "success"
        ] != record["baseline_success"]:
            raise RuntimeError(
                f"baseline reproduction mismatch at dataset index {record['dataset_index']}"
            )
        rows.append({**record, **generation})
        print(
            f"[{label} {position}/{len(records)}] "
            f"dataset_index={record['dataset_index']} success={generation['success']}",
            flush=True,
        )
    return rows


def validate_first_token_prompt_diagnostics(summary: MappingLike, alpha: float) -> None:
    if alpha == 0.0:
        if summary["total_intervention_count"] != 0:
            raise RuntimeError("alpha=0 performed an intervention")
        if summary["reference_cache_transfer_count"] != 0:
            raise RuntimeError("alpha=0 accessed the reference cache")
        return
    if summary["first_forward_detected"] is not True:
        raise RuntimeError("first forward was not detected")
    if summary["first_forward_intervened"] is not True:
        raise RuntimeError("first forward was not intervened")
    if summary["later_forward_intervention_count"] != 0:
        raise RuntimeError("first-token-only intervened after the first forward")
    counts = [item["count"] for item in summary["per_head_intervention_count"]]
    if counts != [1] * len(EXPECTED_TOP5):
        raise RuntimeError(f"first-token-only per-head counts are invalid: {counts}")
    if summary["total_intervention_count"] != len(EXPECTED_TOP5):
        raise RuntimeError("first-token-only total head intervention count is not five")
    if summary["block_changed_across_interventions"]:
        raise RuntimeError("first-token-only intervention block changed")
    if summary["block_start"] != summary["prompt_length"] - 16:
        raise RuntimeError("first-token-only block start mismatch")
    if summary["block_end"] != summary["prompt_length"]:
        raise RuntimeError("first-token-only block end mismatch")


# A small protocol alias keeps runtime type checks out of the hot generation loop.
MappingLike = dict


def run_intervention_condition(
    records: list[dict],
    baseline_by_index: dict[int, dict],
    *,
    timing: str,
    alpha: float,
    model,
    tokenizer,
    artifacts,
    label: str,
) -> tuple[list[dict], dict]:
    rows = []
    with patch_llama_reference_attention(
        model,
        artifacts=artifacts,
        alpha=alpha,
        intervention_timing=timing,
    ) as controller:
        for position, record in enumerate(records, 1):
            controller.clear()
            controller.set_prompt_context(
                prompt_length=record["prompt_length"],
                intervention_timing=timing,
            )
            generation = generate_with_first_logits(
                model,
                tokenizer,
                record["input_ids"],
            )
            diagnostics = controller.summary()
            if generation["generation_failed"]:
                raise RuntimeError(
                    f"intervention generation failed: index={record['dataset_index']} "
                    f"timing={timing} alpha={alpha}"
                )
            baseline = baseline_by_index[record["dataset_index"]]
            comparison = compare_token_sequences(
                baseline["response_token_ids"],
                generation["response_token_ids"],
            )
            first_metrics = compute_logit_step_metrics(
                baseline["first_logits"],
                generation["first_logits"],
                step=0,
            )
            if timing == "first_token_only":
                validate_first_token_response_invariant(
                    response_equal=comparison["response_equal"],
                    first_different_token_position=comparison[
                        "first_different_token_position"
                    ],
                )
                validate_first_token_prompt_diagnostics(diagnostics, alpha)
            if alpha == 0.0:
                if not comparison["response_equal"] or generation["success"] != baseline[
                    "success"
                ]:
                    raise RuntimeError("alpha=0 response/success mismatch")
                if any(
                    first_metrics[key] != 0.0
                    for key in (
                        "max_absolute_logit_difference",
                        "mean_absolute_logit_difference",
                        "kl_baseline_to_intervention",
                        "top1_logit_change",
                        "margin_change",
                    )
                ) or not first_metrics["argmax_equal"]:
                    raise RuntimeError("alpha=0 first-step logits are not exact")
            row = {
                "prompt_id": record["prompt_id"],
                "row_position": record["row_position"],
                "dataset_index": record["dataset_index"],
                "prompt": record["prompt"],
                "formatted_prompt": record["formatted_prompt"],
                "prompt_length": record["prompt_length"],
                "intervention_timing": timing,
                "alpha": alpha,
                "selected_heads": [
                    {"layer": layer, "head": head}
                    for layer, head in artifacts.top5_heads
                ],
                "baseline_response": baseline["response"],
                "baseline_success": baseline["success"],
                **{
                    key: value
                    for key, value in generation.items()
                    if key != "first_logits"
                },
                **comparison,
                "first_token_argmax_equal": first_metrics["argmax_equal"],
                "first_token_logit_metrics": first_metrics,
                "diagnostics": diagnostics,
            }
            rows.append(row)
            print(
                f"[{label} {position}/{len(records)}] index={record['dataset_index']} "
                f"timing={timing} alpha={alpha:.1f} success={row['success']} "
                f"response_equal={row['response_equal']} "
                f"first_argmax_equal={row['first_token_argmax_equal']}",
                flush=True,
            )
    return rows, aggregate_condition_diagnostics(rows, timing=timing, alpha=alpha)


def aggregate_condition_diagnostics(
    rows: list[dict],
    *,
    timing: str,
    alpha: float,
) -> dict:
    per_head = []
    for layer, head in EXPECTED_TOP5:
        count = 0
        for row in rows:
            entry = next(
                item
                for item in row["diagnostics"]["per_head_intervention_count"]
                if (item["layer"], item["head"]) == (layer, head)
            )
            count += int(entry["count"])
        per_head.append({"layer": layer, "head": head, "count": count})
    aggregate = {
        "intervention_timing": timing,
        "alpha": alpha,
        "prompt_count": len(rows),
        "first_forward_detected_count": sum(
            bool(row["diagnostics"]["first_forward_detected"]) for row in rows
        ),
        "first_forward_intervened_count": sum(
            bool(row["diagnostics"]["first_forward_intervened"]) for row in rows
        ),
        "later_forward_intervention_count": sum(
            int(row["diagnostics"]["later_forward_intervention_count"])
            for row in rows
        ),
        "later_forward_skip_count": sum(
            int(row["diagnostics"]["later_forward_skip_count"]) for row in rows
        ),
        "total_intervention_count": sum(
            int(row["diagnostics"]["total_intervention_count"]) for row in rows
        ),
        "patched_attention_call_count": sum(
            int(row["diagnostics"]["attention_forward_call_count"]) for row in rows
        ),
        "per_head_intervention_count": per_head,
        "block_start_by_dataset_index": {
            str(row["dataset_index"]): row["diagnostics"]["block_start"] for row in rows
        },
        "block_end_by_dataset_index": {
            str(row["dataset_index"]): row["diagnostics"]["block_end"] for row in rows
        },
        "block_changed_across_interventions": any(
            bool(row["diagnostics"]["block_changed_across_interventions"])
            for row in rows
        ),
        "max_outside_intervention_block_change": max(
            (float(row["diagnostics"]["max_outside_tail_change"]) for row in rows),
            default=0.0,
        ),
        "short_sequence_skip_count": sum(
            int(row["diagnostics"]["short_sequence_skip_count"]) for row in rows
        ),
        "all_scores_finite": all(
            bool(row["diagnostics"]["all_scores_finite"]) for row in rows
        ),
    }
    for field in ERROR_FIELDS:
        aggregate[field] = max(
            (float(row["diagnostics"][field]) for row in rows),
            default=0.0,
        )
    if timing == "first_token_only" and alpha > 0:
        expected_total = len(rows) * len(EXPECTED_TOP5)
        if aggregate["total_intervention_count"] != expected_total:
            raise RuntimeError("first-token-only aggregate intervention count mismatch")
        if any(item["count"] != len(rows) for item in per_head):
            raise RuntimeError("first-token-only aggregate per-head count mismatch")
        if aggregate["later_forward_intervention_count"] != 0:
            raise RuntimeError("first-token-only aggregate later intervention is nonzero")
    return aggregate


def condition_summary(rows: list[dict], *, timing: str, alpha: float) -> dict:
    return summarize_development_condition(
        rows,
        timing=timing,
        alpha=alpha,
        expected_count=len(rows),
    )


def load_prior_smoke(path: Path) -> dict[float, dict[int, dict]]:
    output = {}
    for alpha in ALPHAS:
        rows = read_jsonl(path / f"alpha_{alpha:.1f}/prompt_results.jsonl")
        output[alpha] = {int(row["dataset_index"]): row for row in rows}
        if tuple(output[alpha]) != SMOKE_DATASET_INDICES:
            raise ValueError("prior dynamic smoke index mismatch")
    return output


def write_condition(
    output_dir: Path,
    *,
    timing: str,
    alpha: float,
    rows: list[dict],
    diagnostics: dict,
) -> None:
    condition_dir = output_dir / timing / f"alpha_{alpha:.1f}"
    condition_dir.mkdir(parents=True)
    write_jsonl(condition_dir / "prompt_results.jsonl", rows)
    write_json(condition_dir / "diagnostics.json", diagnostics)


def finalize_manifest(path: Path, summary: dict) -> None:
    payload = read_json(path)
    if payload.get("status") != "running":
        raise ValueError("run manifest is not in running state")
    payload["status"] = "complete"
    payload["completed_at"] = summary["created_at"]
    payload["smoke_passed"] = summary["smoke_passed"]
    payload["development_passed"] = True
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError("temporary run manifest already exists")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    args = parse_args(argv)
    output_dir = ensure_output_absent(args.output_dir).resolve()
    model_path = Path(args.model).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    jailbreak_input = Path(args.jailbreak_input).resolve()
    baseline_results = Path(args.baseline_results).resolve()
    prior_smoke_dir = Path(args.prior_dynamic_smoke).resolve()

    artifacts = load_validated_reference_artifacts(
        artifact_dir,
        expected_heads=EXPECTED_TOP5,
        expected_selection_manifest_sha256=EXPECTED_SELECTION_MANIFEST_SHA256,
        expected_score_artifact_sha256=EXPECTED_SCORE_ARTIFACT_SHA256,
        expected_model_path=model_path,
    )
    records, split_manifest = load_selection_development_records(
        split_manifest_path=artifact_dir / "split_manifest.json",
        jailbreak_input=jailbreak_input,
        baseline_results=baseline_results,
    )
    prior_smoke = load_prior_smoke(prior_smoke_dir)
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(model_path)
    model_validation = validate_loaded_llama_model(model, artifacts)
    records = format_records(records, tokenizer)

    output_dir.mkdir(parents=True)
    artifact_validation = {
        **artifacts.verification,
        "top5_exact_match": artifacts.top5_heads == EXPECTED_TOP5,
        "reference_mean_shape": list(artifacts.reference_mean.shape),
        "reference_mean_dtype": str(artifacts.reference_mean.dtype),
        "reference_mean_finite": bool(torch.isfinite(artifacts.reference_mean).all()),
        "loaded_model_validation": model_validation,
    }
    write_json(output_dir / "artifact_validation.json", artifact_validation)
    run_manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at": utc_now(),
        "execution_git_head": git_head(REPO_ROOT),
        "development_only": True,
        "selection_data_reused_for_head_selection": True,
        "selection_data_reused_for_reference": True,
        "scope": "selection 20 development comparison; not independent evaluation",
        "model": str(model_path),
        "model_config": llama_model_config_summary(model),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
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
            "seed": 42,
            "system_prompt": None,
            "tokenize_add_special_tokens": False,
        },
        "intervention_timings": list(TIMINGS),
        "default_timing_backward_compatibility": "dynamic_all_steps",
        "alphas": list(ALPHAS),
        "tail_tokens": 16,
        "selected_heads": [
            {"layer": layer, "head": head} for layer, head in artifacts.top5_heads
        ],
        "selection_dataset_indices": list(EXPECTED_SELECTION_DATASET_INDICES),
        "selection_count": 20,
        "selection_all_baseline_attack_success": True,
        "artifact_dir": str(artifact_dir),
        "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
        "score_artifact_sha256": EXPECTED_SCORE_ARTIFACT_SHA256,
        "reference_artifact_sha256": sha256_file(
            artifact_dir / "original_reference.pt"
        ),
        "prior_dynamic_smoke": str(prior_smoke_dir),
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    # Smoke is a hard gate before any 20-row development condition.
    smoke_records = [
        next(record for record in records if record["dataset_index"] == index)
        for index in SMOKE_DATASET_INDICES
    ]
    smoke_baseline = run_baseline(
        smoke_records,
        model=model,
        tokenizer=tokenizer,
        label="smoke baseline",
    )
    smoke_baseline_by_index = {row["dataset_index"]: row for row in smoke_baseline}
    smoke_conditions = []
    dynamic_exact = True
    for timing in TIMINGS:
        for alpha in ALPHAS:
            rows, diagnostics = run_intervention_condition(
                smoke_records,
                smoke_baseline_by_index,
                timing=timing,
                alpha=alpha,
                model=model,
                tokenizer=tokenizer,
                artifacts=artifacts,
                label="smoke",
            )
            if timing == "dynamic_all_steps":
                for row in rows:
                    prior = prior_smoke[alpha][row["dataset_index"]]
                    dynamic_exact &= (
                        row["response"] == prior["response"]
                        and row["success"] == prior["success"]
                        and row["generation_failed"] == prior["generation_failed"]
                    )
            smoke_dir = output_dir / "smoke" / timing / f"alpha_{alpha:.1f}"
            smoke_dir.mkdir(parents=True)
            write_jsonl(smoke_dir / "prompt_results.jsonl", rows)
            write_json(smoke_dir / "diagnostics.json", diagnostics)
            smoke_conditions.append(
                {
                    **condition_summary(rows, timing=timing, alpha=alpha),
                    "diagnostics": diagnostics,
                }
            )
    smoke_passed = (
        dynamic_exact
        and all(item["generation_failure_count"] == 0 for item in smoke_conditions)
        and all(item["nan_or_inf_count"] == 0 for item in smoke_conditions)
    )
    smoke_summary = {
        "passed": smoke_passed,
        "dataset_indices": list(SMOKE_DATASET_INDICES),
        "alpha_zero_exact_for_all_timings": all(
            item["response_changed_count"] == 0
            and item["total_intervention_count"] == 0
            for item in smoke_conditions
            if item["alpha"] == 0.0
        ),
        "dynamic_matches_prior_smoke": dynamic_exact,
        "conditions": smoke_conditions,
    }
    write_json(output_dir / "smoke_summary.json", smoke_summary)
    if not smoke_passed:
        raise RuntimeError("smoke failed; refusing selection 20 development run")

    baseline_rows = run_baseline(
        records,
        model=model,
        tokenizer=tokenizer,
        label="development baseline",
    )
    baseline_by_index = {row["dataset_index"]: row for row in baseline_rows}
    write_jsonl(
        output_dir / "baseline_selection_results.jsonl",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"input_ids", "first_logits"}
            }
            for row in baseline_rows
        ],
    )

    condition_summaries = []
    first_logit_rows = []
    # Full alpha=0 audit for first-token-only; not included in the comparison table.
    alpha_zero_rows, alpha_zero_diagnostics = run_intervention_condition(
        records,
        baseline_by_index,
        timing="first_token_only",
        alpha=0.0,
        model=model,
        tokenizer=tokenizer,
        artifacts=artifacts,
        label="development alpha0 audit",
    )
    for row in alpha_zero_rows:
        first_logit_rows.append(
            {
                "row_position": row["row_position"],
                "dataset_index": row["dataset_index"],
                "intervention_timing": "first_token_only",
                "alpha": 0.0,
                "metrics": row["first_token_logit_metrics"],
            }
        )
    write_condition(
        output_dir,
        timing="first_token_only",
        alpha=0.0,
        rows=alpha_zero_rows,
        diagnostics=alpha_zero_diagnostics,
    )

    for timing in TIMINGS:
        for alpha in DEVELOPMENT_ALPHAS:
            rows, diagnostics = run_intervention_condition(
                records,
                baseline_by_index,
                timing=timing,
                alpha=alpha,
                model=model,
                tokenizer=tokenizer,
                artifacts=artifacts,
                label="development",
            )
            summary = summarize_development_condition(
                rows,
                timing=timing,
                alpha=alpha,
            )
            if summary["generation_failure_count"] or summary["nan_or_inf_count"]:
                raise RuntimeError("development condition contains a failure")
            if timing == "first_token_only":
                if diagnostics["total_intervention_count"] != 100:
                    raise RuntimeError("selection first-token intervention count is not 100")
                for row in rows:
                    first_logit_rows.append(
                        {
                            "row_position": row["row_position"],
                            "dataset_index": row["dataset_index"],
                            "intervention_timing": timing,
                            "alpha": alpha,
                            "metrics": row["first_token_logit_metrics"],
                        }
                    )
            write_condition(
                output_dir,
                timing=timing,
                alpha=alpha,
                rows=rows,
                diagnostics=diagnostics,
            )
            condition_summaries.append({**summary, "diagnostics": diagnostics})

    comparison_rows = [
        {key: value for key, value in summary.items() if key != "diagnostics"}
        for summary in condition_summaries
    ]
    write_comparison_table(output_dir / "comparison_table.csv", comparison_rows)
    first_logit_summary = summarize_first_token_logits(
        first_logit_rows,
        alphas=ALPHAS,
    )
    write_jsonl(
        output_dir / "first_token_logit_diagnostics.jsonl",
        first_logit_rows,
    )
    write_json(
        output_dir / "first_token_logit_summary.json",
        first_logit_summary,
    )
    development_summary = {
        "status": "complete",
        "created_at": utc_now(),
        "development_only": True,
        "selection_data_reused_for_head_selection": True,
        "selection_data_reused_for_reference": True,
        "selection_count": 20,
        "baseline_attack_success_count": 20,
        "conditions": condition_summaries,
        "first_token_logit_summary": first_logit_summary,
        "generation_failure_count": sum(
            item["generation_failure_count"] for item in condition_summaries
        ),
        "empty_response_count": sum(
            item["empty_response_count"] for item in condition_summaries
        ),
        "nan_or_inf_count": sum(
            item["nan_or_inf_count"] for item in condition_summaries
        ),
    }
    write_json(
        output_dir / "selection_development_summary.json",
        development_summary,
    )
    final_summary = {
        "status": "complete",
        "created_at": utc_now(),
        "run_directory": str(output_dir),
        "development_only": True,
        "selection_data_reused_for_head_selection": True,
        "selection_data_reused_for_reference": True,
        "smoke_passed": True,
        "dynamic_matches_prior_smoke": dynamic_exact,
        "conditions": comparison_rows,
    }
    write_json(output_dir / "ablation_summary.json", final_summary)
    finalize_manifest(output_dir / "run_manifest.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
