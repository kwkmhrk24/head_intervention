#!/usr/bin/env python3
"""Evaluate direct S_asym intervention across supported models and attacks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shlex
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.asym_score_dataset import load_dataset_records, read_json_records  # noqa: E402
from utils.asym_score_evaluation import (  # noqa: E402
    attach_baseline_transitions,
    summarize_prompt_results,
)
from utils.asym_score_headsets import load_headsets, sha256_file  # noqa: E402
from utils.asym_score_intervention import (  # noqa: E402
    NonSquareAttentionScoreError,
    SUPPORTED_TRANSFORMERS_VERSION,
    UnsupportedAttentionCacheError,
    patch_attention_scores,
    temporarily_disable_default_dynamic_cache,
    validate_beta,
)
from utils.evaluate_utils import REFUSAL_SUBSTRINGS, check_success_with_details  # noqa: E402
from utils.prompt_utils import apply_template  # noqa: E402


EVALUATION_POSITION = "exploratory intervention screening"
SUPPORTED_CONDITIONS = ("paper", "consensus_increase", "consensus_decrease")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", required=True, choices=("llama", "mistral"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--attack", required=True, choices=("autodan", "gcg"))
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument("--candidate-csv")
    parser.add_argument("--conditions", default="paper")
    parser.add_argument(
        "--beta",
        default="0.5",
        help="One beta or a comma-separated sensitivity list (each in [0,1]).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--equivalence-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--reference-baseline",
        help="Optional auxiliary comparison only; never the transition baseline.",
    )
    return parser.parse_args(argv)


def parse_conditions(value: str) -> list[str]:
    conditions = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not conditions:
        raise ValueError("--conditions must not be empty")
    unknown = [condition for condition in conditions if condition not in SUPPORTED_CONDITIONS]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    if len(conditions) != len(set(conditions)):
        raise ValueError("--conditions contains duplicates")
    return conditions


def parse_betas(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("--beta must not be empty")
    betas = [validate_beta(float(part)) for part in parts]
    if len(betas) != len(set(betas)):
        raise ValueError("--beta contains duplicates")
    return betas


def format_beta(beta: float) -> str:
    return format(beta, ".12g")


def build_condition_plan(conditions, betas, headsets):
    paper = headsets["paper"]
    plan = [
        {"condition": "no_intervention", "heads": [], "beta": None, "patched": False},
        {
            "condition": "beta_zero_control",
            "heads": paper["heads"],
            "beta": 0.0,
            "patched": True,
            "headset_metadata": paper,
        },
    ]
    for condition in conditions:
        metadata = headsets[condition]
        for beta in betas:
            if beta == 0:
                continue
            plan.append(
                {
                    "condition": f"{condition}_beta_{format_beta(beta)}",
                    "headset": condition,
                    "heads": metadata["heads"],
                    "beta": beta,
                    "patched": True,
                    "headset_metadata": metadata,
                }
            )
    return plan


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def check_output_absent(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    return path


def format_records(records, tokenizer, model_family):
    prompts = apply_template(
        [row["prompt"] for row in records],
        system_prompt=None,
        tokenizer=tokenizer,
        model_config={"model_name": "Llama" if model_family == "llama" else "Mistral"},
    )
    return [{**row, "formatted_prompt": prompt} for row, prompt in zip(records, prompts)]


def generate_response(model, tokenizer, formatted_prompt, generation_config):
    inputs = tokenizer(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    prompt_length = inputs.input_ids.shape[-1]
    with torch.inference_mode(), temporarily_disable_default_dynamic_cache(model):
        output_ids = model.generate(inputs.input_ids, **generation_config)
    response_ids = output_ids[0, prompt_length:]
    return tokenizer.decode(response_ids), int(response_ids.numel())


def run_equivalence_preflight(
    model,
    tokenizer,
    formatted_prompt,
    *,
    model_family,
    heads,
    generation_config,
    tolerance,
):
    """Verify beta=0 before any experiment rows are generated."""

    inputs = tokenizer(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    with torch.inference_mode():
        original_logits = model(input_ids=inputs.input_ids, use_cache=False).logits
    with patch_attention_scores(model, model_family, heads, 0.0) as diagnostics:
        diagnostics.clear()
        with torch.inference_mode():
            patched_logits = model(input_ids=inputs.input_ids, use_cache=False).logits
        forward_diagnostics = diagnostics.summary()
    difference = (patched_logits - original_logits).detach().float().abs()
    original_argmax = int(original_logits[0, -1].argmax().item())
    patched_argmax = int(patched_logits[0, -1].argmax().item())
    original_response, _ = generate_response(
        model, tokenizer, formatted_prompt, generation_config
    )
    with patch_attention_scores(model, model_family, heads, 0.0) as diagnostics:
        diagnostics.clear()
        patched_response, _ = generate_response(
            model, tokenizer, formatted_prompt, generation_config
        )
        generation_diagnostics = diagnostics.summary()
    result = {
        "beta": 0.0,
        "heads": [{"layer": layer, "head": head} for layer, head in heads],
        "max_absolute_logit_difference": float(difference.max().cpu().item()),
        "mean_absolute_logit_difference": float(difference.mean().cpu().item()),
        "argmax_token_match": original_argmax == patched_argmax,
        "greedy_response_match": original_response == patched_response,
        "forward_score_diagnostics": forward_diagnostics,
        "generation_score_diagnostics": generation_diagnostics,
        "tolerance": tolerance,
    }
    result["passed"] = (
        result["argmax_token_match"]
        and result["greedy_response_match"]
        and result["max_absolute_logit_difference"] <= tolerance
        and forward_diagnostics["max_intervention_identity_error"] <= tolerance
        and generation_diagnostics["max_intervention_identity_error"] <= tolerance
    )
    return result


def empty_score_diagnostics():
    return {
        "call_count": 0,
        "mean_observed_asym_scale": None,
        "all_score_matrices_square": None,
        "max_intervention_identity_error": 0.0,
        "max_symmetric_component_error": 0.0,
        "max_unselected_head_error": 0.0,
    }


def evaluate_condition(
    records,
    *,
    spec,
    model,
    tokenizer,
    model_family,
    generation_config,
):
    patch = (
        patch_attention_scores(model, model_family, spec["heads"], spec["beta"])
        if spec["patched"]
        else nullcontext(None)
    )
    rows = []
    with patch as diagnostics:
        for record in records:
            if diagnostics is not None:
                diagnostics.clear()
            try:
                response, token_count = generate_response(
                    model, tokenizer, record["formatted_prompt"], generation_config
                )
                success, detail = check_success_with_details(response)
                failed = False
                failure_reason = None
            except (NonSquareAttentionScoreError, UnsupportedAttentionCacheError):
                raise
            except Exception as exc:
                response = None
                token_count = None
                success = None
                detail = {
                    "matched_refusal_substrings": [],
                    "decision_reason": "generation failed; success not evaluated",
                }
                failed = True
                failure_reason = f"{type(exc).__name__}: {exc}"
            score = diagnostics.summary() if diagnostics is not None else empty_score_diagnostics()
            rows.append(
                {
                    **record,
                    "condition": spec["condition"],
                    "heads": [
                        {"layer": layer, "head": head} for layer, head in spec["heads"]
                    ],
                    "beta": spec["beta"],
                    "use_cache": False,
                    "response": response,
                    "success": success,
                    "generation_failed": failed,
                    "generation_failure_reason": failure_reason,
                    "empty_response": response == "" if response is not None else None,
                    "matched_refusal_substrings": detail["matched_refusal_substrings"],
                    "success_decision_reason": detail["decision_reason"],
                    "response_length_chars": len(response) if response is not None else None,
                    "response_length_tokens": token_count,
                    "score_intervention_call_count": score["call_count"],
                    "mean_observed_asym_scale": score["mean_observed_asym_scale"],
                    "all_score_matrices_square": score["all_score_matrices_square"],
                    "max_intervention_identity_error": score["max_intervention_identity_error"],
                    "max_symmetric_component_error": score["max_symmetric_component_error"],
                    "max_unselected_head_error": score["max_unselected_head_error"],
                }
            )
    return rows


def load_reference_baseline(path, records):
    if path is None:
        return None
    rows = read_json_records(path)
    by_index = {}
    expected = {row["source_index"] for row in records}
    for row in rows:
        index = row.get("source_index", row.get("index"))
        if index in expected:
            if index in by_index:
                raise ValueError(f"duplicate reference source_index: {index}")
            if type(row.get("success")) is not bool:
                raise ValueError(f"invalid reference success at source_index {index}")
            reference_prompt = row.get("prompt")
            current_prompt = next(
                record["prompt"]
                for record in records
                if record["source_index"] == index
            )
            if reference_prompt is not None and reference_prompt != current_prompt:
                raise ValueError(
                    f"reference prompt mismatch at source_index {index}"
                )
            by_index[index] = row
    if set(by_index) != expected:
        raise ValueError("reference baseline does not cover selected prompts")
    return by_index


def compare_reference_baseline(run_baseline, reference):
    if reference is None:
        return None
    comparable = [row for row in run_baseline if not row["generation_failed"]]
    matches = sum(reference[row["source_index"]]["success"] == row["success"] for row in comparable)
    return {
        "role": "auxiliary_only_not_transition_reference",
        "compared_count": len(comparable),
        "agreement_count": matches,
        "agreement_rate": matches / len(comparable) if comparable else None,
    }


def comparison_row(summary):
    return {
        key: summary.get(key)
        for key in (
            "condition", "heads", "beta", "total_count", "evaluated_count",
            "success_count", "asr", "asr_denominator", "baseline_success_count",
            "defended_count", "remained_success_count", "regressed_count",
            "remained_failure_count", "defense_success_rate", "regression_rate",
            "generation_failure_count", "empty_response_count",
            "mean_response_length_tokens", "mean_observed_asym_scale",
        )
    }


def main(argv=None):
    args = parse_args(argv)
    conditions = parse_conditions(args.conditions)
    betas = parse_betas(args.beta)
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not math.isfinite(args.equivalence_tolerance) or args.equivalence_tolerance < 0:
        raise ValueError("--equivalence-tolerance must be finite and non-negative")
    output_dir = check_output_absent(args.output_dir)
    records, split_manifest = load_dataset_records(
        args.input_file,
        model_family=args.model_family,
        attack=args.attack,
        split_manifest=args.split_manifest,
    )
    if args.max_prompts is not None:
        if args.max_prompts > len(records):
            raise ValueError("--max-prompts exceeds the normalized input count")
        records = records[: args.max_prompts]
    reference = load_reference_baseline(args.reference_baseline, records)
    headsets = load_headsets(
        model_family=args.model_family,
        conditions=conditions,
        candidate_csv=args.candidate_csv,
    )
    plan = build_condition_plan(conditions, betas, headsets)

    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected Transformers {SUPPORTED_TRANSFORMERS_VERSION}, found {transformers.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to load the real model on CPU")
    set_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model.eval()
    model.config.use_cache = False
    expected_class = "LlamaAttention" if args.model_family == "llama" else "MistralAttention"
    actual_class = type(model.model.layers[0].self_attn).__name__
    if actual_class != expected_class:
        raise TypeError(f"expected eager {expected_class}, found {actual_class}")
    formatted = format_records(records, tokenizer, args.model_family)
    generation_config = {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": False,
    }
    equivalence = run_equivalence_preflight(
        model,
        tokenizer,
        formatted[0]["formatted_prompt"],
        model_family=args.model_family,
        heads=headsets["paper"]["heads"],
        generation_config=generation_config,
        tolerance=args.equivalence_tolerance,
    )
    if not equivalence["passed"]:
        raise RuntimeError("beta=0 equivalence preflight failed")

    all_rows = {}
    summaries = []
    for spec in plan:
        rows = evaluate_condition(
            formatted,
            spec=spec,
            model=model,
            tokenizer=tokenizer,
            model_family=args.model_family,
            generation_config=generation_config,
        )
        if spec["condition"] == "no_intervention":
            baseline_rows = rows
        rows = attach_baseline_transitions(rows, baseline_rows)
        all_rows[spec["condition"]] = rows
        summary = summarize_prompt_results(
            rows,
            condition=spec["condition"],
            heads=spec["heads"],
            beta=spec["beta"],
        )
        summary["headset_metadata"] = spec.get("headset_metadata")
        summaries.append(summary)

    command = shlex.join([sys.executable, *sys.argv])
    manifest = {
        "model": str(Path(args.model).resolve()),
        "model_family": args.model_family,
        "attack": args.attack,
        "input_file": str(Path(args.input_file).resolve()),
        "input_sha256": sha256_file(args.input_file),
        "execution_commit": git_commit(),
        "conditions": [spec["condition"] for spec in plan],
        "heads": {summary["condition"]: summary["heads"] for summary in summaries},
        "betas": betas,
        "generation_settings": {
            **generation_config,
            "seed": args.seed,
            "system_prompt": None,
            "batch_size": 1,
        },
        "baseline_mode": "regenerated_in_run_no_hook_no_cache",
        "reference_baseline": (
            str(Path(args.reference_baseline).resolve()) if args.reference_baseline else None
        ),
        "reference_baseline_comparison": compare_reference_baseline(
            all_rows["no_intervention"], reference
        ),
        "use_cache": False,
        "attention_implementation": "eager",
        "attention_class": actual_class,
        "score_stage": "post-RoPE, after repeat_kv, pre-mask, pre-softmax",
        "intervention_equation": "S_beta = S - beta * (S - S.T) / 2",
        "evaluation_position": EVALUATION_POSITION,
        "success_checker": {
            "name": "case-sensitive refusal substring absence",
            "refusal_substring_count": len(REFUSAL_SUBSTRINGS),
            "normalization": None,
            "generation_failures_excluded_from_asr": True,
            "empty_responses_counted_separately": True,
        },
        "split_manifest": (
            str(Path(args.split_manifest).resolve()) if args.split_manifest else None
        ),
        "split_manifest_sizes": (
            {
                "selection_size": split_manifest.get("selection_size"),
                "evaluation_size": split_manifest.get("evaluation_size"),
            }
            if split_manifest is not None
            else None
        ),
        "normalized_prompt_count": len(records),
        "beta_zero_equivalence": equivalence,
        "command": command,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for summary in summaries:
        summary["run_metadata"] = manifest
        condition_dir = output_dir / summary["condition"]
        write_jsonl(condition_dir / "prompt_results.jsonl", all_rows[summary["condition"]])
        write_json(condition_dir / "summary.json", summary)
    comparison = [comparison_row(summary) for summary in summaries]
    write_json(output_dir / "comparison_summary.json", {"rows": comparison})
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(comparison[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison:
            writer.writerow({**row, "heads": json.dumps(row["heads"])})
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(output_dir / "beta_zero_equivalence.json", equivalence)


if __name__ == "__main__":
    main()
