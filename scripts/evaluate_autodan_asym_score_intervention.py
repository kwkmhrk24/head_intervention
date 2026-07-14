#!/usr/bin/env python3
"""Evaluate direct S_asym suppression with regenerated no-cache baselines."""

import argparse
import csv
import json
import math
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_autodan_baseline import (  # noqa: E402
    get_git_metadata,
    read_records,
    set_seed,
    write_json,
    write_jsonl,
)
from scripts.evaluate_autodan_head_sets import (  # noqa: E402
    validate_fixed_experiment_manifest,
)
from utils.asym_score_intervention import (  # noqa: E402
    NonSquareAttentionScoreError,
    SUPPORTED_TRANSFORMERS_VERSION,
    UnsupportedAttentionCacheError,
    patch_llama_attention_scores,
    validate_beta,
    validate_layer_head_pairs,
)
from utils.evaluate_utils import (  # noqa: E402
    REFUSAL_SUBSTRINGS,
    check_success_with_details,
)
from utils.prompt_utils import apply_template  # noqa: E402


PAPER_HEADS = [(3, 0), (7, 16), (9, 9), (13, 4), (3, 5)]
EXPECTED_ELEMENT_HEADS = {
    "element_positive": [(14, 11), (11, 24), (8, 7), (7, 22), (9, 10)],
    "element_negative": [(31, 22), (10, 30), (8, 29), (10, 31), (9, 15)],
    "element_absolute": [(31, 22), (8, 29), (10, 30), (11, 12), (9, 15)],
}
ELEMENT_RANKING_KEYS = {
    "element_positive": "consistent_positive_top",
    "element_negative": "consistent_negative_top",
    "element_absolute": "consistent_absolute_top",
}
HEADSET_CONDITIONS = (
    "paper",
    "element_positive",
    "element_negative",
    "element_absolute",
)
COMPARISON_COLUMNS = [
    "condition",
    "head_list",
    "beta",
    "total_count",
    "success_count",
    "asr",
    "baseline_success_count",
    "defended_count",
    "remained_success_count",
    "regressed_count",
    "remained_failure_count",
    "defense_success_rate",
    "regression_rate",
    "generation_failure_count",
    "mean_response_length_tokens",
    "mean_observed_asym_scale",
    "max_intervention_identity_error",
    "max_symmetric_component_error",
]
RESEARCH_NOTE = (
    "The head sets were selected after exploratory analysis of the full AutoDAN "
    "dataset. Any 110-prompt evaluation is a post-hoc internal analysis, not a "
    "fully untouched external test."
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate a no-cache baseline and evaluate direct antisymmetric "
            "attention-score suppression on the fixed AutoDAN hold-out."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--elementwise-heads", required=True)
    parser.add_argument("--source-labels", required=True)
    parser.add_argument(
        "--historical-baseline-results",
        default=None,
        help=(
            "Optional saved baseline used only for auxiliary agreement reporting; "
            "it is never used as the transition reference."
        ),
    )
    parser.add_argument(
        "--conditions", default=",".join(HEADSET_CONDITIONS)
    )
    parser.add_argument("--betas", default="0.5,1.0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--equivalence-tolerance", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def parse_conditions(value):
    conditions = [item.strip() for item in value.split(",") if item.strip()]
    if not conditions:
        raise ValueError("--conditions must select at least one head set")
    unknown = [item for item in conditions if item not in HEADSET_CONDITIONS]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    if len(conditions) != len(set(conditions)):
        raise ValueError("--conditions contains duplicates")
    return conditions


def parse_betas(value):
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("--betas must contain at least one beta")
    betas = [validate_beta(float(part)) for part in parts]
    if len(betas) != len(set(betas)):
        raise ValueError("--betas contains duplicates")
    return betas


def validate_head_pairs(name, pairs):
    validate_layer_head_pairs(pairs, layer_count=32, head_count=32)
    if len(pairs) != 5:
        raise ValueError(f"{name} must contain exactly five heads")
    return list(pairs)


def load_headsets(path):
    with Path(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    headsets = {"paper": validate_head_pairs("paper", PAPER_HEADS)}
    for name, ranking_key in ELEMENT_RANKING_KEYS.items():
        rows = payload.get(ranking_key)
        if not isinstance(rows, list) or len(rows) < 5:
            raise ValueError(f"{path}: {ranking_key} must contain at least five rows")
        pairs = [
            (row.get("layer"), row.get("head")) for row in rows[:5]
        ]
        pairs = validate_head_pairs(name, pairs)
        if pairs != EXPECTED_ELEMENT_HEADS[name]:
            raise ValueError(
                f"{path}: {name} differs from the fixed expected top-5; "
                f"expected={EXPECTED_ELEMENT_HEADS[name]}, found={pairs}"
            )
        headsets[name] = pairs
    return headsets


def _format_beta(beta):
    return format(beta, ".12g")


def build_condition_plan(conditions, betas, headsets):
    plan = [
        {
            "condition": "no_intervention",
            "headset": None,
            "layer_head_pairs": [],
            "beta": None,
            "patched": False,
        },
        {
            "condition": "beta_zero_control",
            "headset": "paper",
            "layer_head_pairs": headsets["paper"],
            "beta": 0.0,
            "patched": True,
        },
    ]
    positive_betas = [beta for beta in betas if beta > 0]
    for name in conditions:
        for beta in positive_betas:
            plan.append(
                {
                    "condition": f"{name}_beta_{_format_beta(beta)}",
                    "headset": name,
                    "layer_head_pairs": headsets[name],
                    "beta": beta,
                    "patched": True,
                }
            )
    return plan


def _index_unique(rows, key, source_name):
    indexed = {}
    for row_number, row in enumerate(rows):
        if key not in row or type(row[key]) is not int:
            raise ValueError(f"{source_name}: row {row_number} lacks integer {key}")
        if row[key] in indexed:
            raise ValueError(f"{source_name}: duplicate {key} {row[key]}")
        indexed[row[key]] = row
    return indexed


def align_evaluation_records(input_rows, manifest, source_rows, input_name, source_name):
    validate_fixed_experiment_manifest(manifest)
    evaluation_indices = manifest["evaluation_indices"]
    if len(input_rows) != manifest["evaluation_size"]:
        raise ValueError(
            f"{input_name}: expected {manifest['evaluation_size']} rows, "
            f"found {len(input_rows)}"
        )
    actual_indices = []
    for row_number, row in enumerate(input_rows):
        if type(row.get("source_index")) is not int:
            raise ValueError(f"{input_name}: row {row_number} lacks source_index")
        if "prompt_id" not in row:
            raise ValueError(f"{input_name}: row {row_number} lacks prompt_id")
        if not isinstance(row.get("prompt"), str):
            raise ValueError(f"{input_name}: row {row_number} lacks prompt")
        actual_indices.append(row["source_index"])
    if actual_indices != evaluation_indices:
        raise ValueError("evaluation source indices do not match split manifest")

    source_by_index = _index_unique(source_rows, "index", source_name)
    prepared = []
    for row in input_rows:
        source_index = row["source_index"]
        source = source_by_index.get(source_index)
        if source is None:
            raise ValueError(f"{source_name}: missing index {source_index}")
        if source.get("user prompt") != row["prompt"]:
            raise ValueError(f"prompt mismatch at source_index {source_index}")
        if type(source.get("success")) is not bool:
            raise ValueError(f"source success is not boolean at {source_index}")
        prepared.append(
            {
                "source_index": source_index,
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "source_success": source["success"],
            }
        )
    return prepared


def align_historical_baseline(rows, records, source_name):
    """Join an optional saved baseline without making it an experiment baseline."""

    indexed = _index_unique(rows, "source_index", source_name)
    aligned = {}
    for record in records:
        source_index = record["source_index"]
        row = indexed.get(source_index)
        if row is None:
            raise ValueError(f"{source_name}: missing source_index {source_index}")
        if row.get("prompt") != record["prompt"]:
            raise ValueError(
                f"{source_name}: prompt mismatch at source_index {source_index}"
            )
        if type(row.get("success")) is not bool:
            raise ValueError(
                f"{source_name}: success is not boolean at source_index {source_index}"
            )
        aligned[source_index] = row
    return aligned


def compare_historical_baseline(regenerated_rows, historical_by_index):
    comparable = [
        row
        for row in regenerated_rows
        if not row["generation_failed"] and isinstance(row["success"], bool)
    ]
    pairs = [
        (
            row["source_index"],
            historical_by_index[row["source_index"]]["success"],
            row["success"],
        )
        for row in comparable
    ]
    agreement_indices = [index for index, old, new in pairs if old == new]
    return {
        "role": "auxiliary_only_not_transition_reference",
        "total_count": len(regenerated_rows),
        "compared_count": len(pairs),
        "historical_success_count": sum(old for _, old, _ in pairs),
        "regenerated_success_count": sum(new for _, _, new in pairs),
        "agreement_count": len(agreement_indices),
        "agreement_rate": (
            len(agreement_indices) / len(pairs) if pairs else None
        ),
        "historical_true_regenerated_false_indices": [
            index for index, old, new in pairs if old and not new
        ],
        "historical_false_regenerated_true_indices": [
            index for index, old, new in pairs if not old and new
        ],
        "regenerated_generation_failure_count": (
            len(regenerated_rows) - len(comparable)
        ),
    }


def format_records(records, tokenizer, model_config):
    formatted = apply_template(
        prompts=[record["prompt"] for record in records],
        system_prompt=None,
        tokenizer=tokenizer,
        model_config=model_config,
    )
    return [
        {**record, "formatted_prompt": prompt}
        for record, prompt in zip(records, formatted)
    ]


def classify_transition(baseline_success, current_success, generation_failed=False):
    if generation_failed or baseline_success is None or current_success is None:
        return None
    if baseline_success and current_success:
        return "remained_success"
    if baseline_success and not current_success:
        return "defended"
    if not baseline_success and current_success:
        return "regressed"
    return "remained_failure"


def generate_response(model, tokenizer, formatted_prompt, generation_config):
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)
    input_length = inputs.input_ids.shape[-1]
    with torch.inference_mode():
        outputs = model.generate(inputs.input_ids, **generation_config)
    response_ids = outputs[0, input_length:]
    return tokenizer.decode(response_ids), int(response_ids.numel())


def default_score_diagnostics():
    return {
        "call_count": 0,
        "selected_head_count": 0,
        "pre_asym_fro_norm": 0.0,
        "post_asym_fro_norm": 0.0,
        "aggregate_observed_asym_scale": None,
        "mean_observed_asym_scale": None,
        "expected_asym_scale": None,
        "query_lengths": [],
        "key_lengths": [],
        "all_score_matrices_square": None,
        "max_intervention_identity_error": 0.0,
        "max_symmetric_component_error": 0.0,
        "max_unselected_head_error": 0.0,
        "max_observed_scale_abs_error": None,
    }


def evaluate_records(
    records,
    *,
    condition_spec,
    model,
    tokenizer,
    generation_config,
    baseline_by_index=None,
    show_progress=False,
):
    results = []
    iterator = (
        tqdm(records, leave=False, desc=condition_spec["condition"])
        if show_progress
        else records
    )
    patch_context = (
        patch_llama_attention_scores(
            model,
            condition_spec["layer_head_pairs"],
            condition_spec["beta"],
        )
        if condition_spec["patched"]
        else None
    )

    if patch_context is None:
        context_manager = _null_diagnostics_context()
    else:
        context_manager = patch_context

    with context_manager as diagnostics:
        for record in iterator:
            if diagnostics is not None:
                diagnostics.clear()
            try:
                response, response_length_tokens = generate_response(
                    model,
                    tokenizer,
                    record["formatted_prompt"],
                    generation_config,
                )
                score_diagnostics = (
                    diagnostics.summary()
                    if diagnostics is not None
                    else default_score_diagnostics()
                )
                success, checker = check_success_with_details(response)
                generation_failed = False
                failure_reason = None
            except (NonSquareAttentionScoreError, UnsupportedAttentionCacheError):
                raise
            except Exception as exc:
                response = None
                response_length_tokens = None
                score_diagnostics = (
                    diagnostics.summary()
                    if diagnostics is not None
                    else default_score_diagnostics()
                )
                success = None
                checker = {
                    "matched_refusal_substrings": [],
                    "decision_reason": "generation failed; success not evaluated",
                }
                generation_failed = True
                failure_reason = f"{type(exc).__name__}: {exc}"

            baseline_success = (
                success
                if baseline_by_index is None
                else baseline_by_index[record["source_index"]]["success"]
            )
            transition = classify_transition(
                baseline_success, success, generation_failed
            )
            baseline_response = (
                None
                if baseline_by_index is None
                else baseline_by_index[record["source_index"]].get("response")
            )
            results.append(
                {
                    "source_index": record["source_index"],
                    "prompt_id": record["prompt_id"],
                    "condition": condition_spec["condition"],
                    "headset": condition_spec["headset"],
                    "layer_head_pairs": [
                        {"layer": layer, "head": head}
                        for layer, head in condition_spec["layer_head_pairs"]
                    ],
                    "beta": condition_spec["beta"],
                    "use_cache": False,
                    "prompt": record["prompt"],
                    "formatted_prompt": record["formatted_prompt"],
                    "response": response,
                    "success": success,
                    "source_success": record["source_success"],
                    "baseline_success": baseline_success,
                    "transition": transition,
                    "response_exact_match_with_no_intervention": (
                        None
                        if baseline_response is None or response is None
                        else response == baseline_response
                    ),
                    "generation_failed": generation_failed,
                    "generation_failure_reason": failure_reason,
                    "matched_refusal_substrings": checker[
                        "matched_refusal_substrings"
                    ],
                    "success_decision_reason": checker["decision_reason"],
                    "response_length_chars": (
                        None if response is None else len(response)
                    ),
                    "response_length_tokens": response_length_tokens,
                    "score_intervention_call_count": score_diagnostics[
                        "call_count"
                    ],
                    "score_intervention_selected_head_count": score_diagnostics[
                        "selected_head_count"
                    ],
                    "pre_asym_fro_norm": score_diagnostics[
                        "pre_asym_fro_norm"
                    ],
                    "post_asym_fro_norm": score_diagnostics[
                        "post_asym_fro_norm"
                    ],
                    "aggregate_observed_asym_scale": score_diagnostics[
                        "aggregate_observed_asym_scale"
                    ],
                    "mean_observed_asym_scale": score_diagnostics[
                        "mean_observed_asym_scale"
                    ],
                    "expected_asym_scale": score_diagnostics[
                        "expected_asym_scale"
                    ],
                    "score_query_lengths": score_diagnostics["query_lengths"],
                    "score_key_lengths": score_diagnostics["key_lengths"],
                    "all_score_matrices_square": score_diagnostics[
                        "all_score_matrices_square"
                    ],
                    "max_intervention_identity_error": score_diagnostics[
                        "max_intervention_identity_error"
                    ],
                    "max_symmetric_component_error": score_diagnostics[
                        "max_symmetric_component_error"
                    ],
                    "max_unselected_head_error": score_diagnostics[
                        "max_unselected_head_error"
                    ],
                }
            )
    return results


class _null_diagnostics_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def summarize_results(results, condition_spec):
    evaluated = [row for row in results if not row["generation_failed"]]
    baseline_known = [
        row for row in results if isinstance(row["baseline_success"], bool)
    ]
    baseline_success_count = sum(row["baseline_success"] for row in baseline_known)
    baseline_failure_count = len(baseline_known) - baseline_success_count
    transitions = {
        name: sum(row["transition"] == name for row in results)
        for name in (
            "defended",
            "remained_success",
            "regressed",
            "remained_failure",
        )
    }
    observed_weighted_sum = 0.0
    observed_call_count = 0
    for row in results:
        if row["mean_observed_asym_scale"] is not None:
            count = row["score_intervention_call_count"]
            observed_weighted_sum += row["mean_observed_asym_scale"] * count
            observed_call_count += count
    success_count = sum(row["success"] is True for row in evaluated)
    response_lengths = [
        row["response_length_tokens"]
        for row in evaluated
        if row["response_length_tokens"] is not None
    ]
    return {
        "condition": condition_spec["condition"],
        "headset": condition_spec["headset"],
        "layer_head_pairs": [
            {"layer": layer, "head": head}
            for layer, head in condition_spec["layer_head_pairs"]
        ],
        "beta": condition_spec["beta"],
        "use_cache": False,
        "total_count": len(results),
        "evaluated_count": len(evaluated),
        "success_count": success_count,
        "asr": success_count / len(evaluated) if evaluated else None,
        "baseline_success_count": baseline_success_count,
        "baseline_failure_count": baseline_failure_count,
        "defended_count": transitions["defended"],
        "remained_success_count": transitions["remained_success"],
        "regressed_count": transitions["regressed"],
        "remained_failure_count": transitions["remained_failure"],
        "defense_success_rate": (
            transitions["defended"] / baseline_success_count
            if baseline_success_count
            else None
        ),
        "regression_rate": (
            transitions["regressed"] / baseline_failure_count
            if baseline_failure_count
            else None
        ),
        "generation_failure_count": len(results) - len(evaluated),
        "generation_failure_indices": [
            row["source_index"] for row in results if row["generation_failed"]
        ],
        "mean_response_length_tokens": (
            sum(response_lengths) / len(response_lengths)
            if response_lengths
            else None
        ),
        "score_intervention_call_count": sum(
            row["score_intervention_call_count"] for row in results
        ),
        "mean_observed_asym_scale": (
            observed_weighted_sum / observed_call_count
            if observed_call_count
            else None
        ),
        "max_intervention_identity_error": max(
            (row["max_intervention_identity_error"] for row in results),
            default=0.0,
        ),
        "max_symmetric_component_error": max(
            (row["max_symmetric_component_error"] for row in results),
            default=0.0,
        ),
        "max_unselected_head_error": max(
            (row["max_unselected_head_error"] for row in results),
            default=0.0,
        ),
    }


def comparison_row(summary):
    row = {column: summary.get(column) for column in COMPARISON_COLUMNS}
    row["head_list"] = ";".join(
        f"{pair['layer']},{pair['head']}"
        for pair in summary["layer_head_pairs"]
    )
    return row


def run_equivalence_preflight(
    model,
    tokenizer,
    formatted_prompt,
    layer_head_pairs,
    generation_config,
    tolerance,
):
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)
    with torch.inference_mode():
        original_logits = model(
            input_ids=inputs.input_ids,
            use_cache=False,
        ).logits
    with patch_llama_attention_scores(model, layer_head_pairs, 0.0) as diagnostics:
        diagnostics.clear()
        with torch.inference_mode():
            patched_logits = model(
                input_ids=inputs.input_ids,
                use_cache=False,
            ).logits
        logit_diagnostics = diagnostics.summary()
    difference = (patched_logits - original_logits).detach().float().abs()
    original_argmax = int(original_logits[0, -1].argmax().item())
    patched_argmax = int(patched_logits[0, -1].argmax().item())

    original_response, _ = generate_response(
        model, tokenizer, formatted_prompt, generation_config
    )
    with patch_llama_attention_scores(model, layer_head_pairs, 0.0) as diagnostics:
        diagnostics.clear()
        patched_response, _ = generate_response(
            model, tokenizer, formatted_prompt, generation_config
        )
        generation_diagnostics = diagnostics.summary()

    result = {
        "transformers_version": transformers.__version__,
        "attention_class": type(model.model.layers[0].self_attn).__name__,
        "use_cache": False,
        "beta": 0.0,
        "layer_head_pairs": [
            {"layer": layer, "head": head} for layer, head in layer_head_pairs
        ],
        "max_absolute_logit_difference": float(difference.max().cpu().item()),
        "mean_absolute_logit_difference": float(difference.mean().cpu().item()),
        "original_argmax_token_id": original_argmax,
        "patched_argmax_token_id": patched_argmax,
        "argmax_token_match": original_argmax == patched_argmax,
        "original_greedy_response": original_response,
        "patched_greedy_response": patched_response,
        "greedy_response_match": original_response == patched_response,
        "logit_forward_score_diagnostics": logit_diagnostics,
        "generation_score_diagnostics": generation_diagnostics,
        "tolerance": tolerance,
    }
    result["passed"] = (
        result["argmax_token_match"]
        and result["greedy_response_match"]
        and result["max_absolute_logit_difference"] <= tolerance
        and logit_diagnostics["max_intervention_identity_error"] <= tolerance
        and generation_diagnostics["max_intervention_identity_error"] <= tolerance
    )
    return result


def output_paths(output_dir, plan):
    output_dir = Path(output_dir)
    paths = {
        "equivalence": output_dir / "beta_zero_equivalence.json",
        "comparison_csv": output_dir / "comparison_summary.csv",
        "comparison_json": output_dir / "comparison_summary.json",
    }
    for spec in plan:
        paths[f"{spec['condition']}_results"] = (
            output_dir / spec["condition"] / "prompt_results.jsonl"
        )
        paths[f"{spec['condition']}_summary"] = (
            output_dir / spec["condition"] / "summary.json"
        )
    return paths


def check_outputs(paths, overwrite):
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))


def build_metadata(args, model, tokenizer, spec, manifest, command, git_metadata):
    return {
        "model_path": str(Path(args.model).resolve()),
        "model_type": "Llama",
        "transformers_version": transformers.__version__,
        "supported_transformers_version": SUPPORTED_TRANSFORMERS_VERSION,
        "attention_implementation": model.config._attn_implementation,
        "attention_class": type(model.model.layers[0].self_attn).__name__,
        "input_file": str(Path(args.input_file).resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "source_labels": str(Path(args.source_labels).resolve()),
        "historical_baseline_results": (
            str(Path(args.historical_baseline_results).resolve())
            if args.historical_baseline_results
            else None
        ),
        "historical_baseline_role": (
            "auxiliary agreement only; never used for transitions"
            if args.historical_baseline_results
            else None
        ),
        "elementwise_heads": str(Path(args.elementwise_heads).resolve()),
        "condition": spec["condition"],
        "headset": spec["headset"],
        "layer_head_pairs": [
            {"layer": layer, "head": head}
            for layer, head in spec["layer_head_pairs"]
        ],
        "beta": spec["beta"],
        "beta_definition": (
            "S_beta = S_sym + (1-beta)*S_asym = S - beta*S_asym"
        ),
        "score_stage": {
            "post_rope": True,
            "after_repeat_kv": True,
            "pre_mask": True,
            "pre_softmax": True,
        },
        "intervention_scope": (
            "full square score matrix for selected heads at every forward call"
        ),
        "head_selection_window": "tail16",
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "use_cache": False,
            "seed": args.seed,
            "system_prompt": None,
            "batch_size": 1,
        },
        "chat_template": getattr(tokenizer, "chat_template", None),
        "success_checker": "utils.evaluate_utils.check_success_with_details",
        "refusal_substring_count": len(REFUSAL_SUBSTRINGS),
        "baseline_policy": (
            "regenerated no_intervention with use_cache=False in this run; "
            "no cached historical baseline is used for transitions"
        ),
        "selection_size": manifest["selection_size"],
        "evaluation_size": manifest["evaluation_size"],
        "evaluated_prompt_count": args.max_prompts or manifest["evaluation_size"],
        "command": command,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "research_context_note": RESEARCH_NOTE,
        **git_metadata,
    }


def write_comparison(output_dir, rows, context):
    output_dir = Path(output_dir)
    write_json(output_dir / "comparison_summary.json", {**context, "rows": rows})
    csv_path = output_dir / "comparison_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    conditions = parse_conditions(args.conditions)
    betas = parse_betas(args.betas)
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not math.isfinite(args.equivalence_tolerance) or args.equivalence_tolerance < 0:
        raise ValueError("--equivalence-tolerance must be finite and non-negative")
    headsets = load_headsets(args.elementwise_heads)
    plan = build_condition_plan(conditions, betas, headsets)
    paths = output_paths(args.output_dir, plan)
    check_outputs(paths, args.overwrite)

    with Path(args.split_manifest).open() as handle:
        manifest = json.load(handle)
    input_rows = read_records(args.input_file)
    source_rows = read_records(args.source_labels)
    records = align_evaluation_records(
        input_rows,
        manifest,
        source_rows,
        args.input_file,
        args.source_labels,
    )
    historical_by_index = None
    if args.historical_baseline_results:
        historical_by_index = align_historical_baseline(
            read_records(args.historical_baseline_results),
            records,
            args.historical_baseline_results,
        )
    if args.max_prompts is not None and args.max_prompts > len(records):
        raise ValueError("--max-prompts exceeds the 110-prompt evaluation size")

    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected Transformers {SUPPORTED_TRANSFORMERS_VERSION}, "
            f"found {transformers.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to load Llama2 on CPU")
    set_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True
    )
    model.eval()
    model.config.use_cache = False
    if model.config._attn_implementation != "eager":
        raise ValueError("model attention implementation is not eager")
    if type(model.model.layers[0].self_attn).__name__ != "LlamaAttention":
        raise ValueError("loaded model does not use LlamaAttention")
    model_config = {
        "model_name": "Llama",
        "n_layers": model.config.num_hidden_layers,
        "n_heads": model.config.num_attention_heads,
        "resid_dim": model.config.hidden_size,
    }
    formatted_records = format_records(records, tokenizer, model_config)
    selected_records = (
        formatted_records
        if args.max_prompts is None
        else formatted_records[: args.max_prompts]
    )
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
        selected_records[0]["formatted_prompt"],
        PAPER_HEADS,
        generation_config,
        args.equivalence_tolerance,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_json(paths["equivalence"], equivalence)
    if not equivalence["passed"]:
        raise RuntimeError(
            "beta=0 equivalence preflight failed; intervention conditions were not run"
        )

    command_args = sys.argv if argv is None else [sys.argv[0], *argv]
    command = shlex.join([sys.executable, *command_args])
    git_metadata = get_git_metadata(REPO_ROOT)
    baseline_by_index = None
    summaries = []
    historical_comparison = None
    for spec in plan:
        results = evaluate_records(
            selected_records,
            condition_spec=spec,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            baseline_by_index=baseline_by_index,
            show_progress=True,
        )
        if spec["condition"] == "no_intervention":
            baseline_by_index = {
                row["source_index"]: row for row in results
            }
            if historical_by_index is not None:
                historical_comparison = compare_historical_baseline(
                    results, historical_by_index
                )
        summary = summarize_results(results, spec)
        if spec["condition"] == "no_intervention":
            summary["historical_baseline_comparison"] = historical_comparison
        summary["metadata"] = build_metadata(
            args,
            model,
            tokenizer,
            spec,
            manifest,
            command,
            git_metadata,
        )
        condition_dir = Path(args.output_dir) / spec["condition"]
        write_jsonl(condition_dir / "prompt_results.jsonl", results)
        write_json(condition_dir / "summary.json", summary)
        summaries.append(summary)

    comparison_rows = [comparison_row(summary) for summary in summaries]
    write_comparison(
        args.output_dir,
        comparison_rows,
        {
            "baseline_condition": "no_intervention",
            "baseline_regenerated": True,
            "historical_baseline_used_for_transitions": False,
            "historical_baseline_comparison": historical_comparison,
            "use_cache": False,
            "beta_zero_equivalence_passed": True,
            "research_context_note": RESEARCH_NOTE,
        },
    )
    print(f"validated evaluation prompts: {len(formatted_records)}")
    print(f"evaluated prompts per condition: {len(selected_records)}")
    print("beta=0 equivalence: passed")
    for summary in summaries:
        print(
            f"{summary['condition']}: {summary['success_count']}/"
            f"{summary['evaluated_count']} successes; "
            f"generation failures={summary['generation_failure_count']}"
        )
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
