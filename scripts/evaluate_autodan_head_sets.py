#!/usr/bin/env python3
"""Evaluate fixed AutoDAN head sets on the 20/110 internal hold-out.

This script changes only the attention-head locations used by the repository's
existing head-output intervention.  The intervention activation, interpolation
rate, prompt format, generation configuration, and success checker are fixed
across conditions.
"""

import argparse
import csv
import json
import math
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
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
from utils.evaluate_utils import (  # noqa: E402
    REFUSAL_SUBSTRINGS,
    check_success_with_details,
)
from utils.prompt_utils import apply_template  # noqa: E402


PAPER_HEADS = [(3, 0), (7, 16), (9, 9), (13, 4), (3, 5)]
EXPECTED_S_HEADS = {
    "positive": [(1, 8), (7, 25), (9, 30), (12, 27), (29, 30)],
    "negative": [(13, 5), (18, 29), (6, 1), (17, 23), (24, 13)],
    "absolute": [(1, 8), (7, 25), (13, 5), (9, 30), (12, 27)],
}
SELECTED_HEAD_KEYS = {
    "positive": "positive_top5",
    "negative": "negative_bottom5",
    "absolute": "absolute_top5",
}
ALL_CONDITIONS = ("paper", "positive", "negative", "absolute")

EXPECTED_TOTAL_COUNT = 130
EXPECTED_SELECTION_SIZE = 20
EXPECTED_EVALUATION_SIZE = 110
EXPECTED_INTERPOLATION_RATE = 0.75
EXPECTED_LAYERS = 32
EXPECTED_HEADS = 32
EXPECTED_RESID_DIM = 4096
EXPECTED_SELECTION_INDICES = [
    4,
    5,
    15,
    18,
    19,
    26,
    36,
    39,
    40,
    41,
    43,
    48,
    80,
    82,
    96,
    108,
    110,
    113,
    126,
    129,
]
MAX_NEW_TOKENS = 32
SEED = 42

INTERNAL_HOLDOUT_NOTE = (
    "The full AutoDAN 130-prompt dataset had already been used for exploratory "
    "S-metric analysis. This evaluation is a preliminary internal hold-out "
    "analysis, not a fully untouched external test."
)

COMPARISON_COLUMNS = [
    "condition",
    "head_list",
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
    "source_success_subset_count",
    "source_success_subset_asr",
    "generation_failure_count",
    "mean_response_length_tokens",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed paper and S-metric head sets using the existing "
            "Llama2 attention-head output intervention."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--selected-heads", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--source-labels", required=True)
    parser.add_argument("--intervention-activation", required=True)
    parser.add_argument(
        "--interpolation-rate", type=float, default=EXPECTED_INTERPOLATION_RATE
    )
    parser.add_argument(
        "--conditions",
        default=",".join(ALL_CONDITIONS),
        help="Comma-separated subset of paper,positive,negative,absolute.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help=(
            "Evaluate only the leading N prompts after validating the complete "
            "110-prompt evaluation input. Intended for smoke tests."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the output files selected by this invocation.",
    )
    return parser.parse_args(argv)


def load_json_object(path):
    path = Path(path)
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def parse_conditions(value):
    conditions = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not conditions:
        raise ValueError("--conditions must select at least one condition")
    unknown = [item for item in conditions if item not in ALL_CONDITIONS]
    if unknown:
        raise ValueError(f"unknown condition(s): {', '.join(unknown)}")
    if len(conditions) != len(set(conditions)):
        raise ValueError("--conditions contains duplicate conditions")
    return conditions


def validate_head_set(name, heads, expected_count=5, n_layers=32, n_heads=32):
    normalized = []
    for position, pair in enumerate(heads):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{name}: head {position} is not a (layer, head) pair")
        layer, head = pair
        if type(layer) is not int or type(head) is not int:
            raise ValueError(f"{name}: layer/head indices must be integers")
        if not 0 <= layer < n_layers:
            raise ValueError(
                f"{name}: layer {layer} is outside 0 <= layer < {n_layers}"
            )
        if not 0 <= head < n_heads:
            raise ValueError(f"{name}: head {head} is outside 0 <= head < {n_heads}")
        normalized.append((layer, head))

    if len(normalized) != expected_count:
        raise ValueError(
            f"{name}: expected {expected_count} heads, found {len(normalized)}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name}: duplicate head locations are not allowed")
    return normalized


def load_head_sets(selected_heads_path):
    payload = load_json_object(selected_heads_path)
    head_sets = {"paper": validate_head_set("paper", PAPER_HEADS)}

    for condition, payload_key in SELECTED_HEAD_KEYS.items():
        if payload_key not in payload or not isinstance(payload[payload_key], list):
            raise ValueError(
                f"{selected_heads_path}: missing list field {payload_key!r}"
            )
        pairs = []
        for position, row in enumerate(payload[payload_key]):
            if not isinstance(row, dict) or "layer" not in row or "head" not in row:
                raise ValueError(
                    f"{selected_heads_path}: {payload_key}[{position}] lacks layer/head"
                )
            pairs.append((row["layer"], row["head"]))
        pairs = validate_head_set(condition, pairs)
        if pairs != EXPECTED_S_HEADS[condition]:
            raise ValueError(
                f"{selected_heads_path}: {condition} heads do not match the fixed "
                f"expected set; expected {EXPECTED_S_HEADS[condition]}, found {pairs}"
            )
        head_sets[condition] = pairs
    return head_sets


def _validate_index_list(manifest, field, expected_size):
    values = manifest.get(field)
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise ValueError(f"split manifest field {field!r} must be a list of integers")
    if len(values) != expected_size:
        raise ValueError(
            f"split manifest {field} has {len(values)} entries, expected {expected_size}"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"split manifest {field} contains duplicate indices")
    return values


def validate_split_manifest(
    manifest,
    *,
    expected_total=None,
    expected_selection_size=None,
    expected_evaluation_size=None,
):
    for field in ("total_count", "selection_size", "evaluation_size", "seed"):
        if type(manifest.get(field)) is not int:
            raise ValueError(f"split manifest field {field!r} must be an integer")

    total_count = manifest["total_count"]
    selection_size = manifest["selection_size"]
    evaluation_size = manifest["evaluation_size"]
    if total_count <= 0 or selection_size < 0 or evaluation_size < 0:
        raise ValueError("split manifest counts must be non-negative and total positive")
    if selection_size + evaluation_size != total_count:
        raise ValueError("split manifest selection/evaluation sizes do not sum to total")

    if expected_total is not None and total_count != expected_total:
        raise ValueError(f"expected total_count={expected_total}, found {total_count}")
    if (
        expected_selection_size is not None
        and selection_size != expected_selection_size
    ):
        raise ValueError(
            f"expected selection_size={expected_selection_size}, found {selection_size}"
        )
    if (
        expected_evaluation_size is not None
        and evaluation_size != expected_evaluation_size
    ):
        raise ValueError(
            f"expected evaluation_size={expected_evaluation_size}, found {evaluation_size}"
        )

    selection_indices = _validate_index_list(
        manifest, "selection_indices", selection_size
    )
    evaluation_indices = _validate_index_list(
        manifest, "evaluation_indices", evaluation_size
    )
    overlap = set(selection_indices) & set(evaluation_indices)
    if overlap:
        raise ValueError(
            "selection and evaluation indices overlap: "
            + ", ".join(str(index) for index in sorted(overlap))
        )
    union = set(selection_indices) | set(evaluation_indices)
    if union != set(range(total_count)):
        raise ValueError(
            "selection/evaluation union does not equal all source indices "
            f"0..{total_count - 1}"
        )
    return selection_indices, evaluation_indices


def validate_fixed_experiment_manifest(manifest):
    selection_indices, evaluation_indices = validate_split_manifest(
        manifest,
        expected_total=EXPECTED_TOTAL_COUNT,
        expected_selection_size=EXPECTED_SELECTION_SIZE,
        expected_evaluation_size=EXPECTED_EVALUATION_SIZE,
    )
    if manifest["seed"] != SEED:
        raise ValueError(f"expected split seed={SEED}, found {manifest['seed']}")
    if selection_indices != EXPECTED_SELECTION_INDICES:
        raise ValueError(
            "split manifest selection_indices do not match the fixed seed-42 "
            "AutoDAN hold-out"
        )
    expected_selection_set = set(EXPECTED_SELECTION_INDICES)
    expected_evaluation_indices = [
        index
        for index in range(EXPECTED_TOTAL_COUNT)
        if index not in expected_selection_set
    ]
    if evaluation_indices != expected_evaluation_indices:
        raise ValueError(
            "split manifest evaluation_indices do not match the fixed seed-42 "
            "AutoDAN hold-out"
        )
    expected_fields = {
        "model": "Llama-2-7b-chat-hf",
        "attack": "AutoDAN",
        "selection_from_baseline_success_only": True,
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"split manifest expected {field}={expected!r}, "
                f"found {manifest.get(field)!r}"
            )
    return selection_indices, evaluation_indices


def _index_unique_rows(rows, index_key, source_name):
    indexed = {}
    for row_number, row in enumerate(rows):
        if index_key not in row or type(row[index_key]) is not int:
            raise ValueError(
                f"{source_name}: row {row_number} lacks integer {index_key!r}"
            )
        index = row[index_key]
        if index in indexed:
            raise ValueError(f"{source_name}: duplicate {index_key} {index}")
        indexed[index] = row
    return indexed


def align_evaluation_records(
    input_rows,
    manifest,
    baseline_rows,
    source_rows,
    *,
    input_name="input",
    baseline_name="baseline results",
    source_name="source labels",
):
    selection_indices, evaluation_indices = validate_split_manifest(manifest)
    if len(input_rows) != manifest["evaluation_size"]:
        raise ValueError(
            f"{input_name}: expected {manifest['evaluation_size']} evaluation rows, "
            f"found {len(input_rows)}"
        )

    actual_indices = []
    for row_number, row in enumerate(input_rows):
        if "source_index" not in row or type(row["source_index"]) is not int:
            raise ValueError(
                f"{input_name}: row {row_number} lacks an integer source_index"
            )
        if "prompt_id" not in row:
            raise ValueError(f"{input_name}: row {row_number} lacks prompt_id")
        if "prompt" not in row or not isinstance(row["prompt"], str):
            raise ValueError(f"{input_name}: row {row_number} lacks a string prompt")
        actual_indices.append(row["source_index"])

    if actual_indices != evaluation_indices:
        leaked = sorted(set(actual_indices) & set(selection_indices))
        if leaked:
            raise ValueError(
                f"{input_name}: selection indices appear in evaluation input: {leaked}"
            )
        raise ValueError(
            f"{input_name}: source_index order/content does not match evaluation_indices"
        )

    baseline_by_index = _index_unique_rows(
        baseline_rows, "source_index", baseline_name
    )
    source_by_index = _index_unique_rows(source_rows, "index", source_name)
    prepared = []
    for row in input_rows:
        source_index = row["source_index"]
        baseline = baseline_by_index.get(source_index)
        if baseline is None:
            raise ValueError(
                f"{baseline_name}: missing evaluation source_index {source_index}"
            )
        source = source_by_index.get(source_index)
        if source is None:
            raise ValueError(f"{source_name}: missing evaluation index {source_index}")

        if baseline.get("prompt") != row["prompt"]:
            raise ValueError(
                f"prompt mismatch between {input_name} and {baseline_name} at "
                f"source_index {source_index}"
            )
        if source.get("user prompt") != row["prompt"]:
            raise ValueError(
                f"prompt mismatch between {input_name} and {source_name} at "
                f"source_index {source_index}"
            )
        if type(baseline.get("success")) is not bool:
            raise ValueError(
                f"{baseline_name}: success at source_index {source_index} is not boolean"
            )
        if baseline.get("generation_failed") is not False:
            raise ValueError(
                f"{baseline_name}: source_index {source_index} is not a completed "
                "baseline generation"
            )
        if type(source.get("success")) is not bool:
            raise ValueError(
                f"{source_name}: success at index {source_index} is not boolean"
            )
        if not isinstance(baseline.get("formatted_prompt"), str):
            raise ValueError(
                f"{baseline_name}: formatted_prompt at source_index {source_index} "
                "is unavailable"
            )

        prepared.append(
            {
                "source_index": source_index,
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "baseline_success": baseline["success"],
                "baseline_response_length_chars": baseline.get(
                    "response_length_chars"
                ),
                "baseline_response_length_tokens": baseline.get(
                    "response_length_tokens"
                ),
                "source_success": source["success"],
                "_baseline_formatted_prompt": baseline["formatted_prompt"],
            }
        )
    return prepared


def format_and_validate_prompts(records, tokenizer, model_config):
    prompts = [record["prompt"] for record in records]
    formatted_prompts = apply_template(
        prompts=prompts,
        system_prompt=None,
        tokenizer=tokenizer,
        model_config=model_config,
    )
    if len(formatted_prompts) != len(records):
        raise ValueError("prompt formatter returned an unexpected number of prompts")

    formatted_records = []
    for record, formatted_prompt in zip(records, formatted_prompts):
        if formatted_prompt != record["_baseline_formatted_prompt"]:
            raise ValueError(
                "formatted prompt mismatch with baseline at source_index "
                f"{record['source_index']}"
            )
        public_record = {
            key: value for key, value in record.items() if not key.startswith("_")
        }
        public_record["formatted_prompt"] = formatted_prompt
        formatted_records.append(public_record)
    return formatted_records


def validate_model_and_activation(model_config, activation):
    expected_model = {
        "model_name": "Llama",
        "n_layers": EXPECTED_LAYERS,
        "n_heads": EXPECTED_HEADS,
        "resid_dim": EXPECTED_RESID_DIM,
    }
    for key, expected in expected_model.items():
        if model_config.get(key) != expected:
            raise ValueError(
                f"unsupported model configuration: expected {key}={expected!r}, "
                f"found {model_config.get(key)!r}"
            )
    if not isinstance(activation, torch.Tensor):
        raise ValueError("intervention activation must be a torch.Tensor")
    expected_shape = (
        model_config["n_layers"],
        model_config["n_heads"],
        1,
        model_config["resid_dim"] // model_config["n_heads"],
    )
    if tuple(activation.shape) != expected_shape:
        raise ValueError(
            f"intervention activation shape mismatch: expected {expected_shape}, "
            f"found {tuple(activation.shape)}"
        )


def classify_transition(baseline_success, intervention_success, generation_failed=False):
    if generation_failed or intervention_success is None:
        return None
    if baseline_success and intervention_success:
        return "remained_success"
    if baseline_success and not intervention_success:
        return "defended"
    if not baseline_success and intervention_success:
        return "regressed"
    return "remained_failure"


def _heads_for_json(heads):
    return [{"layer": layer, "head": head} for layer, head in heads]


def evaluate_condition_records(
    records,
    *,
    condition,
    heads,
    interpolation_rate,
    generate_response,
    response_token_length,
    show_progress=False,
):
    results = []
    iterator = tqdm(records, leave=False, desc=condition) if show_progress else records
    selected_heads = _heads_for_json(heads)
    for record in iterator:
        try:
            response = generate_response(record["formatted_prompt"], heads)
            success, diagnostics = check_success_with_details(response)
            generation_failed = False
            failure_reason = None
            response_length_chars = len(response)
            response_length_tokens = response_token_length(response)
            matched_refusal_substrings = diagnostics["matched_refusal_substrings"]
            success_decision_reason = diagnostics["decision_reason"]
            contains_refusal_substring = diagnostics[
                "contains_refusal_substring"
            ]
        except Exception as exc:  # Keep prompt alignment visible after failures.
            response = None
            success = None
            generation_failed = True
            failure_reason = f"{type(exc).__name__}: {exc}"
            response_length_chars = None
            response_length_tokens = None
            matched_refusal_substrings = []
            success_decision_reason = "generation failed; success not evaluated"
            contains_refusal_substring = None

        transition = classify_transition(
            record["baseline_success"], success, generation_failed
        )
        results.append(
            {
                "source_index": record["source_index"],
                "prompt_id": record["prompt_id"],
                "condition": condition,
                "selected_heads": selected_heads,
                "interpolation_rate": interpolation_rate,
                "prompt": record["prompt"],
                "formatted_prompt": record["formatted_prompt"],
                "response": response,
                "success": success,
                "baseline_success": record["baseline_success"],
                "source_success": record["source_success"],
                "success_agreement_with_baseline": (
                    None if success is None else success == record["baseline_success"]
                ),
                "generation_failed": generation_failed,
                "failure_reason": failure_reason,
                "contains_refusal_substring": contains_refusal_substring,
                "matched_refusal_substrings": matched_refusal_substrings,
                "success_decision_reason": success_decision_reason,
                "response_length_chars": response_length_chars,
                "response_length_tokens": response_length_tokens,
                "transition": transition,
            }
        )
    return results


def build_baseline_comparison_results(records):
    results = []
    for record in records:
        success = record["baseline_success"]
        results.append(
            {
                "source_index": record["source_index"],
                "condition": "no_intervention",
                "success": success,
                "baseline_success": success,
                "source_success": record["source_success"],
                "generation_failed": False,
                "response_length_chars": record.get(
                    "baseline_response_length_chars"
                ),
                "response_length_tokens": record.get(
                    "baseline_response_length_tokens"
                ),
                "transition": (
                    "remained_success" if success else "remained_failure"
                ),
            }
        )
    return results


def _mean_or_none(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize_condition(results, condition, heads, interpolation_rate):
    evaluated = [row for row in results if not row["generation_failed"]]
    baseline_success_count = sum(row["baseline_success"] for row in results)
    baseline_failure_count = len(results) - baseline_success_count
    transitions = {
        transition: sum(row["transition"] == transition for row in results)
        for transition in (
            "defended",
            "remained_success",
            "regressed",
            "remained_failure",
        )
    }
    source_subset = [row for row in results if row["source_success"]]
    source_subset_evaluated = [
        row for row in source_subset if not row["generation_failed"]
    ]
    success_count = sum(row["success"] is True for row in evaluated)
    source_subset_success_count = sum(
        row["success"] is True for row in source_subset_evaluated
    )

    return {
        "condition": condition,
        "selected_heads": _heads_for_json(heads),
        "interpolation_rate": interpolation_rate,
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
        "source_success_subset_count": len(source_subset),
        "source_success_subset_evaluated_count": len(source_subset_evaluated),
        "source_success_subset_success_count": source_subset_success_count,
        "source_success_subset_asr": (
            source_subset_success_count / len(source_subset_evaluated)
            if source_subset_evaluated
            else None
        ),
        "generation_failure_count": len(results) - len(evaluated),
        "generation_failure_indices": [
            row["source_index"] for row in results if row["generation_failed"]
        ],
        "mean_response_length_chars": _mean_or_none(
            [row.get("response_length_chars") for row in evaluated]
        ),
        "mean_response_length_tokens": _mean_or_none(
            [row.get("response_length_tokens") for row in evaluated]
        ),
    }


def _head_list_string(heads):
    return ";".join(f"{layer},{head}" for layer, head in heads)


def comparison_row(summary, heads):
    row = {key: summary.get(key) for key in COMPARISON_COLUMNS}
    row["head_list"] = _head_list_string(heads)
    return row


def build_comparison_rows(baseline_summary, condition_summaries, head_sets):
    rows = [comparison_row(baseline_summary, [])]
    for summary in condition_summaries:
        rows.append(comparison_row(summary, head_sets[summary["condition"]]))
    return rows


def write_comparison_outputs(output_dir, rows, context):
    output_dir = Path(output_dir)
    json_path = output_dir / "comparison_summary.json"
    csv_path = output_dir / "comparison_summary.csv"
    write_json(json_path, {**context, "rows": rows})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def check_output_paths(output_dir, conditions, overwrite):
    output_dir = Path(output_dir)
    paths = [
        output_dir / "comparison_summary.csv",
        output_dir / "comparison_summary.json",
    ]
    for condition in conditions:
        paths.extend(
            [
                output_dir / condition / "prompt_results.jsonl",
                output_dir / condition / "summary.json",
            ]
        )
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing output(s): " + ", ".join(existing)
        )


def build_metadata(
    *,
    args,
    model,
    tokenizer,
    model_config,
    manifest,
    condition,
    heads,
    evaluated_prompt_count,
    command,
    git_metadata,
):
    selection_methods = {
        "paper": "fixed Llama paper heads from evaluate.py",
        "positive": "selection-20 mean_delta_asym_ratio positive top-5",
        "negative": "selection-20 mean_delta_asym_ratio negative bottom-5",
        "absolute": "selection-20 abs(mean_delta_asym_ratio) top-5",
    }
    compact_model_config = {
        key: model_config.get(key)
        for key in ("model_name", "name_or_path", "n_layers", "n_heads", "resid_dim")
    }
    generation = getattr(model, "generation_config", None)
    return {
        "model_path": str(Path(args.model).resolve()),
        "model_config": compact_model_config,
        "model_config_type": getattr(model.config, "model_type", None),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "dataset": "AutoDAN 130 prompt pairs; fixed 20/110 internal hold-out",
        "attack": "AutoDAN",
        "input_file": str(Path(args.input_file).resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "baseline_results": str(Path(args.baseline_results).resolve()),
        "source_labels": str(Path(args.source_labels).resolve()),
        "intervention_activation": str(
            Path(args.intervention_activation).resolve()
        ),
        "intervention_activation_shape": [
            EXPECTED_LAYERS,
            EXPECTED_HEADS,
            1,
            EXPECTED_RESID_DIM // EXPECTED_HEADS,
        ],
        "intervention_enabled": True,
        "intervened_head_num": len(heads),
        "intervention_method": (
            "utils.intervention_utils.generate_with_intervention; selected "
            "attention-head output replacement/interpolation at o_proj input"
        ),
        "interpolation_rate": args.interpolation_rate,
        "generation_config": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "top_k_passed_to_generate": False,
            "num_beams": getattr(generation, "num_beams", 1),
            "max_length": getattr(generation, "max_length", None),
            "eos_token_id": getattr(generation, "eos_token_id", None),
            "pad_token_id": getattr(generation, "pad_token_id", None),
            "bos_token_id": getattr(generation, "bos_token_id", None),
            "system_prompt": None,
            "seed": SEED,
            "batch_size": 1,
            "add_special_tokens": False,
            "attention_mask_passed_to_generate": False,
            "decode_skip_special_tokens": False,
        },
        "chat_template": getattr(tokenizer, "chat_template", None),
        "prompt_format": (
            "utils.prompt_utils.apply_template; user message only; "
            "tokenizer.apply_chat_template(tokenize=False)"
        ),
        "success_checker": "utils.evaluate_utils.check_success_with_details",
        "success_checker_rule": (
            "success iff none of the configured refusal substrings occurs; "
            "case-sensitive direct substring matching with no normalization"
        ),
        "refusal_substring_count": len(REFUSAL_SUBSTRINGS),
        "refusal_substring_source": "utils.evaluate_utils.REFUSAL_SUBSTRINGS",
        "condition": condition,
        "selected_heads": _heads_for_json(heads),
        "selected_heads_file": str(Path(args.selected_heads).resolve()),
        "selection_method": selection_methods[condition],
        "selection_size": manifest["selection_size"],
        "evaluation_size": manifest["evaluation_size"],
        "evaluated_prompt_count": evaluated_prompt_count,
        "seed": manifest["seed"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "research_context_note": INTERNAL_HOLDOUT_NOTE,
        "generation_failure_policy": (
            "success and transition are null; failed rows are excluded from "
            "ASR and source-success-subset ASR denominators"
        ),
        "response_length_tokens_method": (
            "re-tokenize the decoded response with add_special_tokens=False"
        ),
        **git_metadata,
    }


def main(argv=None):
    args = parse_args(argv)
    conditions = parse_conditions(args.conditions)
    if not math.isclose(
        args.interpolation_rate,
        EXPECTED_INTERPOLATION_RATE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "this comparison fixes --interpolation-rate at "
            f"{EXPECTED_INTERPOLATION_RATE}"
        )
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive")
    check_output_paths(args.output_dir, conditions, args.overwrite)

    manifest = load_json_object(args.split_manifest)
    validate_fixed_experiment_manifest(manifest)
    head_sets = load_head_sets(args.selected_heads)
    input_rows = read_records(args.input_file)
    baseline_rows = read_records(args.baseline_results)
    source_rows = read_records(args.source_labels)
    aligned_records = align_evaluation_records(
        input_rows,
        manifest,
        baseline_rows,
        source_rows,
        input_name=args.input_file,
        baseline_name=args.baseline_results,
        source_name=args.source_labels,
    )
    if args.max_prompts is not None and args.max_prompts > len(aligned_records):
        raise ValueError(
            f"--max-prompts={args.max_prompts} exceeds the evaluation size "
            f"{len(aligned_records)}"
        )

    intervention_activation = torch.load(
        args.intervention_activation, map_location="cpu"
    )
    validate_model_and_activation(
        {
            "model_name": "Llama",
            "n_layers": EXPECTED_LAYERS,
            "n_heads": EXPECTED_HEADS,
            "resid_dim": EXPECTED_RESID_DIM,
        },
        intervention_activation,
    )
    set_seed(SEED)
    from utils.intervention_utils import generate_with_intervention
    from utils.model_utils import load_model_and_tokenizer

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    model.eval()
    validate_model_and_activation(model_config, intervention_activation)
    for condition in ALL_CONDITIONS:
        validate_head_set(
            condition,
            head_sets[condition],
            n_layers=model_config["n_layers"],
            n_heads=model_config["n_heads"],
        )

    # Validate all 110 formatted prompts before applying a smoke-test limit.
    formatted_records = format_and_validate_prompts(
        aligned_records, tokenizer, model_config
    )
    records = (
        formatted_records
        if args.max_prompts is None
        else formatted_records[: args.max_prompts]
    )
    generation_config = {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    def generate_response(formatted_prompt, heads):
        return generate_with_intervention(
            prompt=formatted_prompt,
            intervention_locations=heads,
            interpolation_rate=args.interpolation_rate,
            intervention_activation=intervention_activation,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )

    def response_token_length(response):
        token_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        return len(token_ids)

    command = shlex.join([sys.executable, *(sys.argv if argv is None else argv)])
    git_metadata = get_git_metadata(REPO_ROOT)
    condition_summaries = []
    for condition in conditions:
        heads = head_sets[condition]
        results = evaluate_condition_records(
            records,
            condition=condition,
            heads=heads,
            interpolation_rate=args.interpolation_rate,
            generate_response=generate_response,
            response_token_length=response_token_length,
            show_progress=True,
        )
        summary = summarize_condition(
            results, condition, heads, args.interpolation_rate
        )
        summary["metadata"] = build_metadata(
            args=args,
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            manifest=manifest,
            condition=condition,
            heads=heads,
            evaluated_prompt_count=len(records),
            command=command,
            git_metadata=git_metadata,
        )
        condition_dir = Path(args.output_dir) / condition
        write_jsonl(condition_dir / "prompt_results.jsonl", results)
        write_json(condition_dir / "summary.json", summary)
        condition_summaries.append(summary)

    baseline_summary = summarize_condition(
        build_baseline_comparison_results(records),
        "no_intervention",
        [],
        0.0,
    )
    comparison_rows = build_comparison_rows(
        baseline_summary, condition_summaries, head_sets
    )
    write_comparison_outputs(
        args.output_dir,
        comparison_rows,
        {
            "research_context_note": INTERNAL_HOLDOUT_NOTE,
            "baseline_source": str(Path(args.baseline_results).resolve()),
            "baseline_regenerated": False,
            "evaluated_prompt_count": len(records),
            "evaluation_size": manifest["evaluation_size"],
            "conditions": conditions,
            "rows_include_no_intervention_from_saved_baseline": True,
        },
    )

    print(f"validated evaluation prompts: {len(aligned_records)}")
    print(f"evaluated prompts per condition: {len(records)}")
    print(f"baseline successes in evaluated rows: {baseline_summary['success_count']}")
    for summary in condition_summaries:
        print(
            f"{summary['condition']}: {summary['success_count']}/"
            f"{summary['evaluated_count']} successes; "
            f"generation failures={summary['generation_failure_count']}"
        )
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
