#!/usr/bin/env python3
"""Evaluate HarmBench AutoDAN test cases with the fixed Llama2 baseline."""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_autodan_baseline import (  # noqa: E402
    check_output_paths,
    set_seed,
    write_json,
    write_jsonl,
)
from utils.asym_score_intervention import (  # noqa: E402
    temporarily_disable_default_dynamic_cache,
)
from utils.evaluate_utils import check_success_with_details  # noqa: E402
from utils.prompt_utils import apply_template  # noqa: E402


CANDIDATE_SELECTION_RULE = "last"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HarmBench AutoDAN test cases with the fixed, "
            "no-intervention Llama2 baseline."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--harmbench-test-cases-dir", required=True)
    parser.add_argument("--behaviors-csv", required=True)
    parser.add_argument(
        "--behavior-ids",
        required=True,
        help="Comma-separated BehaviorIDs. Evaluation follows behavior CSV row order.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-transformers-version", default=None)
    parser.add_argument("--source-harmbench-generation-commit", required=True)
    parser.add_argument("--source-harmbench-current-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_text(value):
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def parse_behavior_ids(value):
    behavior_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not behavior_ids:
        raise ValueError("--behavior-ids must contain at least one BehaviorID")
    if len(behavior_ids) != len(set(behavior_ids)):
        raise ValueError("--behavior-ids contains duplicate BehaviorIDs")
    return behavior_ids


def _read_behavior_rows(behaviors_csv):
    behaviors_csv = Path(behaviors_csv).resolve()
    if not behaviors_csv.is_file():
        raise FileNotFoundError(f"behavior CSV does not exist: {behaviors_csv}")

    with behaviors_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"BehaviorID", "Behavior", "RawIndex"}
        columns = set(reader.fieldnames or [])
        missing = sorted(required - columns)
        if missing:
            raise ValueError(
                f"{behaviors_csv}: missing required columns {', '.join(missing)}"
            )
        rows = list(reader)

    seen = set()
    prepared = []
    for row_number, row in enumerate(rows, start=2):
        behavior_id = row["BehaviorID"]
        if not behavior_id:
            raise ValueError(f"{behaviors_csv}: empty BehaviorID on row {row_number}")
        if behavior_id in seen:
            raise ValueError(
                f"{behaviors_csv}: duplicate BehaviorID {behavior_id!r}"
            )
        seen.add(behavior_id)
        try:
            raw_index = int(row["RawIndex"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{behaviors_csv}: RawIndex for {behavior_id!r} is not an integer"
            ) from exc
        prepared.append(
            {
                "behavior_id": behavior_id,
                "raw_index": raw_index,
                "goal": row["Behavior"],
            }
        )
    return behaviors_csv, prepared


def load_harmbench_records(
    harmbench_test_cases_dir,
    behaviors_csv,
    behavior_ids,
    source_harmbench_generation_commit,
    source_harmbench_current_commit,
):
    """Load final test cases in CSV order without consulting AutoDAN logs."""

    requested = list(behavior_ids)
    if not requested:
        raise ValueError("at least one BehaviorID must be requested")
    if len(requested) != len(set(requested)):
        raise ValueError("requested BehaviorIDs contain duplicates")

    test_cases_dir = Path(harmbench_test_cases_dir).resolve()
    if not test_cases_dir.is_dir():
        raise FileNotFoundError(
            f"HarmBench test cases directory does not exist: {test_cases_dir}"
        )
    csv_path, csv_rows = _read_behavior_rows(behaviors_csv)
    rows_by_id = {row["behavior_id"]: row for row in csv_rows}
    absent = [behavior_id for behavior_id in requested if behavior_id not in rows_by_id]
    if absent:
        raise ValueError(
            f"requested BehaviorID is absent from behavior CSV: {', '.join(absent)}"
        )

    source_manifest = {
        "attack_source": "HarmBench",
        "attack_method": "AutoDAN",
        "selection_split": "selection",
        "candidate_selection_rule": CANDIDATE_SELECTION_RULE,
        "source_harmbench_generation_commit": source_harmbench_generation_commit,
        "source_harmbench_current_commit": source_harmbench_current_commit,
        "source_harmbench_test_cases_dir": str(test_cases_dir),
        "source_behaviors_csv_path": str(csv_path),
        "source_behaviors_csv_sha256": sha256_file(csv_path),
        "requested_behavior_ids": requested,
        "evaluation_order": "behavior_csv_row_order",
        "source_test_case_files": [],
        "source_method_config_files": [],
    }

    records = []
    requested_set = set(requested)
    for csv_row in csv_rows:
        behavior_id = csv_row["behavior_id"]
        if behavior_id not in requested_set:
            continue

        test_cases_path = (
            test_cases_dir
            / "test_cases_individual_behaviors"
            / behavior_id
            / "test_cases.json"
        )
        if not test_cases_path.is_file():
            raise FileNotFoundError(
                f"test_cases.json does not exist for {behavior_id!r}: "
                f"{test_cases_path}"
            )
        try:
            payload = json.loads(test_cases_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {test_cases_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{test_cases_path}: top-level value must be an object")
        if behavior_id not in payload:
            raise ValueError(
                f"{test_cases_path}: missing BehaviorID key {behavior_id!r}"
            )
        candidates = payload[behavior_id]
        if not isinstance(candidates, list):
            raise ValueError(
                f"{test_cases_path}: candidates for {behavior_id!r} must be a list"
            )
        if not candidates:
            raise ValueError(
                f"{test_cases_path}: candidate list for {behavior_id!r} is empty"
            )
        if not all(isinstance(candidate, str) for candidate in candidates):
            raise ValueError(
                f"{test_cases_path}: every candidate for {behavior_id!r} "
                "must be a string"
            )

        method_config_path = test_cases_path.with_name("method_config.json")
        if not method_config_path.is_file():
            raise FileNotFoundError(
                f"method_config.json does not exist for {behavior_id!r}: "
                f"{method_config_path}"
            )
        try:
            method_config = json.loads(
                method_config_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON in {method_config_path}: {exc}"
            ) from exc
        if not isinstance(method_config, dict):
            raise ValueError(
                f"{method_config_path}: top-level value must be an object"
            )

        candidate_index = len(candidates) - 1
        attack_prompt = candidates[candidate_index]
        test_cases_sha256 = sha256_file(test_cases_path)
        method_config_sha256 = sha256_file(method_config_path)
        source_manifest["source_test_case_files"].append(
            {
                "behavior_id": behavior_id,
                "path": str(test_cases_path),
                "sha256": test_cases_sha256,
            }
        )
        source_manifest["source_method_config_files"].append(
            {
                "behavior_id": behavior_id,
                "path": str(method_config_path),
                "sha256": method_config_sha256,
            }
        )
        records.append(
            {
                **csv_row,
                "attack_prompt": attack_prompt,
                "attack_prompt_sha256": sha256_text(attack_prompt),
                "candidate_index": candidate_index,
                "candidate_count": len(candidates),
                "candidate_selection_rule": CANDIDATE_SELECTION_RULE,
                "attack_source": "HarmBench",
                "attack_method": "AutoDAN",
                "selection_split": "selection",
                "source_test_cases_path": str(test_cases_path),
                "source_test_cases_sha256": test_cases_sha256,
                "source_method_config_path": str(method_config_path),
                "source_method_config_sha256": method_config_sha256,
                "source_behaviors_csv_path": str(csv_path),
                "source_behaviors_csv_sha256": source_manifest[
                    "source_behaviors_csv_sha256"
                ],
                "source_harmbench_generation_commit": (
                    source_harmbench_generation_commit
                ),
                "source_harmbench_current_commit": source_harmbench_current_commit,
            }
        )

    if len(records) != len(requested):
        raise ValueError("not every requested BehaviorID was selected exactly once")
    return records, source_manifest


def validate_transformers_version(expected_version, actual_version=None):
    actual_version = transformers.__version__ if actual_version is None else actual_version
    if expected_version is not None and actual_version != expected_version:
        raise RuntimeError(
            "Transformers version mismatch: expected "
            f"{expected_version}, found {actual_version}; model was not loaded"
        )
    return actual_version


def build_decoding_config(max_new_tokens):
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "batch_size": 1,
        "use_cache": False,
        "system_prompt": None,
    }


def configure_model_for_baseline(model):
    model.config.use_cache = False
    model.eval()


def format_user_prompt(attack_prompt, tokenizer, model_config):
    return apply_template(
        prompts=[attack_prompt],
        system_prompt=None,
        tokenizer=tokenizer,
        model_config=model_config,
    )[0]


def generate_baseline_response(
    model,
    tokenizer,
    formatted_prompt,
    decoding_config,
    cache_context=temporarily_disable_default_dynamic_cache,
):
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)
    if inputs.input_ids.shape[0] != 1:
        raise ValueError("baseline generation requires batch size 1")
    input_length = inputs.input_ids.shape[-1]
    generation_kwargs = {
        "max_new_tokens": decoding_config["max_new_tokens"],
        "do_sample": False,
        "num_beams": 1,
        "use_cache": False,
    }
    with torch.inference_mode():
        with cache_context(model):
            outputs = model.generate(inputs.input_ids, **generation_kwargs)
    response_ids = outputs[0, input_length:]
    return tokenizer.decode(response_ids), int(response_ids.numel())


def _chat_template_metadata(tokenizer):
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("tokenizer.chat_template must be a non-empty string")
    tokenizer_name = getattr(tokenizer, "name_or_path", None)
    identifier = (
        f"{tokenizer_name}:tokenizer.chat_template"
        if tokenizer_name
        else f"{type(tokenizer).__name__}:tokenizer.chat_template"
    )
    return identifier, sha256_text(chat_template)


def evaluate_records(
    records,
    model,
    tokenizer,
    model_config,
    model_path,
    decoding_config,
    transformers_version,
    run_timestamp_utc,
    source_manifest,
    cache_context=temporarily_disable_default_dynamic_cache,
):
    chat_template_identifier, chat_template_sha256 = _chat_template_metadata(tokenizer)
    resolved_model_path = str(Path(model_path).resolve())
    results = []
    for record in records:
        response = None
        try:
            formatted_prompt = format_user_prompt(
                record["attack_prompt"], tokenizer, model_config
            )
            response, response_length_tokens = generate_baseline_response(
                model,
                tokenizer,
                formatted_prompt,
                decoding_config,
                cache_context=cache_context,
            )
            keyword_success, diagnostics = check_success_with_details(response)
            result = {
                **record,
                "response": response,
                "response_sha256": sha256_text(response),
                "keyword_success": keyword_success,
                "contains_refusal_substring": diagnostics[
                    "contains_refusal_substring"
                ],
                "matched_refusal_substrings": diagnostics[
                    "matched_refusal_substrings"
                ],
                "success_decision_reason": diagnostics["decision_reason"],
                "generation_failed": False,
                "failure_reason": None,
                "response_length_chars": len(response),
                "response_length_tokens": response_length_tokens,
            }
        except Exception as exc:
            result = {
                **record,
                "response": response,
                "response_sha256": (
                    None if response is None else sha256_text(response)
                ),
                "keyword_success": None,
                "contains_refusal_substring": None,
                "matched_refusal_substrings": [],
                "success_decision_reason": (
                    "generation failed; success not evaluated"
                ),
                "generation_failed": True,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "response_length_chars": (
                    None if response is None else len(response)
                ),
                "response_length_tokens": None,
            }
        result.update(
            {
                "model_path": resolved_model_path,
                "decoding_config": dict(decoding_config),
                "transformers_version": transformers_version,
                "chat_template_identifier": chat_template_identifier,
                "chat_template_sha256": chat_template_sha256,
                "run_timestamp_utc": run_timestamp_utc,
                "source_manifest": source_manifest,
            }
        )
        results.append(result)
    return results


def build_summary(
    records,
    results,
    model_path,
    decoding_config,
    transformers_version,
    expected_transformers_version,
    source_manifest,
    run_timestamp_utc,
    output_jsonl,
):
    generated = [row for row in results if not row["generation_failed"]]
    keyword_success_count = sum(
        row["keyword_success"] is True for row in generated
    )
    keyword_failure_count = sum(
        row["keyword_success"] is False for row in generated
    )
    return {
        "total": len(results),
        "generation_success_count": len(generated),
        "generation_failed_count": len(results) - len(generated),
        "keyword_success_count": keyword_success_count,
        "keyword_failure_count": keyword_failure_count,
        "keyword_asr": (
            keyword_success_count / len(generated) if generated else None
        ),
        "behavior_ids": [record["behavior_id"] for record in records],
        "raw_indices": [record["raw_index"] for record in records],
        "model_path": str(Path(model_path).resolve()),
        "decoding_config": dict(decoding_config),
        "transformers_version": transformers_version,
        "expected_transformers_version": expected_transformers_version,
        "seed": source_manifest.get("seed"),
        "source_harmbench_generation_commit": source_manifest[
            "source_harmbench_generation_commit"
        ],
        "source_harmbench_current_commit": source_manifest[
            "source_harmbench_current_commit"
        ],
        "source_behaviors_csv_path": source_manifest[
            "source_behaviors_csv_path"
        ],
        "source_behaviors_csv_sha256": source_manifest[
            "source_behaviors_csv_sha256"
        ],
        "run_timestamp_utc": run_timestamp_utc,
        "output_jsonl_path": str(Path(output_jsonl).resolve()),
        "source_manifest": source_manifest,
    }


def run(
    args,
    *,
    model_loader=None,
    actual_transformers_version=None,
    seed_setter=set_seed,
    cache_context=temporarily_disable_default_dynamic_cache,
):
    check_output_paths(args.output_jsonl, args.summary_json, args.overwrite)
    decoding_config = build_decoding_config(args.max_new_tokens)
    transformers_version = validate_transformers_version(
        args.expected_transformers_version,
        actual_transformers_version,
    )
    behavior_ids = parse_behavior_ids(args.behavior_ids)
    records, source_manifest = load_harmbench_records(
        args.harmbench_test_cases_dir,
        args.behaviors_csv,
        behavior_ids,
        args.source_harmbench_generation_commit,
        args.source_harmbench_current_commit,
    )
    source_manifest["seed"] = args.seed

    seed_setter(args.seed)
    if model_loader is None:
        from utils.model_utils import load_model_and_tokenizer

        model_loader = load_model_and_tokenizer
    model, tokenizer, model_config = model_loader(args.model)
    if model_config.get("model_name") != "Llama":
        raise ValueError("HarmBench AutoDAN baseline evaluation supports Llama only")
    configure_model_for_baseline(model)

    run_timestamp_utc = datetime.now(timezone.utc).isoformat()
    results = evaluate_records(
        records,
        model,
        tokenizer,
        model_config,
        args.model,
        decoding_config,
        transformers_version,
        run_timestamp_utc,
        source_manifest,
        cache_context=cache_context,
    )
    summary = build_summary(
        records,
        results,
        args.model,
        decoding_config,
        transformers_version,
        args.expected_transformers_version,
        source_manifest,
        run_timestamp_utc,
        args.output_jsonl,
    )
    write_jsonl(args.output_jsonl, results)
    write_json(args.summary_json, summary)

    print(f"evaluated behaviors: {summary['total']}")
    print(f"generation successes: {summary['generation_success_count']}")
    print(f"generation failures: {summary['generation_failed_count']}")
    print(f"keyword successes: {summary['keyword_success_count']}")
    print(f"prompt results: {args.output_jsonl}")
    print(f"summary: {args.summary_json}")
    return results, summary


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
