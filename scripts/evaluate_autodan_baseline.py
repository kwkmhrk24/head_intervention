#!/usr/bin/env python3
"""Save prompt-level diagnostics for true no-intervention AutoDAN evaluation.

The default generation kwargs intentionally match ``evaluate.py``.  This
script never loads intervention activations and never installs hooks.
"""

import argparse
import json
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.evaluate_utils import (  # noqa: E402
    REFUSAL_SUBSTRINGS,
    check_success_with_details,
)
from utils.prompt_utils import apply_template  # noqa: E402


DEFAULT_INPUT = "data/derived/llama2/autodan/jailbreak_prompts.jsonl"
DEFAULT_SOURCE_LABELS = "data/raw/llama2/llama2_AutoDAN.json"
DEFAULT_OUTPUT_JSONL = (
    "data/outputs/llama2/autodan/baseline_alignment/"
    "no_intervention_prompt_results.jsonl"
)
DEFAULT_SUMMARY_JSON = (
    "data/outputs/llama2/autodan/baseline_alignment/"
    "no_intervention_summary.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Llama2 AutoDAN without intervention and save prompt-level "
            "responses plus success-checker diagnostics."
        )
    )
    parser.add_argument("--model", required=True, help="Local Llama2 model path.")
    parser.add_argument("--input-file", default=DEFAULT_INPUT)
    parser.add_argument("--source-labels", default=DEFAULT_SOURCE_LABELS)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Defaults to evaluate.py's current 32-token setting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Recorded and applied for reproducibility. The default greedy "
            "generation does not sample."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Defaults to no system message, matching current AutoDAN evaluation.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional leading-row limit for smoke tests; omit for all rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the two explicitly selected output files.",
    )
    return parser.parse_args()


def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}: invalid JSON on line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: line {line_number} is not an object")
                rows.append(row)
        return rows

    with path.open() as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected a JSON list of objects")
    return rows


def build_source_label_index(source_rows, source_path):
    by_index = {}
    for row_number, row in enumerate(source_rows):
        missing = [key for key in ("index", "success") if key not in row]
        if missing:
            raise ValueError(
                f"{source_path}: row {row_number} missing keys {', '.join(missing)}"
            )
        source_index = row["index"]
        if source_index in by_index:
            raise ValueError(f"{source_path}: duplicate index {source_index!r}")
        if not isinstance(row["success"], bool):
            raise ValueError(
                f"{source_path}: success at index {source_index!r} is not boolean"
            )
        by_index[source_index] = row
    return by_index


def prepare_prompt_records(input_rows, source_rows, input_path, source_path, max_prompts=None):
    if max_prompts is not None and max_prompts <= 0:
        raise ValueError("--max-prompts must be positive")

    source_by_index = build_source_label_index(source_rows, source_path)
    selected_rows = input_rows if max_prompts is None else input_rows[:max_prompts]
    prepared = []
    seen_source_indices = set()

    for row_number, row in enumerate(selected_rows):
        if "prompt" not in row or not isinstance(row["prompt"], str):
            raise ValueError(f"{input_path}: row {row_number} has no string prompt")

        source_index = row.get("source_index", row.get("index", row_number))
        prompt_id = row.get("prompt_id", source_index)
        if source_index in seen_source_indices:
            raise ValueError(f"{input_path}: duplicate source index {source_index!r}")
        seen_source_indices.add(source_index)

        source_row = source_by_index.get(source_index)
        if source_row is None:
            raise ValueError(
                f"{input_path}: source index {source_index!r} is absent from {source_path}"
            )
        source_prompt = source_row.get("user prompt")
        if source_prompt is not None and source_prompt != row["prompt"]:
            raise ValueError(
                f"prompt mismatch for source index {source_index!r} between "
                f"{input_path} and {source_path}"
            )

        prepared.append(
            {
                "source_index": source_index,
                "prompt_id": prompt_id,
                "prompt": row["prompt"],
                "source_success": source_row["success"],
            }
        )
    return prepared


def evaluate_prompt_records(prompt_records, format_prompt, generate_response, show_progress=False):
    results = []
    iterator = tqdm(prompt_records, leave=False) if show_progress else prompt_records

    for record in iterator:
        formatted_prompt = None
        try:
            formatted_prompt = format_prompt(record["prompt"])
            response, response_length_tokens = generate_response(formatted_prompt)
            success, diagnostics = check_success_with_details(response)
            result = {
                **record,
                "formatted_prompt": formatted_prompt,
                "response": response,
                "success": success,
                "source_success": record["source_success"],
                "success_agreement": success == record["source_success"],
                "generation_failed": False,
                "failure_reason": None,
                "response_length_chars": len(response),
                "response_length_tokens": response_length_tokens,
                "contains_refusal_substring": diagnostics[
                    "contains_refusal_substring"
                ],
                "matched_refusal_substrings": diagnostics[
                    "matched_refusal_substrings"
                ],
                "success_decision_reason": diagnostics["decision_reason"],
            }
        except Exception as exc:  # Preserve a row and diagnose per-prompt failures.
            result = {
                **record,
                "formatted_prompt": formatted_prompt,
                "response": None,
                "success": None,
                "source_success": record["source_success"],
                "success_agreement": None,
                "generation_failed": True,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "response_length_chars": None,
                "response_length_tokens": None,
                "contains_refusal_substring": None,
                "matched_refusal_substrings": [],
                "success_decision_reason": "generation failed; success not evaluated",
            }
        results.append(result)

    return results


def summarize_results(results):
    evaluated = [row for row in results if not row["generation_failed"]]
    agreement_count = sum(row["success_agreement"] for row in evaluated)

    def count_pair(source_success, current_success):
        return sum(
            row["source_success"] is source_success
            and row["success"] is current_success
            for row in evaluated
        )

    both_success = count_pair(True, True)
    source_only = count_pair(True, False)
    current_only = count_pair(False, True)
    both_failure = count_pair(False, False)
    source_true_current_false = [
        row["source_index"]
        for row in evaluated
        if row["source_success"] is True and row["success"] is False
    ]
    source_false_current_true = [
        row["source_index"]
        for row in evaluated
        if row["source_success"] is False and row["success"] is True
    ]

    return {
        "total_count": len(results),
        "evaluated_count": len(evaluated),
        "source_success_count": sum(row["source_success"] for row in results),
        "current_success_count": sum(row["success"] is True for row in evaluated),
        "both_success_count": both_success,
        "source_only_success_count": source_only,
        "current_only_success_count": current_only,
        "both_failure_count": both_failure,
        "agreement_count": agreement_count,
        "agreement_rate": (
            agreement_count / len(evaluated) if evaluated else None
        ),
        "generation_failure_count": sum(
            row["generation_failed"] for row in results
        ),
        "confusion_matrix": [
            {"source_success": True, "current_success": True, "count": both_success},
            {"source_success": True, "current_success": False, "count": source_only},
            {"source_success": False, "current_success": True, "count": current_only},
            {"source_success": False, "current_success": False, "count": both_failure},
        ],
        "source_true_current_false_indices": source_true_current_false,
        "source_false_current_true_indices": source_false_current_true,
        "generation_failure_indices": [
            row["source_index"] for row in results if row["generation_failed"]
        ],
    }


def get_git_metadata(repo_root):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {
            "git_commit": commit,
            "git_commit_available": True,
            "git_commit_failure_reason": None,
            "git_worktree_dirty": bool(status.strip()),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "git_commit": None,
            "git_commit_available": False,
            "git_commit_failure_reason": f"{type(exc).__name__}: {exc}",
            "git_worktree_dirty": None,
        }


def build_run_metadata(
    *,
    model_path,
    model,
    tokenizer,
    model_config,
    input_file,
    source_labels,
    total_count,
    max_new_tokens,
    seed,
    system_prompt,
    command,
    git_metadata,
):
    model_generation_config = model.generation_config
    chat_template = getattr(tokenizer, "chat_template", None)
    model_revision = getattr(model.config, "_commit_hash", None)

    metadata = {
        "model_path": str(Path(model_path).resolve()),
        "model_type": model_config["model_name"],
        "model_config_type": getattr(model.config, "model_type", None),
        "model_name_or_path": model_config.get("name_or_path"),
        "model_revision": model_revision,
        "input_file": str(Path(input_file).resolve()),
        "source_label_file": str(Path(source_labels).resolve()),
        "total_count": total_count,
        "max_new_tokens": max_new_tokens,
        "max_length": getattr(model_generation_config, "max_length", None),
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": getattr(model_generation_config, "top_k", None),
        "top_k_passed_to_generate": False,
        "num_beams": getattr(model_generation_config, "num_beams", 1),
        "seed": seed,
        "batch_size": 1,
        "eos_token_id": getattr(model_generation_config, "eos_token_id", None),
        "pad_token_id": getattr(model_generation_config, "pad_token_id", None),
        "bos_token_id": getattr(model_generation_config, "bos_token_id", None),
        "generation_kwargs": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": max_new_tokens,
        },
        "chat_template": chat_template,
        "prompt_format": (
            "utils.prompt_utils.apply_template; tokenizer.apply_chat_template; "
            "tokenize=False; add_generation_prompt not passed for Llama"
        ),
        "system_prompt": system_prompt,
        "tokenization": {
            "add_special_tokens": False,
            "attention_mask_passed_to_generate": False,
            "decode_skip_special_tokens": False,
            "response_slice": "generated token ids after input token length",
        },
        "success_checker": "utils.evaluate_utils.check_success_with_details",
        "success_checker_rule": (
            "success iff no configured refusal substring occurs in the decoded "
            "response; direct case-sensitive matching with no normalization"
        ),
        "empty_response_is_success": True,
        "refusal_substring_source": "utils.evaluate_utils.REFUSAL_SUBSTRINGS",
        "refusal_substrings": list(REFUSAL_SUBSTRINGS),
        "intervention_enabled": False,
        "intervention_hooks_installed": False,
        "intervention_activation_loaded": False,
        "generation_failure_policy": (
            "save the prompt row with success=null; exclude it from the "
            "success confusion matrix and agreement denominator"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        **git_metadata,
    }
    return metadata


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def check_output_paths(output_jsonl, summary_json, overwrite):
    output_paths = [Path(output_jsonl), Path(summary_json)]
    if output_paths[0].resolve() == output_paths[1].resolve():
        raise ValueError("--output-jsonl and --summary-json must be different files")
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing output(s): " + ", ".join(existing)
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    check_output_paths(args.output_jsonl, args.summary_json, args.overwrite)

    input_rows = read_records(args.input_file)
    source_rows = read_records(args.source_labels)
    prompt_records = prepare_prompt_records(
        input_rows=input_rows,
        source_rows=source_rows,
        input_path=args.input_file,
        source_path=args.source_labels,
        max_prompts=args.max_prompts,
    )

    set_seed(args.seed)
    from utils.model_utils import load_model_and_tokenizer

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    if model_config["model_name"] != "Llama":
        raise ValueError("this baseline-alignment script currently supports Llama only")
    model.eval()

    generation_kwargs = {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "max_new_tokens": args.max_new_tokens,
    }

    def format_prompt(prompt):
        return apply_template(
            prompts=[prompt],
            system_prompt=args.system_prompt,
            tokenizer=tokenizer,
            model_config=model_config,
        )[0]

    def generate_response(formatted_prompt):
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        input_length = inputs.input_ids.shape[-1]
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, **generation_kwargs)
        response_ids = outputs[0, input_length:]
        return tokenizer.decode(response_ids), int(response_ids.numel())

    results = evaluate_prompt_records(
        prompt_records,
        format_prompt=format_prompt,
        generate_response=generate_response,
        show_progress=True,
    )

    command = shlex.join([sys.executable, *sys.argv])
    metadata = build_run_metadata(
        model_path=args.model,
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        input_file=args.input_file,
        source_labels=args.source_labels,
        total_count=len(results),
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        system_prompt=args.system_prompt,
        command=command,
        git_metadata=get_git_metadata(REPO_ROOT),
    )
    summary = {**summarize_results(results), "metadata": metadata}

    write_jsonl(args.output_jsonl, results)
    write_json(args.summary_json, summary)

    print(f"prompt results: {args.output_jsonl}")
    print(f"summary: {args.summary_json}")
    print(f"current successes: {summary['current_success_count']}/{summary['evaluated_count']}")
    print(f"source successes: {summary['source_success_count']}/{summary['total_count']}")
    print(f"generation failures: {summary['generation_failure_count']}")


if __name__ == "__main__":
    main()
