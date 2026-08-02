#!/usr/bin/env python3
"""Generate and diagnose original-prompt responses for the fixed GCG selection."""

from __future__ import annotations

import argparse
import json
import random
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
from utils.gcg_wasserstein_reference import (  # noqa: E402
    read_json,
    read_jsonl,
    selection_records,
    sha256_file,
    sha256_text,
    utc_now,
    verify_artifact_directory,
    write_json,
    write_jsonl,
)
from utils.prompt_utils import apply_template  # noqa: E402


DEFAULT_MODEL = (
    "/mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf"
)
DEFAULT_ORIGINAL = "data/derived/llama2/gcg/original_prompts.jsonl"
RESULT_NAME = "selection_original_prompt_results.jsonl"
SUMMARY_NAME = "selection_original_refusal_summary.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--original-input", default=DEFAULT_ORIGINAL)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_and_tokenizer(model_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for original-prompt diagnosis")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama import modeling_llama

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model.eval()
    model.config.use_cache = False
    if type(model.model.layers[0].self_attn) is not modeling_llama.LlamaAttention:
        raise TypeError("original-prompt diagnosis requires eager LlamaAttention")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str) -> tuple[str, int, str]:
    formatted_prompt = apply_template(
        [prompt],
        system_prompt=None,
        tokenizer=tokenizer,
        model_config={"model_name": "Llama"},
    )[0]
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)
    prompt_length = int(inputs.input_ids.shape[-1])
    generation_config = {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "max_new_tokens": 32,
        "use_cache": False,
    }
    with torch.inference_mode(), temporarily_disable_default_dynamic_cache(model):
        output_ids = model.generate(inputs.input_ids, **generation_config)
    response_ids = output_ids[0, prompt_length:]
    return (
        tokenizer.decode(response_ids),
        int(response_ids.numel()),
        formatted_prompt,
    )


def main(argv=None):
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    verification = verify_artifact_directory(run_dir)
    result_path = run_dir / RESULT_NAME
    summary_path = run_dir / SUMMARY_NAME
    for path in (result_path, summary_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    split_path = run_dir / "split_manifest.json"
    split_manifest = read_json(split_path)
    if sha256_file(args.original_input) != split_manifest["original_input_sha256"]:
        raise ValueError("original input hash does not match split manifest")
    original_rows = read_jsonl(args.original_input)
    selected = selection_records(split_manifest)
    set_seed(42)
    model, tokenizer = load_model_and_tokenizer(args.model)

    rows = []
    for selection_position, record in enumerate(selected):
        row_position = record["row_position"]
        dataset_index = record["dataset_index"]
        source = original_rows[row_position]
        if source.get("index") != dataset_index:
            raise ValueError(
                f"original input dataset_index mismatch at row_position={row_position}"
            )
        if sha256_text(source.get("prompt", "")) != record["original_prompt_sha256"]:
            raise ValueError(f"original prompt hash mismatch at dataset_index={dataset_index}")
        print(
            f"[{selection_position + 1}/{len(selected)}] "
            f"row_position={row_position} dataset_index={dataset_index}"
        )
        try:
            response, response_tokens, formatted_prompt = generate_response(
                model,
                tokenizer,
                source["prompt"],
            )
            success, detail = check_success_with_details(response)
            generation_failed = False
            generation_failure_reason = None
            refusal = not success
            non_refusal = success
        except Exception as exc:
            response = None
            response_tokens = None
            formatted_prompt = None
            detail = {
                "matched_refusal_substrings": [],
                "decision_reason": "generation failed; refusal not evaluated",
            }
            generation_failed = True
            generation_failure_reason = f"{type(exc).__name__}: {exc}"
            refusal = None
            non_refusal = None
        rows.append(
            {
                "selection_position": selection_position,
                "row_position": row_position,
                "dataset_index": dataset_index,
                "prompt_id": record["prompt_id"],
                "prompt": source["prompt"],
                "original_prompt_sha256": record["original_prompt_sha256"],
                "formatted_prompt": formatted_prompt,
                "response": response,
                "response_length_tokens": response_tokens,
                "refusal": refusal,
                "non_refusal": non_refusal,
                "generation_failed": generation_failed,
                "generation_failure_reason": generation_failure_reason,
                "matched_refusal_substrings": detail[
                    "matched_refusal_substrings"
                ],
                "decision_reason": detail["decision_reason"],
            }
        )

    refusal_indices = [
        row["dataset_index"] for row in rows if row["refusal"] is True
    ]
    non_refusal_indices = [
        row["dataset_index"] for row in rows if row["non_refusal"] is True
    ]
    failure_indices = [
        row["dataset_index"] for row in rows if row["generation_failed"]
    ]
    summary = {
        "reference_name": "original reference",
        "alternative_name": "non-jailbreak reference",
        "selection_count": len(rows),
        "refusal_count": len(refusal_indices),
        "non_refusal_count": len(non_refusal_indices),
        "refusal_dataset_indices": refusal_indices,
        "non_refusal_dataset_indices": non_refusal_indices,
        "generation_failure_count": len(failure_indices),
        "generation_failure_dataset_indices": failure_indices,
        "diagnostic_role": (
            "diagnostic only; not used for Head ranking or reference construction"
        ),
        "checker": "check_success_with_details case-sensitive refusal substrings",
        "generation_settings": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": 32,
            "use_cache": False,
            "batch_size": 1,
            "system_prompt": None,
            "seed": 42,
        },
        "model": str(Path(args.model).resolve()),
        "original_input": str(Path(args.original_input).resolve()),
        "original_input_sha256": sha256_file(args.original_input),
        "selection_manifest_sha256": verification[
            "selection_manifest_sha256"
        ],
        "created_at": utc_now(),
    }
    write_jsonl(result_path, rows)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
