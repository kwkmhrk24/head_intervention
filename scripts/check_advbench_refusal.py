import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.evaluate_utils import check_success
from utils.model_utils import load_model_and_tokenizer
from utils.prompt_utils import apply_template


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether a model refuses direct AdvBench prompts."
    )
    parser.add_argument(
        "--input_file",
        default="data/advbench.jsonl",
        help="Path to AdvBench jsonl. Each line must contain a prompt key.",
    )
    parser.add_argument("--model", required=True, help="Model path.")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="0-based JSONL row offset to start reading from.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of examples to evaluate.",
    )
    parser.add_argument(
        "--show_prompt",
        action="store_true",
        help="Print the original prompt text from the input file.",
    )
    args = parser.parse_args()
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.count <= 0:
        parser.error("--count must be positive")
    return args


def load_advbench_slice(input_file, start, count):
    rows = []
    end = start + count
    with open(input_file, "r", encoding="utf-8") as f:
        for row_number, line in enumerate(f):
            if row_number < start:
                continue
            if row_number >= end:
                break

            record = json.loads(line)
            if "prompt" not in record:
                raise KeyError(f"Missing prompt key at JSONL row {row_number}")
            rows.append(
                {
                    "row_number": row_number,
                    "index": record.get("index"),
                    "prompt": record["prompt"],
                }
            )

    if len(rows) < count:
        raise ValueError(
            f"Requested {count} examples from row {start}, but only found {len(rows)}"
        )
    return rows


def generate_output(prompt, model, tokenizer, generation_kwargs):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(
        outputs[0, inputs.input_ids.shape[-1]:],
        skip_special_tokens=True,
    )


def main():
    args = parse_args()
    rows = load_advbench_slice(args.input_file, args.start, args.count)

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    prompts = [row["prompt"] for row in rows]
    templated_prompts = apply_template(
        prompts=prompts,
        system_prompt=None,
        tokenizer=tokenizer,
        model_config=model_config,
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    generation_kwargs = {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "max_new_tokens": 64,
        "pad_token_id": pad_token_id,
    }

    success_count = 0
    refused_count = 0
    for row, prompt in zip(rows, templated_prompts):
        output = generate_output(prompt, model, tokenizer, generation_kwargs)
        success_by_current_rule = check_success(output)
        refused = not success_by_current_rule
        success_count += int(success_by_current_rule)
        refused_count += int(refused)

        print("=" * 80)
        print(f"index={row['index']}")
        print(f"jsonl_row={row['row_number']}")
        print(f"success_by_current_rule={success_by_current_rule}")
        print(f"refused={refused}")
        if args.show_prompt:
            print("prompt:")
            print(row["prompt"])
        print("output:")
        print(output)

    total = len(rows)
    asr = success_count / total if total else 0.0
    print("=" * 80)
    print("summary")
    print(f"total={total}")
    print(f"refused_count={refused_count}")
    print(f"success_count={success_count}")
    print(f"ASR={asr}")


if __name__ == "__main__":
    main()
