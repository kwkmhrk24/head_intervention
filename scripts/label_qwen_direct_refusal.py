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
        description="Label direct refusal status for Qwen2.5 seed dataset rows."
    )
    parser.add_argument("--input", required=True, help="Input seed dataset jsonl.")
    parser.add_argument("--output", required=True, help="Output labeled jsonl.")
    parser.add_argument("--model", required=True, help="Model path.")
    parser.add_argument(
        "--save_output_prefix",
        action="store_true",
        help="Save a short generated output prefix in each row.",
    )
    parser.add_argument(
        "--output_prefix_chars",
        type=int,
        default=200,
        help="Number of output characters to keep when --save_output_prefix is set.",
    )
    args = parser.parse_args()
    if args.output_prefix_chars <= 0:
        parser.error("--output_prefix_chars must be positive")
    return args


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_generation_kwargs(tokenizer):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    return {
        "do_sample": False,
        "max_new_tokens": 64,
        "pad_token_id": pad_token_id,
    }


def generate_output(prompt, model, tokenizer, generation_kwargs):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(outputs[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = read_jsonl(input_path)

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    generation_kwargs = get_generation_kwargs(tokenizer)
    prompts = [row["original_prompt"] for row in rows]
    templated_prompts = apply_template(
        prompts=prompts,
        system_prompt=None,
        tokenizer=tokenizer,
        model_config=model_config,
    )

    refused_count = 0
    success_count = 0
    labeled_rows = []
    for row, prompt in zip(rows, templated_prompts):
        output = generate_output(prompt, model, tokenizer, generation_kwargs)
        success = check_success(output)
        refused = not success
        refused_count += int(refused)
        success_count += int(success)

        labeled = dict(row)
        labeled["direct_refused"] = refused
        labeled["direct_success_by_current_rule"] = success
        if args.save_output_prefix:
            labeled["direct_output_prefix"] = output[: args.output_prefix_chars]
        labeled_rows.append(labeled)

    write_jsonl(output_path, labeled_rows)

    total = len(labeled_rows)
    direct_asr = success_count / total if total else 0.0
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"total={total}")
    print(f"refused_count={refused_count}")
    print(f"success_count={success_count}")
    print(f"direct_ASR={direct_asr}")


if __name__ == "__main__":
    main()
