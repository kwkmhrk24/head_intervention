import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.evaluate_utils import check_success
from utils.model_utils import load_model_and_tokenizer
from utils.prompt_utils import apply_template


def get_args():
    parser = argparse.ArgumentParser(description="Select AdvBench prompts that the model safely refuses.")
    parser.add_argument("--model", required=True, help="Llama2 model path.")
    parser.add_argument("--advbench_file", default="data/raw/advbench.jsonl", help="AdvBench JSONL file.")
    parser.add_argument("--exclude_file", default="data/raw/llama2/AutoDAN_llama2.jsonl", help="JSONL file whose records should be excluded from candidates.")
    parser.add_argument("--output_file", default="data/derived/llama2/refusal/advbench_safe20_prompts.jsonl", help="Output JSONL file.")
    parser.add_argument("--num_prompts", type=int, default=20, help="Number of safe rejection prompts to select.")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Maximum new tokens for refusal checking.")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt passed to apply_template().")
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_no} is not a JSON object")
            rows.append(row)
    return rows


def require_keys(rows, path, keys):
    for line_no, row in enumerate(rows, start=1):
        missing = [key for key in keys if key not in row]
        if missing:
            raise ValueError(f"{path}: line {line_no} missing required keys: {', '.join(missing)}")


def get_exclusions(exclude_rows):
    excluded_indices = {row["index"] for row in exclude_rows if "index" in row}
    excluded_prompts = {row["prompt"] for row in exclude_rows if "prompt" in row}
    return excluded_indices, excluded_prompts


def is_excluded(row, excluded_indices, excluded_prompts):
    if "index" in row and excluded_indices:
        return row["index"] in excluded_indices
    if "prompt" in row and excluded_prompts:
        return row["prompt"] in excluded_prompts
    return False


def generate_response(prompt, model, tokenizer, generation_config):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, **generation_config)
    return tokenizer.decode(outputs.squeeze()[inputs.input_ids.shape[-1]:])


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = get_args()
    if args.num_prompts <= 0:
        raise ValueError("--num_prompts must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")

    advbench_rows = read_jsonl(args.advbench_file)
    exclude_rows = read_jsonl(args.exclude_file)
    require_keys(advbench_rows, args.advbench_file, ["index", "prompt"])

    excluded_indices, excluded_prompts = get_exclusions(exclude_rows)
    candidates = [
        row for row in advbench_rows
        if not is_excluded(row, excluded_indices, excluded_prompts)
    ]

    print(f"advbench_file: {args.advbench_file}")
    print(f"exclude_file: {args.exclude_file}")
    print(f"output_file: {args.output_file}")
    print(f"num_prompts target: {args.num_prompts}")
    print(f"num excluded indices: {len(excluded_indices)}")
    print(f"num excluded prompts: {len(excluded_prompts)}")
    print(f"num candidates after exclusion: {len(candidates)}")

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    model.eval()
    generation_config = {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "max_new_tokens": args.max_new_tokens,
    }

    selected = []
    for row in candidates:
        templated_prompt = apply_template(
            prompts=[row["prompt"]],
            system_prompt=args.system_prompt,
            tokenizer=tokenizer,
            model_config=model_config,
        )[0]
        output = generate_response(
            prompt=templated_prompt,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )
        if not check_success(output):
            selected.append({
                "index": row["index"],
                "prompt": row["prompt"],
                "model_output": output,
                "source": "advbench_safe_rejection",
            })

        if len(selected) >= args.num_prompts:
            break

    if len(selected) < args.num_prompts:
        raise ValueError(
            f"Only found {len(selected)} safe rejection prompts, "
            f"but --num_prompts requested {args.num_prompts}"
        )

    write_jsonl(args.output_file, selected)
    selected_indices = [row["index"] for row in selected]
    print(f"num selected: {len(selected)}")
    print(f"selected indices: {selected_indices}")


if __name__ == "__main__":
    main()
