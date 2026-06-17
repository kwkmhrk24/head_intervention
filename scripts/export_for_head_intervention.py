import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Qwen2.5 dataset rows for head intervention scripts."
    )
    parser.add_argument("--input", required=True, help="Input master/labeled jsonl.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--only-successful-attacks",
        action="store_true",
        help="Export only rows with attack_success=true for paired prompt files.",
    )
    return parser.parse_args()


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_prompt_jsonl(path, prompts):
    with path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    paired_rows = [row for row in rows if row.get("jailbreak_prompt")]
    if args.only_successful_attacks:
        paired_rows = [row for row in paired_rows if row.get("attack_success") is True]

    original_prompts = [row["original_prompt"] for row in paired_rows]
    jailbreak_prompts = [row["jailbreak_prompt"] for row in paired_rows]
    refusal_prompts = [
        row["original_prompt"]
        for row in paired_rows
        if row.get("direct_refused") is True
    ]

    write_prompt_jsonl(output_dir / "original_prompts.jsonl", original_prompts)
    write_prompt_jsonl(output_dir / "jailbreak_prompts.jsonl", jailbreak_prompts)
    write_prompt_jsonl(output_dir / "refusal_prompts.jsonl", refusal_prompts)

    print(f"input={input_path}")
    print(f"output_dir={output_dir}")
    print(f"input_rows={len(rows)}")
    print(f"paired_rows_exported={len(paired_rows)}")
    print(f"original_prompts={len(original_prompts)}")
    print(f"jailbreak_prompts={len(jailbreak_prompts)}")
    print(f"refusal_prompts={len(refusal_prompts)}")
    print(f"only_successful_attacks={args.only_successful_attacks}")


if __name__ == "__main__":
    main()
