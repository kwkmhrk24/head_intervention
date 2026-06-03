import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create small prompt-only AdvBench subsets for reproduction smoke tests."
    )
    parser.add_argument(
        "--input",
        default="data/advbench.jsonl",
        help="Path to AdvBench jsonl. Each line must contain a prompt key.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where subset jsonl files are written.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of examples to copy.",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    return args


def iter_prompt_records(path, count):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line_number > count:
                break
            record = json.loads(line)
            if "prompt" not in record:
                raise KeyError(f"Missing prompt key at line {line_number}")
            yield {"prompt": record["prompt"]}


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = list(iter_prompt_records(input_path, args.count))
    if len(records) < args.count:
        raise ValueError(
            f"Requested {args.count} examples but only found {len(records)} in {input_path}"
        )

    original_path = output_dir / f"original_prompts_{args.count}.jsonl"
    refusal_path = output_dir / f"refusal_prompts_{args.count}.jsonl"
    write_jsonl(original_path, records)
    write_jsonl(refusal_path, records)

    print(f"Wrote {len(records)} records to {original_path}")
    print(f"Wrote {len(records)} records to {refusal_path}")


if __name__ == "__main__":
    main()
