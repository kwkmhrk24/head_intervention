import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a Qwen2.5 seed dataset from AdvBench original prompts."
    )
    parser.add_argument(
        "--advbench",
        default="data/advbench.jsonl",
        help="Path to AdvBench jsonl. Each line must contain a prompt key.",
    )
    parser.add_argument(
        "--output",
        default="data/qwen2_dataset/seed_20.jsonl",
        help="Output seed dataset jsonl path.",
    )
    parser.add_argument("--count", type=int, default=20, help="Number of rows to export.")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="0-based AdvBench row offset to start from.",
    )
    parser.add_argument(
        "--split",
        default="aie_train",
        help="Split label to assign to all rows.",
    )
    parser.add_argument(
        "--target_model",
        default="Qwen2.5-7B-Instruct",
        help="Target model label stored in the dataset.",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.start < 0:
        parser.error("--start must be non-negative")
    return args


def read_advbench_rows(path, start, count):
    rows = []
    end = start + count
    with path.open("r", encoding="utf-8") as f:
        for row_number, line in enumerate(f):
            if row_number < start:
                continue
            if row_number >= end:
                break
            record = json.loads(line)
            if "prompt" not in record:
                raise KeyError(f"Missing prompt key at AdvBench row {row_number}")
            rows.append((row_number, record))

    if len(rows) < count:
        raise ValueError(
            f"Requested {count} rows from {path} starting at {start}, found {len(rows)}"
        )
    return rows


def main():
    args = parse_args()
    advbench_path = Path(args.advbench)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_advbench_rows(advbench_path, args.start, args.count)
    with output_path.open("w", encoding="utf-8") as f:
        for row_number, record in rows:
            original_index = record.get("index", row_number)
            dataset_record = {
                "id": f"qwen2_advbench_{int(original_index):03d}_direct",
                "source": "advbench",
                "original_index": original_index,
                "target_model": args.target_model,
                "original_prompt": record["prompt"],
                "jailbreak_prompt": None,
                "attack_method": "direct",
                "split": args.split,
                "notes": "",
            }
            f.write(json.dumps(dataset_record, ensure_ascii=False) + "\n")

    print(f"wrote={len(rows)}")
    print(f"advbench={advbench_path}")
    print(f"output={output_path}")
    print(f"split={args.split}")


if __name__ == "__main__":
    main()
