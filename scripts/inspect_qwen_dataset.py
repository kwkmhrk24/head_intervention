import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect Qwen2.5 master/labeled dataset jsonl files."
    )
    parser.add_argument("--input", required=True, help="Input dataset jsonl.")
    return parser.parse_args()


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def print_counter(name, counter):
    print(name)
    if not counter:
        print("  <empty>")
        return
    for key, value in sorted(counter.items(), key=lambda item: str(item[0])):
        print(f"  {key}: {value}")


def summarize_lengths(name, values):
    if not values:
        print(f"{name}: count=0")
        return
    values = sorted(values)
    count = len(values)
    mean = sum(values) / count
    print(
        f"{name}: count={count} min={values[0]} mean={mean:.1f} "
        f"median={values[count // 2]} max={values[-1]}"
    )


def rate(true_count, total):
    return true_count / total if total else None


def main():
    args = parse_args()
    input_path = Path(args.input)
    rows = read_jsonl(input_path)
    total = len(rows)

    attack_method_counts = Counter(row.get("attack_method") for row in rows)
    split_counts = Counter(row.get("split") for row in rows)
    direct_refused_known = [row for row in rows if row.get("direct_refused") is not None]
    attack_success_known = [row for row in rows if row.get("attack_success") is not None]
    direct_refused_count = sum(row.get("direct_refused") is True for row in direct_refused_known)
    attack_success_count = sum(row.get("attack_success") is True for row in attack_success_known)
    missing_jailbreak_count = sum(not row.get("jailbreak_prompt") for row in rows)

    original_lengths = [len(row["original_prompt"]) for row in rows if row.get("original_prompt")]
    jailbreak_lengths = [len(row["jailbreak_prompt"]) for row in rows if row.get("jailbreak_prompt")]

    print(f"input={input_path}")
    print(f"total={total}")
    print_counter("attack_method_counts", attack_method_counts)
    print_counter("split_counts", split_counts)
    print(f"direct_refused_known={len(direct_refused_known)}")
    print(f"direct_refused_count={direct_refused_count}")
    print(f"direct_refused_rate={rate(direct_refused_count, len(direct_refused_known))}")
    print(f"attack_success_known={len(attack_success_known)}")
    print(f"attack_success_count={attack_success_count}")
    print(f"attack_success_rate={rate(attack_success_count, len(attack_success_known))}")
    print(f"missing_jailbreak_prompt_count={missing_jailbreak_count}")
    summarize_lengths("original_prompt_chars", original_lengths)
    summarize_lengths("jailbreak_prompt_chars", jailbreak_lengths)


if __name__ == "__main__":
    main()
