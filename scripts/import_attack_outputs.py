import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import existing GCG/AutoDAN outputs into a Qwen2.5 master jsonl."
    )
    parser.add_argument("--seed", required=True, help="Direct-labeled seed dataset jsonl.")
    parser.add_argument(
        "--attack-file",
        required=True,
        help="Existing attack output jsonl. Each line must contain a prompt key.",
    )
    parser.add_argument(
        "--attack-method",
        required=True,
        choices=["gcg", "autodan"],
        help="Attack method label for imported prompts.",
    )
    parser.add_argument("--output", required=True, help="Output master jsonl path.")
    args = parser.parse_args()
    return args


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    args = parse_args()
    seed_path = Path(args.seed)
    attack_path = Path(args.attack_file)
    output_path = Path(args.output)

    seed_rows = read_jsonl(seed_path)
    attack_rows = read_jsonl(attack_path)
    if len(seed_rows) != len(attack_rows):
        raise ValueError(
            f"Line count mismatch: seed={len(seed_rows)} attack_file={len(attack_rows)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row_number, (seed, attack) in enumerate(zip(seed_rows, attack_rows)):
            if "prompt" not in attack:
                raise KeyError(f"Missing prompt key in attack file at row {row_number}")
            original_index = seed["original_index"]
            master = {
                "id": f"qwen2_advbench_{int(original_index):03d}_{args.attack_method}",
                "source": seed.get("source", "advbench"),
                "original_index": original_index,
                "attack_method": args.attack_method,
                "target_model": seed.get("target_model", "Qwen2.5-7B-Instruct"),
                "original_prompt": seed["original_prompt"],
                "jailbreak_prompt": attack["prompt"],
                "direct_refused": seed.get("direct_refused"),
                "direct_success_by_current_rule": seed.get(
                    "direct_success_by_current_rule"
                ),
                "attack_refused": None,
                "attack_success": None,
                "split": seed.get("split"),
                "notes": seed.get("notes", ""),
            }
            f.write(json.dumps(master, ensure_ascii=False) + "\n")

    print("import_summary")
    print(f"seed={seed_path}")
    print(f"attack_file={attack_path}")
    print(f"attack_method={args.attack_method}")
    print(f"output={output_path}")
    print(f"rows={len(seed_rows)}")


if __name__ == "__main__":
    main()
