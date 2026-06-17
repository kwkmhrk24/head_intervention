import argparse
import json
from pathlib import Path


GCG_WARNING = (
    "WARNING: GCG_llama2.jsonl indices exist in advbench.jsonl, but mapping to "
    "llama2_GCG.json is not fully aligned. Treat generated GCG pairs as tentative "
    "until confirmed."
)


def get_args():
    parser = argparse.ArgumentParser(description="Prepare prompt-only JSONL files for Llama2 reproduction.")
    parser.add_argument("--advbench", default="data/advbench.jsonl", help="AdvBench JSONL with index/prompt.")
    parser.add_argument("--autodan", default="data/AutoDAN_llama2.jsonl", help="AutoDAN JSONL with index/prompt.")
    parser.add_argument("--gcg", default="data/GCG_llama2.jsonl", help="GCG JSONL with index/prompt.")
    parser.add_argument("--xstest", default="data/xstest.jsonl", help="XSTest JSONL with prompt.")
    parser.add_argument("--output-dir", default="data", help="Directory for derived JSONL files.")
    parser.add_argument("--include-gcg", action="store_true", help="Also write tentative GCG prompt pairs.")
    parser.add_argument("--keep-index", action="store_true", help="Include index in output rows for inspection.")
    parser.add_argument("--dry-run", action="store_true", help="Print checks and planned outputs without writing files.")
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


def build_advbench_index(advbench_rows):
    index = {}
    for line_no, row in enumerate(advbench_rows, start=1):
        idx = row["index"]
        if idx in index:
            raise ValueError(f"data/advbench.jsonl: duplicate index {idx!r} at line {line_no}")
        index[idx] = row
    return index


def build_pairs(attack_rows, advbench_by_index, label):
    missing = []
    original_rows = []
    jailbreak_rows = []

    for row in attack_rows:
        idx = row["index"]
        advbench_row = advbench_by_index.get(idx)
        if advbench_row is None:
            missing.append(idx)
            continue

        original_rows.append({"index": idx, "prompt": advbench_row["prompt"]})
        jailbreak_rows.append({"index": idx, "prompt": row["prompt"]})

    if missing:
        first_missing = ", ".join(repr(idx) for idx in missing[:10])
        raise ValueError(f"{label}: {len(missing)} indices missing from advbench: {first_missing}")

    if len(original_rows) != len(jailbreak_rows):
        raise ValueError(f"{label}: original/jailbreak count mismatch")

    return original_rows, jailbreak_rows


def prompt_rows(rows, keep_index):
    output = []
    for row in rows:
        out_row = {"prompt": row["prompt"]}
        if keep_index and "index" in row:
            out_row = {"index": row["index"], **out_row}
        output.append(out_row)
    return output


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def report_plan(name, input_count, outputs, dry_run):
    mode = "DRY-RUN" if dry_run else "WRITE"
    print(f"[{mode}] {name}")
    print(f"  input records: {input_count}")
    for path, rows in outputs:
        print(f"  output: {path} records={len(rows)}")


def main():
    args = get_args()
    output_dir = Path(args.output_dir)

    advbench_rows = read_jsonl(args.advbench)
    autodan_rows = read_jsonl(args.autodan)
    xstest_rows = read_jsonl(args.xstest)

    require_keys(advbench_rows, args.advbench, ["index", "prompt"])
    require_keys(autodan_rows, args.autodan, ["index", "prompt"])
    require_keys(xstest_rows, args.xstest, ["prompt"])

    advbench_by_index = build_advbench_index(advbench_rows)

    autodan_original, autodan_jailbreak = build_pairs(autodan_rows, advbench_by_index, "AutoDAN")
    refusal_rows = prompt_rows(xstest_rows, keep_index=args.keep_index)

    planned_outputs = [
        (
            output_dir / "original_prompts_autodan_llama2.jsonl",
            prompt_rows(autodan_original, keep_index=args.keep_index),
        ),
        (
            output_dir / "jailbreak_prompts_autodan_llama2.jsonl",
            prompt_rows(autodan_jailbreak, keep_index=args.keep_index),
        ),
        (
            output_dir / "refusal_prompts_llama2.jsonl",
            refusal_rows,
        ),
    ]

    print("Input summary")
    print(f"  advbench: {len(advbench_rows)} records")
    print(f"  AutoDAN: {len(autodan_rows)} records")
    print(f"  xstest: {len(xstest_rows)} records")
    print("  AutoDAN missing advbench indices: 0")
    report_plan("AutoDAN/refusal outputs", len(autodan_rows), planned_outputs, args.dry_run)

    if args.include_gcg:
        print(GCG_WARNING)
        gcg_rows = read_jsonl(args.gcg)
        require_keys(gcg_rows, args.gcg, ["index", "prompt"])
        gcg_original, gcg_jailbreak = build_pairs(gcg_rows, advbench_by_index, "GCG")
        gcg_outputs = [
            (
                output_dir / "original_prompts_gcg_llama2.jsonl",
                prompt_rows(gcg_original, keep_index=args.keep_index),
            ),
            (
                output_dir / "jailbreak_prompts_gcg_llama2.jsonl",
                prompt_rows(gcg_jailbreak, keep_index=args.keep_index),
            ),
        ]
        print(f"  GCG: {len(gcg_rows)} records")
        print("  GCG missing advbench indices: 0")
        report_plan("tentative GCG outputs", len(gcg_rows), gcg_outputs, args.dry_run)
        planned_outputs.extend(gcg_outputs)

    if args.dry_run:
        return

    for path, rows in planned_outputs:
        write_jsonl(path, rows)


if __name__ == "__main__":
    main()
