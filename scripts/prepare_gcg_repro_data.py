import argparse
import json
from pathlib import Path


LLAMA2_FALLBACK_OFFSET = 60
EXPECTED_LLAMA2_FALLBACK_INDICES = {63, 93, 105}


def get_args():
    parser = argparse.ArgumentParser(description="Prepare paired GCG reproduction datasets.")
    parser.add_argument("--advbench", default="data/raw/advbench.jsonl")
    parser.add_argument("--llama2-gcg", default="data/raw/llama2/GCG_llama2.jsonl")
    parser.add_argument("--mistral-gcg", default="data/raw/mistral/GCG_mistral.jsonl")
    parser.add_argument("--derived-root", default="data/derived")
    return parser.parse_args()


def read_jsonl(path, required_keys):
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
            missing = [key for key in required_keys if key not in row]
            if missing:
                raise ValueError(f"{path}: line {line_no} missing keys: {missing}")
            if not isinstance(row["index"], int):
                raise ValueError(f"{path}: line {line_no} index must be an integer")
            if not isinstance(row["prompt"], str) or not row["prompt"]:
                raise ValueError(f"{path}: line {line_no} prompt must be a non-empty string")
            rows.append(row)

    indices = [row["index"] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{path}: duplicate indices found")
    return rows


def find_longest_prefix_match(gcg_prompt, advbench_rows):
    matches = [row for row in advbench_rows if gcg_prompt.startswith(row["prompt"])]
    if not matches:
        return None

    longest_length = max(len(row["prompt"]) for row in matches)
    longest_matches = [row for row in matches if len(row["prompt"]) == longest_length]
    if len(longest_matches) != 1:
        indices = [row["index"] for row in longest_matches]
        raise ValueError(f"Ambiguous longest AdvBench prefix matches: {indices}")
    return longest_matches[0]


def build_pairs(label, gcg_rows, advbench_rows, advbench_by_index, fallback_offset=None):
    prefix_by_raw_index = {}
    unmatched_rows = []

    for row in gcg_rows:
        match = find_longest_prefix_match(row["prompt"], advbench_rows)
        if match is None:
            unmatched_rows.append(row)
        else:
            prefix_by_raw_index[row["index"]] = match

    if fallback_offset is not None:
        bad_offsets = [
            (raw_index, match["index"], raw_index + fallback_offset)
            for raw_index, match in prefix_by_raw_index.items()
            if match["index"] != raw_index + fallback_offset
        ]
        if bad_offsets:
            raise ValueError(
                f"{label}: prefix matches violate raw_index + {fallback_offset}: "
                f"{bad_offsets[:10]}"
            )
    elif unmatched_rows:
        unmatched_indices = [row["index"] for row in unmatched_rows]
        raise ValueError(f"{label}: unmatched GCG rows: {unmatched_indices}")

    fallback_by_raw_index = {}
    if fallback_offset is not None:
        for row in unmatched_rows:
            advbench_index = row["index"] + fallback_offset
            if advbench_index not in advbench_by_index:
                raise ValueError(
                    f"{label}: fallback AdvBench index not found: {advbench_index}"
                )
            fallback_by_raw_index[row["index"]] = advbench_by_index[advbench_index]

    original_rows = []
    jailbreak_rows = []
    for row in gcg_rows:
        match = prefix_by_raw_index.get(row["index"])
        if match is None:
            match = fallback_by_raw_index.get(row["index"])
        if match is None:
            raise ValueError(f"{label}: no AdvBench mapping for raw index {row['index']}")
        original_rows.append({"index": match["index"], "prompt": match["prompt"]})
        jailbreak_rows.append({"index": match["index"], "prompt": row["prompt"]})

    stats = {
        "rows": len(gcg_rows),
        "prefix_matched": len(prefix_by_raw_index),
        "fallback": len(fallback_by_raw_index),
        "unmatched": len(gcg_rows) - len(prefix_by_raw_index) - len(fallback_by_raw_index),
        "fallback_advbench_indices": sorted(
            row["index"] for row in fallback_by_raw_index.values()
        ),
    }
    validate_pairs(label, original_rows, jailbreak_rows, set(advbench_by_index))
    return original_rows, jailbreak_rows, stats


def validate_pairs(label, original_rows, jailbreak_rows, advbench_indices):
    if len(original_rows) != len(jailbreak_rows):
        raise ValueError(f"{label}: original/jailbreak row count mismatch")
    for row_number, (original, jailbreak) in enumerate(
        zip(original_rows, jailbreak_rows), start=1
    ):
        if original["index"] != jailbreak["index"]:
            raise ValueError(f"{label}: index mismatch at row {row_number}")
        if original["index"] not in advbench_indices:
            raise ValueError(f"{label}: non-AdvBench index at row {row_number}")
        if not original["prompt"] or not jailbreak["prompt"]:
            raise ValueError(f"{label}: empty prompt at row {row_number}")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(derived_root, model_name, original_rows, jailbreak_rows):
    output_dir = Path(derived_root) / model_name / "gcg"
    outputs = {
        output_dir / "original_prompts.jsonl": original_rows,
        output_dir / "jailbreak_prompts.jsonl": jailbreak_rows,
        output_dir / "original_prompts_10.jsonl": original_rows[:10],
        output_dir / "jailbreak_prompts_10.jsonl": jailbreak_rows[:10],
    }
    for path, rows in outputs.items():
        write_jsonl(path, rows)
    return outputs


def print_report(label, stats, outputs):
    print(f"{label} rows: {stats['rows']}")
    print(f"{label} prefix matched: {stats['prefix_matched']}")
    print(f"{label} fallback: {stats['fallback']}")
    print(f"{label} unmatched: {stats['unmatched']}")
    if stats["fallback_advbench_indices"]:
        print(f"{label} fallback AdvBench indices: {stats['fallback_advbench_indices']}")
    for path, rows in outputs.items():
        print(f"{label} output: {path} ({len(rows)} rows)")


def main():
    args = get_args()
    advbench_rows = read_jsonl(args.advbench, ["index", "prompt", "target"])
    llama2_rows = read_jsonl(args.llama2_gcg, ["index", "prompt"])
    mistral_rows = read_jsonl(args.mistral_gcg, ["index", "prompt"])
    advbench_by_index = {row["index"]: row for row in advbench_rows}

    llama2_original, llama2_jailbreak, llama2_stats = build_pairs(
        "llama2",
        llama2_rows,
        advbench_rows,
        advbench_by_index,
        fallback_offset=LLAMA2_FALLBACK_OFFSET,
    )
    if set(llama2_stats["fallback_advbench_indices"]) != EXPECTED_LLAMA2_FALLBACK_INDICES:
        raise ValueError(
            "llama2: unexpected fallback AdvBench indices: "
            f"{llama2_stats['fallback_advbench_indices']}"
        )

    mistral_original, mistral_jailbreak, mistral_stats = build_pairs(
        "mistral",
        mistral_rows,
        advbench_rows,
        advbench_by_index,
    )
    if mistral_stats["fallback"] != 0:
        raise ValueError("mistral: fallback must remain zero")

    llama2_outputs = write_outputs(
        args.derived_root, "llama2", llama2_original, llama2_jailbreak
    )
    mistral_outputs = write_outputs(
        args.derived_root, "mistral", mistral_original, mistral_jailbreak
    )
    print_report("llama2", llama2_stats, llama2_outputs)
    print_report("mistral", mistral_stats, mistral_outputs)


if __name__ == "__main__":
    main()
