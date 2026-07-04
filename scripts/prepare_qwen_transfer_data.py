import argparse
import json
from pathlib import Path


DATASETS = [
    (
        "qwen2.5/autodan_from_llama2",
        "data/derived/llama2/autodan/original_prompts.jsonl",
        "data/derived/llama2/autodan/jailbreak_prompts.jsonl",
    ),
    (
        "qwen2.5/gcg_from_llama2",
        "data/derived/llama2/gcg/original_prompts.jsonl",
        "data/derived/llama2/gcg/jailbreak_prompts.jsonl",
    ),
    (
        "qwen2.5/autodan_from_mistral",
        "data/derived/mistral/autodan/original_prompts.jsonl",
        "data/derived/mistral/autodan/jailbreak_prompts.jsonl",
    ),
    (
        "qwen2.5/gcg_from_mistral",
        "data/derived/mistral/gcg/original_prompts.jsonl",
        "data/derived/mistral/gcg/jailbreak_prompts.jsonl",
    ),
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare Qwen transfer datasets from existing derived prompts."
    )
    parser.add_argument("--derived-root", default="data/derived")
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
            if "prompt" not in row:
                raise ValueError(f"{path}: line {line_no} missing prompt key")
            if not isinstance(row["prompt"], str) or not row["prompt"]:
                raise ValueError(f"{path}: line {line_no} prompt must be a non-empty string")
            rows.append(row)

    if not rows:
        raise ValueError(f"{path}: no rows found")
    index_flags = {"index" in row for row in rows}
    if len(index_flags) != 1:
        raise ValueError(f"{path}: index schema is inconsistent across rows")
    return rows, index_flags.pop()


def project_rows(rows, has_index):
    if has_index:
        return [{"index": row["index"], "prompt": row["prompt"]} for row in rows]
    return [{"prompt": row["prompt"]} for row in rows]


def prepare_pair(label, original_path, jailbreak_path):
    original_rows, original_has_index = read_jsonl(original_path)
    jailbreak_rows, jailbreak_has_index = read_jsonl(jailbreak_path)

    if len(original_rows) != len(jailbreak_rows):
        raise ValueError(
            f"{label}: original/jailbreak row count mismatch: "
            f"{len(original_rows)} != {len(jailbreak_rows)}"
        )
    if original_has_index != jailbreak_has_index:
        raise ValueError(f"{label}: only one side contains index keys")
    if original_has_index:
        for row_number, (original, jailbreak) in enumerate(
            zip(original_rows, jailbreak_rows), start=1
        ):
            if original["index"] != jailbreak["index"]:
                raise ValueError(f"{label}: index mismatch at row {row_number}")

    original_output = project_rows(original_rows, original_has_index)
    jailbreak_output = project_rows(jailbreak_rows, jailbreak_has_index)
    schema = ["index", "prompt"] if original_has_index else ["prompt"]
    return original_output, jailbreak_output, schema


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_outputs(derived_root, label, original_rows, jailbreak_rows):
    output_dir = Path(derived_root) / label.removeprefix("qwen2.5/")
    return {
        output_dir / "original_prompts.jsonl": original_rows,
        output_dir / "jailbreak_prompts.jsonl": jailbreak_rows,
        output_dir / "original_prompts_10.jsonl": original_rows[:10],
        output_dir / "jailbreak_prompts_10.jsonl": jailbreak_rows[:10],
    }


def main():
    args = get_args()
    qwen_root = Path(args.derived_root) / "qwen2.5"
    prepared = []

    for label, original_path, jailbreak_path in DATASETS:
        original_rows, jailbreak_rows, schema = prepare_pair(
            label, original_path, jailbreak_path
        )
        outputs = build_outputs(qwen_root, label, original_rows, jailbreak_rows)
        prepared.append((label, len(original_rows), schema, outputs))

    for label, row_count, schema, outputs in prepared:
        for path, rows in outputs.items():
            write_jsonl(path, rows)
        print(f"{label} rows: {row_count}")
        print(f"{label} schema: {schema}")
        for path, rows in outputs.items():
            print(f"{label} output: {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
