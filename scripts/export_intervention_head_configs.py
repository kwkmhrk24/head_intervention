#!/usr/bin/env python3
"""Export validated intervention head candidates from the summary CSV.

This module intentionally uses only the Python standard library.  It does not
load a model, tokenizer, tensor, or any GPU-related package.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path


SCHEMA_VERSION = 1
GENERATED_BY = "scripts/export_intervention_head_configs.py"
REQUIRED_COLUMNS = {
    "model",
    "candidate_type",
    "source_attack",
    "direction",
    "k",
    "layer",
    "head",
    "rank",
    "selection_metric",
    "selection_value",
    "attack_consistency",
}
EXPECTED_DIRECTIONS = ("increase", "decrease")
CONDITION_BY_DIRECTION = {
    "increase": "candidate_consensus_increase",
    "decrease": "candidate_consensus_decrease",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Export fixed, provenance-bearing intervention head sets from a "
            "cross-attack candidate CSV."
        )
    )
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--candidate-type",
        action="append",
        required=True,
        help=(
            "Exact candidate_type value from the CSV. Repeat once for the "
            "increase type and once for the decrease type."
        ),
    )
    parser.add_argument("--source-attack", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-num-layers", required=True, type=int)
    parser.add_argument("--expected-num-heads", required=True, type=int)
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_identifier(value):
    """Return a deterministic lowercase identifier component."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return normalized.strip("_")


def candidate_set_id(model, candidate_type, source_attack, k):
    parts = (
        normalize_identifier(model),
        normalize_identifier(candidate_type),
        normalize_identifier(source_attack),
        f"k{k}",
    )
    if any(not part for part in parts):
        raise ValueError("candidate set ID fields must normalize to non-empty values")
    return "_".join(parts)


def _parse_int(row, field, row_number):
    try:
        value = int(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CSV row {row_number}: {field} must be an integer, found {row[field]!r}"
        ) from exc
    return value


def _parse_float(row, field, row_number):
    try:
        value = float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CSV row {row_number}: {field} must be numeric, found {row[field]!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"CSV row {row_number}: {field} must be finite")
    return value


def _parse_bool(row, field, row_number):
    value = row[field].strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(
        f"CSV row {row_number}: {field} must be True or False, found {row[field]!r}"
    )


def read_candidate_rows(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        return [(row_number, row) for row_number, row in enumerate(reader, start=2)]


def extract_candidate_set(
    numbered_rows,
    *,
    model,
    candidate_type,
    source_attack,
    k,
    expected_num_layers,
    expected_num_heads,
):
    selected = [
        (row_number, row)
        for row_number, row in numbered_rows
        if row["model"] == model
        and row["candidate_type"] == candidate_type
        and row["source_attack"] == source_attack
        and row["k"] == str(k)
    ]
    if len(selected) != k:
        raise ValueError(
            f"{candidate_type}: expected {k} matching rows, found {len(selected)}"
        )

    parsed = []
    for row_number, row in selected:
        parsed.append(
            {
                "model": row["model"],
                "candidate_type": row["candidate_type"],
                "source_attack": row["source_attack"],
                "direction": row["direction"],
                "k": _parse_int(row, "k", row_number),
                "layer": _parse_int(row, "layer", row_number),
                "head": _parse_int(row, "head", row_number),
                "rank": _parse_int(row, "rank", row_number),
                "selection_metric": row["selection_metric"],
                "selection_value": _parse_float(row, "selection_value", row_number),
                "attack_consistency": _parse_bool(
                    row, "attack_consistency", row_number
                ),
            }
        )

    consistent_fields = (
        "model",
        "candidate_type",
        "source_attack",
        "direction",
        "k",
        "selection_metric",
        "attack_consistency",
    )
    for field in consistent_fields:
        values = {row[field] for row in parsed}
        if len(values) != 1:
            raise ValueError(
                f"{candidate_type}: {field} must be consistent, found {sorted(values)!r}"
            )

    direction = parsed[0]["direction"]
    if direction not in EXPECTED_DIRECTIONS:
        raise ValueError(
            f"{candidate_type}: direction must be increase or decrease, found {direction!r}"
        )
    if parsed[0]["model"] != model:
        raise ValueError(f"{candidate_type}: model does not match requested model")
    if parsed[0]["candidate_type"] != candidate_type:
        raise ValueError(f"{candidate_type}: candidate_type mismatch")
    if parsed[0]["source_attack"] != source_attack:
        raise ValueError(f"{candidate_type}: source_attack mismatch")
    if parsed[0]["k"] != k:
        raise ValueError(f"{candidate_type}: k mismatch")
    if not parsed[0]["selection_metric"]:
        raise ValueError(f"{candidate_type}: selection_metric must not be empty")
    if parsed[0]["attack_consistency"] is not True:
        raise ValueError(f"{candidate_type}: attack_consistency must be true")

    ranks = [row["rank"] for row in parsed]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"{candidate_type}: duplicate ranks are not allowed")
    if sorted(ranks) != list(range(1, k + 1)):
        raise ValueError(f"{candidate_type}: ranks must be exactly 1 through {k}")

    pairs = [(row["layer"], row["head"]) for row in parsed]
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{candidate_type}: duplicate layer/head pairs are not allowed")
    for layer, head in pairs:
        if not 0 <= layer < expected_num_layers:
            raise ValueError(
                f"{candidate_type}: layer {layer} is outside "
                f"0 <= layer < {expected_num_layers}"
            )
        if not 0 <= head < expected_num_heads:
            raise ValueError(
                f"{candidate_type}: head {head} is outside "
                f"0 <= head < {expected_num_heads}"
            )

    parsed.sort(key=lambda row: row["rank"])
    set_identifier = candidate_set_id(model, candidate_type, source_attack, k)
    return {
        "candidate_set_id": set_identifier,
        "model": model,
        "candidate_type": candidate_type,
        "source_attack": source_attack,
        "direction": direction,
        "k": k,
        "selection_metric": parsed[0]["selection_metric"],
        "attack_consistency": True,
        "heads": [
            {
                "layer": row["layer"],
                "head": row["head"],
                "rank": row["rank"],
                "selection_value": row["selection_value"],
            }
            for row in parsed
        ],
    }


def build_export_payload(
    candidate_csv,
    *,
    model,
    candidate_types,
    source_attack,
    k,
    source_commit,
    expected_num_layers,
    expected_num_heads,
):
    candidate_csv = Path(candidate_csv)
    if k <= 0:
        raise ValueError("k must be positive")
    if expected_num_layers <= 0 or expected_num_heads <= 0:
        raise ValueError("expected model dimensions must be positive")
    if not source_commit.strip():
        raise ValueError("source_commit must not be empty")
    if len(candidate_types) != len(set(candidate_types)):
        raise ValueError("candidate types must not contain duplicates")

    numbered_rows = read_candidate_rows(candidate_csv)
    sets_by_direction = {}
    for candidate_type in candidate_types:
        candidate_set = extract_candidate_set(
            numbered_rows,
            model=model,
            candidate_type=candidate_type,
            source_attack=source_attack,
            k=k,
            expected_num_layers=expected_num_layers,
            expected_num_heads=expected_num_heads,
        )
        direction = candidate_set["direction"]
        if direction in sets_by_direction:
            raise ValueError(f"multiple candidate types selected direction {direction!r}")
        sets_by_direction[direction] = candidate_set

    if set(sets_by_direction) != set(EXPECTED_DIRECTIONS):
        raise ValueError(
            "candidate types must resolve exactly one increase and one decrease set"
        )

    candidate_sets = {
        CONDITION_BY_DIRECTION[direction]: sets_by_direction[direction]
        for direction in EXPECTED_DIRECTIONS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_csv": str(candidate_csv.resolve()),
        "source_csv_sha256": sha256_file(candidate_csv),
        "source_commit": source_commit,
        "generated_by": GENERATED_BY,
        "candidate_set_ids": {
            condition: candidate_set["candidate_set_id"]
            for condition, candidate_set in candidate_sets.items()
        },
        "candidate_sets": candidate_sets,
    }


def write_json_atomic(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {path.parent}")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
        os.rename(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def export_candidate_config(output_file, **kwargs):
    output_file = Path(output_file)
    if output_file.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_file}")
    payload = build_export_payload(**kwargs)
    write_json_atomic(output_file, payload)
    return payload


def main(argv=None):
    args = parse_args(argv)
    payload = export_candidate_config(
        args.output_file,
        candidate_csv=args.candidate_csv,
        model=args.model,
        candidate_types=args.candidate_type,
        source_attack=args.source_attack,
        k=args.k,
        source_commit=args.source_commit,
        expected_num_layers=args.expected_num_layers,
        expected_num_heads=args.expected_num_heads,
    )
    print(f"candidate config: {args.output_file}")
    for condition, candidate_set in payload["candidate_sets"].items():
        pairs = [(row["layer"], row["head"]) for row in candidate_set["heads"]]
        print(f"{condition}: {pairs}")
    print(f"source CSV SHA-256: {payload['source_csv_sha256']}")


if __name__ == "__main__":
    main()
