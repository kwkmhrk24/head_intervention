"""Model-specific head-set loading for direct score intervention."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Sequence


PAPER_HEADS = {
    "llama": [(3, 0), (7, 16), (9, 9), (13, 4), (3, 5)],
    "mistral": [(16, 14), (31, 14), (15, 14), (30, 3), (16, 31)],
}
MODEL_CSV_NAMES = {
    "llama": "Llama-2-7b-chat-hf",
    "mistral": "Mistral-7B-Instruct-v0.2",
}
CONSENSUS_CONDITIONS = ("consensus_increase", "consensus_decrease")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_head_pairs(
    pairs: Sequence[tuple[int, int]],
    *,
    name: str,
    expected_count: int,
    num_layers: int = 32,
    num_heads: int = 32,
) -> list[tuple[int, int]]:
    normalized = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{name}: every head must be a layer/head pair")
        layer, head = pair
        if type(layer) is not int or type(head) is not int:
            raise ValueError(f"{name}: layer/head indices must be integers")
        if not 0 <= layer < num_layers or not 0 <= head < num_heads:
            raise ValueError(f"{name}: layer/head pair out of range: {(layer, head)}")
        normalized.append((layer, head))
    if len(normalized) != expected_count:
        raise ValueError(f"{name}: expected {expected_count} heads, found {len(normalized)}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name}: duplicate layer/head pair")
    return normalized


def load_paper_heads(model_family: str) -> dict:
    model_family = model_family.lower()
    if model_family not in PAPER_HEADS:
        raise ValueError(f"unsupported model family: {model_family!r}")
    heads = validate_head_pairs(
        PAPER_HEADS[model_family], name=f"{model_family} paper", expected_count=5
    )
    return {
        "condition": "paper",
        "heads": heads,
        "selection_family": "paper",
        "selection_method": "model_specific_paper_top5",
        "model": MODEL_CSV_NAMES[model_family],
    }


def validate_model_specific_paper_heads(
    model_family: str, pairs: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    validated = validate_head_pairs(pairs, name="paper", expected_count=5)
    expected = PAPER_HEADS[model_family.lower()]
    if validated != expected:
        raise ValueError(
            f"paper heads do not match {model_family}: expected={expected}, "
            f"found={validated}"
        )
    return validated


def _parse_bool(value: str, *, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{field}: expected True or False, found {value!r}")


def load_consensus_headsets(
    candidate_csv: str | Path,
    *,
    model_family: str,
    k: int = 5,
) -> dict[str, dict]:
    """Load rank-ordered, model-specific AutoDAN+GCG consensus heads."""

    model_family = model_family.lower()
    if model_family not in MODEL_CSV_NAMES:
        raise ValueError(f"unsupported model family: {model_family!r}")
    path = Path(candidate_csv)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
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
    if not rows:
        raise ValueError(f"{path}: candidate CSV is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    result = {}
    model_name = MODEL_CSV_NAMES[model_family]
    for direction in ("increase", "decrease"):
        candidate_type = f"attack_consistent_{direction}"
        selected = [
            row
            for row in rows
            if row["model"] == model_name
            and row["candidate_type"] == candidate_type
            and row["source_attack"] == "AutoDAN+GCG"
            and row["direction"] == direction
            and row["k"] == str(k)
        ]
        if len(selected) != k:
            raise ValueError(
                f"{path}: {model_name}/{candidate_type}/k={k} expected {k} rows, "
                f"found {len(selected)}"
            )
        converted = []
        for row in selected:
            try:
                converted.append(
                    {
                        **row,
                        "layer": int(row["layer"]),
                        "head": int(row["head"]),
                        "rank": int(row["rank"]),
                        "k": int(row["k"]),
                        "selection_value": float(row["selection_value"]),
                        "attack_consistency": _parse_bool(
                            row["attack_consistency"], field="attack_consistency"
                        ),
                    }
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid candidate row") from exc
        converted.sort(key=lambda row: row["rank"])
        if [row["rank"] for row in converted] != list(range(1, k + 1)):
            raise ValueError(f"{path}: {candidate_type} ranks must be 1..{k}")
        if not all(row["attack_consistency"] for row in converted):
            raise ValueError(f"{path}: {candidate_type} is not attack-consistent")
        for key in (
            "model",
            "candidate_type",
            "source_attack",
            "direction",
            "k",
            "selection_metric",
            "attack_consistency",
        ):
            if len({row[key] for row in converted}) != 1:
                raise ValueError(f"{path}: {candidate_type} has inconsistent {key}")
        heads = validate_head_pairs(
            [(row["layer"], row["head"]) for row in converted],
            name=candidate_type,
            expected_count=k,
        )
        condition = f"consensus_{direction}"
        result[condition] = {
            "condition": condition,
            "heads": heads,
            "selection_family": "delta_asym_ratio_cross_attack_consensus",
            "selection_method": f"attack_consistent_consensus_{direction}_k{k}",
            "model": model_name,
            "candidate_type": candidate_type,
            "source_attack": "AutoDAN+GCG",
            "direction": direction,
            "k": k,
            "selection_metric": converted[0]["selection_metric"],
            "attack_consistency": True,
            "ranked_heads": [
                {
                    "layer": row["layer"],
                    "head": row["head"],
                    "rank": row["rank"],
                    "selection_value": row["selection_value"],
                }
                for row in converted
            ],
            "candidate_csv": str(path.resolve()),
            "candidate_csv_sha256": sha256_file(path),
        }
    return result


def load_headsets(
    *,
    model_family: str,
    conditions: Sequence[str],
    candidate_csv: str | Path | None,
) -> dict[str, dict]:
    allowed = {"paper", *CONSENSUS_CONDITIONS}
    unknown = [condition for condition in conditions if condition not in allowed]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    if len(conditions) != len(set(conditions)):
        raise ValueError("conditions contain duplicates")
    result = {"paper": load_paper_heads(model_family)}
    requested_consensus = [c for c in conditions if c in CONSENSUS_CONDITIONS]
    if requested_consensus:
        if candidate_csv is None:
            raise ValueError("--candidate-csv is required for consensus conditions")
        result.update(
            load_consensus_headsets(candidate_csv, model_family=model_family)
        )
    return result
