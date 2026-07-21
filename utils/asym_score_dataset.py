"""Dataset adapters for multi-dataset asymmetric-score evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


SUPPORTED_MODEL_FAMILIES = ("llama", "mistral")
SUPPORTED_ATTACKS = ("autodan", "gcg")


def read_json_records(path: str | Path) -> list[dict]:
    """Read a JSON array or JSONL file without changing record order."""

    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}: invalid JSON on line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: line {line_number} is not an object")
                rows.append(row)
        return rows

    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected a JSON list of objects")
    return rows


def _require_family_and_attack(model_family: str, attack: str) -> tuple[str, str]:
    model_family = model_family.strip().lower()
    attack = attack.strip().lower()
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"unsupported model family: {model_family!r}")
    if attack not in SUPPORTED_ATTACKS:
        raise ValueError(f"unsupported attack: {attack!r}")
    return model_family, attack


def normalize_dataset_records(
    rows: Sequence[Mapping[str, object]],
    *,
    model_family: str,
    attack: str,
    source_name: str = "input records",
) -> list[dict]:
    """Normalize all supported prompt files to one deterministic schema.

    Existing ``source_index`` and ``prompt_id`` values are retained. Otherwise
    an integer ``index`` is the semantic source index; prompt-only AutoDAN rows
    use their zero-based row number. A missing prompt id is a stable string made
    from model family, attack, and source index.
    """

    model_family, attack = _require_family_and_attack(model_family, attack)
    normalized = []
    seen_source_indices = set()
    seen_prompt_ids = set()
    for row_number, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{source_name}: row {row_number} is not an object")
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{source_name}: row {row_number} lacks a non-empty prompt")

        raw_index = row.get("index", row_number)
        if type(raw_index) is not int:
            raise ValueError(f"{source_name}: row {row_number} has a non-integer index")
        source_index = row.get("source_index", raw_index)
        if type(source_index) is not int:
            raise ValueError(
                f"{source_name}: row {row_number} has a non-integer source_index"
            )
        if "source_index" in row and "index" in row and source_index != raw_index:
            raise ValueError(
                f"{source_name}: row {row_number} index/source_index mismatch"
            )
        prompt_id = row.get(
            "prompt_id", f"{model_family}:{attack}:{source_index}"
        )
        if not isinstance(prompt_id, (str, int)) or isinstance(prompt_id, bool):
            raise ValueError(f"{source_name}: row {row_number} has an invalid prompt_id")
        prompt_id_key = (type(prompt_id), prompt_id)
        if source_index in seen_source_indices:
            raise ValueError(f"{source_name}: duplicate source_index {source_index}")
        if prompt_id_key in seen_prompt_ids:
            raise ValueError(f"{source_name}: duplicate prompt_id {prompt_id!r}")
        seen_source_indices.add(source_index)
        seen_prompt_ids.add(prompt_id_key)
        normalized.append(
            {
                "prompt_id": prompt_id,
                "source_index": source_index,
                "prompt": prompt,
                "attack": attack,
                "model": model_family,
                "raw_index": raw_index,
            }
        )
    if not normalized:
        raise ValueError(f"{source_name}: no prompt records")
    return normalized


def apply_evaluation_split(
    records: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object] | None,
    *,
    source_name: str = "input records",
) -> list[dict]:
    """Select or validate a manifest's evaluation indices in manifest order."""

    if manifest is None:
        return [dict(row) for row in records]
    selection = manifest.get("selection_indices")
    evaluation = manifest.get("evaluation_indices")
    if not isinstance(selection, list) or not isinstance(evaluation, list):
        raise ValueError("split manifest requires selection_indices/evaluation_indices")
    if any(type(value) is not int for value in selection + evaluation):
        raise ValueError("split manifest indices must be integers")
    if len(selection) != len(set(selection)) or len(evaluation) != len(set(evaluation)):
        raise ValueError("split manifest contains duplicate indices")
    if set(selection) & set(evaluation):
        raise ValueError("split manifest selection/evaluation indices overlap")
    total_count = manifest.get("total_count")
    if type(total_count) is not int or set(selection) | set(evaluation) != set(
        range(total_count)
    ):
        raise ValueError("split manifest indices do not partition range(total_count)")
    if manifest.get("selection_size") != len(selection):
        raise ValueError("split manifest selection_size mismatch")
    if manifest.get("evaluation_size") != len(evaluation):
        raise ValueError("split manifest evaluation_size mismatch")

    by_index = {row["source_index"]: dict(row) for row in records}
    actual = [row["source_index"] for row in records]
    if actual == evaluation:
        return [dict(row) for row in records]
    missing = [index for index in evaluation if index not in by_index]
    if missing:
        raise ValueError(f"{source_name}: missing evaluation indices {missing[:10]}")
    return [by_index[index] for index in evaluation]


def load_dataset_records(
    path: str | Path,
    *,
    model_family: str,
    attack: str,
    split_manifest: str | Path | None = None,
) -> tuple[list[dict], dict | None]:
    rows = read_json_records(path)
    records = normalize_dataset_records(
        rows,
        model_family=model_family,
        attack=attack,
        source_name=str(path),
    )
    manifest = None
    if split_manifest is not None:
        with Path(split_manifest).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError(f"{split_manifest}: expected a JSON object")
    return apply_evaluation_split(records, manifest, source_name=str(path)), manifest
