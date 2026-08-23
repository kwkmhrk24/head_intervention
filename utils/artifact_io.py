"""Strict JSON, hashing, and output-safety helpers for experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    """Read non-empty JSONL lines and require every value to be an object."""

    rows = []
    with Path(path).open(encoding="utf-8") as handle:
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


def read_json(path: str | Path) -> dict:
    """Read a JSON artifact and require a top-level object."""

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def ensure_output_directory_absent(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {path}")
    return path


def write_json(path: str | Path, payload) -> None:
    """Write a JSON artifact without replacing an existing file."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write object rows as JSONL without replacing an existing file."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
