#!/usr/bin/env python3
"""Compare AutoDAN and GCG attention changes on matched source indices.

This is a CPU-only post-hoc analysis. It reads existing prompt-level metrics and
memory-mapped raw arrays; it never loads a model or writes subset NPY arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = f"/tmp/matplotlib-{os.getuid()}"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_attention_head_metric_relationships import (
    aggregate_attention_metrics,
    pearson_correlation,
    spearman_correlation,
    validate_head_grid,
)
from scripts.plot_attention_score_all_heads_distributions import (
    RAW_ARRAY_FILENAMES,
    load_array,
    load_metadata,
    validate_array_group,
)
from scripts.plot_top_delta_asym_ratio_head_histograms import (
    build_summary_row,
    output_subdirectory_name,
    save_histograms,
    select_signed_heads,
    validate_delta,
    write_dict_rows,
)


COMPARISON_METRICS = [
    "delta_asym_ratio",
    "element_delta_mean",
    "element_std_shift",
    "element_positive_rate_shift",
]
TOP_K_VALUES = (3, 5, 10, 20, 50)
MAPPING_COLUMNS = [
    "raw_prompt_index",
    "input_row_index",
    "source_index",
    "mapping_basis",
    "mapping_verified",
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only matched-source comparison of AutoDAN and GCG attention "
            "score changes from existing raw NPY and prompt metrics."
        )
    )
    parser.add_argument("--autodan_raw_dir", required=True)
    parser.add_argument("--gcg_raw_dir", required=True)
    parser.add_argument("--autodan_original_metrics", required=True)
    parser.add_argument("--autodan_jailbreak_metrics", required=True)
    parser.add_argument("--gcg_original_metrics", required=True)
    parser.add_argument("--gcg_jailbreak_metrics", required=True)
    parser.add_argument("--gcg_manifest", required=True)
    parser.add_argument(
        "--autodan_mapping_mode",
        required=True,
        choices=("validated_row_index",),
    )
    parser.add_argument(
        "--autodan_original_prompts",
        default="data/derived/llama2/autodan/original_prompts.jsonl",
    )
    parser.add_argument(
        "--autodan_jailbreak_prompts",
        default="data/derived/llama2/autodan/jailbreak_prompts.jsonl",
    )
    parser.add_argument(
        "--autodan_source_data",
        default="data/raw/llama2/AutoDAN_llama2.jsonl",
    )
    parser.add_argument(
        "--autodan_labels",
        default="data/raw/llama2/llama2_AutoDAN.json",
    )
    parser.add_argument("--advbench", default="data/raw/advbench.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--expected_common_count", type=int, default=70)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry_run_mapping",
        action="store_true",
        help="Validate mappings and common source indices without creating outputs.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"row is not an object: line={line_number}")
                rows.append(row)
    except Exception as exc:
        raise ValueError(
            f"Failed to read JSONL: path={path}, error_type={type(exc).__name__}"
        ) from exc
    return rows


def read_json_list(path: Path) -> list[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"Failed to read JSON: path={path}, error_type={type(exc).__name__}"
        ) from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a JSON list of objects: path={path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".write_test_"):
            pass
    except OSError as exc:
        raise OSError(f"Output directory is not writable: {path}") from exc


def validate_unique_mapping(mapping: pd.DataFrame, label: str) -> None:
    required = {"raw_prompt_index", "source_index"}
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"{label} mapping missing columns: {missing}")
    for column in required:
        values = pd.to_numeric(mapping[column], errors="raise").to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{label} mapping contains NaN/inf: column={column}")
    if mapping["raw_prompt_index"].duplicated().any():
        raise ValueError(f"{label} mapping has duplicate raw_prompt_index")
    if mapping["source_index"].duplicated().any():
        raise ValueError(f"{label} mapping has duplicate source_index")


def validate_prompt_rows(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    for row_index, row in enumerate(rows):
        if not isinstance(row.get("prompt"), str):
            raise ValueError(f"{label} row lacks string prompt: row={row_index}")


def build_autodan_mapping(
    raw_prompt_count: int,
    metadata: Mapping[str, Any],
    original_prompts_path: Path,
    jailbreak_prompts_path: Path,
    source_data_path: Path,
    labels_path: Path,
    advbench_path: Path,
    mapping_mode: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if mapping_mode != "validated_row_index":
        raise ValueError(f"Unsupported AutoDAN mapping mode: {mapping_mode}")
    original = read_jsonl(original_prompts_path)
    jailbreak = read_jsonl(jailbreak_prompts_path)
    source = read_jsonl(source_data_path)
    labels = read_json_list(labels_path)
    advbench = read_jsonl(advbench_path)
    counts = [raw_prompt_count, len(original), len(jailbreak), len(source), len(labels)]
    if len(set(counts)) != 1:
        raise ValueError(
            "AutoDAN mapping input counts differ: "
            f"raw/original/jailbreak/source/labels={counts}"
        )
    validate_prompt_rows(original, "AutoDAN original")
    validate_prompt_rows(jailbreak, "AutoDAN jailbreak")
    indices = [row.get("index") for row in source]
    if indices != list(range(raw_prompt_count)):
        raise ValueError("AutoDAN source indices must equal ordered rows 0..N-1")
    label_indices = [row.get("index") for row in labels]
    if label_indices != indices:
        raise ValueError("AutoDAN label indices/order differ from source data")
    advbench_by_index = {row.get("index"): row for row in advbench}
    if len(advbench_by_index) != len(advbench):
        raise ValueError("AdvBench contains duplicate index values")
    for row_index in range(raw_prompt_count):
        source_row = source[row_index]
        label_row = labels[row_index]
        advbench_row = advbench_by_index.get(row_index)
        if advbench_row is None:
            raise ValueError(f"AdvBench index is missing: source_index={row_index}")
        if source_row.get("prompt") != jailbreak[row_index]["prompt"]:
            raise ValueError(f"AutoDAN jailbreak/source mismatch: row={row_index}")
        if label_row.get("user prompt") != jailbreak[row_index]["prompt"]:
            raise ValueError(f"AutoDAN jailbreak/label mismatch: row={row_index}")
        if advbench_row.get("prompt") != original[row_index]["prompt"]:
            raise ValueError(f"AutoDAN original/AdvBench mismatch: row={row_index}")
        if type(label_row.get("success")) is not bool:
            raise ValueError(f"AutoDAN label success is not boolean: row={row_index}")
    prompt_definition = str(metadata.get("prompt_id_definition", "")).lower()
    source_definition = str(metadata.get("source_index_definition", "")).lower()
    if "jsonl row index" not in prompt_definition:
        raise ValueError("AutoDAN metadata does not define prompt_id as JSONL row index")
    if "jsonl row index" not in source_definition:
        raise ValueError("AutoDAN metadata does not define source_index as JSONL row index")
    mapping = pd.DataFrame(
        {
            "raw_prompt_index": np.arange(raw_prompt_count, dtype=np.int64),
            "input_row_index": np.arange(raw_prompt_count, dtype=np.int64),
            "source_index": np.asarray(indices, dtype=np.int64),
            "mapping_basis": "validated JSONL row index and AdvBench/source content match",
            "mapping_verified": True,
        }
    )
    validate_unique_mapping(mapping, "AutoDAN")
    details = {
        "mapping_mode": mapping_mode,
        "mapping_basis": (
            "raw metadata JSONL-row contract plus exact original/AdvBench and "
            "jailbreak/AutoDAN source/label content matching"
        ),
        "source_index_min": int(mapping["source_index"].min()),
        "source_index_max": int(mapping["source_index"].max()),
        "success_true_count": int(sum(row["success"] for row in labels)),
        "success_false_count": int(sum(not row["success"] for row in labels)),
        "success_filter_applied": False,
    }
    return mapping[MAPPING_COLUMNS], details


def parse_included(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("GCG manifest included column must be boolean")
    return normalized.eq("true")


def build_gcg_mapping(
    manifest_path: Path,
    raw_prompt_count: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        manifest = pd.read_csv(manifest_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read GCG manifest: path={manifest_path}, "
            f"error_type={type(exc).__name__}"
        ) from exc
    required = {
        "raw_prompt_index",
        "input_row_index",
        "source_index",
        "original_source_index",
        "jailbreak_source_index",
        "included",
        "exclusion_reason",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"GCG manifest missing columns: {missing}")
    included_mask = parse_included(manifest["included"])
    included = manifest.loc[included_mask].copy()
    excluded = manifest.loc[~included_mask].copy()
    if len(included) != raw_prompt_count:
        raise ValueError(
            "GCG included manifest/raw count mismatch: "
            f"included={len(included)}, raw={raw_prompt_count}"
        )
    if not (
        included["source_index"].astype(str)
        == included["original_source_index"].astype(str)
    ).all() or not (
        included["source_index"].astype(str)
        == included["jailbreak_source_index"].astype(str)
    ).all():
        raise ValueError("GCG original/jailbreak source indices differ")
    mapping = included[["raw_prompt_index", "input_row_index", "source_index"]].copy()
    for column in ("raw_prompt_index", "input_row_index", "source_index"):
        mapping[column] = pd.to_numeric(mapping[column], errors="raise").astype(int)
    if mapping["raw_prompt_index"].tolist() != list(range(raw_prompt_count)):
        raise ValueError("GCG included raw_prompt_index must be contiguous 0..N-1")
    mapping["mapping_basis"] = "explicit included prompt_manifest.csv row"
    mapping["mapping_verified"] = True
    validate_unique_mapping(mapping, "GCG")
    details = {
        "mapping_basis": "explicit included GCG prompt manifest",
        "input_pair_count": int(len(manifest)),
        "included_pair_count": int(len(included)),
        "excluded_pair_count": int(len(excluded)),
        "excluded_source_indices": [
            int(value) for value in excluded["source_index"].tolist()
        ],
    }
    return mapping[MAPPING_COLUMNS], details


def compute_common_mapping(
    autodan_mapping: pd.DataFrame,
    gcg_mapping: pd.DataFrame,
    expected_count: int,
    expected_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    validate_unique_mapping(autodan_mapping, "AutoDAN")
    validate_unique_mapping(gcg_mapping, "GCG")
    common = autodan_mapping.merge(
        gcg_mapping,
        on="source_index",
        how="inner",
        validate="one_to_one",
        suffixes=("_autodan", "_gcg"),
    )
    common = common.sort_values("source_index").reset_index(drop=True)
    if len(common) != expected_count:
        raise ValueError(
            f"Expected common source count {expected_count}; received={len(common)}"
        )
    if expected_indices is not None and common["source_index"].tolist() != list(
        expected_indices
    ):
        raise ValueError("Common source indices do not match the expected sequence")
    return common[
        [
            "source_index",
            "raw_prompt_index_autodan",
            "input_row_index_autodan",
            "raw_prompt_index_gcg",
            "input_row_index_gcg",
        ]
    ]


def load_raw_bundle(raw_dir: Path, validate_values: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    metadata = load_metadata(raw_dir / "metadata.json")
    arrays = {
        condition: load_array(raw_dir / filename)
        for condition, filename in RAW_ARRAY_FILENAMES.items()
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"Raw array shapes differ: raw_dir={raw_dir}")
    shape = next(iter(shapes))
    if len(shape) != 4:
        raise ValueError(f"Raw arrays must be 4D: raw_dir={raw_dir}, shape={shape}")
    if tuple(metadata.get("array_shape", ())) != shape:
        raise ValueError(f"Raw metadata shape mismatch: raw_dir={raw_dir}")
    if metadata.get("array_axes") != [
        "prompt",
        "layer",
        "head",
        "strict_lower_position",
    ]:
        raise ValueError(f"Unexpected raw array axes: raw_dir={raw_dir}")
    if int(metadata.get("tail_tokens", -1)) != 16:
        raise ValueError(f"Expected tail_tokens=16: raw_dir={raw_dir}")
    if int(metadata.get("elements_per_head", -1)) != 120:
        raise ValueError(f"Expected elements_per_head=120: raw_dir={raw_dir}")
    if validate_values:
        validate_array_group(arrays, metadata, expected_ndim=4, label=str(raw_dir))
    return arrays, metadata


def validate_subset_indices(
    raw_indices: Sequence[int],
    prompt_count: int,
    label: str,
) -> np.ndarray:
    indices = np.asarray(raw_indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError(f"{label} matched subset is empty")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{label} matched raw indices are not unique")
    if np.any(indices < 0) or np.any(indices >= prompt_count):
        raise ValueError(f"{label} matched raw index is out of range")
    return indices


def _update_moments(
    state: Dict[str, Any],
    chunk: np.ndarray,
) -> None:
    chunk64 = np.asarray(chunk, dtype=np.float64)
    chunk_count = int(chunk64.shape[-1])
    chunk_mean = chunk64.mean(axis=-1)
    chunk_m2 = np.sum((chunk64 - chunk_mean[..., None]) ** 2, axis=-1)
    count = int(state["count"])
    if count == 0:
        state["mean"] = chunk_mean
        state["m2"] = chunk_m2
    else:
        total = count + chunk_count
        difference = chunk_mean - state["mean"]
        state["mean"] += difference * chunk_count / total
        state["m2"] += (
            chunk_m2 + difference**2 * count * chunk_count / total
        )
    state["positive"] += np.sum(chunk64 > 0, axis=-1)
    state["count"] = count + chunk_count


def aggregate_raw_head_metrics(
    arrays: Mapping[str, np.ndarray],
    raw_indices: Sequence[int],
) -> Tuple[pd.DataFrame, float]:
    shape = arrays["original"].shape
    indices = validate_subset_indices(raw_indices, shape[0], "raw")
    states: Dict[str, Dict[str, Any]] = {
        condition: {
            "count": 0,
            "mean": np.zeros(shape[1:3], dtype=np.float64),
            "m2": np.zeros(shape[1:3], dtype=np.float64),
            "positive": np.zeros(shape[1:3], dtype=np.int64),
        }
        for condition in ("original", "jailbreak", "delta")
    }
    max_delta_error = 0.0
    for raw_prompt_index in indices:
        chunks = {
            condition: np.asarray(values[int(raw_prompt_index)])
            for condition, values in arrays.items()
        }
        for condition, chunk in chunks.items():
            if not np.isfinite(chunk).all():
                raise ValueError(
                    f"Matched raw contains NaN/inf: condition={condition}, "
                    f"raw_prompt_index={int(raw_prompt_index)}"
                )
        error = float(
            np.max(
                np.abs(
                    chunks["delta"]
                    - (chunks["jailbreak"] - chunks["original"])
                )
            )
        )
        max_delta_error = max(max_delta_error, error)
        if not np.allclose(
            chunks["delta"],
            chunks["jailbreak"] - chunks["original"],
            rtol=1e-5,
            atol=2e-6,
        ):
            raise ValueError(
                "Matched raw delta mismatch: "
                f"raw_prompt_index={int(raw_prompt_index)}, max_abs_error={error}"
            )
        for condition, chunk in chunks.items():
            _update_moments(states[condition], chunk)
    expected_count = len(indices) * shape[3]
    if any(int(state["count"]) != expected_count for state in states.values()):
        raise ValueError("Matched raw element count is inconsistent")
    statistics: Dict[str, Dict[str, np.ndarray]] = {}
    for condition, state in states.items():
        statistics[condition] = {
            "mean": state["mean"],
            "std": np.sqrt(np.maximum(state["m2"] / state["count"], 0.0)),
            "positive_rate": state["positive"] / state["count"],
        }
    rows = []
    for layer in range(shape[1]):
        for head in range(shape[2]):
            original_mean = float(statistics["original"]["mean"][layer, head])
            jailbreak_mean = float(statistics["jailbreak"]["mean"][layer, head])
            delta_mean = float(statistics["delta"]["mean"][layer, head])
            if not np.isclose(
                delta_mean,
                jailbreak_mean - original_mean,
                rtol=1e-5,
                atol=2e-6,
            ):
                raise ValueError(
                    f"Element delta mean identity failed: layer={layer}, head={head}"
                )
            original_rate = float(
                statistics["original"]["positive_rate"][layer, head]
            )
            jailbreak_rate = float(
                statistics["jailbreak"]["positive_rate"][layer, head]
            )
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "head_label": f"L{layer}H{head}",
                    "element_value_count": expected_count,
                    "original_element_mean": original_mean,
                    "jailbreak_element_mean": jailbreak_mean,
                    "element_delta_mean": delta_mean,
                    "original_element_std": float(
                        statistics["original"]["std"][layer, head]
                    ),
                    "jailbreak_element_std": float(
                        statistics["jailbreak"]["std"][layer, head]
                    ),
                    "element_std_shift": float(
                        statistics["jailbreak"]["std"][layer, head]
                        - statistics["original"]["std"][layer, head]
                    ),
                    "original_positive_rate": original_rate,
                    "jailbreak_positive_rate": jailbreak_rate,
                    "element_positive_rate_shift": jailbreak_rate - original_rate,
                }
            )
    result = pd.DataFrame(rows)
    validate_head_grid(result)
    return result, max_delta_error


def read_metrics(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read attention metrics: path={path}, "
            f"error_type={type(exc).__name__}"
        ) from exc


def prepare_matched_attention_metrics(
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
    common_sources: set[int],
    label: str,
    source_mode: str,
) -> pd.DataFrame:
    required = {"prompt_id", "source_index", "layer", "head", "window_size", "asym_ratio"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} attention metrics missing columns: {missing}")
    result = frame.copy()
    if source_mode == "prompt_id_mapping":
        source_by_input = dict(
            zip(mapping["input_row_index"], mapping["source_index"])
        )
        prompt_ids = pd.to_numeric(result["prompt_id"], errors="raise").astype(int)
        if not set(prompt_ids.unique()).issubset(source_by_input):
            raise ValueError(f"{label} prompt_id is absent from AutoDAN mapping")
        nonnull_source = result["source_index"].notna()
        if nonnull_source.any():
            expected = prompt_ids.loc[nonnull_source].map(source_by_input).astype(str)
            actual = result.loc[nonnull_source, "source_index"].astype(str)
            if not actual.reset_index(drop=True).equals(expected.reset_index(drop=True)):
                raise ValueError(f"{label} source_index conflicts with mapping")
        result["source_index"] = prompt_ids.map(source_by_input).astype(int)
    elif source_mode == "source_index":
        if result["source_index"].isna().any():
            raise ValueError(f"{label} source_index contains missing values")
        result["source_index"] = pd.to_numeric(
            result["source_index"], errors="raise"
        ).astype(int)
    else:
        raise ValueError(f"Unsupported source mode: {source_mode}")
    result = result[result["source_index"].isin(common_sources)].copy()
    result["prompt_id"] = result["source_index"]
    if result["source_index"].nunique() != len(common_sources):
        raise ValueError(f"{label} does not cover every common source index")
    if sorted(result["source_index"].unique()) != sorted(common_sources):
        raise ValueError(f"{label} common source index set differs")
    if set(result["window_size"].unique()) != {16}:
        raise ValueError(f"{label} must contain only window_size=16")
    if result.duplicated(["prompt_id", "layer", "head"]).any():
        raise ValueError(f"{label} has duplicate matched prompt/layer/head keys")
    head_grid = result[["layer", "head"]].drop_duplicates()
    validate_head_grid(head_grid)
    expected_heads = len(head_grid)
    per_prompt = result.groupby("prompt_id").size()
    if not per_prompt.eq(expected_heads).all():
        raise ValueError(
            f"{label} must contain {expected_heads} layer-head rows per prompt"
        )
    return result


def aggregate_matched_attention(
    original: pd.DataFrame,
    jailbreak: pd.DataFrame,
) -> pd.DataFrame:
    aggregated, _, window_size = aggregate_attention_metrics(original, jailbreak)
    if window_size != 16:
        raise ValueError("Matched attention metrics must use window_size=16")
    return aggregated[
        [
            "layer",
            "head",
            "original_asym_ratio",
            "jailbreak_asym_ratio",
            "delta_asym_ratio",
        ]
    ]


def build_attack_metrics(
    raw_metrics: pd.DataFrame,
    attention_metrics: pd.DataFrame,
) -> pd.DataFrame:
    result = raw_metrics.merge(
        attention_metrics,
        on=["layer", "head"],
        how="inner",
        validate="one_to_one",
    )
    if len(result) != len(raw_metrics) or len(result) != len(attention_metrics):
        raise ValueError("Raw and attention head grids differ")
    validate_head_grid(result)
    numeric = [column for column in result.columns if column != "head_label"]
    if not np.isfinite(result[numeric].to_numpy(np.float64)).all():
        raise ValueError("Matched attack metrics contain NaN/inf")
    return result.sort_values(["layer", "head"]).reset_index(drop=True)


def build_joined_metrics(
    autodan: pd.DataFrame,
    gcg: pd.DataFrame,
) -> pd.DataFrame:
    keep = ["layer", "head", "head_label", *COMPARISON_METRICS]
    left = autodan[keep].rename(
        columns={metric: f"autodan_{metric}" for metric in COMPARISON_METRICS}
    )
    right = gcg[keep].rename(
        columns={metric: f"gcg_{metric}" for metric in COMPARISON_METRICS}
    )
    result = left.merge(
        right,
        on=["layer", "head"],
        validate="one_to_one",
        suffixes=("_autodan", "_gcg"),
    )
    if len(result) != len(autodan) or len(result) != len(gcg):
        raise ValueError("Attack head grids differ")
    if not (result["head_label_autodan"] == result["head_label_gcg"]).all():
        raise ValueError("Attack head labels differ")
    result["head_label"] = result.pop("head_label_autodan")
    result = result.drop(columns="head_label_gcg")
    for metric in COMPARISON_METRICS:
        auto = result[f"autodan_{metric}"]
        gcg_values = result[f"gcg_{metric}"]
        result[f"{metric}_gcg_minus_autodan"] = gcg_values - auto
        result[f"{metric}_sign_agreement"] = np.sign(auto) == np.sign(gcg_values)
    result = result.sort_values(["layer", "head"]).reset_index(drop=True)
    validate_head_grid(result)
    return result


def build_metric_comparisons(joined: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    correlations = []
    signs = []
    for metric in COMPARISON_METRICS:
        auto = joined[f"autodan_{metric}"].to_numpy(np.float64)
        gcg_values = joined[f"gcg_{metric}"].to_numpy(np.float64)
        agreement = joined[f"{metric}_sign_agreement"]
        correlations.append(
            {
                "metric": metric,
                "pearson": pearson_correlation(auto, gcg_values),
                "spearman": spearman_correlation(auto, gcg_values),
                "n": len(joined),
            }
        )
        signs.append(
            {
                "metric": metric,
                "n": len(joined),
                "sign_agreement_count": int(agreement.sum()),
                "sign_agreement_rate": float(agreement.mean()),
                "both_positive_count": int(((auto > 0) & (gcg_values > 0)).sum()),
                "both_negative_count": int(((auto < 0) & (gcg_values < 0)).sum()),
                "autodan_only_positive_count": int(
                    ((auto > 0) & (gcg_values <= 0)).sum()
                ),
                "gcg_only_positive_count": int(
                    ((gcg_values > 0) & (auto <= 0)).sum()
                ),
            }
        )
    return pd.DataFrame(correlations), pd.DataFrame(signs)


def _ranked_heads(frame: pd.DataFrame, metric: str, ascending: bool) -> pd.DataFrame:
    return frame.sort_values(
        [metric, "layer", "head"],
        ascending=[ascending, True, True],
        kind="mergesort",
    )


def build_topk_overlap(
    autodan: pd.DataFrame,
    gcg: pd.DataFrame,
    k_values: Sequence[int] = TOP_K_VALUES,
) -> pd.DataFrame:
    rows = []
    for metric in COMPARISON_METRICS:
        for direction, ascending in (("increase", False), ("decrease", True)):
            auto_ranked = _ranked_heads(autodan, metric, ascending)
            gcg_ranked = _ranked_heads(gcg, metric, ascending)
            for k in k_values:
                if k <= 0 or k > len(auto_ranked) or k > len(gcg_ranked):
                    raise ValueError(f"Invalid top-k value: k={k}")
                auto_heads = list(
                    auto_ranked.head(k)[["layer", "head"]].itertuples(
                        index=False, name=None
                    )
                )
                gcg_heads = list(
                    gcg_ranked.head(k)[["layer", "head"]].itertuples(
                        index=False, name=None
                    )
                )
                auto_set = set(auto_heads)
                gcg_set = set(gcg_heads)
                common = sorted(auto_set & gcg_set)
                union = auto_set | gcg_set
                label = lambda pair: f"L{int(pair[0])}H{int(pair[1])}"
                rows.append(
                    {
                        "metric": metric,
                        "direction": direction,
                        "k": int(k),
                        "overlap_count": len(common),
                        "overlap_rate": len(common) / k,
                        "jaccard": len(common) / len(union),
                        "common_heads": ";".join(label(pair) for pair in common),
                    }
                )
    return pd.DataFrame(rows)


def build_layer_outputs(
    attack_frames: Mapping[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for attack, frame in attack_frames.items():
        for layer, group in frame.groupby("layer", sort=True):
            row: Dict[str, Any] = {
                "attack": attack,
                "layer": int(layer),
                "head_count": int(len(group)),
            }
            for metric in COMPARISON_METRICS:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_median"] = float(group[metric].median())
                row[f"{metric}_positive_head_rate"] = float(
                    (group[metric] > 0).mean()
                )
            rows.append(row)
    summary = pd.DataFrame(rows)
    if len(summary) != 64 or not summary["head_count"].eq(32).all():
        raise ValueError("Matched layer summary must contain 32 layers per attack")
    auto = summary[summary["attack"] == "AutoDAN"].set_index("layer")
    gcg = summary[summary["attack"] == "GCG"].set_index("layer")
    correlations = []
    for column in summary.columns:
        if column in {"attack", "layer", "head_count"}:
            continue
        correlations.append(
            {
                "metric": column,
                "pearson": pearson_correlation(auto[column], gcg[column]),
                "spearman": spearman_correlation(auto[column], gcg[column]),
                "n_layers": 32,
            }
        )
    return summary, pd.DataFrame(correlations)


def build_l13h5_summary(
    attack_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for attack, frame in attack_frames.items():
        target = frame[(frame["layer"] == 13) & (frame["head"] == 5)]
        if len(target) != 1:
            raise ValueError(f"{attack}: expected exactly one L13H5")
        values = target.iloc[0]
        for metric in COMPARISON_METRICS:
            increase = _ranked_heads(frame, metric, False).reset_index(drop=True)
            decrease = _ranked_heads(frame, metric, True).reset_index(drop=True)
            increase_rank = int(
                increase.index[
                    (increase["layer"] == 13) & (increase["head"] == 5)
                ][0]
            ) + 1
            decrease_rank = int(
                decrease.index[
                    (decrease["layer"] == 13) & (decrease["head"] == 5)
                ][0]
            ) + 1
            rows.append(
                {
                    "attack": attack,
                    "layer": 13,
                    "head": 5,
                    "head_label": "L13H5",
                    "metric": metric,
                    "value": float(values[metric]),
                    "increase_rank": increase_rank,
                    "decrease_rank": decrease_rank,
                }
            )
    return pd.DataFrame(rows)


def extract_head_subset(
    arrays: Mapping[str, np.ndarray],
    raw_indices: Sequence[int],
    layer: int,
    head: int,
) -> Dict[str, np.ndarray]:
    indices = validate_subset_indices(raw_indices, arrays["original"].shape[0], "histogram")
    extracted = {
        condition: np.stack(
            [np.asarray(values[int(index), layer, head, :]) for index in indices]
        )
        for condition, values in arrays.items()
    }
    expected_shape = (len(indices), arrays["original"].shape[3])
    if any(values.shape != expected_shape for values in extracted.values()):
        raise ValueError("Histogram matched subset shape mismatch")
    validate_delta(extracted, layer, head)
    return extracted


def write_histograms(
    output_dir: Path,
    attack: str,
    head_metrics: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    raw_indices: Sequence[int],
    top_k: int,
    bins: int,
    dpi: int,
) -> Tuple[pd.DataFrame, list[Path]]:
    selected = select_signed_heads(
        head_metrics,
        "delta_asym_ratio",
        top_k=top_k,
        bottom_k=top_k,
    )
    attack_dir = output_dir / "histograms" / attack.lower()
    attack_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    pngs: list[Path] = []
    for row in selected.itertuples(index=False):
        head_dir = attack_dir / output_subdirectory_name(row)
        head_dir.mkdir(parents=True, exist_ok=True)
        arrays_for_head = extract_head_subset(
            arrays,
            raw_indices,
            int(row.layer),
            int(row.head),
        )
        delta_error = validate_delta(arrays_for_head, int(row.layer), int(row.head))
        summary = build_summary_row(
            row,
            arrays_for_head,
            "delta_asym_ratio",
            prompt_count=len(raw_indices),
            values_per_prompt=arrays_for_head["original"].shape[1],
            delta_max_abs_error=delta_error,
            output_directory=str(head_dir.relative_to(output_dir)),
        )
        if summary["total_values"] != len(raw_indices) * 120:
            raise ValueError("Histogram element count is not matched_prompt_count * 120")
        expected = head_metrics[
            (head_metrics["layer"] == int(row.layer))
            & (head_metrics["head"] == int(row.head))
        ].iloc[0]
        comparisons = {
            "original_mean": expected["original_element_mean"],
            "jailbreak_mean": expected["jailbreak_element_mean"],
            "delta_mean": expected["element_delta_mean"],
        }
        for actual_column, expected_value in comparisons.items():
            if not np.isclose(
                summary[actual_column], expected_value, rtol=1e-5, atol=1e-6
            ):
                raise ValueError(
                    f"Histogram/head metric mismatch: attack={attack}, "
                    f"layer={int(row.layer)}, head={int(row.head)}, "
                    f"column={actual_column}"
                )
        summary.update(
            {
                "analysis_type": "matched",
                "exploratory": True,
                "matched_source_count": len(raw_indices),
                "attack": attack,
            }
        )
        write_dict_rows(head_dir / "summary.csv", [summary])
        pngs.extend(
            save_histograms(
                head_dir,
                arrays_for_head,
                bins,
                dpi,
                "Llama-2-7B-chat",
                f"{attack} matched70 exploratory",
                row,
                "delta_asym_ratio",
                16,
                len(raw_indices),
            )
        )
        summaries.append(summary)
    selected.to_csv(attack_dir / "selected_heads.csv", index=False)
    pd.DataFrame(summaries).to_csv(attack_dir / "selection_summary.csv", index=False)
    return selected, pngs


def save_comparison_plots(
    output_dir: Path,
    joined: pd.DataFrame,
    layer_summary: pd.DataFrame,
    dpi: int,
) -> list[Path]:
    paths = []
    for metric in COMPARISON_METRICS:
        x = joined[f"autodan_{metric}"].to_numpy(np.float64)
        y = joined[f"gcg_{metric}"].to_numpy(np.float64)
        low = float(min(x.min(), y.min()))
        high = float(max(x.max(), y.max()))
        padding = max((high - low) * 0.05, 1e-12)
        fig, axis = plt.subplots(figsize=(7.5, 7))
        try:
            axis.scatter(x, y, s=18, alpha=0.55, edgecolors="none")
            axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
            axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
            axis.plot(
                [low - padding, high + padding],
                [low - padding, high + padding],
                color="gray",
                linestyle=":",
                linewidth=1,
            )
            target = joined[(joined["layer"] == 13) & (joined["head"] == 5)].iloc[0]
            target_x = float(target[f"autodan_{metric}"])
            target_y = float(target[f"gcg_{metric}"])
            axis.scatter(
                [target_x],
                [target_y],
                marker="*",
                s=180,
                color="red",
                edgecolors="black",
                linewidths=0.7,
                zorder=4,
            )
            axis.annotate(
                "L13H5",
                (target_x, target_y),
                xytext=(6, 6),
                textcoords="offset points",
            )
            axis.set_xlim(low - padding, high + padding)
            axis.set_ylim(low - padding, high + padding)
            axis.set_xlabel(f"AutoDAN {metric}")
            axis.set_ylabel(f"GCG {metric}")
            axis.set_title(
                f"Llama-2 matched70 exploratory: {metric}\n"
                f"Pearson r={pearson_correlation(x, y):.3f}, "
                f"Spearman rho={spearman_correlation(x, y):.3f}, n=1024 heads"
            )
            axis.grid(alpha=0.2)
            fig.tight_layout()
            path = output_dir / f"scatter_matched_{metric}.png"
            fig.savefig(path, dpi=dpi)
            paths.append(path)
        finally:
            plt.close(fig)
    auto = layer_summary[layer_summary["attack"] == "AutoDAN"].set_index("layer")
    gcg = layer_summary[layer_summary["attack"] == "GCG"].set_index("layer")
    fig, axis = plt.subplots(figsize=(9, 5))
    try:
        axis.plot(
            auto.index,
            auto["delta_asym_ratio_mean"],
            marker="o",
            markersize=4,
            label="AutoDAN",
        )
        axis.plot(
            gcg.index,
            gcg["delta_asym_ratio_mean"],
            marker="o",
            markersize=4,
            label="GCG",
        )
        axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Layer")
        axis.set_ylabel("Mean matched delta_asym_ratio across heads")
        axis.set_title("Llama-2 matched70 exploratory layer profile")
        axis.set_xticks(range(0, 32, 2))
        axis.legend()
        axis.grid(alpha=0.2)
        fig.tight_layout()
        path = output_dir / "layer_profile_matched_delta_asym_ratio.png"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    finally:
        plt.close(fig)
    return paths


def current_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def build_metadata(
    args: argparse.Namespace,
    elapsed_seconds: float,
    common: pd.DataFrame,
    auto_metadata: Mapping[str, Any],
    gcg_metadata: Mapping[str, Any],
    auto_details: Mapping[str, Any],
    gcg_details: Mapping[str, Any],
    raw_delta_errors: Mapping[str, float],
) -> Dict[str, Any]:
    source_paths = {
        "autodan_raw_dir": args.autodan_raw_dir,
        "gcg_raw_dir": args.gcg_raw_dir,
        "autodan_original_metrics": args.autodan_original_metrics,
        "autodan_jailbreak_metrics": args.autodan_jailbreak_metrics,
        "gcg_original_metrics": args.gcg_original_metrics,
        "gcg_jailbreak_metrics": args.gcg_jailbreak_metrics,
        "gcg_manifest": args.gcg_manifest,
        "autodan_original_prompts": args.autodan_original_prompts,
        "autodan_jailbreak_prompts": args.autodan_jailbreak_prompts,
        "autodan_source_data": args.autodan_source_data,
        "autodan_labels": args.autodan_labels,
        "advbench": args.advbench,
    }
    hashes = {
        label: sha256_file(Path(path))
        for label, path in source_paths.items()
        if Path(path).is_file()
    }
    return {
        "analysis_type": "matched",
        "exploratory": True,
        "held_out": False,
        "model": "Llama-2-7b-chat-hf",
        "attacks": ["AutoDAN", "GCG"],
        "common_source_count": len(common),
        "common_source_min": int(common["source_index"].min()),
        "common_source_max": int(common["source_index"].max()),
        "common_source_indices_path": "common_source_indices.csv",
        "autodan_mapping_basis": auto_details["mapping_basis"],
        "gcg_mapping_basis": gcg_details["mapping_basis"],
        "tail_tokens": 16,
        "elements_per_head": 120,
        "elements_per_matched_head": len(common) * 120,
        "raw_shapes": {
            "AutoDAN": auto_metadata["array_shape"],
            "GCG": gcg_metadata["array_shape"],
        },
        "raw_array_axes": [
            "prompt",
            "layer",
            "head",
            "strict_lower_position",
        ],
        "metric_definitions": {
            "delta_asym_ratio": (
                "mean over matched prompts of jailbreak asym_ratio - original asym_ratio"
            ),
            "element_delta_mean": "mean(raw_jailbreak - raw_original)",
            "element_std_shift": "std(raw_jailbreak) - std(raw_original)",
            "element_positive_rate_shift": (
                "positive_rate(raw_jailbreak) - positive_rate(raw_original)"
            ),
        },
        "std_ddof": 0,
        "success_definition_limitation": (
            "AutoDAN contains 84 source-success and 46 source-failure pairs with no "
            "success filter; a comparable GCG success definition is unavailable."
        ),
        "analysis_positioning": (
            "exploratory post-hoc internal analysis; not an independent held-out evaluation"
        ),
        "source_input_paths": source_paths,
        "source_input_sha256": hashes,
        "source_commits": {
            "analysis_code": current_commit(),
            "AutoDAN_raw": auto_metadata.get("source_commit"),
            "GCG_raw": gcg_metadata.get("source_commit"),
        },
        "raw_matched_delta_max_abs_error": dict(raw_delta_errors),
        "cli_args": vars(args),
        "execution_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "execution_time_seconds": elapsed_seconds,
    }


def main() -> None:
    args = get_args()
    start = time.perf_counter()
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    if args.expected_common_count <= 0:
        raise ValueError("--expected_common_count must be positive")
    auto_arrays, auto_metadata = load_raw_bundle(
        Path(args.autodan_raw_dir), validate_values=not args.dry_run_mapping
    )
    gcg_arrays, gcg_metadata = load_raw_bundle(
        Path(args.gcg_raw_dir), validate_values=not args.dry_run_mapping
    )
    if auto_arrays["original"].shape[1:] != gcg_arrays["original"].shape[1:]:
        raise ValueError("AutoDAN/GCG layer-head-position shapes differ")
    autodan_mapping, auto_details = build_autodan_mapping(
        auto_arrays["original"].shape[0],
        auto_metadata,
        Path(args.autodan_original_prompts),
        Path(args.autodan_jailbreak_prompts),
        Path(args.autodan_source_data),
        Path(args.autodan_labels),
        Path(args.advbench),
        args.autodan_mapping_mode,
    )
    gcg_mapping, gcg_details = build_gcg_mapping(
        Path(args.gcg_manifest), gcg_arrays["original"].shape[0]
    )
    common = compute_common_mapping(
        autodan_mapping,
        gcg_mapping,
        args.expected_common_count,
        expected_indices=range(60, 130),
    )
    print(f"AutoDAN mapping rows: {len(autodan_mapping)}")
    print(f"GCG mapping rows: {len(gcg_mapping)}")
    print(
        "common source indices: "
        f"count={len(common)}, min={common.source_index.min()}, "
        f"max={common.source_index.max()}"
    )
    if args.dry_run_mapping:
        print("mapping dry-run: passed")
        return
    output_dir = Path(args.output_dir)
    validate_output_dir(output_dir, args.overwrite)
    autodan_mapping.to_csv(output_dir / "autodan_source_mapping.csv", index=False)
    gcg_mapping.to_csv(output_dir / "gcg_source_mapping.csv", index=False)
    common.to_csv(output_dir / "common_source_indices.csv", index=False)

    auto_raw, auto_delta_error = aggregate_raw_head_metrics(
        auto_arrays, common["raw_prompt_index_autodan"].to_numpy(np.int64)
    )
    gcg_raw, gcg_delta_error = aggregate_raw_head_metrics(
        gcg_arrays, common["raw_prompt_index_gcg"].to_numpy(np.int64)
    )
    common_sources = set(int(value) for value in common["source_index"])
    auto_original = prepare_matched_attention_metrics(
        read_metrics(Path(args.autodan_original_metrics), "AutoDAN original"),
        autodan_mapping,
        common_sources,
        "AutoDAN original",
        "prompt_id_mapping",
    )
    auto_jailbreak = prepare_matched_attention_metrics(
        read_metrics(Path(args.autodan_jailbreak_metrics), "AutoDAN jailbreak"),
        autodan_mapping,
        common_sources,
        "AutoDAN jailbreak",
        "prompt_id_mapping",
    )
    gcg_original = prepare_matched_attention_metrics(
        read_metrics(Path(args.gcg_original_metrics), "GCG original"),
        gcg_mapping,
        common_sources,
        "GCG original",
        "source_index",
    )
    gcg_jailbreak = prepare_matched_attention_metrics(
        read_metrics(Path(args.gcg_jailbreak_metrics), "GCG jailbreak"),
        gcg_mapping,
        common_sources,
        "GCG jailbreak",
        "source_index",
    )
    autodan_metrics = build_attack_metrics(
        auto_raw, aggregate_matched_attention(auto_original, auto_jailbreak)
    )
    gcg_metrics = build_attack_metrics(
        gcg_raw, aggregate_matched_attention(gcg_original, gcg_jailbreak)
    )
    autodan_metrics.to_csv(output_dir / "autodan_matched_head_metrics.csv", index=False)
    gcg_metrics.to_csv(output_dir / "gcg_matched_head_metrics.csv", index=False)
    joined = build_joined_metrics(autodan_metrics, gcg_metrics)
    joined.to_csv(output_dir / "matched_head_metrics_joined.csv", index=False)
    correlations, signs = build_metric_comparisons(joined)
    correlations.to_csv(output_dir / "matched_metric_correlations.csv", index=False)
    signs.to_csv(output_dir / "matched_sign_agreement.csv", index=False)
    topk = build_topk_overlap(autodan_metrics, gcg_metrics)
    topk.to_csv(output_dir / "matched_topk_overlap.csv", index=False)
    layer_summary, layer_correlations = build_layer_outputs(
        {"AutoDAN": autodan_metrics, "GCG": gcg_metrics}
    )
    layer_summary.to_csv(output_dir / "matched_layer_summary.csv", index=False)
    layer_correlations.to_csv(
        output_dir / "matched_layer_correlations.csv", index=False
    )
    l13h5 = build_l13h5_summary(
        {"AutoDAN": autodan_metrics, "GCG": gcg_metrics}
    )
    l13h5.to_csv(output_dir / "l13h5_matched_summary.csv", index=False)
    auto_selected, auto_pngs = write_histograms(
        output_dir,
        "AutoDAN",
        autodan_metrics,
        auto_arrays,
        common["raw_prompt_index_autodan"].to_numpy(np.int64),
        args.top_k,
        args.bins,
        args.dpi,
    )
    gcg_selected, gcg_pngs = write_histograms(
        output_dir,
        "GCG",
        gcg_metrics,
        gcg_arrays,
        common["raw_prompt_index_gcg"].to_numpy(np.int64),
        args.top_k,
        args.bins,
        args.dpi,
    )
    comparison_pngs = save_comparison_plots(
        output_dir, joined, layer_summary, args.dpi
    )
    pngs = [*auto_pngs, *gcg_pngs, *comparison_pngs]
    expected_histogram_pngs = 4 * args.top_k
    if (
        len(auto_pngs) != expected_histogram_pngs
        or len(gcg_pngs) != expected_histogram_pngs
        or len(comparison_pngs) != 5
    ):
        raise ValueError("Unexpected PNG count")
    if any(not path.is_file() or path.stat().st_size == 0 for path in pngs):
        raise ValueError("A generated PNG is empty or missing")
    elapsed = time.perf_counter() - start
    metadata = build_metadata(
        args,
        elapsed,
        common,
        auto_metadata,
        gcg_metadata,
        auto_details,
        gcg_details,
        {"AutoDAN": auto_delta_error, "GCG": gcg_delta_error},
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"matched head rows per attack: {len(autodan_metrics)}")
    print(f"elements per matched head: {int(auto_raw.element_value_count.iloc[0])}")
    print(f"joined head rows: {len(joined)}")
    print(f"layer summary rows: {len(layer_summary)}")
    print(f"AutoDAN selected heads: {len(auto_selected)}")
    print(f"GCG selected heads: {len(gcg_selected)}")
    print(f"histogram PNGs: {len(auto_pngs) + len(gcg_pngs)}")
    print(f"comparison PNGs: {len(comparison_pngs)}")
    print(f"execution time seconds: {elapsed:.3f}")
    print(f"output directory: {output_dir}")


if __name__ == "__main__":
    main()
