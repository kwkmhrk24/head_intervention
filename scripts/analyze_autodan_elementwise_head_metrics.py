#!/usr/bin/env python3
"""Compute selection-only elementwise S_asym head metrics offline."""

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SELECTION_INDICES = [
    4, 5, 15, 18, 19, 26, 36, 39, 40, 41,
    43, 48, 80, 82, 96, 108, 110, 113, 126, 129,
]
PAPER_HEADS = [(3, 0), (7, 16), (9, 9), (13, 4), (3, 5)]
EXPECTED_AIE_HEADS = [(13, 4), (16, 15), (15, 30), (12, 1), (21, 14)]
SELECTED_HEAD_KEYS = {
    "asym_ratio_positive": "positive_top5",
    "asym_ratio_negative": "negative_bottom5",
    "asym_ratio_absolute": "absolute_top5",
}
MAIN_METRICS = (
    "consistent_positive",
    "consistent_negative",
    "consistent_absolute",
)
CORRELATION_METRICS = [
    "signed_mean",
    "positive_mass",
    "negative_mass",
    "absolute_mass",
    "consistent_positive",
    "consistent_negative",
    "consistent_absolute",
    "last_row_consistent_positive",
    "last_row_consistent_negative",
    "last_row_consistent_absolute",
    "lag_profile_absolute",
    "mean_cosine_distance",
    "sign_flip_rate",
]
METRIC_COLUMNS = [
    "signed_mean",
    "signed_median",
    "positive_mass",
    "negative_mass",
    "absolute_mass",
    "positive_rate",
    "negative_rate",
    "sign_imbalance",
    "consistent_positive",
    "consistent_negative",
    "consistent_absolute",
    "strong_positive_position_rate",
    "strong_negative_position_rate",
    "last_row_signed_mean",
    "last_row_positive_mass",
    "last_row_negative_mass",
    "last_row_absolute_mass",
    "last_row_consistent_positive",
    "last_row_consistent_negative",
    "last_row_consistent_absolute",
    "lag_profile_absolute",
    "lag_profile_mass",
    "mean_cosine_similarity",
    "mean_cosine_distance",
    "sign_flip_rate",
    "mean_pearson_correlation",
]
RESEARCH_NOTE = (
    "The full AutoDAN dataset had already been used for exploratory metric "
    "analysis. The new elementwise metrics were computed only from the fixed "
    "20-prompt selection subset. Any later reuse of the 110-prompt evaluation "
    "subset is post-hoc internal analysis rather than a fully untouched test."
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compute offline elementwise antisymmetric attention-score metrics "
            "using only the fixed AutoDAN selection prompts."
        )
    )
    parser.add_argument("--raw-original", required=True)
    parser.add_argument("--raw-jailbreak", required=True)
    parser.add_argument("--raw-delta", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--selected-heads", required=True)
    parser.add_argument("--aie-ranking", required=True)
    parser.add_argument("--ratio-ranking", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--consistency-threshold", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_json(path):
    with Path(path).open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            rows.append(row)
    return rows


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def strict_lower_positions(window_size):
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    rows, columns = np.tril_indices(window_size, k=-1)
    return rows.astype(np.int64), columns.astype(np.int64)


def validate_selection_indices(indices, total_count, expected_size=20):
    if not isinstance(indices, list) or any(type(index) is not int for index in indices):
        raise ValueError("selection_indices must be a list of integers")
    if len(indices) != expected_size:
        raise ValueError(
            f"expected {expected_size} selection indices, found {len(indices)}"
        )
    if len(set(indices)) != len(indices):
        raise ValueError("selection_indices contain duplicates")
    invalid = [index for index in indices if not 0 <= index < total_count]
    if invalid:
        raise ValueError(f"selection_indices are out of range: {invalid}")
    return indices


def validate_manifest(manifest):
    expected = {
        "dataset": "AutoDAN 130 paired prompts",
        "model": "Llama-2-7b-chat-hf",
        "attack": "AutoDAN",
        "total_count": 130,
        "selection_size": 20,
        "evaluation_size": 110,
        "seed": 42,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"split manifest expected {key}={value!r}, found {manifest.get(key)!r}"
            )
    selection = validate_selection_indices(
        manifest.get("selection_indices"), manifest["total_count"]
    )
    if selection != EXPECTED_SELECTION_INDICES:
        raise ValueError("selection_indices do not match the fixed seed-42 split")
    evaluation = manifest.get("evaluation_indices")
    if not isinstance(evaluation, list) or len(evaluation) != 110:
        raise ValueError("evaluation_indices must contain 110 entries")
    if set(selection) & set(evaluation):
        raise ValueError("selection and evaluation indices overlap")
    if set(selection) | set(evaluation) != set(range(130)):
        raise ValueError("selection/evaluation indices do not cover all 130 prompts")
    return selection


def validate_raw_metadata(metadata, manifest):
    expected = {
        "processed_prompt_count": 130,
        "tail_tokens": 16,
        "window_size": 16,
        "elements_per_head": 120,
        "num_layers": 32,
        "num_attention_heads": 32,
        "array_shape": [130, 32, 32, 120],
        "array_axes": ["prompt", "layer", "head", "strict_lower_position"],
        "score_definition": "S = QK^T / sqrt(d_head)",
        "post_rope": True,
        "pre_mask": True,
        "pre_softmax": True,
        "strict_lower_triangle": "row_pos > col_pos",
        "strict_lower_position_order": "row-major order",
        "prompt_id_definition": "zero-based JSONL row index",
        "source_index_definition": "zero-based JSONL row index",
        "delta_definition": (
            "jailbreak - original at paired prompt/layer/head/position"
        ),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"raw metadata expected {key}={value!r}, found {metadata.get(key)!r}"
            )
    rows, columns = strict_lower_positions(16)
    if metadata.get("strict_lower_row_positions") != rows.tolist():
        raise ValueError("metadata strict-lower row positions are inconsistent")
    if metadata.get("strict_lower_col_positions") != columns.tolist():
        raise ValueError("metadata strict-lower column positions are inconsistent")

    original_path = metadata.get("original_prompts")
    jailbreak_path = metadata.get("jailbreak_prompts")
    manifest_paths = manifest.get("original_input_paths", {})
    if manifest_paths.get("original_prompts") != original_path:
        raise ValueError("original prompt path differs between metadata and manifest")
    if manifest_paths.get("jailbreak_prompts") != jailbreak_path:
        raise ValueError("jailbreak prompt path differs between metadata and manifest")
    original_rows = read_jsonl(resolve_repo_path(original_path))
    jailbreak_rows = read_jsonl(resolve_repo_path(jailbreak_path))
    if len(original_rows) != 130 or len(jailbreak_rows) != 130:
        raise ValueError("metadata prompt files must each contain 130 rows")
    for source_index, (original, jailbreak) in enumerate(
        zip(original_rows, jailbreak_rows)
    ):
        for condition, row in (("original", original), ("jailbreak", jailbreak)):
            if row.get("source_index", source_index) != source_index:
                raise ValueError(
                    f"{condition} source_index mismatch at row {source_index}"
                )
            if not isinstance(row.get("prompt"), str):
                raise ValueError(f"{condition} row {source_index} lacks prompt")
    return rows, columns


def load_selection_arrays(paths, metadata, selection_indices):
    arrays = {
        name: np.load(path, mmap_mode="r")
        for name, path in paths.items()
    }
    expected_shape = tuple(metadata["array_shape"])
    for name, array in arrays.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_shape}, found {array.shape}"
            )
        if array.dtype != np.float32:
            raise ValueError(f"{name} dtype must be float32, found {array.dtype}")

    # Fancy indexing copies only the fixed 20 selection rows from each memmap.
    selected = {
        name: np.asarray(array[selection_indices], dtype=np.float32)
        for name, array in arrays.items()
    }
    expected_selected_shape = (20, 32, 32, 120)
    for name, array in selected.items():
        if array.shape != expected_selected_shape:
            raise ValueError(f"selected {name} shape mismatch: {array.shape}")
    max_error = validate_delta_identity(
        selected["original"], selected["jailbreak"], selected["delta"]
    )
    return selected, max_error


def validate_delta_identity(original, jailbreak, delta):
    if original.shape != jailbreak.shape or original.shape != delta.shape:
        raise ValueError("original/jailbreak/delta shapes differ")
    for name, array in (
        ("original", original),
        ("jailbreak", jailbreak),
        ("delta", delta),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinity")
    difference = jailbreak - original
    max_error = float(np.max(np.abs(delta - difference)))
    if not np.allclose(delta, difference, rtol=1e-5, atol=2e-6):
        raise ValueError(f"raw delta identity failed; max error={max_error}")
    return max_error


def _positive_negative_absolute(values, axes):
    positive = np.mean(np.maximum(values, 0), axis=axes, dtype=np.float64)
    negative = np.mean(np.maximum(-values, 0), axis=axes, dtype=np.float64)
    absolute = np.mean(np.abs(values), axis=axes, dtype=np.float64)
    if not np.allclose(absolute, positive + negative, rtol=1e-10, atol=1e-10):
        raise ValueError("positive/negative/absolute mass identity failed")
    return positive, negative, absolute


def compute_structural_metrics(original, jailbreak):
    original64 = np.asarray(original, dtype=np.float64)
    jailbreak64 = np.asarray(jailbreak, dtype=np.float64)
    dot = np.sum(original64 * jailbreak64, axis=-1, dtype=np.float64)
    original_norm = np.sqrt(
        np.sum(original64 * original64, axis=-1, dtype=np.float64)
    )
    jailbreak_norm = np.sqrt(
        np.sum(jailbreak64 * jailbreak64, axis=-1, dtype=np.float64)
    )
    both_zero = (original_norm == 0) & (jailbreak_norm == 0)
    one_zero = (original_norm == 0) ^ (jailbreak_norm == 0)
    regular = ~(both_zero | one_zero)
    cosine = np.empty_like(dot)
    cosine[both_zero] = 1.0
    cosine[one_zero] = 0.0
    cosine[regular] = dot[regular] / (
        original_norm[regular] * jailbreak_norm[regular]
    )
    cosine = np.clip(cosine, -1.0, 1.0)

    original_mean = np.mean(original64, axis=-1, keepdims=True)
    jailbreak_mean = np.mean(jailbreak64, axis=-1, keepdims=True)
    original_centered = original64 - original_mean
    jailbreak_centered = jailbreak64 - jailbreak_mean
    covariance = np.sum(
        original_centered * jailbreak_centered, axis=-1, dtype=np.float64
    )
    denominator = np.sqrt(
        np.sum(original_centered**2, axis=-1, dtype=np.float64)
        * np.sum(jailbreak_centered**2, axis=-1, dtype=np.float64)
    )
    valid = denominator > 0
    correlations = np.zeros_like(covariance)
    correlations[valid] = covariance[valid] / denominator[valid]
    correlations = np.clip(correlations, -1.0, 1.0)
    valid_count = np.sum(valid, axis=0)
    correlation_sum = np.sum(np.where(valid, correlations, 0.0), axis=0)
    mean_correlation = np.full(valid_count.shape, np.nan, dtype=np.float64)
    np.divide(
        correlation_sum,
        valid_count,
        out=mean_correlation,
        where=valid_count > 0,
    )

    return {
        "mean_cosine_similarity": np.mean(cosine, axis=0, dtype=np.float64),
        "mean_cosine_distance": 1.0
        - np.mean(cosine, axis=0, dtype=np.float64),
        "cosine_both_zero_prompt_count": np.sum(both_zero, axis=0),
        "cosine_one_zero_prompt_count": np.sum(one_zero, axis=0),
        "sign_flip_rate": np.mean(
            np.sign(original) != np.sign(jailbreak),
            axis=(0, 3),
            dtype=np.float64,
        ),
        "mean_pearson_correlation": mean_correlation,
        "valid_pearson_prompt_count": valid_count,
    }


def compute_elementwise_metrics(
    original,
    jailbreak,
    delta,
    row_positions,
    col_positions,
    consistency_threshold=0.75,
):
    if original.shape != jailbreak.shape or original.shape != delta.shape:
        raise ValueError("original/jailbreak/delta selection shapes differ")
    if original.ndim != 4:
        raise ValueError("selection arrays must have axes prompt,layer,head,position")
    if original.shape[-1] != len(row_positions) or len(row_positions) != len(col_positions):
        raise ValueError("strict-lower position count does not match arrays")
    if not 0 <= consistency_threshold <= 1:
        raise ValueError("consistency_threshold must be in [0, 1]")

    metrics = {
        "signed_mean": np.mean(delta, axis=(0, 3), dtype=np.float64),
        "signed_median": np.median(delta, axis=(0, 3)),
        "positive_rate": np.mean(delta > 0, axis=(0, 3), dtype=np.float64),
        "negative_rate": np.mean(delta < 0, axis=(0, 3), dtype=np.float64),
    }
    (
        metrics["positive_mass"],
        metrics["negative_mass"],
        metrics["absolute_mass"],
    ) = _positive_negative_absolute(delta, (0, 3))
    metrics["sign_imbalance"] = metrics["positive_rate"] - metrics["negative_rate"]
    if np.any(np.abs(metrics["sign_imbalance"]) > 1 + 1e-12):
        raise ValueError("sign_imbalance is outside [-1, 1]")

    position_mean = np.mean(delta, axis=0, dtype=np.float64)
    consistent_positive = np.mean(
        np.maximum(position_mean, 0), axis=-1, dtype=np.float64
    )
    consistent_negative = np.mean(
        np.maximum(-position_mean, 0), axis=-1, dtype=np.float64
    )
    consistent_absolute = np.mean(
        np.abs(position_mean), axis=-1, dtype=np.float64
    )
    if not np.allclose(
        consistent_absolute,
        consistent_positive + consistent_negative,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError("consistent metric identity failed")
    metrics.update(
        {
            "consistent_positive": consistent_positive,
            "consistent_negative": consistent_negative,
            "consistent_absolute": consistent_absolute,
            "strong_positive_position_rate": np.mean(
                np.mean(delta > 0, axis=0) >= consistency_threshold,
                axis=-1,
            ),
            "strong_negative_position_rate": np.mean(
                np.mean(delta < 0, axis=0) >= consistency_threshold,
                axis=-1,
            ),
        }
    )

    last_row_mask = row_positions == 15
    if int(np.sum(last_row_mask)) != 15:
        raise ValueError("tail16 last-row mask must contain 15 positions")
    last_delta = delta[..., last_row_mask]
    metrics["last_row_signed_mean"] = np.mean(
        last_delta, axis=(0, 3), dtype=np.float64
    )
    (
        metrics["last_row_positive_mass"],
        metrics["last_row_negative_mass"],
        metrics["last_row_absolute_mass"],
    ) = _positive_negative_absolute(last_delta, (0, 3))
    last_position_mean = position_mean[..., last_row_mask]
    metrics["last_row_consistent_positive"] = np.mean(
        np.maximum(last_position_mean, 0), axis=-1, dtype=np.float64
    )
    metrics["last_row_consistent_negative"] = np.mean(
        np.maximum(-last_position_mean, 0), axis=-1, dtype=np.float64
    )
    metrics["last_row_consistent_absolute"] = np.mean(
        np.abs(last_position_mean), axis=-1, dtype=np.float64
    )
    if not np.allclose(
        metrics["last_row_consistent_absolute"],
        metrics["last_row_consistent_positive"]
        + metrics["last_row_consistent_negative"],
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError("last-row consistent metric identity failed")

    lags = row_positions - col_positions
    if sorted(np.unique(lags).tolist()) != list(range(1, 16)):
        raise ValueError("tail16 lags must cover 1 through 15")
    lag_signed = np.empty((original.shape[1], original.shape[2], 15), dtype=np.float64)
    lag_absolute = np.empty_like(lag_signed)
    for lag in range(1, 16):
        mask = lags == lag
        lag_signed[..., lag - 1] = np.mean(
            delta[..., mask], axis=(0, 3), dtype=np.float64
        )
        lag_absolute[..., lag - 1] = np.mean(
            np.abs(delta[..., mask]), axis=(0, 3), dtype=np.float64
        )
    metrics["lag_profile_absolute"] = np.mean(
        np.abs(lag_signed), axis=-1, dtype=np.float64
    )
    metrics["lag_profile_mass"] = np.mean(
        lag_absolute, axis=-1, dtype=np.float64
    )
    metrics.update(compute_structural_metrics(original, jailbreak))
    return metrics, position_mean, lag_signed, lag_absolute


def ranking_specs():
    specs = [
        ("signed_mean_positive_top", "signed_mean", False),
        ("signed_mean_negative_bottom", "signed_mean", True),
    ]
    specs.extend(
        (f"{metric}_top", metric, False)
        for metric in METRIC_COLUMNS
        if metric != "signed_mean"
    )
    return specs


def metrics_dataframe(metrics):
    layers, heads = metrics["signed_mean"].shape
    frame = pd.DataFrame(
        {
            "layer": np.repeat(np.arange(layers), heads),
            "head": np.tile(np.arange(heads), layers),
        }
    )
    for name, values in metrics.items():
        if values.shape == (layers, heads):
            frame[name] = values.reshape(-1)
    for ranking_name, metric, ascending in ranking_specs():
        frame[f"{ranking_name}_rank"] = frame[metric].rank(
            method="min", ascending=ascending, na_option="bottom"
        ).astype(int)
    return frame


def ranked_rows(frame, metric, ascending, limit):
    return frame.sort_values(
        [metric, "layer", "head"],
        ascending=[ascending, True, True],
        na_position="last",
        kind="mergesort",
    ).head(limit)


def build_top_head_outputs(frame, top_n=20):
    json_payload = {}
    csv_rows = []
    for ranking_name, metric, ascending in ranking_specs():
        ranked = ranked_rows(frame, metric, ascending, top_n)
        rows = []
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            value = getattr(row, metric)
            item = {
                "rank": rank,
                "layer": int(row.layer),
                "head": int(row.head),
                "value": None if pd.isna(value) else float(value),
            }
            rows.append(item)
            csv_rows.append(
                {
                    "ranking": ranking_name,
                    "metric": metric,
                    "direction": "ascending" if ascending else "descending",
                    **item,
                }
            )
        json_payload[ranking_name] = rows
    return json_payload, pd.DataFrame(csv_rows)


def bootstrap_prompt_indices(prompt_count, sample_count, seed):
    if prompt_count <= 0 or sample_count <= 0:
        raise ValueError("prompt_count and sample_count must be positive")
    return np.random.default_rng(seed).integers(
        0, prompt_count, size=(sample_count, prompt_count)
    )


def _top_flat_indices(values, top_k):
    flat = np.asarray(values).reshape(-1)
    if not np.isfinite(flat).all():
        raise ValueError("bootstrap ranking values contain NaN or infinity")
    order = np.lexsort((np.arange(flat.size), -flat))
    return order[:top_k]


def jaccard_similarity(first, second):
    first, second = set(first), set(second)
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def bootstrap_consistent_stability(delta, top_k=5, sample_count=1000, seed=42):
    if delta.ndim != 4:
        raise ValueError("delta must have prompt,layer,head,position axes")
    head_count = delta.shape[1] * delta.shape[2]
    if not 1 <= top_k <= head_count:
        raise ValueError("top_k is outside the available head count")
    samples = bootstrap_prompt_indices(delta.shape[0], sample_count, seed)
    original_position_mean = np.mean(delta, axis=0, dtype=np.float64)
    original_scores = {
        "consistent_positive": np.mean(
            np.maximum(original_position_mean, 0), axis=-1
        ),
        "consistent_negative": np.mean(
            np.maximum(-original_position_mean, 0), axis=-1
        ),
        "consistent_absolute": np.mean(np.abs(original_position_mean), axis=-1),
    }
    original_top = {
        metric: _top_flat_indices(values, top_k) for metric, values in original_scores.items()
    }
    counts = {metric: np.zeros(head_count, dtype=np.int64) for metric in MAIN_METRICS}
    jaccards = {metric: [] for metric in MAIN_METRICS}

    for sample in samples:
        sampled_position_mean = np.mean(delta[sample], axis=0, dtype=np.float64)
        sampled_scores = {
            "consistent_positive": np.mean(
                np.maximum(sampled_position_mean, 0), axis=-1
            ),
            "consistent_negative": np.mean(
                np.maximum(-sampled_position_mean, 0), axis=-1
            ),
            "consistent_absolute": np.mean(
                np.abs(sampled_position_mean), axis=-1
            ),
        }
        for metric, values in sampled_scores.items():
            selected = _top_flat_indices(values, top_k)
            counts[metric][selected] += 1
            jaccards[metric].append(
                jaccard_similarity(original_top[metric], selected)
            )

    inclusion = pd.DataFrame(
        {
            "layer": np.repeat(np.arange(delta.shape[1]), delta.shape[2]),
            "head": np.tile(np.arange(delta.shape[2]), delta.shape[1]),
        }
    )
    summary = {
        "bootstrap_samples": sample_count,
        "seed": seed,
        "top_k": top_k,
        "resampling_unit": "selection prompt; positions are never resampled",
        "metrics": {},
    }
    for metric in MAIN_METRICS:
        inclusion[f"{metric}_inclusion_count"] = counts[metric]
        inclusion[f"{metric}_top_k_inclusion_probability"] = (
            counts[metric] / sample_count
        )
        values = np.asarray(jaccards[metric], dtype=np.float64)
        summary["metrics"][metric] = {
            "original_top_k": [
                {
                    "layer": int(index // delta.shape[2]),
                    "head": int(index % delta.shape[2]),
                }
                for index in original_top[metric]
            ],
            "jaccard_similarity": {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "q05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "q95": float(np.quantile(values, 0.95)),
            },
        }
    return inclusion, summary, samples


def validate_head_list(name, heads, expected_count=5):
    normalized = []
    for pair in heads:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{name} contains an invalid head pair")
        layer, head = pair
        if type(layer) is not int or type(head) is not int:
            raise ValueError(f"{name} layer/head must be integers")
        if not 0 <= layer < 32 or not 0 <= head < 32:
            raise ValueError(f"{name} contains an out-of-range head {(layer, head)}")
        normalized.append((layer, head))
    if len(normalized) != expected_count:
        raise ValueError(f"{name} must contain {expected_count} heads")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicate heads")
    return normalized


def load_reference_headsets(selected_heads_path, aie_ranking_path):
    selected_payload = read_json(selected_heads_path)
    references = {"paper": validate_head_list("paper", PAPER_HEADS)}
    for name, key in SELECTED_HEAD_KEYS.items():
        rows = selected_payload.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"selected-heads JSON lacks {key}")
        references[name] = validate_head_list(
            name, [(row.get("layer"), row.get("head")) for row in rows]
        )

    aie = pd.read_csv(aie_ranking_path)
    required = {"rank", "layer", "head"}
    if not required.issubset(aie.columns):
        raise ValueError(f"AIE ranking lacks columns {sorted(required - set(aie.columns))}")
    aie = aie.sort_values("rank").head(5)
    references["aie"] = validate_head_list(
        "aie",
        [(int(row.layer), int(row.head)) for row in aie.itertuples()],
    )
    if references["aie"] != EXPECTED_AIE_HEADS:
        raise ValueError(
            f"AIE top-5 differs from the verified expected set: {references['aie']}"
        )
    return references, selected_payload


def attach_ratio_metric(frame, ratio_ranking_path, selection_indices):
    ratio = pd.read_csv(ratio_ranking_path)
    required = {"layer", "head", "count", "mean_delta_asym_ratio"}
    if not required.issubset(ratio.columns):
        raise ValueError(
            f"ratio ranking lacks columns {sorted(required - set(ratio.columns))}"
        )
    if len(ratio) != 1024 or ratio.duplicated(["layer", "head"]).any():
        raise ValueError("ratio ranking must contain one row for each of 1024 heads")
    if not (ratio["count"] == len(selection_indices)).all():
        raise ValueError("ratio ranking was not aggregated over selection 20")
    merged = frame.merge(
        ratio[["layer", "head", "mean_delta_asym_ratio"]],
        on=["layer", "head"],
        how="left",
        validate="one_to_one",
    )
    if merged["mean_delta_asym_ratio"].isna().any():
        raise ValueError("ratio ranking does not cover all layer/head pairs")
    return merged


def build_overlap_outputs(frame, references, top_sizes=(5, 10)):
    rows = []
    payload = {"reference_headsets": {}, "comparisons": []}
    for name, heads in references.items():
        payload["reference_headsets"][name] = [
            {"layer": layer, "head": head} for layer, head in heads
        ]
    for metric in MAIN_METRICS:
        for top_n in top_sizes:
            ranked = ranked_rows(frame, metric, False, top_n)
            new_heads = [
                (int(row.layer), int(row.head)) for row in ranked.itertuples()
            ]
            for reference_name, reference_heads in references.items():
                common = sorted(set(new_heads) & set(reference_heads))
                row = {
                    "metric": metric,
                    "new_head_count": top_n,
                    "reference_set": reference_name,
                    "reference_head_count": len(reference_heads),
                    "overlap_count": len(common),
                    "jaccard_similarity": jaccard_similarity(
                        new_heads, reference_heads
                    ),
                    "common_heads": json.dumps(common),
                }
                rows.append(row)
                payload["comparisons"].append(
                    {
                        **row,
                        "new_heads": [
                            {"layer": layer, "head": head}
                            for layer, head in new_heads
                        ],
                        "common_heads": [
                            {"layer": layer, "head": head}
                            for layer, head in common
                        ],
                    }
                )
    return pd.DataFrame(rows), payload


def build_reference_rank_rows(frame, references):
    indexed = frame.set_index(["layer", "head"])
    rows = []
    for reference_name, heads in references.items():
        for layer, head in heads:
            values = indexed.loc[(layer, head)]
            for ranking_name, metric, ascending in ranking_specs():
                value = values[metric]
                rows.append(
                    {
                        "reference_set": reference_name,
                        "layer": layer,
                        "head": head,
                        "ranking": ranking_name,
                        "metric": metric,
                        "direction": "ascending" if ascending else "descending",
                        "value": None if pd.isna(value) else float(value),
                        "rank": int(values[f"{ranking_name}_rank"]),
                    }
                )
    return pd.DataFrame(rows)


def spearman_correlation(frame, metrics):
    missing = [metric for metric in metrics if metric not in frame.columns]
    if missing:
        raise ValueError(f"correlation metrics are unavailable: {missing}")
    ranked = frame[metrics].rank(method="average", na_option="keep")
    return ranked.corr(method="pearson")


def build_lag_frame(lag_signed, lag_absolute):
    rows = []
    for layer in range(lag_signed.shape[0]):
        for head in range(lag_signed.shape[1]):
            for lag in range(1, 16):
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "lag": lag,
                        "lag_signed_mean": float(lag_signed[layer, head, lag - 1]),
                        "lag_absolute_mean": float(
                            lag_absolute[layer, head, lag - 1]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    with Path(path).open("w") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def output_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "metrics": output_dir / "per_head_elementwise_metrics.csv",
        "top_json": output_dir / "top_heads_by_metric.json",
        "top_csv": output_dir / "top_heads_by_metric.csv",
        "correlations": output_dir / "metric_rank_correlations.csv",
        "overlap_csv": output_dir / "headset_overlap_summary.csv",
        "overlap_json": output_dir / "headset_overlap_summary.json",
        "reference_ranks": output_dir / "reference_head_metric_ranks.csv",
        "lags": output_dir / "lag_profiles.csv",
        "positions": output_dir / "position_mean_delta.npy",
        "bootstrap_csv": output_dir / "bootstrap_head_inclusion.csv",
        "bootstrap_json": output_dir / "bootstrap_stability_summary.json",
        "metadata": output_dir / "metadata.json",
    }


def check_outputs(paths, overwrite):
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))


def main(argv=None):
    args = parse_args(argv)
    if args.top_k <= 0 or args.top_k > 1024:
        raise ValueError("--top-k must be in 1..1024")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if not 0 <= args.consistency_threshold <= 1:
        raise ValueError("--consistency-threshold must be in [0, 1]")
    paths = output_paths(args.output_dir)
    check_outputs(paths, args.overwrite)

    manifest = read_json(args.split_manifest)
    selection_indices = validate_manifest(manifest)
    metadata = read_json(args.metadata)
    row_positions, col_positions = validate_raw_metadata(metadata, manifest)
    raw_paths = {
        "original": args.raw_original,
        "jailbreak": args.raw_jailbreak,
        "delta": args.raw_delta,
    }
    arrays, delta_max_error = load_selection_arrays(
        raw_paths, metadata, selection_indices
    )
    references, selected_payload = load_reference_headsets(
        args.selected_heads, args.aie_ranking
    )
    if selected_payload.get("selection_indices") != selection_indices:
        raise ValueError("selected-heads JSON selection indices differ from manifest")

    metrics, position_mean, lag_signed, lag_absolute = compute_elementwise_metrics(
        arrays["original"],
        arrays["jailbreak"],
        arrays["delta"],
        row_positions,
        col_positions,
        args.consistency_threshold,
    )
    frame = attach_ratio_metric(
        metrics_dataframe(metrics), args.ratio_ranking, selection_indices
    )
    if frame.shape[0] != 1024:
        raise ValueError(f"expected 1024 head rows, found {len(frame)}")
    numeric = frame.select_dtypes(include=[np.number]).drop(
        columns=["mean_pearson_correlation"], errors="ignore"
    )
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("per-head metrics contain unexpected NaN or infinity")
    pearson_invalid = frame["valid_pearson_prompt_count"] == 0
    if frame.loc[~pearson_invalid, "mean_pearson_correlation"].isna().any():
        raise ValueError("valid Pearson rows contain NaN")

    top_json, top_csv = build_top_head_outputs(frame, top_n=20)
    inclusion, bootstrap_summary, _ = bootstrap_consistent_stability(
        arrays["delta"], args.top_k, args.bootstrap_samples, args.seed
    )
    overlap_csv, overlap_json = build_overlap_outputs(frame, references)
    reference_ranks = build_reference_rank_rows(frame, references)
    correlation_metrics = [*CORRELATION_METRICS, "mean_delta_asym_ratio"]
    correlations = spearman_correlation(frame, correlation_metrics)
    lag_frame = build_lag_frame(lag_signed, lag_absolute)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths["metrics"], index=False, na_rep="")
    write_json(paths["top_json"], top_json)
    top_csv.to_csv(paths["top_csv"], index=False)
    correlations.to_csv(paths["correlations"], index_label="metric")
    overlap_csv.to_csv(paths["overlap_csv"], index=False)
    write_json(paths["overlap_json"], overlap_json)
    reference_ranks.to_csv(paths["reference_ranks"], index=False, na_rep="")
    lag_frame.to_csv(paths["lags"], index=False)
    np.save(paths["positions"], position_mean.astype(np.float32))
    inclusion.to_csv(paths["bootstrap_csv"], index=False)
    write_json(paths["bootstrap_json"], bootstrap_summary)

    command_args = sys.argv if argv is None else [sys.argv[0], *argv]
    output_metadata = {
        "model": metadata["model"],
        "attack": "AutoDAN",
        "tail_tokens": 16,
        "matrix_component": "S_asym strict-lower triangle",
        "score_definition": metadata["score_definition"],
        "score_stage": {
            "post_rope": True,
            "pre_mask": True,
            "pre_softmax": True,
        },
        "raw_array_paths": {
            name: str(Path(path).resolve()) for name, path in raw_paths.items()
        },
        "raw_array_shape": metadata["array_shape"],
        "axis_order": metadata["array_axes"],
        "strict_lower_position_count": 120,
        "strict_lower_position_order": "row-major order",
        "last_row_definition": (
            "row_pos == 15 inside the tail16 window; not an absolute sequence index"
        ),
        "selection_indices": selection_indices,
        "selection_size": 20,
        "evaluation_size": 110,
        "evaluation_values_used_for_metrics": False,
        "selection_array_shape": list(arrays["delta"].shape),
        "seed": args.seed,
        "split_seed": manifest["seed"],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_resampling_unit": "prompt",
        "top_k": args.top_k,
        "top_head_output_count_per_ranking": 20,
        "consistency_threshold": args.consistency_threshold,
        "delta_identity_max_abs_error_selection": delta_max_error,
        "cosine_zero_vector_policy": (
            "both zero vectors -> similarity 1; exactly one zero vector -> 0"
        ),
        "sign_flip_definition": "np.sign(original) != np.sign(jailbreak)",
        "pearson_constant_vector_policy": (
            "prompt/head correlation is null when either vector is constant; "
            "head mean uses valid prompts only"
        ),
        "reference_head_sources": {
            "paper": "evaluate.py fixed Llama heads",
            "aie": str(Path(args.aie_ranking).resolve()),
            "asym_ratio": str(Path(args.selected_heads).resolve()),
            "mean_delta_asym_ratio": str(Path(args.ratio_ranking).resolve()),
        },
        "git_commit": git_commit(),
        "command": shlex.join([sys.executable, *command_args]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "research_context_note": RESEARCH_NOTE,
    }
    write_json(paths["metadata"], output_metadata)

    print(f"selection prompts used: {len(selection_indices)}")
    print(f"per-head rows: {len(frame)}")
    print(f"bootstrap samples: {args.bootstrap_samples}")
    for metric in MAIN_METRICS:
        top = ranked_rows(frame, metric, False, args.top_k)
        heads = [(int(row.layer), int(row.head)) for row in top.itertuples()]
        print(f"{metric} top-{args.top_k}: {heads}")
    print(f"output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
