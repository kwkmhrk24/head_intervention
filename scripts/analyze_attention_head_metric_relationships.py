import argparse
import json
import tempfile
from datetime import datetime, timezone
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


PAPER_HEADS = {(3, 0), (7, 16), (9, 9), (13, 4), (3, 5)}
JOIN_KEYS = ["prompt_id", "layer", "head"]
HEAD_KEYS = ["layer", "head"]
ATTENTION_METRICS = [
    "sym_ratio",
    "asym_ratio",
    "asym_lower_mean",
    "asym_lower_abs_mean",
    "asym_lower_pos_rate",
    "sym_norm_per_token",
    "asym_norm_per_token",
]
PER_HEAD_REQUIRED = {
    "layer",
    "head",
    "value_count",
    "original_mean",
    "jailbreak_mean",
    "delta_mean",
    "original_std",
    "jailbreak_std",
    "delta_std",
    "original_positive_rate",
    "jailbreak_positive_rate",
    "delta_positive_rate",
}
METRICS_REQUIRED = {
    "prompt_id",
    "source_index",
    "layer",
    "head",
    "window_size",
    *ATTENTION_METRICS,
}
CORRELATION_METRICS = [
    "element_delta_mean",
    "element_std_shift",
    "element_positive_rate_shift",
    "element_delta_positive_rate",
    "delta_sym_ratio",
    "delta_asym_ratio",
    "delta_asym_lower_mean",
    "delta_asym_lower_abs_mean",
    "aie",
    "abs_element_delta_mean",
    "abs_element_std_shift",
    "abs_element_positive_rate_shift",
    "abs_delta_sym_ratio",
    "abs_delta_asym_ratio",
    "abs_delta_asym_lower_mean",
    "abs_aie",
]
OUTPUT_FILENAMES = [
    "head_metrics_merged.csv",
    "head_metrics_layer_centered.csv",
    "layer_summary.csv",
    "ranking_element_delta_mean_most_negative.csv",
    "ranking_element_delta_mean_most_positive.csv",
    "ranking_element_std_shift_largest.csv",
    "ranking_element_std_shift_smallest.csv",
    "ranking_aie_highest.csv",
    "ranking_abs_aie_highest.csv",
    "correlation_spearman_full.csv",
    "correlation_pearson_full.csv",
    "correlation_spearman_layer_centered.csv",
    "correlation_pearson_layer_centered.csv",
    "metadata.json",
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate element, attention, and AIE metrics for every layer-head."
    )
    parser.add_argument("--per_head_summary", required=True)
    parser.add_argument("--original_metrics", required=True)
    parser.add_argument("--jailbreak_metrics", required=True)
    parser.add_argument(
        "--aie",
        default=None,
        help="Optional AIE tensor. Omit for metric-only datasets such as GCG.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Target files already exist in {output_dir}; use --overwrite: {existing}"
        )
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".write_test_"):
            pass
    except OSError as exc:
        raise OSError(f"Output directory is not writable: {output_dir}") from exc


def read_csv(path: str, label: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read {label}: path={path}, error_type={type(exc).__name__}"
        ) from exc
    if frame.empty:
        raise ValueError(f"{label} contains no rows: {path}")
    return frame


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def validate_finite(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(np.float64)
        except Exception as exc:
            raise ValueError(f"{label} column is not numeric: {column}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains NaN/inf: column={column}")


def validate_unique(frame: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    duplicates = int(frame.duplicated(list(keys)).sum())
    if duplicates:
        raise ValueError(f"{label} has duplicate keys {list(keys)}: count={duplicates}")


def validate_head_grid(
    frame: pd.DataFrame,
    expected_layers: int | None = None,
    expected_heads_per_layer: int | None = None,
) -> Tuple[List[int], List[int]]:
    validate_unique(frame, HEAD_KEYS, "head grid")
    layers = sorted(int(value) for value in frame["layer"].unique())
    heads = sorted(int(value) for value in frame["head"].unique())
    inferred_layers = len(layers) if expected_layers is None else expected_layers
    inferred_heads = (
        len(heads) if expected_heads_per_layer is None else expected_heads_per_layer
    )
    if layers != list(range(inferred_layers)):
        raise ValueError(f"Expected layers 0..{inferred_layers - 1}; received={layers}")
    if heads != list(range(inferred_heads)):
        raise ValueError(
            f"Expected heads 0..{inferred_heads - 1}; received={heads}"
        )
    counts = frame.groupby("layer")["head"].nunique()
    if not counts.eq(inferred_heads).all():
        raise ValueError(f"Each layer must have {inferred_heads} heads")
    if len(frame) != inferred_layers * inferred_heads:
        raise ValueError(
            "Head grid is incomplete: "
            f"rows={len(frame)}, expected={inferred_layers * inferred_heads}"
        )
    return layers, heads


def build_element_metrics(per_head: pd.DataFrame) -> pd.DataFrame:
    require_columns(per_head, PER_HEAD_REQUIRED, "per_head_summary")
    validate_unique(per_head, HEAD_KEYS, "per_head_summary")
    numeric = sorted(PER_HEAD_REQUIRED - {"layer", "head"})
    validate_finite(per_head, ["layer", "head", *numeric], "per_head_summary")
    result = per_head.rename(
        columns={
            "value_count": "element_value_count",
            "original_mean": "element_original_mean",
            "jailbreak_mean": "element_jailbreak_mean",
            "delta_mean": "element_delta_mean",
            "original_std": "element_original_std",
            "jailbreak_std": "element_jailbreak_std",
            "delta_std": "element_delta_std",
            "original_positive_rate": "element_original_positive_rate",
            "jailbreak_positive_rate": "element_jailbreak_positive_rate",
            "delta_positive_rate": "element_delta_positive_rate",
        }
    ).copy()
    result["element_std_shift"] = (
        result["element_jailbreak_std"] - result["element_original_std"]
    )
    result["element_positive_rate_shift"] = (
        result["element_jailbreak_positive_rate"]
        - result["element_original_positive_rate"]
    )
    return result


def validate_metric_input(frame: pd.DataFrame, label: str) -> None:
    require_columns(frame, METRICS_REQUIRED, label)
    validate_unique(frame, JOIN_KEYS, label)
    validate_finite(
        frame,
        ["prompt_id", "layer", "head", "window_size", *ATTENTION_METRICS],
        label,
    )
    if frame["window_size"].nunique() != 1:
        raise ValueError(f"{label} must contain one window_size")


def aggregate_attention_metrics(
    original: pd.DataFrame,
    jailbreak: pd.DataFrame,
) -> Tuple[pd.DataFrame, int, int]:
    validate_metric_input(original, "original_metrics")
    validate_metric_input(jailbreak, "jailbreak_metrics")
    original_keys = set(original[JOIN_KEYS].itertuples(index=False, name=None))
    jailbreak_keys = set(jailbreak[JOIN_KEYS].itertuples(index=False, name=None))
    if original_keys != jailbreak_keys:
        raise ValueError(
            "Original/jailbreak prompt-level keys do not match: "
            f"original_only={len(original_keys - jailbreak_keys)}, "
            f"jailbreak_only={len(jailbreak_keys - original_keys)}"
        )
    original_window = int(original["window_size"].iloc[0])
    jailbreak_window = int(jailbreak["window_size"].iloc[0])
    if original_window != jailbreak_window:
        raise ValueError("Original/jailbreak window_size values differ")

    source_pairs = original[JOIN_KEYS + ["source_index"]].merge(
        jailbreak[JOIN_KEYS + ["source_index"]],
        on=JOIN_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_original", "_jailbreak"),
    )
    both_present = (
        source_pairs["source_index_original"].notna()
        & source_pairs["source_index_jailbreak"].notna()
    )
    if not source_pairs.loc[both_present, "source_index_original"].astype(str).equals(
        source_pairs.loc[both_present, "source_index_jailbreak"].astype(str)
    ):
        raise ValueError("Original/jailbreak source_index values do not match")

    original_columns = JOIN_KEYS + ATTENTION_METRICS
    jailbreak_columns = JOIN_KEYS + ATTENTION_METRICS
    merged = original[original_columns].merge(
        jailbreak[jailbreak_columns],
        on=JOIN_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_original", "_jailbreak"),
    )
    for metric in ATTENTION_METRICS:
        merged[f"delta_{metric}"] = (
            merged[f"{metric}_jailbreak"] - merged[f"{metric}_original"]
        )

    prompt_count = int(merged["prompt_id"].nunique())
    per_head_prompt_counts = merged.groupby(HEAD_KEYS).size()
    if prompt_count == 0 or not per_head_prompt_counts.eq(prompt_count).all():
        mismatched = per_head_prompt_counts[per_head_prompt_counts != prompt_count]
        example = mismatched.index[0] if not mismatched.empty else None
        raise ValueError(
            "Each layer-head must contain every prompt: "
            f"expected_prompt_count={prompt_count}, example_layer_head={example}"
        )

    rows: List[Dict[str, Any]] = []
    for (layer, head), group in merged.groupby(HEAD_KEYS, sort=True):
        row: Dict[str, Any] = {"layer": int(layer), "head": int(head)}
        for metric in ATTENTION_METRICS:
            original_mean = float(group[f"{metric}_original"].mean())
            jailbreak_mean = float(group[f"{metric}_jailbreak"].mean())
            delta_mean = float(group[f"delta_{metric}"].mean())
            if not np.isclose(
                delta_mean,
                jailbreak_mean - original_mean,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    "Prompt-delta aggregation mismatch: "
                    f"layer={layer}, head={head}, metric={metric}"
                )
            row[f"original_{metric}"] = original_mean
            row[f"jailbreak_{metric}"] = jailbreak_mean
            row[f"delta_{metric}"] = delta_mean
        rows.append(row)
    return pd.DataFrame(rows), prompt_count, original_window


def load_aie(path: str) -> np.ndarray:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Failed to load AIE: path={path}, error_type={type(exc).__name__}"
        ) from exc
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(f"AIE must be a 2D tensor: path={path}")
    values = tensor.detach().float().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"AIE contains NaN/inf: path={path}")
    return values


def attach_aie_and_flags(
    frame: pd.DataFrame,
    aie: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    result = frame.copy().sort_values(HEAD_KEYS).reset_index(drop=True)
    layers, heads = validate_head_grid(result)
    expected_shape = (len(layers), len(heads))
    if aie.shape != expected_shape:
        raise ValueError(
            f"Expected AIE shape {expected_shape}; received={aie.shape}"
        )
    result["aie"] = [
        aie[int(layer), int(head)]
        for layer, head in result[HEAD_KEYS].itertuples(index=False, name=None)
    ]
    result["abs_aie"] = result["aie"].abs()
    result["aie_rank"] = result["aie"].rank(
        ascending=False, method="first"
    ).astype(int)
    result["abs_aie_rank"] = result["abs_aie"].rank(
        ascending=False, method="first"
    ).astype(int)
    result["aie_positive"] = result["aie"] > 0
    result["aie_negative"] = result["aie"] < 0
    result["aie_top_k"] = result["aie_rank"] <= top_k
    result["abs_aie_top_k"] = result["abs_aie_rank"] <= top_k
    result["paper_head"] = [
        (int(layer), int(head)) in PAPER_HEADS
        for layer, head in result[HEAD_KEYS].itertuples(index=False, name=None)
    ]
    result["head_label"] = [
        f"L{int(layer)}H{int(head)}"
        for layer, head in result[HEAD_KEYS].itertuples(index=False, name=None)
    ]
    return result


def attach_head_labels_and_paper_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().sort_values(HEAD_KEYS).reset_index(drop=True)
    result["paper_head"] = [
        (int(layer), int(head)) in PAPER_HEADS
        for layer, head in result[HEAD_KEYS].itertuples(index=False, name=None)
    ]
    result["head_label"] = [
        f"L{int(layer)}H{int(head)}"
        for layer, head in result[HEAD_KEYS].itertuples(index=False, name=None)
    ]
    return result


def add_absolute_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source, target in (
        ("element_delta_mean", "abs_element_delta_mean"),
        ("element_std_shift", "abs_element_std_shift"),
        ("element_positive_rate_shift", "abs_element_positive_rate_shift"),
        ("delta_sym_ratio", "abs_delta_sym_ratio"),
        ("delta_asym_ratio", "abs_delta_asym_ratio"),
        ("delta_asym_lower_mean", "abs_delta_asym_lower_mean"),
    ):
        result[target] = result[source].abs()
    return result


def validate_aggregation_consistency(frame: pd.DataFrame) -> None:
    comparisons = [
        ("element_delta_mean", "delta_asym_lower_mean"),
        ("element_positive_rate_shift", "delta_asym_lower_pos_rate"),
        ("element_original_mean", "original_asym_lower_mean"),
        ("element_jailbreak_mean", "jailbreak_asym_lower_mean"),
        ("element_original_positive_rate", "original_asym_lower_pos_rate"),
        ("element_jailbreak_positive_rate", "jailbreak_asym_lower_pos_rate"),
    ]
    for element_column, attention_column in comparisons:
        if not np.allclose(
            frame[element_column],
            frame[attention_column],
            rtol=1e-5,
            atol=1e-6,
        ):
            difference = np.abs(frame[element_column] - frame[attention_column])
            index = int(difference.idxmax())
            row = frame.loc[index]
            raise ValueError(
                "Aggregation consistency failed: "
                f"element_column={element_column}, attention_column={attention_column}, "
                f"layer={int(row['layer'])}, head={int(row['head'])}, "
                f"max_abs_difference={float(difference.loc[index])}"
            )


def build_merged_head_metrics(
    per_head: pd.DataFrame,
    original: pd.DataFrame,
    jailbreak: pd.DataFrame,
    aie: np.ndarray | None = None,
    top_k: int = 20,
) -> Tuple[pd.DataFrame, int, int]:
    element_metrics = build_element_metrics(per_head)
    attention_metrics, prompt_count, window_size = aggregate_attention_metrics(
        original, jailbreak
    )
    merged = element_metrics.merge(
        attention_metrics,
        on=HEAD_KEYS,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(element_metrics) or len(merged) != len(attention_metrics):
        raise ValueError("Element and attention head sets do not match")
    validate_head_grid(merged)
    if aie is None:
        merged = attach_head_labels_and_paper_flags(merged)
    else:
        merged = attach_aie_and_flags(merged, aie, top_k)
    merged = add_absolute_columns(merged)
    validate_aggregation_consistency(merged)
    return merged, prompt_count, window_size


def build_layer_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for layer, group in frame.groupby("layer", sort=True):
        row = {
                "layer": int(layer),
                "head_count": int(len(group)),
                "element_delta_mean_mean": float(group["element_delta_mean"].mean()),
                "element_delta_mean_median": float(group["element_delta_mean"].median()),
                "element_delta_mean_negative_rate": float((group["element_delta_mean"] < 0).mean()),
                "element_delta_mean_positive_rate": float((group["element_delta_mean"] > 0).mean()),
                "element_std_shift_mean": float(group["element_std_shift"].mean()),
                "element_std_shift_median": float(group["element_std_shift"].median()),
                "element_std_shift_positive_rate": float((group["element_std_shift"] > 0).mean()),
                "element_positive_rate_shift_mean": float(
                    group["element_positive_rate_shift"].mean()
                ),
                "element_positive_rate_shift_median": float(
                    group["element_positive_rate_shift"].median()
                ),
                "delta_sym_ratio_mean": float(group["delta_sym_ratio"].mean()),
                "delta_sym_ratio_median": float(group["delta_sym_ratio"].median()),
                "delta_asym_ratio_mean": float(group["delta_asym_ratio"].mean()),
                "delta_asym_ratio_median": float(group["delta_asym_ratio"].median()),
                "delta_asym_lower_mean_mean": float(group["delta_asym_lower_mean"].mean()),
            }
        if "aie" in group.columns:
            row.update(
                {
                    "aie_mean": float(group["aie"].mean()),
                    "aie_median": float(group["aie"].median()),
                    "aie_positive_rate": float((group["aie"] > 0).mean()),
                    "abs_aie_mean": float(group["abs_aie"].mean()),
                }
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    expected_heads = int(frame.groupby("layer")["head"].nunique().iloc[0])
    if not result["head_count"].eq(expected_heads).all():
        raise ValueError(
            f"Each layer_summary row must aggregate {expected_heads} heads"
        )
    return result


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.size != y_values.size or x_values.size < 2:
        raise ValueError("Correlation inputs must have equal length >= 2")
    if np.std(x_values) == 0 or np.std(y_values) == 0:
        raise ValueError("Correlation is undefined for a constant metric")
    return float(np.corrcoef(x_values, y_values)[0, 1])


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy(np.float64)
    y_rank = pd.Series(y).rank(method="average").to_numpy(np.float64)
    return pearson_correlation(x_rank, y_rank)


def correlation_long(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    method: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    function = pearson_correlation if method == "pearson" else spearman_correlation
    for metric_x, metric_y in combinations_with_replacement(metrics, 2):
        x = frame[metric_x].to_numpy(np.float64)
        y = frame[metric_y].to_numpy(np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        rows.append(
            {
                "metric_x": metric_x,
                "metric_y": metric_y,
                "correlation": function(x[finite], y[finite]),
                "n": int(finite.sum()),
            }
        )
    return pd.DataFrame(rows)


def nonconstant_metrics(
    frame: pd.DataFrame,
    metrics: Sequence[str],
) -> Tuple[List[str], List[str]]:
    included: List[str] = []
    excluded: List[str] = []
    for metric in metrics:
        values = frame[metric].to_numpy(np.float64)
        if np.std(values) == 0:
            excluded.append(metric)
        else:
            included.append(metric)
    if not included:
        raise ValueError("No non-constant metrics are available for correlation")
    return included, excluded


def layer_center(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    result = frame[["layer", "head", "head_label"]].copy()
    for metric in metrics:
        result[metric] = frame[metric] - frame.groupby("layer")[metric].transform("mean")
        means = result.groupby("layer")[metric].mean().to_numpy(np.float64)
        if not np.allclose(means, 0, atol=1e-7):
            raise ValueError(f"Layer-centering failed for metric={metric}")
    return result


def write_rankings(frame: pd.DataFrame, output_dir: Path, top_k: int) -> None:
    specifications = [
        ("ranking_element_delta_mean_most_negative.csv", "element_delta_mean", True),
        ("ranking_element_delta_mean_most_positive.csv", "element_delta_mean", False),
        ("ranking_element_std_shift_largest.csv", "element_std_shift", False),
        ("ranking_element_std_shift_smallest.csv", "element_std_shift", True),
        ("ranking_aie_highest.csv", "aie", False),
        ("ranking_abs_aie_highest.csv", "abs_aie", False),
    ]
    base_columns = [
        "layer",
        "head",
        "head_label",
        "paper_head",
        "aie",
        "abs_aie",
        "element_delta_mean",
        "element_std_shift",
        "delta_sym_ratio",
        "delta_asym_ratio",
    ]
    for filename, metric, ascending in specifications:
        if metric not in frame.columns:
            continue
        ranking = frame.sort_values(
            [metric, "layer", "head"], ascending=[ascending, True, True]
        ).head(top_k).copy()
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
        columns = [
            "rank",
            *[column for column in base_columns if column in frame.columns],
        ]
        if metric not in columns:
            columns.insert(5, metric)
        ranking[columns].to_csv(output_dir / filename, index=False)


def main() -> None:
    args = get_args()
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)

    per_head = read_csv(args.per_head_summary, "per_head_summary")
    original = read_csv(args.original_metrics, "original_metrics")
    jailbreak = read_csv(args.jailbreak_metrics, "jailbreak_metrics")
    aie = load_aie(args.aie) if args.aie else None
    merged, prompt_count, window_size = build_merged_head_metrics(
        per_head, original, jailbreak, aie=aie, top_k=args.top_k
    )
    layers, heads = validate_head_grid(merged)

    required_finite = [
        column
        for column in merged.columns
        if column
        not in {
            "head_label",
            "paper_head",
            "aie_positive",
            "aie_negative",
            "aie_top_k",
            "abs_aie_top_k",
        }
    ]
    validate_finite(merged, required_finite, "head_metrics_merged")
    merged.to_csv(output_dir / "head_metrics_merged.csv", index=False)

    layer_summary = build_layer_summary(merged)
    validate_finite(
        layer_summary,
        [column for column in layer_summary.columns],
        "layer_summary",
    )
    layer_summary.to_csv(output_dir / "layer_summary.csv", index=False)
    write_rankings(merged, output_dir, min(args.top_k, len(merged)))

    available_correlation_metrics = [
        metric for metric in CORRELATION_METRICS if metric in merged.columns
    ]
    correlation_metrics, constant_full_metrics = nonconstant_metrics(
        merged, available_correlation_metrics
    )
    correlation_long(merged, correlation_metrics, "spearman").to_csv(
        output_dir / "correlation_spearman_full.csv", index=False
    )
    correlation_long(merged, correlation_metrics, "pearson").to_csv(
        output_dir / "correlation_pearson_full.csv", index=False
    )
    centered = layer_center(merged, available_correlation_metrics)
    centered.to_csv(output_dir / "head_metrics_layer_centered.csv", index=False)
    centered_correlation_metrics, constant_centered_metrics = nonconstant_metrics(
        centered, available_correlation_metrics
    )
    correlation_long(centered, centered_correlation_metrics, "spearman").to_csv(
        output_dir / "correlation_spearman_layer_centered.csv", index=False
    )
    correlation_long(centered, centered_correlation_metrics, "pearson").to_csv(
        output_dir / "correlation_pearson_layer_centered.csv", index=False
    )

    metadata = {
        "inputs": {
            "per_head_summary": args.per_head_summary,
            "original_metrics": args.original_metrics,
            "jailbreak_metrics": args.jailbreak_metrics,
            "aie": args.aie,
        },
        "output_dir": args.output_dir,
        "head_count": int(len(merged)),
        "layer_count": int(merged["layer"].nunique()),
        "heads_per_layer": len(heads),
        "prompt_count": prompt_count,
        "window_size": window_size,
        "aie_shape": list(aie.shape) if aie is not None else None,
        "aie_available": aie is not None,
        "paper_heads": [list(head) for head in sorted(PAPER_HEADS)],
        "top_k": args.top_k,
        "rank_tie_method": "first after stable layer/head ordering",
        "derived_metric_definitions": {
            "element_std_shift": "element_jailbreak_std - element_original_std",
            "element_delta_std": "std of paired element deltas; not a std shift",
            "element_positive_rate_shift": (
                "element_jailbreak_positive_rate - element_original_positive_rate"
            ),
            "delta_attention_metrics": "mean over prompts of jailbreak - original",
            "abs_columns": "absolute value of the corresponding signed metric",
        },
        "join_keys": JOIN_KEYS,
        "source_index_join_policy": (
            "source_index is not a join key; when present in both conditions, "
            "paired values must match"
        ),
        "correlation_statistical_unit": f"one layer-head (n={len(merged)})",
        "correlation_p_values": "not computed; heads are not assumed independent",
        "constant_metrics_excluded_from_full_correlation": constant_full_metrics,
        "constant_metrics_excluded_from_layer_centered_correlation": (
            constant_centered_metrics
        ),
        "definitional_consistency_pairs": [
            ["element_delta_mean", "delta_asym_lower_mean"],
            ["element_positive_rate_shift", "delta_asym_lower_pos_rate"],
        ],
        "layer_centering": (
            "value - mean(value within the same layer), then correlate 1024 "
            "centered values"
        ),
        "execution_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "cli_arguments": vars(args),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"merged head rows: {len(merged)}")
    print(f"layer rows: {len(layer_summary)}")
    print(f"prompt count: {prompt_count}")
    print(f"window_size: {window_size}")
    print(f"paper heads: {int(merged['paper_head'].sum())}")
    if aie is not None:
        rank_one = merged.loc[merged["aie_rank"] == 1].iloc[0]
        print(f"AIE rank 1: {rank_one['head_label']}")
    else:
        print("AIE: not provided")
    print("mean consistency: passed")
    print("positive-rate consistency: passed")
    print(f"output directory: {output_dir}")


if __name__ == "__main__":
    main()
