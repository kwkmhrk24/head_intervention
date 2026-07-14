import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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

from scripts.plot_attention_score_all_heads_distributions import (
    RAW_ARRAY_FILENAMES,
    common_histogram_bin_edges,
    density_from_counts,
    histogram_counts,
    load_array,
    load_metadata,
    symmetric_delta_bin_edges,
    validate_finite_chunked,
    write_comparison_bins,
    write_delta_bins,
)


QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
QUANTILE_NAMES = ("q01", "q05", "q25", "q50", "q75", "q95", "q99")
SELECTION_AUXILIARY_COLUMNS = [
    "element_delta_mean",
    "element_std_shift",
    "element_positive_rate_shift",
    "delta_sym_ratio",
    "aie",
    "aie_rank",
    "paper_head",
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select signed metric top/bottom attention heads and plot their "
            "S_asym strict-lower distributions from existing all-head raw arrays."
        )
    )
    parser.add_argument(
        "--head_metrics_csv",
        "--head_metrics",
        dest="head_metrics_csv",
        required=True,
        help="Path to head_metrics_merged.csv.",
    )
    parser.add_argument(
        "--raw_output_dir",
        required=True,
        help="Directory containing raw NPY arrays and metadata.json.",
    )
    parser.add_argument(
        "--metric",
        default="delta_asym_ratio",
        help="Signed head-selection metric (default: delta_asym_ratio).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of largest signed metric values to select (default: 3).",
    )
    parser.add_argument(
        "--bottom_k",
        type=int,
        default=3,
        help="Number of smallest signed metric values to select (default: 3).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for selected-head summaries and histograms.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of histogram bins (default: 100).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution (default: 200).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing target files to be overwritten.",
    )
    return parser.parse_args()


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".write_test_"):
            pass
    except OSError as exc:
        raise OSError(f"Output directory is not writable: {output_dir}") from exc


def read_head_metrics(path: Path, metric: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read head metrics: path={path}, "
            f"error_type={type(exc).__name__}"
        ) from exc
    if frame.empty:
        raise ValueError(f"Head metrics CSV contains no rows: {path}")
    required = {"layer", "head", "delta_asym_ratio", metric}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Head metrics CSV missing required columns: {missing}")
    duplicate_count = int(frame.duplicated(["layer", "head"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Head metrics contains duplicate layer/head keys: count={duplicate_count}"
        )
    for column in dict.fromkeys(("layer", "head", "delta_asym_ratio", metric)):
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(np.float64)
        except Exception as exc:
            raise ValueError(f"Head metrics column is not numeric: {column}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"Head metrics contains NaN/inf: column={column}")
    if (frame[["layer", "head"]] < 0).any().any():
        raise ValueError("Head metrics contains negative layer/head indices")
    return frame


def select_signed_heads(
    frame: pd.DataFrame,
    metric: str,
    top_k: int,
    bottom_k: int,
) -> pd.DataFrame:
    if top_k <= 0 or bottom_k <= 0:
        raise ValueError("--top_k and --bottom_k must be positive")
    if top_k + bottom_k > len(frame):
        raise ValueError(
            "Not enough heads for disjoint signed groups: "
            f"rows={len(frame)}, top_k={top_k}, bottom_k={bottom_k}"
        )

    parts: List[pd.DataFrame] = []
    specifications = [
        (f"increase_top{top_k}", top_k, False),
        (f"decrease_top{bottom_k}", bottom_k, True),
    ]
    for group, count, ascending in specifications:
        ranked = frame.sort_values(
            [metric, "layer", "head"],
            ascending=[ascending, True, True],
            kind="mergesort",
        ).head(count).copy()
        ranked.insert(0, "rank_within_group", np.arange(1, count + 1))
        ranked.insert(0, "group", group)
        parts.append(ranked)

    selected = pd.concat(parts, ignore_index=True)
    if selected.duplicated(["layer", "head"]).any():
        duplicates = selected.loc[
            selected.duplicated(["layer", "head"], keep=False),
            ["layer", "head"],
        ].drop_duplicates()
        raise ValueError(
            "Increase/decrease selections overlap: "
            f"layer_heads={duplicates.values.tolist()}"
        )
    if not selected.iloc[:top_k][metric].is_monotonic_decreasing:
        raise ValueError("Increase selection is not sorted in descending order")
    if not selected.iloc[top_k:][metric].is_monotonic_increasing:
        raise ValueError("Decrease selection is not sorted in ascending order")

    selected["head_label"] = [
        f"L{int(layer)}H{int(head)}"
        for layer, head in selected[["layer", "head"]].itertuples(
            index=False,
            name=None,
        )
    ]
    return selected


def validate_raw_metadata(metadata: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    required = [
        "processed_prompt_count",
        "window_size",
        "elements_per_head",
        "num_layers",
        "num_attention_heads",
        "array_shape",
        "array_axes",
        "delta_definition",
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Raw metadata missing required keys: {missing}")

    prompt_count = int(metadata["processed_prompt_count"])
    window_size = int(metadata["window_size"])
    elements_per_head = int(metadata["elements_per_head"])
    num_layers = int(metadata["num_layers"])
    num_heads = int(metadata["num_attention_heads"])
    expected_elements = window_size * (window_size - 1) // 2
    expected_shape = (prompt_count, num_layers, num_heads, elements_per_head)
    expected_axes = ["prompt", "layer", "head", "strict_lower_position"]

    if prompt_count <= 0 or window_size <= 1 or num_layers <= 0 or num_heads <= 0:
        raise ValueError("Raw metadata contains non-positive dimensions")
    if elements_per_head != expected_elements:
        raise ValueError(
            "elements_per_head does not match the strict lower triangle: "
            f"actual={elements_per_head}, expected={expected_elements}"
        )
    if tuple(metadata["array_shape"]) != expected_shape:
        raise ValueError(
            f"Raw metadata shape mismatch: actual={metadata['array_shape']}, "
            f"expected={expected_shape}"
        )
    if metadata["array_axes"] != expected_axes:
        raise ValueError(
            f"Raw array axes mismatch: actual={metadata['array_axes']}, "
            f"expected={expected_axes}"
        )
    expected_delta_definition = (
        "jailbreak - original at paired prompt/layer/head/position"
    )
    if metadata["delta_definition"] != expected_delta_definition:
        raise ValueError(
            f"Unexpected delta definition: {metadata['delta_definition']}"
        )
    return prompt_count, elements_per_head, num_layers, num_heads


def load_and_validate_raw_arrays(
    raw_output_dir: Path,
    metadata: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    arrays = {
        condition: load_array(raw_output_dir / filename)
        for condition, filename in RAW_ARRAY_FILENAMES.items()
    }
    shapes = {condition: array.shape for condition, array in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Raw original/jailbreak/delta shapes differ: {shapes}")
    expected_shape = tuple(metadata["array_shape"])
    if next(iter(shapes.values())) != expected_shape:
        raise ValueError(
            f"Raw arrays do not match metadata: arrays={shapes}, "
            f"metadata={expected_shape}"
        )
    for condition, array in arrays.items():
        validate_finite_chunked(condition, array)
    return arrays


def extract_head_arrays(
    arrays: Mapping[str, np.ndarray],
    layer: int,
    head: int,
    expected_shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    extracted = {
        condition: np.array(array[:, layer, head, :], dtype=np.float32, copy=True)
        for condition, array in arrays.items()
    }
    shapes = {condition: values.shape for condition, values in extracted.items()}
    if any(shape != expected_shape for shape in shapes.values()):
        raise ValueError(
            f"Extracted head shape mismatch: layer={layer}, head={head}, "
            f"actual={shapes}, expected={expected_shape}"
        )
    for condition, values in extracted.items():
        if not np.isfinite(values).all():
            raise ValueError(
                f"Extracted values contain NaN/inf: condition={condition}, "
                f"layer={layer}, head={head}"
            )
    return extracted


def validate_delta(
    arrays: Mapping[str, np.ndarray],
    layer: int,
    head: int,
) -> float:
    difference = arrays["jailbreak"] - arrays["original"]
    absolute_error = np.abs(arrays["delta"] - difference)
    max_error = float(np.max(absolute_error))
    if not np.allclose(
        arrays["delta"],
        difference,
        rtol=1e-5,
        atol=2e-6,
    ):
        raise ValueError(
            f"Raw delta mismatch: layer={layer}, head={head}, "
            f"max_abs_error={max_error}"
        )
    return max_error


def distribution_statistics(values: np.ndarray, prefix: str) -> Dict[str, float]:
    flattened = values.reshape(-1)
    result = {
        f"{prefix}_mean": float(np.mean(flattened, dtype=np.float64)),
        f"{prefix}_median": float(np.median(flattened)),
        f"{prefix}_std": float(np.std(flattened, dtype=np.float64)),
        f"{prefix}_positive_rate": float(np.mean(flattened > 0)),
    }
    quantile_values = np.quantile(flattened, QUANTILES)
    result.update(
        {
            f"{prefix}_{name}": float(value)
            for name, value in zip(QUANTILE_NAMES, quantile_values)
        }
    )
    return result


def build_summary_row(
    selected_row: Any,
    arrays: Mapping[str, np.ndarray],
    metric: str,
    prompt_count: int,
    values_per_prompt: int,
    delta_max_abs_error: float,
    output_directory: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "group": selected_row.group,
        "rank": int(selected_row.rank_within_group),
        "layer": int(selected_row.layer),
        "head": int(selected_row.head),
        "head_label": selected_row.head_label,
        "selection_metric": metric,
        "selection_metric_value": float(getattr(selected_row, metric)),
        "delta_asym_ratio": float(selected_row.delta_asym_ratio),
        "prompt_count": prompt_count,
        "values_per_prompt": values_per_prompt,
        "total_values": int(arrays["original"].size),
        "output_directory": output_directory,
        "delta_max_abs_error": delta_max_abs_error,
    }
    for condition in ("original", "jailbreak", "delta"):
        row.update(distribution_statistics(arrays[condition], condition))
    row["std_shift"] = row["jailbreak_std"] - row["original_std"]
    row["mean_delta_identity_error"] = abs(
        (row["jailbreak_mean"] - row["original_mean"]) - row["delta_mean"]
    )
    if row["mean_delta_identity_error"] > 2e-6:
        raise ValueError(
            "Mean delta identity failed: "
            f"layer={row['layer']}, head={row['head']}, "
            f"error={row['mean_delta_identity_error']}"
        )
    if hasattr(selected_row, "paper_head"):
        row["paper_head"] = bool(selected_row.paper_head)
    return row


def validate_raw_means(selected_row: Any, summary: Mapping[str, Any]) -> None:
    comparisons = [
        ("element_original_mean", "original_mean"),
        ("element_jailbreak_mean", "jailbreak_mean"),
        ("element_delta_mean", "delta_mean"),
    ]
    for metric_column, summary_column in comparisons:
        if not hasattr(selected_row, metric_column):
            continue
        expected = float(getattr(selected_row, metric_column))
        actual = float(summary[summary_column])
        if not np.isclose(actual, expected, rtol=1e-5, atol=1e-6):
            raise ValueError(
                "Raw mean does not match head metrics: "
                f"layer={int(selected_row.layer)}, head={int(selected_row.head)}, "
                f"raw_column={summary_column}, head_metric_column={metric_column}, "
                f"raw_value={actual}, head_metric_value={expected}"
            )


def infer_plot_labels(metadata: Mapping[str, Any]) -> Tuple[str, str]:
    model_path = str(metadata.get("model", metadata.get("model_name", "Model")))
    model_label = Path(model_path.rstrip("/")).name or model_path
    if model_label.lower() == "llama-2-7b-chat-hf":
        model_label = "Llama-2-7B-chat"
    jailbreak_prompts = str(metadata.get("jailbreak_prompts", ""))
    attack_name = Path(jailbreak_prompts).parent.name
    attack_label = "AutoDAN" if attack_name.lower() == "autodan" else attack_name
    return model_label, attack_label or "Attack"


def save_histograms(
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    bins: int,
    dpi: int,
    model_label: str,
    attack_label: str,
    selected_row: Any,
    metric: str,
    window_size: int,
    prompt_count: int,
) -> List[Path]:
    comparison_edges = common_histogram_bin_edges(
        arrays["original"],
        arrays["jailbreak"],
        bins,
    )
    original_counts = histogram_counts(arrays["original"], comparison_edges)
    jailbreak_counts = histogram_counts(arrays["jailbreak"], comparison_edges)
    original_density = density_from_counts(original_counts, comparison_edges)
    jailbreak_density = density_from_counts(jailbreak_counts, comparison_edges)
    delta_edges = symmetric_delta_bin_edges(arrays["delta"], bins)
    delta_counts = histogram_counts(arrays["delta"], delta_edges)
    delta_density = density_from_counts(delta_counts, delta_edges)

    comparison_path = output_dir / "histogram_original_vs_jailbreak.png"
    delta_path = output_dir / "histogram_delta.png"
    selection_value = float(getattr(selected_row, metric))
    group_label = "Increase" if selected_row.group.startswith("increase") else "Decrease"
    rank = int(selected_row.rank_within_group)
    context = (
        f"{model_label} / {attack_label} | {group_label} Top {rank} "
        f"{selected_row.head_label} | {metric}={selection_value:+.6f}"
    )
    experiment = (
        f"S_asym strict lower | tail{window_size} | full {prompt_count} prompts"
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        ax.stairs(
            original_density,
            comparison_edges,
            fill=True,
            alpha=0.45,
            linewidth=1.3,
            label="Original",
        )
        ax.stairs(
            jailbreak_density,
            comparison_edges,
            fill=True,
            alpha=0.45,
            linewidth=1.3,
            label="Jailbreak",
        )
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("S_asym strict-lower value")
        ax.set_ylabel("Density")
        ax.set_title(f"{context}\nOriginal vs Jailbreak | {experiment}")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(comparison_path, dpi=dpi)
    finally:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        ax.stairs(
            delta_density,
            delta_edges,
            fill=True,
            alpha=0.65,
            color="#7A5195",
            linewidth=1.3,
        )
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Jailbreak - Original")
        ax.set_ylabel("Density")
        ax.set_title(f"{context}\nElement Delta | {experiment}")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(delta_path, dpi=dpi)
    finally:
        plt.close(fig)

    write_comparison_bins(
        output_dir / "histogram_original_vs_jailbreak_bins.csv",
        comparison_edges,
        original_counts,
        jailbreak_counts,
    )
    write_delta_bins(
        output_dir / "histogram_delta_bins.csv",
        delta_edges,
        delta_counts,
    )
    return [comparison_path, delta_path]


def output_subdirectory_name(row: Any) -> str:
    direction = "increase" if row.group.startswith("increase") else "decrease"
    return (
        f"{direction}_top{int(row.rank_within_group)}_"
        f"layer{int(row.layer)}_head{int(row.head)}"
    )


def output_targets(output_dir: Path, selected: pd.DataFrame) -> List[Path]:
    targets = [
        output_dir / "selected_heads_delta_asym_ratio_signed.csv",
        output_dir / "selection_summary.csv",
    ]
    for row in selected.itertuples(index=False):
        head_dir = output_dir / output_subdirectory_name(row)
        targets.extend(
            [
                head_dir / "histogram_original_vs_jailbreak.png",
                head_dir / "histogram_delta.png",
                head_dir / "histogram_original_vs_jailbreak_bins.csv",
                head_dir / "histogram_delta_bins.csv",
                head_dir / "summary.csv",
            ]
        )
    return targets


def reject_existing_targets(targets: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Target files already exist; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def write_dict_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    columns = list(rows[0].keys())
    if any(list(row.keys()) != columns for row in rows):
        raise ValueError(f"CSV rows have inconsistent schemas: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def selected_output_rows(
    selected: pd.DataFrame,
    metric: str,
) -> List[Dict[str, Any]]:
    columns = [
        "group",
        "rank_within_group",
        "layer",
        "head",
        "head_label",
        metric,
        *[
            column
            for column in SELECTION_AUXILIARY_COLUMNS
            if column in selected.columns and column != metric
        ],
    ]
    rows: List[Dict[str, Any]] = []
    for row in selected[columns].to_dict(orient="records"):
        rows.append(row)
    return rows


def main() -> None:
    args = get_args()
    if args.bins <= 0 or args.dpi <= 0:
        raise ValueError("--bins and --dpi must be positive")
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir)

    metrics = read_head_metrics(Path(args.head_metrics_csv), args.metric)
    selected = select_signed_heads(metrics, args.metric, args.top_k, args.bottom_k)
    targets = output_targets(output_dir, selected)
    reject_existing_targets(targets, args.overwrite)
    metadata = load_metadata(Path(args.raw_output_dir) / "metadata.json")
    prompt_count, elements_per_head, num_layers, num_heads = validate_raw_metadata(
        metadata
    )
    arrays = load_and_validate_raw_arrays(Path(args.raw_output_dir), metadata)
    if selected["layer"].max() >= num_layers or selected["head"].max() >= num_heads:
        raise ValueError(
            "Selected layer/head is outside raw array bounds: "
            f"num_layers={num_layers}, num_heads={num_heads}"
        )

    model_label, attack_label = infer_plot_labels(metadata)
    expected_shape = (prompt_count, elements_per_head)
    summary_rows: List[Dict[str, Any]] = []
    png_paths: List[Path] = []
    maximum_delta_error = 0.0

    for selected_row in selected.itertuples(index=False):
        layer = int(selected_row.layer)
        head = int(selected_row.head)
        head_arrays = extract_head_arrays(
            arrays,
            layer,
            head,
            expected_shape,
        )
        delta_error = validate_delta(head_arrays, layer, head)
        maximum_delta_error = max(maximum_delta_error, delta_error)
        subdirectory_name = output_subdirectory_name(selected_row)
        head_output_dir = output_dir / subdirectory_name
        head_output_dir.mkdir(parents=True, exist_ok=True)
        summary = build_summary_row(
            selected_row,
            head_arrays,
            args.metric,
            prompt_count,
            elements_per_head,
            delta_error,
            subdirectory_name,
        )
        validate_raw_means(selected_row, summary)
        summary_rows.append(summary)
        write_dict_rows(head_output_dir / "summary.csv", [summary])
        png_paths.extend(
            save_histograms(
                head_output_dir,
                head_arrays,
                args.bins,
                args.dpi,
                model_label,
                attack_label,
                selected_row,
                args.metric,
                int(metadata["window_size"]),
                prompt_count,
            )
        )

    write_dict_rows(
        output_dir / "selected_heads_delta_asym_ratio_signed.csv",
        selected_output_rows(selected, args.metric),
    )
    write_dict_rows(output_dir / "selection_summary.csv", summary_rows)

    expected_total = prompt_count * elements_per_head
    if any(int(row["total_values"]) != expected_total for row in summary_rows):
        raise ValueError("Unexpected flattened value count in head summary")
    if len(png_paths) != (args.top_k + args.bottom_k) * 2:
        raise ValueError(f"Unexpected PNG count: actual={len(png_paths)}")
    for path in png_paths:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"PNG is missing or empty: {path}")

    print(f"head metrics rows: {len(metrics)}")
    print(f"selection metric: {args.metric}")
    for row in selected.itertuples(index=False):
        print(
            f"{row.group} rank={int(row.rank_within_group)}: "
            f"{row.head_label}, {args.metric}={float(getattr(row, args.metric)):+.6f}"
        )
    print(f"raw array shape: {arrays['original'].shape}")
    print(f"extracted shape per head: {expected_shape}")
    print(f"values per condition/head: {expected_total}")
    print(f"delta max absolute error: {maximum_delta_error:.9g}")
    print("mean(jailbreak) - mean(original) consistency: passed")
    print(f"non-empty PNG files: {len(png_paths)}")
    print(f"output directory: {output_dir}")


if __name__ == "__main__":
    main()
