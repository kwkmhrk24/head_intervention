import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = f"/tmp/matplotlib-{os.getuid()}"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_attention_head_metric_relationships import (
    pearson_correlation,
    spearman_correlation,
    validate_finite,
    validate_head_grid,
)


HEATMAP_METRICS = [
    "element_delta_mean",
    "element_std_shift",
    "element_positive_rate_shift",
    "element_delta_positive_rate",
    "delta_sym_ratio",
    "delta_asym_ratio",
    "delta_asym_lower_mean",
    "aie",
]
HISTOGRAM_METRICS = ["element_delta_mean", "element_std_shift", "aie"]
SCATTER_SPECS = [
    ("element_delta_mean", "aie", "scatter_element_delta_mean_vs_aie.png"),
    ("abs_element_delta_mean", "abs_aie", "scatter_abs_element_delta_mean_vs_abs_aie.png"),
    ("element_std_shift", "aie", "scatter_element_std_shift_vs_aie.png"),
    ("abs_element_std_shift", "abs_aie", "scatter_abs_element_std_shift_vs_abs_aie.png"),
    ("element_positive_rate_shift", "aie", "scatter_element_positive_rate_shift_vs_aie.png"),
    ("element_std_shift", "delta_asym_ratio", "scatter_element_std_shift_vs_delta_asym_ratio.png"),
    ("element_delta_mean", "delta_sym_ratio", "scatter_element_delta_mean_vs_delta_sym_ratio.png"),
]
OUTPUT_FILENAMES = [
    *[f"heatmap_{metric}.png" for metric in HEATMAP_METRICS],
    *[f"histogram_head_{metric}.png" for metric in HISTOGRAM_METRICS],
    "layer_profile_element_delta_mean.png",
    "layer_profile_element_std_shift.png",
    "layer_profile_negative_head_rate.png",
    "layer_profile_aie.png",
    *[filename for _, _, filename in SCATTER_SPECS],
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot layer-head structure and metric relationships from merged CSVs."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_outputs(input_dir: Path, overwrite: bool) -> None:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    existing = [name for name in OUTPUT_FILENAMES if (input_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Plot files already exist in {input_dir}; use --overwrite: {existing}"
        )
    try:
        with tempfile.NamedTemporaryFile(dir=input_dir, prefix=".write_test_"):
            pass
    except OSError as exc:
        raise OSError(f"Input directory is not writable: {input_dir}") from exc


def read_required_csv(path: Path, required: Sequence[str], label: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read {label}: path={path}, error_type={type(exc).__name__}"
        ) from exc
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
    return frame


def heatmap_pivot(frame: pd.DataFrame, metric: str) -> np.ndarray:
    pivot = frame.pivot(index="layer", columns="head", values=metric)
    pivot = pivot.reindex(index=range(32), columns=range(32))
    values = pivot.to_numpy(np.float64)
    if values.shape != (32, 32) or not np.isfinite(values).all():
        raise ValueError(f"Invalid heatmap grid for metric={metric}: shape={values.shape}")
    return values


def save_heatmap(
    frame: pd.DataFrame,
    metric: str,
    path: Path,
    dpi: int,
) -> None:
    values = heatmap_pivot(frame, metric)
    sequential = metric == "element_delta_positive_rate"
    if sequential:
        vmin, vmax, cmap = 0.0, 1.0, "viridis"
    else:
        max_abs = float(np.max(np.abs(values)))
        if max_abs == 0:
            max_abs = 1.0
        vmin, vmax, cmap = -max_abs, max_abs, "coolwarm"

    fig, ax = plt.subplots(figsize=(10, 9))
    try:
        image = ax.imshow(values, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
        colorbar.set_label(metric)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"Layer × Head: {metric}")
        ax.set_xticks(range(0, 32, 4))
        ax.set_yticks(range(0, 32, 4))

        paper = frame[frame["paper_head"].astype(bool)]
        for row in paper.itertuples(index=False):
            ax.add_patch(
                Rectangle(
                    (row.head - 0.48, row.layer - 0.48),
                    0.96,
                    0.96,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.8,
                )
            )
        aie_top10 = frame[frame["aie_rank"] <= 10]
        ax.scatter(
            aie_top10["head"],
            aie_top10["layer"],
            marker="o",
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.5,
            s=75,
        )
        ax.legend(
            handles=[
                Patch(facecolor="none", edgecolor="red", label="paper head"),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="none",
                    markeredgecolor="cyan",
                    label="AIE top 10",
                ),
            ],
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


def save_head_histogram(frame: pd.DataFrame, metric: str, path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        ax.hist(frame[metric], bins=50, density=False, alpha=0.8)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(metric)
        ax.set_ylabel("Head count")
        ax.set_title(f"Head Distribution: {metric} (n={len(frame)})")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


def save_layer_mean_median_profile(
    layer_summary: pd.DataFrame,
    mean_column: str,
    median_column: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        ax.plot(layer_summary["layer"], layer_summary[mean_column], marker="o", label="mean")
        ax.plot(layer_summary["layer"], layer_summary[median_column], marker="s", label="median")
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(range(0, 32, 2))
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


def save_single_layer_profile(
    layer_summary: pd.DataFrame,
    columns: Sequence[Tuple[str, str]],
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
    zero_line: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        for column, label in columns:
            ax.plot(layer_summary["layer"], layer_summary[column], marker="o", label=label)
        if zero_line:
            ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(range(0, 32, 2))
        if len(columns) > 1:
            ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


def save_scatter(
    frame: pd.DataFrame,
    metric_x: str,
    metric_y: str,
    path: Path,
    dpi: int,
) -> None:
    x = frame[metric_x].to_numpy(np.float64)
    y = frame[metric_y].to_numpy(np.float64)
    spearman = spearman_correlation(x, y)
    pearson = pearson_correlation(x, y)
    paper = frame[frame["paper_head"].astype(bool)]
    top10 = frame[frame["aie_rank"] <= 10]
    focus = frame[(frame["layer"] == 13) & (frame["head"] == 4)]

    fig, ax = plt.subplots(figsize=(8, 7))
    try:
        ax.scatter(x, y, s=22, alpha=0.45, color="gray", label="all heads")
        ax.scatter(
            paper[metric_x],
            paper[metric_y],
            marker="x",
            s=75,
            color="red",
            linewidths=1.8,
            label="paper heads",
        )
        ax.scatter(
            top10[metric_x],
            top10[metric_y],
            marker="o",
            s=70,
            facecolors="none",
            edgecolors="blue",
            linewidths=1.5,
            label="AIE top 10",
        )
        ax.scatter(
            focus[metric_x],
            focus[metric_y],
            marker="*",
            s=180,
            color="gold",
            edgecolors="black",
            label="L13H4",
            zorder=5,
        )
        labels = pd.concat([paper, top10]).drop_duplicates(["layer", "head"])
        for row in labels.itertuples(index=False):
            ax.annotate(
                row.head_label,
                (getattr(row, metric_x), getattr(row, metric_y)),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel(metric_x)
        ax.set_ylabel(metric_y)
        ax.set_title(
            f"{metric_x} vs {metric_y}\n"
            f"Spearman rho={spearman:.3f}, Pearson r={pearson:.3f} (n={len(frame)} heads)"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


def main() -> None:
    args = get_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    input_dir = Path(args.input_dir)
    prepare_outputs(input_dir, args.overwrite)
    merged_required = [
        "layer",
        "head",
        "head_label",
        "paper_head",
        "aie_rank",
        *HEATMAP_METRICS,
        *[metric for pair in SCATTER_SPECS for metric in pair[:2]],
    ]
    merged = read_required_csv(
        input_dir / "head_metrics_merged.csv",
        merged_required,
        "head_metrics_merged",
    )
    validate_head_grid(merged)
    validate_finite(
        merged,
        [column for column in merged_required if column not in {"head_label", "paper_head"}],
        "head_metrics_merged",
    )
    layer_required = [
        "layer",
        "element_delta_mean_mean",
        "element_delta_mean_median",
        "element_delta_mean_negative_rate",
        "element_std_shift_mean",
        "element_std_shift_median",
        "aie_mean",
        "aie_median",
        "abs_aie_mean",
    ]
    layer_summary = read_required_csv(
        input_dir / "layer_summary.csv",
        layer_required,
        "layer_summary",
    )
    if len(layer_summary) != 32:
        raise ValueError(f"Expected 32 layer rows; received={len(layer_summary)}")
    if layer_summary["layer"].duplicated().any():
        raise ValueError("layer_summary contains duplicate layer rows")
    validate_finite(layer_summary, layer_required, "layer_summary")

    for metric in HEATMAP_METRICS:
        save_heatmap(merged, metric, input_dir / f"heatmap_{metric}.png", args.dpi)
    for metric in HISTOGRAM_METRICS:
        save_head_histogram(
            merged,
            metric,
            input_dir / f"histogram_head_{metric}.png",
            args.dpi,
        )
    save_layer_mean_median_profile(
        layer_summary,
        "element_delta_mean_mean",
        "element_delta_mean_median",
        "element_delta_mean",
        "Layer Profile: Element Delta Mean",
        input_dir / "layer_profile_element_delta_mean.png",
        args.dpi,
    )
    save_layer_mean_median_profile(
        layer_summary,
        "element_std_shift_mean",
        "element_std_shift_median",
        "element_std_shift",
        "Layer Profile: Element Std Shift",
        input_dir / "layer_profile_element_std_shift.png",
        args.dpi,
    )
    save_single_layer_profile(
        layer_summary,
        [("element_delta_mean_negative_rate", "negative head rate")],
        "Rate",
        "Layer Profile: Negative Element-Delta Heads",
        input_dir / "layer_profile_negative_head_rate.png",
        args.dpi,
        zero_line=False,
    )
    save_single_layer_profile(
        layer_summary,
        [("aie_mean", "mean"), ("aie_median", "median"), ("abs_aie_mean", "abs mean")],
        "AIE",
        "Layer Profile: AIE",
        input_dir / "layer_profile_aie.png",
        args.dpi,
        zero_line=True,
    )
    for metric_x, metric_y, filename in SCATTER_SPECS:
        save_scatter(merged, metric_x, metric_y, input_dir / filename, args.dpi)

    for filename in OUTPUT_FILENAMES:
        path = input_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"Plot output missing or empty: {path}")
    print(f"head rows: {len(merged)}")
    print(f"layer rows: {len(layer_summary)}")
    print(f"generated plots: {len(OUTPUT_FILENAMES)}")
    print(f"output directory: {input_dir}")


if __name__ == "__main__":
    main()
