import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = f"/tmp/matplotlib-{os.getuid()}"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_attention_score_element_distributions import (
    DEFAULT_QUANTILES,
    parse_quantiles,
)


RAW_ARRAY_FILENAMES = {
    "original": "raw_original_values.npy",
    "jailbreak": "raw_jailbreak_values.npy",
    "delta": "raw_delta_values.npy",
}

MEAN_ARRAY_FILENAMES = {
    "original": "all_heads_mean_original.npy",
    "jailbreak": "all_heads_mean_jailbreak.npy",
    "delta": "all_heads_mean_delta.npy",
}

COMPARISON_BIN_COLUMNS = [
    "bin_left",
    "bin_right",
    "bin_center",
    "original_count",
    "jailbreak_count",
    "original_density",
    "jailbreak_density",
]

DELTA_BIN_COLUMNS = [
    "bin_left",
    "bin_right",
    "bin_center",
    "delta_count",
    "delta_density",
]

MEAN_QUANTILE_COLUMNS = [
    "layer_head_count",
    "window_size",
    "prompt_count",
    "element_count",
    "quantile",
    "quantile_percent",
    "original_quantile",
    "jailbreak_quantile",
    "quantile_shift",
    "delta_distribution_quantile",
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all-head raw and all-head mean attention score distributions "
            "from previously saved NPY arrays."
        )
    )
    parser.add_argument(
        "--raw_output_dir",
        required=True,
        help="Directory containing all-head raw arrays and metadata.",
    )
    parser.add_argument(
        "--mean_output_dir",
        required=True,
        help="Directory containing all-head mean arrays and metadata.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=200,
        help="Number of histogram bins (default: 200).",
    )
    parser.add_argument(
        "--quantiles",
        default=DEFAULT_QUANTILES,
        help=f"Comma-separated mean-distribution quantiles (default: {DEFAULT_QUANTILES}).",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        raise ValueError(
            f"Failed to read metadata: path={path}, error_type={type(exc).__name__}"
        ) from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a JSON object: {path}")
    return metadata


def load_array(path: Path) -> np.ndarray:
    try:
        array = np.load(path, mmap_mode="r")
    except Exception as exc:
        raise ValueError(
            f"Failed to load NPY array: path={path}, error_type={type(exc).__name__}"
        ) from exc
    if array.dtype != np.float32:
        raise ValueError(f"Expected float32 NPY array: path={path}, dtype={array.dtype}")
    return array


def validate_finite_chunked(name: str, values: np.ndarray) -> None:
    for prompt_index in range(values.shape[0]):
        if not np.isfinite(values[prompt_index]).all():
            raise ValueError(
                f"Array contains NaN or infinity: name={name}, "
                f"prompt_index={prompt_index}"
            )


def validate_array_group(
    arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    expected_ndim: int,
    label: str,
) -> None:
    shapes = {name: array.shape for name, array in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"{label} array shapes do not match: {shapes}")
    shape = next(iter(shapes.values()))
    if len(shape) != expected_ndim:
        raise ValueError(
            f"{label} arrays must have ndim={expected_ndim}: shape={shape}"
        )
    metadata_shape = tuple(metadata.get("array_shape", ()))
    if metadata_shape and metadata_shape != shape:
        raise ValueError(
            f"{label} metadata shape does not match arrays: "
            f"metadata={metadata_shape}, array={shape}"
        )
    for name, array in arrays.items():
        validate_finite_chunked(f"{label}_{name}", array)

    for prompt_index in range(shape[0]):
        if not np.allclose(
            arrays["delta"][prompt_index],
            arrays["jailbreak"][prompt_index]
            - arrays["original"][prompt_index],
            rtol=1e-5,
            atol=2e-6,
        ):
            raise ValueError(
                f"{label} delta does not equal jailbreak - original: "
                f"prompt_index={prompt_index}"
            )


def chunked_extrema(values: np.ndarray) -> Tuple[float, float]:
    minimum = float("inf")
    maximum = float("-inf")
    for prompt_index in range(values.shape[0]):
        chunk = values[prompt_index]
        minimum = min(minimum, float(np.min(chunk)))
        maximum = max(maximum, float(np.max(chunk)))
    return minimum, maximum


def common_histogram_bin_edges(
    original: np.ndarray,
    jailbreak: np.ndarray,
    bins: int,
) -> np.ndarray:
    original_min, original_max = chunked_extrema(original)
    jailbreak_min, jailbreak_max = chunked_extrema(jailbreak)
    minimum = min(original_min, jailbreak_min)
    maximum = max(original_max, jailbreak_max)
    if minimum == maximum:
        padding = max(abs(minimum) * 0.01, 1.0)
        minimum -= padding
        maximum += padding
    return np.linspace(minimum, maximum, bins + 1, dtype=np.float64)


def symmetric_delta_bin_edges(delta: np.ndarray, bins: int) -> np.ndarray:
    minimum, maximum = chunked_extrema(delta)
    max_abs = max(abs(minimum), abs(maximum))
    if max_abs == 0:
        max_abs = 1.0
    return np.linspace(-max_abs, max_abs, bins + 1, dtype=np.float64)


def histogram_counts(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    for prompt_index in range(values.shape[0]):
        chunk = np.asarray(values[prompt_index]).reshape(-1)
        chunk_counts, _ = np.histogram(chunk, bins=bin_edges)
        counts += chunk_counts
    if int(counts.sum()) != int(values.size):
        raise ValueError(
            "Histogram counts do not cover all elements: "
            f"counted={int(counts.sum())}, expected={int(values.size)}"
        )
    return counts


def density_from_counts(counts: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    widths = np.diff(bin_edges)
    return counts.astype(np.float64) / (int(counts.sum()) * widths)


def write_comparison_bins(
    path: Path,
    bin_edges: np.ndarray,
    original_counts: np.ndarray,
    jailbreak_counts: np.ndarray,
) -> None:
    original_density = density_from_counts(original_counts, bin_edges)
    jailbreak_density = density_from_counts(jailbreak_counts, bin_edges)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_BIN_COLUMNS)
        writer.writeheader()
        for index in range(len(original_counts)):
            left = float(bin_edges[index])
            right = float(bin_edges[index + 1])
            writer.writerow(
                {
                    "bin_left": left,
                    "bin_right": right,
                    "bin_center": (left + right) / 2,
                    "original_count": int(original_counts[index]),
                    "jailbreak_count": int(jailbreak_counts[index]),
                    "original_density": float(original_density[index]),
                    "jailbreak_density": float(jailbreak_density[index]),
                }
            )


def write_delta_bins(
    path: Path,
    bin_edges: np.ndarray,
    delta_counts: np.ndarray,
) -> None:
    delta_density = density_from_counts(delta_counts, bin_edges)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DELTA_BIN_COLUMNS)
        writer.writeheader()
        for index in range(len(delta_counts)):
            left = float(bin_edges[index])
            right = float(bin_edges[index + 1])
            writer.writerow(
                {
                    "bin_left": left,
                    "bin_right": right,
                    "bin_center": (left + right) / 2,
                    "delta_count": int(delta_counts[index]),
                    "delta_density": float(delta_density[index]),
                }
            )


def title_metadata(metadata: Dict[str, Any]) -> str:
    return (
        f"tail={metadata.get('window_size')}, "
        f"prompts={metadata.get('processed_prompt_count')}, "
        f"layer-heads={metadata.get('layer_head_count')}"
    )


def save_histogram_comparison(
    path: Path,
    bin_edges: np.ndarray,
    original_counts: np.ndarray,
    jailbreak_counts: np.ndarray,
    title_prefix: str,
    metadata: Dict[str, Any],
) -> None:
    original_density = density_from_counts(original_counts, bin_edges)
    jailbreak_density = density_from_counts(jailbreak_counts, bin_edges)
    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        ax.stairs(
            original_density,
            bin_edges,
            label="original",
            linewidth=1.5,
        )
        ax.stairs(
            jailbreak_density,
            bin_edges,
            label="jailbreak",
            linewidth=1.5,
        )
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Antisymmetric strict-lower-triangle value")
        ax.set_ylabel("Density")
        ax.set_title(f"{title_prefix} | {title_metadata(metadata)}")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)


def save_histogram_delta(
    path: Path,
    bin_edges: np.ndarray,
    delta_counts: np.ndarray,
    title_prefix: str,
    metadata: Dict[str, Any],
) -> None:
    delta_density = density_from_counts(delta_counts, bin_edges)
    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        ax.stairs(
            delta_density,
            bin_edges,
            color="#7A5195",
            linewidth=1.5,
        )
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Delta (jailbreak - original)")
        ax.set_ylabel("Density")
        ax.set_title(f"{title_prefix} | {title_metadata(metadata)}")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)


def plot_histogram_set(
    output_dir: Path,
    arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    bins: int,
    title_label: str,
    write_bin_csvs: bool,
) -> None:
    comparison_edges = common_histogram_bin_edges(
        arrays["original"],
        arrays["jailbreak"],
        bins,
    )
    original_counts = histogram_counts(arrays["original"], comparison_edges)
    jailbreak_counts = histogram_counts(arrays["jailbreak"], comparison_edges)
    delta_edges = symmetric_delta_bin_edges(arrays["delta"], bins)
    delta_counts = histogram_counts(arrays["delta"], delta_edges)

    save_histogram_comparison(
        output_dir / "histogram_original_vs_jailbreak.png",
        comparison_edges,
        original_counts,
        jailbreak_counts,
        f"{title_label}: Original vs Jailbreak",
        metadata,
    )
    save_histogram_delta(
        output_dir / "histogram_delta.png",
        delta_edges,
        delta_counts,
        f"{title_label}: Element Delta",
        metadata,
    )
    if write_bin_csvs:
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


def flattened_mean_values(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def shared_x_limits(original: np.ndarray, jailbreak: np.ndarray) -> Tuple[float, float]:
    minimum = min(float(original.min()), float(jailbreak.min()), 0.0)
    maximum = max(float(original.max()), float(jailbreak.max()), 0.0)
    padding = (maximum - minimum) * 0.02 if minimum != maximum else 1.0
    return minimum - padding, maximum + padding


def save_mean_ecdf(
    path: Path,
    original: np.ndarray,
    jailbreak: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    original_sorted = np.sort(flattened_mean_values(original))
    jailbreak_sorted = np.sort(flattened_mean_values(jailbreak))
    original_ecdf = np.arange(1, original_sorted.size + 1) / original_sorted.size
    jailbreak_ecdf = np.arange(1, jailbreak_sorted.size + 1) / jailbreak_sorted.size
    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        ax.step(original_sorted, original_ecdf, where="post", label="original")
        ax.step(jailbreak_sorted, jailbreak_ecdf, where="post", label="jailbreak")
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlim(*shared_x_limits(original_sorted, jailbreak_sorted))
        ax.set_xlabel("Antisymmetric strict-lower-triangle value")
        ax.set_ylabel("ECDF")
        ax.set_title(f"All-head Mean ECDF | {title_metadata(metadata)}")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)


def compute_mean_quantile_rows(
    arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    quantiles: Sequence[float],
) -> List[Dict[str, Any]]:
    original = flattened_mean_values(arrays["original"])
    jailbreak = flattened_mean_values(arrays["jailbreak"])
    delta = flattened_mean_values(arrays["delta"])
    original_quantiles = np.quantile(original, quantiles)
    jailbreak_quantiles = np.quantile(jailbreak, quantiles)
    delta_quantiles = np.quantile(delta, quantiles)
    rows: List[Dict[str, Any]] = []
    for quantile, original_value, jailbreak_value, delta_value in zip(
        quantiles,
        original_quantiles,
        jailbreak_quantiles,
        delta_quantiles,
    ):
        rows.append(
            {
                "layer_head_count": metadata["layer_head_count"],
                "window_size": metadata["window_size"],
                "prompt_count": metadata["processed_prompt_count"],
                "element_count": int(original.size),
                "quantile": quantile,
                "quantile_percent": quantile * 100,
                "original_quantile": float(original_value),
                "jailbreak_quantile": float(jailbreak_value),
                "quantile_shift": float(jailbreak_value - original_value),
                "delta_distribution_quantile": float(delta_value),
            }
        )
    return rows


def write_mean_quantile_summary(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MEAN_QUANTILE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_mean_quantile_plot(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    quantile_percent = [row["quantile_percent"] for row in rows]
    original = [row["original_quantile"] for row in rows]
    jailbreak = [row["jailbreak_quantile"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        ax.plot(quantile_percent, original, marker="o", label="original_quantile")
        ax.plot(
            quantile_percent,
            jailbreak,
            marker="o",
            label="jailbreak_quantile",
        )
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Quantile percent")
        ax.set_ylabel("Antisymmetric value")
        ax.set_xticks(quantile_percent)
        ax.set_title(f"All-head Mean Quantiles | {title_metadata(metadata)}")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)


def load_output_arrays(
    output_dir: Path,
    filenames: Dict[str, str],
) -> Dict[str, np.ndarray]:
    return {
        condition: load_array(output_dir / filename)
        for condition, filename in filenames.items()
    }


def main() -> None:
    args = get_args()
    if args.bins <= 0:
        raise ValueError("--bins must be positive")
    quantiles = parse_quantiles(args.quantiles)
    raw_output_dir = Path(args.raw_output_dir)
    mean_output_dir = Path(args.mean_output_dir)

    raw_metadata = load_metadata(raw_output_dir / "metadata.json")
    mean_metadata = load_metadata(mean_output_dir / "metadata.json")
    raw_arrays = load_output_arrays(raw_output_dir, RAW_ARRAY_FILENAMES)
    mean_arrays = load_output_arrays(mean_output_dir, MEAN_ARRAY_FILENAMES)
    validate_array_group(raw_arrays, raw_metadata, expected_ndim=4, label="raw")
    validate_array_group(mean_arrays, mean_metadata, expected_ndim=2, label="mean")

    expected_mean_shape = (raw_arrays["original"].shape[0], raw_arrays["original"].shape[3])
    if mean_arrays["original"].shape != expected_mean_shape:
        raise ValueError(
            "Mean array shape does not match raw prompt/position axes: "
            f"mean={mean_arrays['original'].shape}, expected={expected_mean_shape}"
        )
    for key in (
        "processed_prompt_count",
        "window_size",
        "elements_per_head",
        "num_layers",
        "num_attention_heads",
        "layer_head_count",
    ):
        if raw_metadata.get(key) != mean_metadata.get(key):
            raise ValueError(
                f"Raw/mean metadata mismatch for {key}: "
                f"raw={raw_metadata.get(key)!r}, mean={mean_metadata.get(key)!r}"
            )
    if raw_arrays["original"].shape[1] != raw_metadata.get("num_layers"):
        raise ValueError("Raw layer axis does not match metadata num_layers")
    if raw_arrays["original"].shape[2] != raw_metadata.get("num_attention_heads"):
        raise ValueError("Raw head axis does not match metadata num_attention_heads")

    expected_mean_original = np.mean(
        raw_arrays["original"], axis=(1, 2), dtype=np.float64
    )
    expected_mean_jailbreak = np.mean(
        raw_arrays["jailbreak"], axis=(1, 2), dtype=np.float64
    )
    expected_mean_delta = np.mean(
        raw_arrays["delta"], axis=(1, 2), dtype=np.float64
    )
    if not np.allclose(mean_arrays["original"], expected_mean_original):
        raise ValueError("Mean original array does not match raw all-head mean")
    if not np.allclose(mean_arrays["jailbreak"], expected_mean_jailbreak):
        raise ValueError("Mean jailbreak array does not match raw all-head mean")
    if not np.allclose(mean_arrays["delta"], expected_mean_delta):
        raise ValueError("Mean delta array does not match raw all-head mean")

    plot_histogram_set(
        raw_output_dir,
        raw_arrays,
        raw_metadata,
        args.bins,
        "All-head Raw",
        write_bin_csvs=True,
    )
    plot_histogram_set(
        mean_output_dir,
        mean_arrays,
        mean_metadata,
        args.bins,
        "All-head Mean",
        write_bin_csvs=False,
    )
    save_mean_ecdf(
        mean_output_dir / "ecdf_original_vs_jailbreak.png",
        mean_arrays["original"],
        mean_arrays["jailbreak"],
        mean_metadata,
    )
    quantile_rows = compute_mean_quantile_rows(
        mean_arrays,
        mean_metadata,
        quantiles,
    )
    write_mean_quantile_summary(
        mean_output_dir / "quantile_summary.csv",
        quantile_rows,
    )
    save_mean_quantile_plot(
        mean_output_dir / "quantile_comparison.png",
        quantile_rows,
        mean_metadata,
    )

    print(f"raw values per condition: {int(raw_arrays['original'].size)}")
    print(f"mean values per condition: {int(mean_arrays['original'].size)}")
    print(f"histogram bins: {args.bins}")
    print(f"quantiles: {','.join(f'{quantile:g}' for quantile in quantiles)}")
    print(f"raw output directory: {raw_output_dir}")
    print(f"mean output directory: {mean_output_dir}")


if __name__ == "__main__":
    main()
