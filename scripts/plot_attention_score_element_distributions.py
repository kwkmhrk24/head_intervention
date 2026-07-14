import argparse
import csv
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = f"/tmp/matplotlib-{os.getuid()}"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_QUANTILES = "0.01,0.05,0.25,0.50,0.75,0.95,0.99"

ELEMENT_REQUIRED_COLUMNS = {
    "condition",
    "prompt_id",
    "source_index",
    "layer",
    "head",
    "row_pos",
    "col_pos",
    "value",
    "seq_len",
    "window_size",
}

DELTA_REQUIRED_COLUMNS = {
    "prompt_id",
    "source_index",
    "layer",
    "head",
    "row_pos",
    "col_pos",
    "original_value",
    "jailbreak_value",
    "delta",
}

PAIR_KEY_COLUMNS = [
    "prompt_id",
    "source_index",
    "layer",
    "head",
    "row_pos",
    "col_pos",
]

ELEMENT_INTEGER_COLUMNS = [
    "prompt_id",
    "layer",
    "head",
    "row_pos",
    "col_pos",
    "seq_len",
    "window_size",
]

DELTA_INTEGER_COLUMNS = [
    "prompt_id",
    "layer",
    "head",
    "row_pos",
    "col_pos",
]

QUANTILE_SUMMARY_COLUMNS = [
    "layer",
    "head",
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


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot ECDF and quantile comparisons from saved attention score "
            "element-distribution CSVs."
        )
    )
    parser.add_argument(
        "--element_csv",
        required=True,
        help="element_values_selected_heads.csv from the extraction script.",
    )
    parser.add_argument(
        "--delta_csv",
        required=True,
        help="delta_element_values_selected_heads.csv from the extraction script.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for PNG and quantile summary outputs.",
    )
    parser.add_argument(
        "--quantiles",
        default=DEFAULT_QUANTILES,
        help=f"Comma-separated quantiles (default: {DEFAULT_QUANTILES}).",
    )
    return parser.parse_args()


def parse_quantiles(value):
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("--quantiles must be a non-empty comma-separated list")
    try:
        quantiles = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--quantiles must contain only numeric values") from exc
    if not np.isfinite(np.asarray(quantiles, dtype=np.float64)).all():
        raise ValueError("--quantiles must contain only finite values")
    if any(not 0 < quantile < 1 for quantile in quantiles):
        raise ValueError("Each quantile must be greater than 0 and less than 1")
    if any(current >= following for current, following in zip(quantiles, quantiles[1:])):
        raise ValueError("--quantiles must be strictly increasing without duplicates")
    return quantiles


def require_columns(dataframe, required_columns, label):
    missing = sorted(required_columns - set(dataframe.columns))
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def read_csv(path, label, has_condition):
    dtype = {"source_index": "string"}
    if has_condition:
        dtype["condition"] = "string"
    try:
        dataframe = pd.read_csv(path, dtype=dtype)
    except Exception as exc:
        raise ValueError(
            f"Failed to read {label} CSV: error_type={type(exc).__name__}"
        ) from None
    if dataframe.empty:
        raise ValueError(f"{label} CSV contains no data rows")
    return dataframe


def coerce_numeric_columns(dataframe, columns, integer_columns, label):
    for column in columns:
        try:
            numeric = pd.to_numeric(dataframe[column], errors="raise")
        except Exception:
            raise ValueError(
                f"{label}: column {column!r} must be numeric"
            ) from None
        values = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}: column {column!r} contains NaN or infinity")
        if column in integer_columns:
            if not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{label}: column {column!r} must contain integers")
            dataframe[column] = values.astype(np.int64)
        else:
            dataframe[column] = values


def keys_as_set(dataframe):
    return set(dataframe[PAIR_KEY_COLUMNS].itertuples(index=False, name=None))


def head_set(dataframe):
    return set(dataframe[["layer", "head"]].itertuples(index=False, name=None))


def validate_and_load_inputs(element_path, delta_path):
    element = read_csv(element_path, "element", has_condition=True)
    delta = read_csv(delta_path, "delta", has_condition=False)
    require_columns(element, ELEMENT_REQUIRED_COLUMNS, "element CSV")
    require_columns(delta, DELTA_REQUIRED_COLUMNS, "delta CSV")

    if element["condition"].isna().any():
        raise ValueError("element CSV: condition contains missing values")
    conditions = set(element["condition"].tolist())
    if conditions != {"original", "jailbreak"}:
        raise ValueError(
            "element CSV: condition values must be exactly original and jailbreak"
        )
    for label, dataframe in (("element CSV", element), ("delta CSV", delta)):
        if dataframe["source_index"].isna().any() or (
            dataframe["source_index"].str.len() == 0
        ).any():
            raise ValueError(f"{label}: source_index contains missing values")

    coerce_numeric_columns(
        element,
        ELEMENT_INTEGER_COLUMNS + ["value"],
        set(ELEMENT_INTEGER_COLUMNS),
        "element CSV",
    )
    coerce_numeric_columns(
        delta,
        DELTA_INTEGER_COLUMNS + ["original_value", "jailbreak_value", "delta"],
        set(DELTA_INTEGER_COLUMNS),
        "delta CSV",
    )
    if (element["window_size"] <= 0).any():
        raise ValueError("element CSV: window_size must be positive")

    element_key_columns = ["condition", *PAIR_KEY_COLUMNS]
    if element.duplicated(element_key_columns).any():
        duplicate_count = int(element.duplicated(element_key_columns).sum())
        raise ValueError(
            f"element CSV: duplicate composite keys detected; count={duplicate_count}"
        )
    if delta.duplicated(PAIR_KEY_COLUMNS).any():
        duplicate_count = int(delta.duplicated(PAIR_KEY_COLUMNS).sum())
        raise ValueError(
            f"delta CSV: duplicate composite keys detected; count={duplicate_count}"
        )

    original = element[element["condition"] == "original"].copy()
    jailbreak = element[element["condition"] == "jailbreak"].copy()
    if not (len(original) == len(jailbreak) == len(delta)):
        raise ValueError(
            "original/jailbreak/delta element counts differ: "
            f"original={len(original)}, jailbreak={len(jailbreak)}, delta={len(delta)}"
        )

    original_heads = head_set(original)
    jailbreak_heads = head_set(jailbreak)
    delta_heads = head_set(delta)
    if original_heads != jailbreak_heads:
        raise ValueError("original and jailbreak layer/head sets do not match")
    if delta_heads != original_heads:
        raise ValueError("delta and element layer/head sets do not match")

    original_keys = keys_as_set(original)
    jailbreak_keys = keys_as_set(jailbreak)
    delta_keys = keys_as_set(delta)
    if original_keys != jailbreak_keys:
        raise ValueError("original and jailbreak composite-key sets do not match")
    if delta_keys != original_keys:
        raise ValueError("delta and element composite-key sets do not match")

    for (layer, head), group in element.groupby(["layer", "head"], sort=True):
        window_sizes = group["window_size"].unique()
        if len(window_sizes) != 1:
            raise ValueError(
                "window_size must be unique within each head: "
                f"layer={layer}, head={head}, count={len(window_sizes)}"
            )

    original_indexed = original.set_index(PAIR_KEY_COLUMNS)["value"]
    jailbreak_indexed = jailbreak.set_index(PAIR_KEY_COLUMNS)["value"]
    delta_indexed = delta.set_index(PAIR_KEY_COLUMNS)
    original_aligned = original_indexed.reindex(delta_indexed.index).to_numpy(
        dtype=np.float64
    )
    jailbreak_aligned = jailbreak_indexed.reindex(delta_indexed.index).to_numpy(
        dtype=np.float64
    )
    if not np.allclose(
        original_aligned,
        delta_indexed["original_value"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("delta original_value does not match element CSV")
    if not np.allclose(
        jailbreak_aligned,
        delta_indexed["jailbreak_value"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("delta jailbreak_value does not match element CSV")
    if not np.allclose(
        delta_indexed["delta"].to_numpy(dtype=np.float64),
        delta_indexed["jailbreak_value"].to_numpy(dtype=np.float64)
        - delta_indexed["original_value"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("delta values do not equal jailbreak_value - original_value")

    for layer, head in sorted(original_heads):
        original_count = len(
            original[(original["layer"] == layer) & (original["head"] == head)]
        )
        jailbreak_count = len(
            jailbreak[(jailbreak["layer"] == layer) & (jailbreak["head"] == head)]
        )
        delta_count = len(delta[(delta["layer"] == layer) & (delta["head"] == head)])
        if not (original_count == jailbreak_count == delta_count):
            raise ValueError(
                "Per-head element counts differ: "
                f"layer={layer}, head={head}, original={original_count}, "
                f"jailbreak={jailbreak_count}, delta={delta_count}"
            )

    return original, jailbreak, delta, sorted(original_heads)


def get_head_data(original, jailbreak, delta, layer, head):
    original_head = original[
        (original["layer"] == layer) & (original["head"] == head)
    ]
    jailbreak_head = jailbreak[
        (jailbreak["layer"] == layer) & (jailbreak["head"] == head)
    ]
    delta_head = delta[(delta["layer"] == layer) & (delta["head"] == head)]
    window_size = int(original_head["window_size"].iloc[0])
    prompt_count = len(
        set(
            original_head[["prompt_id", "source_index"]].itertuples(
                index=False,
                name=None,
            )
        )
    )
    return {
        "layer": layer,
        "head": head,
        "window_size": window_size,
        "prompt_count": prompt_count,
        "element_count": len(original_head),
        "original_values": original_head["value"].to_numpy(dtype=np.float64),
        "jailbreak_values": jailbreak_head["value"].to_numpy(dtype=np.float64),
        "delta_values": delta_head["delta"].to_numpy(dtype=np.float64),
    }


def output_path(output_dir, stem, layer, head, multiple_heads):
    suffix = f"_layer{layer}_head{head}" if multiple_heads else ""
    return output_dir / f"{stem}{suffix}.png"


def shared_x_limits(original_values, jailbreak_values):
    minimum = min(float(original_values.min()), float(jailbreak_values.min()), 0.0)
    maximum = max(float(original_values.max()), float(jailbreak_values.max()), 0.0)
    if minimum == maximum:
        padding = 1.0
    else:
        padding = (maximum - minimum) * 0.02
    return minimum - padding, maximum + padding


def save_ecdf_plot(head_data, output_dir, multiple_heads):
    layer = head_data["layer"]
    head = head_data["head"]
    original_sorted = np.sort(head_data["original_values"])
    jailbreak_sorted = np.sort(head_data["jailbreak_values"])
    original_ecdf = np.arange(1, len(original_sorted) + 1) / len(original_sorted)
    jailbreak_ecdf = np.arange(1, len(jailbreak_sorted) + 1) / len(
        jailbreak_sorted
    )
    x_limits = shared_x_limits(original_sorted, jailbreak_sorted)
    path = output_path(
        output_dir,
        "ecdf_original_vs_jailbreak",
        layer,
        head,
        multiple_heads,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        ax.step(original_sorted, original_ecdf, where="post", label="original")
        ax.step(jailbreak_sorted, jailbreak_ecdf, where="post", label="jailbreak")
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlim(*x_limits)
        ax.set_xlabel("Antisymmetric strict-lower-triangle value")
        ax.set_ylabel("ECDF")
        ax.set_title(
            f"Original vs Jailbreak ECDF | layer={layer}, head={head} | "
            f"window_size={head_data['window_size']}, "
            f"prompts={head_data['prompt_count']}"
        )
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)
    return path


def compute_quantile_rows(head_data, quantiles):
    original_quantiles = np.quantile(head_data["original_values"], quantiles)
    jailbreak_quantiles = np.quantile(head_data["jailbreak_values"], quantiles)
    delta_quantiles = np.quantile(head_data["delta_values"], quantiles)
    rows = []
    for quantile, original_value, jailbreak_value, delta_value in zip(
        quantiles,
        original_quantiles,
        jailbreak_quantiles,
        delta_quantiles,
    ):
        rows.append(
            {
                "layer": head_data["layer"],
                "head": head_data["head"],
                "window_size": head_data["window_size"],
                "prompt_count": head_data["prompt_count"],
                "element_count": head_data["element_count"],
                "quantile": quantile,
                "quantile_percent": quantile * 100,
                "original_quantile": float(original_value),
                "jailbreak_quantile": float(jailbreak_value),
                "quantile_shift": float(jailbreak_value - original_value),
                "delta_distribution_quantile": float(delta_value),
            }
        )
    return rows


def save_quantile_plot(head_data, quantile_rows, output_dir, multiple_heads):
    layer = head_data["layer"]
    head = head_data["head"]
    quantile_percent = [row["quantile_percent"] for row in quantile_rows]
    original_quantiles = [row["original_quantile"] for row in quantile_rows]
    jailbreak_quantiles = [row["jailbreak_quantile"] for row in quantile_rows]
    path = output_path(
        output_dir,
        "quantile_comparison",
        layer,
        head,
        multiple_heads,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        ax.plot(
            quantile_percent,
            original_quantiles,
            marker="o",
            label="original_quantile",
        )
        ax.plot(
            quantile_percent,
            jailbreak_quantiles,
            marker="o",
            label="jailbreak_quantile",
        )
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Quantile percent")
        ax.set_ylabel("Antisymmetric value")
        ax.set_title(
            f"Quantile Comparison | layer={layer}, head={head} | "
            f"window_size={head_data['window_size']}, "
            f"prompts={head_data['prompt_count']}"
        )
        ax.set_xticks(quantile_percent)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)
    return path


def write_quantile_summary(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUANTILE_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = get_args()
    quantiles = parse_quantiles(args.quantiles)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original, jailbreak, delta, heads = validate_and_load_inputs(
        args.element_csv,
        args.delta_csv,
    )
    multiple_heads = len(heads) > 1
    quantile_summary_rows = []
    generated_paths = []

    for layer, head in heads:
        head_data = get_head_data(original, jailbreak, delta, layer, head)
        generated_paths.append(
            save_ecdf_plot(head_data, output_dir, multiple_heads)
        )
        head_quantile_rows = compute_quantile_rows(head_data, quantiles)
        quantile_summary_rows.extend(head_quantile_rows)
        generated_paths.append(
            save_quantile_plot(
                head_data,
                head_quantile_rows,
                output_dir,
                multiple_heads,
            )
        )

    summary_path = output_dir / "quantile_summary.csv"
    write_quantile_summary(summary_path, quantile_summary_rows)
    generated_paths.append(summary_path)

    heads_label = ";".join(f"{layer},{head}" for layer, head in heads)
    print(f"processed heads: {heads_label}")
    print(f"quantiles: {','.join(f'{quantile:g}' for quantile in quantiles)}")
    print(f"quantile summary rows: {len(quantile_summary_rows)}")
    print(f"generated files: {len(generated_paths)}")
    print(f"output directory: {output_dir}")


if __name__ == "__main__":
    main()
