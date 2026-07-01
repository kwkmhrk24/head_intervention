import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DELTA_CONDITION = "delta_jailbreak_minus_original"
DEFAULT_GROUPS = [
    "all_heads",
    "selected_heads_mean",
    "layer13_head4",
    "layer3_head5",
    "layer3_head0",
    "layer7_head16",
    "layer9_head9",
]


def get_args():
    parser = argparse.ArgumentParser(description="Plot attention score symmetry summaries.")
    parser.add_argument("--summary_csv", help="Summary CSV from summarize_attention_score_metrics.py.")
    parser.add_argument("--summary_csvs", nargs="+", help="Multiple summary CSVs for tail-window comparison.")
    parser.add_argument("--original_csv", help="Raw original all-head metrics CSV.")
    parser.add_argument("--jailbreak_csv", help="Raw jailbreak all-head metrics CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory for PNG outputs.")
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS, help="Groups to plot from summary CSV.")
    parser.add_argument("--paper_heads", default="3,0;7,16;9,9;13,4;3,5", help="Paper heads to mark on heatmaps.")
    return parser.parse_args()


def require_columns(df, columns, label):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def normalize_legacy_asym_columns(df):
    rename_columns = {
        column: column.replace("skew", "asym")
        for column in df.columns
        if "skew" in column and column.replace("skew", "asym") not in df.columns
    }
    return df.rename(columns=rename_columns)


def parse_heads(heads):
    parsed = []
    for item in heads.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid head spec {item!r}; expected 'layer,head'")
        try:
            parsed.append((int(parts[0]), int(parts[1])))
        except ValueError as exc:
            raise ValueError(f"Invalid head spec {item!r}; layer/head must be integers") from exc
    return parsed


def infer_window_label(df, fallback=""):
    if "window_type" not in df.columns:
        return fallback
    values = [str(value) for value in df["window_type"].dropna().unique()]
    if len(values) == 1:
        return values[0]
    return fallback


def filename_window_label(label):
    return label.replace("_", "")


def infer_window_size_from_path(path):
    match = re.search(r"tail(\d+)", Path(path).name)
    if match:
        return int(match.group(1))
    return None


def infer_window_label_from_path(path, fallback="tail20"):
    window_size = infer_window_size_from_path(path)
    if window_size is not None:
        return f"tail{window_size}"
    return fallback


def get_delta_rows(summary_csv, groups):
    df = normalize_legacy_asym_columns(pd.read_csv(summary_csv))
    required = [
        "group",
        "condition",
        "delta_sym_ratio_mean",
        "delta_asym_ratio_mean",
        "sym_up_rate",
        "asym_down_rate",
    ]
    require_columns(df, required, summary_csv)
    delta = df[df["condition"] == DELTA_CONDITION].copy()
    if delta.empty:
        raise ValueError(f"{summary_csv}: no rows with condition={DELTA_CONDITION!r}")
    missing = [group for group in groups if group not in set(delta["group"])]
    if missing:
        raise ValueError(f"{summary_csv}: requested groups not found: {missing}")
    delta = delta[delta["group"].isin(groups)].copy()
    delta["group"] = pd.Categorical(delta["group"], categories=groups, ordered=True)
    return delta.sort_values("group")


def setup_slide_style():
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 18,
    })


def save_bar_plot(summary_csv, groups, output_dir):
    delta = get_delta_rows(summary_csv, groups)
    window_label = infer_window_label(delta, fallback="tail")
    output_label = filename_window_label(window_label)
    x = np.arange(len(delta))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(delta) * 1.4), 6))
    ax.bar(x - width / 2, delta["delta_sym_ratio_mean"], width, label="delta_sym_ratio_mean", color="#4477AA")
    ax.bar(x + width / 2, delta["delta_asym_ratio_mean"], width, label="delta_asym_ratio_mean", color="#CC6677")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(delta["group"], rotation=30, ha="right")
    ax.set_ylabel("delta ratio")
    ax.set_title(f"Jailbreak - Original: Symmetric / Asymmetric Ratio Change ({window_label})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path = output_dir / f"{output_label}_delta_sym_asym_bar.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_rate_plot(summary_csv, groups, output_dir):
    delta = get_delta_rows(summary_csv, groups)
    window_label = infer_window_label(delta, fallback="tail")
    output_label = filename_window_label(window_label)
    x = np.arange(len(delta))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(delta) * 1.4), 6))
    ax.bar(x - width / 2, delta["sym_up_rate"], width, label="sym_up_rate", color="#228833")
    ax.bar(x + width / 2, delta["asym_down_rate"], width, label="asym_down_rate", color="#AA3377")
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(delta["group"], rotation=30, ha="right")
    ax.set_ylabel("rate")
    ax.set_title(f"Prompt-wise Direction Agreement ({window_label})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path = output_dir / f"{output_label}_sym_up_asym_down_rate_bar.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_lower_triangle_bar(summary_csv, groups, output_dir, metric, title, filename_suffix, color):
    delta = get_delta_rows(summary_csv, groups)
    require_columns(delta, [metric], summary_csv)
    if delta[metric].isna().all():
        raise ValueError(f"{summary_csv}: no values found for {metric}")

    window_label = infer_window_label(delta, fallback="tail")
    output_label = filename_window_label(window_label)
    x = np.arange(len(delta))

    fig, ax = plt.subplots(figsize=(max(10, len(delta) * 1.4), 6))
    ax.bar(x, delta[metric], color=color)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(delta["group"], rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{title} ({window_label})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path = output_dir / f"{output_label}_{filename_suffix}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def summary_has_lower_triangle_metrics(summary_csv):
    df = normalize_legacy_asym_columns(pd.read_csv(summary_csv))
    required = [
        "delta_asym_lower_pos_rate_mean",
        "asym_lower_pos_up_rate",
        "delta_asym_lower_mean_mean",
        "delta_asym_lower_abs_mean_mean",
    ]
    return all(column in df.columns and df[column].notna().any() for column in required)


def load_raw_metrics(path, label):
    df = normalize_legacy_asym_columns(pd.read_csv(path))
    required = ["prompt_id", "source_index", "layer", "head", "sym_ratio", "asym_ratio"]
    require_columns(df, required, label)
    return df


def compute_head_delta_grid(original_csv, jailbreak_csv, metric):
    original = load_raw_metrics(original_csv, original_csv)
    jailbreak = load_raw_metrics(jailbreak_csv, jailbreak_csv)
    merge_keys = ["prompt_id", "source_index", "layer", "head"]
    merged = jailbreak.merge(
        original,
        on=merge_keys,
        suffixes=("_jailbreak", "_original"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("No raw metric rows matched by prompt_id, source_index, layer, head")
    merged[f"delta_{metric}"] = merged[f"{metric}_jailbreak"] - merged[f"{metric}_original"]
    head_delta = merged.groupby(["layer", "head"], as_index=False)[f"delta_{metric}"].mean()
    max_layer = int(head_delta["layer"].max())
    max_head = int(head_delta["head"].max())
    grid = np.full((max_layer + 1, max_head + 1), np.nan)
    for row in head_delta.itertuples(index=False):
        grid[int(row.layer), int(row.head)] = getattr(row, f"delta_{metric}")
    return grid


def save_heatmap(grid, metric, output_dir, paper_heads, window_label):
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        raise ValueError(f"No finite values for {metric} heatmap")
    vmax = float(np.max(np.abs(finite)))
    if vmax == 0:
        vmax = 1e-12

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(grid, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(f"mean delta {metric}")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(f"Jailbreak - Original: mean delta {metric}")

    for layer, head in paper_heads:
        ax.scatter(head, layer, marker="s", s=120, facecolors="none", edgecolors="black", linewidths=1.8)
    if (13, 4) in paper_heads:
        ax.scatter(4, 13, marker="*", s=220, color="gold", edgecolors="black", linewidths=1.0)
        ax.annotate("13,4", xy=(4, 13), xytext=(6, 11), color="black",
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2})

    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    output_path = output_dir / f"{window_label}_delta_{metric}_heatmap.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_window_comparison(summary_csvs, groups, output_dir, metric):
    records = []
    for path in summary_csvs:
        delta = get_delta_rows(path, groups)
        window_size = infer_window_size_from_path(path)
        if window_size is None and "window_size_mean" in delta.columns:
            window_size = int(round(float(delta["window_size_mean"].dropna().iloc[0])))
        if window_size is None:
            raise ValueError(f"Could not infer tail window size from {path}")
        for row in delta.itertuples(index=False):
            records.append({
                "window_size": window_size,
                "group": row.group,
                "value": getattr(row, f"delta_{metric}_mean"),
            })

    df = pd.DataFrame.from_records(records)
    fig, ax = plt.subplots(figsize=(9, 6))
    for group in groups:
        group_df = df[df["group"] == group].sort_values("window_size")
        ax.plot(group_df["window_size"], group_df["value"], marker="o", label=group)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("tail window size")
    ax.set_ylabel(f"delta {metric}")
    ax.set_title(f"Window Comparison: delta {metric}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path = output_dir / f"window_comparison_delta_{metric}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    args = get_args()
    setup_slide_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    if args.summary_csv:
        outputs.append(save_bar_plot(args.summary_csv, args.groups, output_dir))
        outputs.append(save_rate_plot(args.summary_csv, args.groups, output_dir))
        if summary_has_lower_triangle_metrics(args.summary_csv):
            outputs.append(save_lower_triangle_bar(
                args.summary_csv,
                args.groups,
                output_dir,
                "delta_asym_lower_pos_rate_mean",
                "Jailbreak - Original: Lower-triangle Positive Rate in S_asym",
                "delta_asym_lower_pos_rate_bar",
                "#4477AA",
            ))
            outputs.append(save_lower_triangle_bar(
                args.summary_csv,
                args.groups,
                output_dir,
                "asym_lower_pos_up_rate",
                "Prompt-wise Direction Agreement: Lower-triangle Positive Rate",
                "asym_lower_pos_up_rate_bar",
                "#228833",
            ))
            outputs.append(save_lower_triangle_bar(
                args.summary_csv,
                args.groups,
                output_dir,
                "delta_asym_lower_mean_mean",
                "Jailbreak - Original: Lower-triangle Mean in S_asym",
                "delta_asym_lower_mean_bar",
                "#CC6677",
            ))
            outputs.append(save_lower_triangle_bar(
                args.summary_csv,
                args.groups,
                output_dir,
                "delta_asym_lower_abs_mean_mean",
                "Jailbreak - Original: Lower-triangle Absolute Mean in S_asym",
                "delta_asym_lower_abs_mean_bar",
                "#AA3377",
            ))

    if args.original_csv and args.jailbreak_csv:
        paper_heads = parse_heads(args.paper_heads)
        window_label = infer_window_label_from_path(args.jailbreak_csv)
        outputs.append(save_heatmap(
            compute_head_delta_grid(args.original_csv, args.jailbreak_csv, "sym_ratio"),
            "sym_ratio",
            output_dir,
            paper_heads,
            window_label,
        ))
        outputs.append(save_heatmap(
            compute_head_delta_grid(args.original_csv, args.jailbreak_csv, "asym_ratio"),
            "asym_ratio",
            output_dir,
            paper_heads,
            window_label,
        ))
    elif args.original_csv or args.jailbreak_csv:
        raise ValueError("--original_csv and --jailbreak_csv must be provided together for heatmaps")

    if args.summary_csvs:
        outputs.append(save_window_comparison(args.summary_csvs, args.groups, output_dir, "sym_ratio"))
        outputs.append(save_window_comparison(args.summary_csvs, args.groups, output_dir, "asym_ratio"))

    if not outputs:
        raise ValueError("No plots requested. Provide --summary_csv, raw CSV pair, or --summary_csvs.")

    for path in outputs:
        print(f"wrote: {path}")


if __name__ == "__main__":
    main()
