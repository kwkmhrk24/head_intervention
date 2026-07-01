import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


REQUIRED_COLUMNS = {
    "prompt_id",
    "source_index",
    "layer",
    "head",
    "seq_len",
    "window_type",
    "window_size",
    "fro_norm_per_token",
    "sym_norm_per_token",
    "asym_norm_per_token",
    "sym_ratio",
    "asym_ratio",
    "reconstruction_rel_error",
    "energy_rel_error",
}

SUMMARY_METRICS = [
    "fro_norm_per_token",
    "sym_norm_per_token",
    "asym_norm_per_token",
    "sym_ratio",
    "asym_ratio",
]

ERROR_METRICS = [
    "reconstruction_rel_error",
    "energy_rel_error",
]

DELTA_METRICS = [
    "fro_norm_per_token",
    "sym_norm_per_token",
    "asym_norm_per_token",
    "sym_ratio",
    "asym_ratio",
]

FIELDNAMES = [
    "group",
    "condition",
    "n_rows",
    "n_prompts",
    "n_heads",
    "window_type",
    "window_size_mean",
    "seq_len_mean",
    "fro_norm_per_token_mean",
    "fro_norm_per_token_std",
    "sym_norm_per_token_mean",
    "sym_norm_per_token_std",
    "asym_norm_per_token_mean",
    "asym_norm_per_token_std",
    "sym_ratio_mean",
    "sym_ratio_std",
    "asym_ratio_mean",
    "asym_ratio_std",
    "reconstruction_rel_error_mean",
    "energy_rel_error_mean",
    "delta_fro_norm_per_token_mean",
    "delta_fro_norm_per_token_std",
    "delta_sym_norm_per_token_mean",
    "delta_sym_norm_per_token_std",
    "delta_asym_norm_per_token_mean",
    "delta_asym_norm_per_token_std",
    "delta_sym_ratio_mean",
    "delta_sym_ratio_std",
    "delta_asym_ratio_mean",
    "delta_asym_ratio_std",
    "sym_up_rate",
    "asym_down_rate",
]


def get_args():
    parser = argparse.ArgumentParser(description="Summarize all-head attention score metrics against selected heads.")
    parser.add_argument("--original_csv", required=True, help="Original prompt metrics CSV.")
    parser.add_argument("--jailbreak_csv", required=True, help="Jailbreak prompt metrics CSV.")
    parser.add_argument("--safe_csv", default=None, help="Optional safe/refusal prompt metrics CSV.")
    parser.add_argument("--heads", required=True, help="Selected heads, e.g. '3,0;7,16;9,9;13,4;3,5'.")
    parser.add_argument("--output_csv", required=True, help="Output summary CSV path.")
    return parser.parse_args()


def parse_heads(heads):
    parsed = []
    if heads.strip().lower() == "all":
        raise ValueError("--heads for summary must be an explicit list like '3,0;7,16', not 'all'")

    for item in heads.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid head spec {item!r}; expected 'layer,head'")
        try:
            layer, head = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"Invalid head spec {item!r}; layer/head must be integers") from exc
        parsed.append((layer, head))

    if not parsed:
        raise ValueError("--heads did not contain any valid layer,head pairs")
    return parsed


def to_float(value):
    if value == "" or value is None:
        return None
    return float(value)


def normalize_legacy_asym_columns(row):
    normalized = dict(row)
    for column, value in row.items():
        if "skew" in column:
            normalized.setdefault(column.replace("skew", "asym"), value)
    return normalized


def read_metrics_csv(path):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV or missing header")
        available_columns = set(reader.fieldnames)
        available_columns.update(
            column.replace("skew", "asym")
            for column in reader.fieldnames
            if "skew" in column
        )
        missing = REQUIRED_COLUMNS - available_columns
        if missing:
            raise ValueError(f"{path}: missing required columns: {sorted(missing)}")

        rows = []
        for row in reader:
            parsed = normalize_legacy_asym_columns(row)
            parsed["layer"] = int(parsed["layer"])
            parsed["head"] = int(parsed["head"])
            parsed["seq_len"] = to_float(parsed["seq_len"])
            parsed["window_size"] = to_float(parsed["window_size"])
            for metric in SUMMARY_METRICS + ERROR_METRICS:
                parsed[metric] = to_float(parsed[metric])
            rows.append(parsed)
    return rows


def validate_heads_present(rows, heads, label):
    present = {(row["layer"], row["head"]) for row in rows}
    missing = [head for head in heads if head not in present]
    if missing:
        raise ValueError(f"{label}: selected heads not found in CSV: {missing}")


def mean_or_blank(values):
    values = [value for value in values if value is not None]
    if not values:
        return ""
    return mean(values)


def std_or_blank(values):
    values = [value for value in values if value is not None]
    if len(values) <= 1:
        return 0.0 if values else ""
    return stdev(values)


def rate_or_blank(values):
    values = [value for value in values if value is not None]
    if not values:
        return ""
    return mean(values)


def group_rows(rows, group, selected_heads):
    if group == "all_heads":
        return rows
    if group == "selected_heads_mean":
        selected = set(selected_heads)
        return [row for row in rows if (row["layer"], row["head"]) in selected]

    layer, head = group
    return [row for row in rows if row["layer"] == layer and row["head"] == head]


def group_name(group):
    if isinstance(group, str):
        return group
    layer, head = group
    return f"layer{layer}_head{head}"


def base_summary_row(rows, group, condition):
    if not rows:
        raise ValueError(f"No rows found for group={group_name(group)} condition={condition}")

    window_types = sorted({row["window_type"] for row in rows})
    return {
        "group": group_name(group),
        "condition": condition,
        "n_rows": len(rows),
        "n_prompts": len({row["prompt_id"] for row in rows}),
        "n_heads": len({(row["layer"], row["head"]) for row in rows}),
        "window_type": window_types[0] if len(window_types) == 1 else "mixed",
        "window_size_mean": mean_or_blank([row["window_size"] for row in rows]),
        "seq_len_mean": mean_or_blank([row["seq_len"] for row in rows]),
    }


def summarize_condition(rows, group, condition):
    selected_rows = group_rows(rows, group, [])
    summary = base_summary_row(selected_rows, group, condition)

    for metric in SUMMARY_METRICS:
        summary[f"{metric}_mean"] = mean_or_blank([row[metric] for row in selected_rows])
        summary[f"{metric}_std"] = std_or_blank([row[metric] for row in selected_rows])

    for metric in ERROR_METRICS:
        summary[f"{metric}_mean"] = mean_or_blank([row[metric] for row in selected_rows])

    return summary


def summarize_condition_group(rows, group, condition, selected_heads):
    selected_rows = group_rows(rows, group, selected_heads)
    summary = base_summary_row(selected_rows, group, condition)

    for metric in SUMMARY_METRICS:
        summary[f"{metric}_mean"] = mean_or_blank([row[metric] for row in selected_rows])
        summary[f"{metric}_std"] = std_or_blank([row[metric] for row in selected_rows])

    for metric in ERROR_METRICS:
        summary[f"{metric}_mean"] = mean_or_blank([row[metric] for row in selected_rows])

    return summary


def make_pair_key(row):
    return (row["prompt_id"], row["source_index"], row["layer"], row["head"])


def compute_delta_rows(original_rows, jailbreak_rows):
    original_by_key = {make_pair_key(row): row for row in original_rows}
    jailbreak_by_key = {make_pair_key(row): row for row in jailbreak_rows}
    shared_keys = sorted(set(original_by_key) & set(jailbreak_by_key))
    if not shared_keys:
        raise ValueError("No paired original/jailbreak rows matched by prompt_id, source_index, layer, head")

    delta_rows = []
    for key in shared_keys:
        original = original_by_key[key]
        jailbreak = jailbreak_by_key[key]
        row = {
            "prompt_id": jailbreak["prompt_id"],
            "source_index": jailbreak["source_index"],
            "layer": jailbreak["layer"],
            "head": jailbreak["head"],
            "seq_len": jailbreak["seq_len"],
            "window_type": jailbreak["window_type"],
            "window_size": jailbreak["window_size"],
        }
        for metric in DELTA_METRICS:
            row[f"delta_{metric}"] = jailbreak[metric] - original[metric]
        delta_rows.append(row)

    return delta_rows


def summarize_delta_group(rows, group, selected_heads):
    selected_rows = group_rows(rows, group, selected_heads)
    summary = base_summary_row(selected_rows, group, "delta_jailbreak_minus_original")

    for metric in DELTA_METRICS:
        delta_metric = f"delta_{metric}"
        summary[f"{delta_metric}_mean"] = mean_or_blank([row[delta_metric] for row in selected_rows])
        summary[f"{delta_metric}_std"] = std_or_blank([row[delta_metric] for row in selected_rows])

    summary["sym_up_rate"] = rate_or_blank([
        row["delta_sym_ratio"] > 0 for row in selected_rows
    ])
    summary["asym_down_rate"] = rate_or_blank([
        row["delta_asym_ratio"] < 0 for row in selected_rows
    ])
    return summary


def main():
    args = get_args()
    selected_heads = parse_heads(args.heads)

    original_rows = read_metrics_csv(args.original_csv)
    jailbreak_rows = read_metrics_csv(args.jailbreak_csv)
    safe_rows = read_metrics_csv(args.safe_csv) if args.safe_csv else None

    validate_heads_present(original_rows, selected_heads, args.original_csv)
    validate_heads_present(jailbreak_rows, selected_heads, args.jailbreak_csv)
    if safe_rows is not None:
        validate_heads_present(safe_rows, selected_heads, args.safe_csv)

    groups = ["all_heads", "selected_heads_mean", *selected_heads]
    output_rows = []

    for group in groups:
        output_rows.append(summarize_condition_group(original_rows, group, "original", selected_heads))
        output_rows.append(summarize_condition_group(jailbreak_rows, group, "jailbreak", selected_heads))
        if safe_rows is not None:
            output_rows.append(summarize_condition_group(safe_rows, group, "safe", selected_heads))

    delta_rows = compute_delta_rows(original_rows, jailbreak_rows)
    for group in groups:
        output_rows.append(summarize_delta_group(delta_rows, group, selected_heads))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"wrote rows: {len(output_rows)}")
    print(f"output_csv: {output_path}")


if __name__ == "__main__":
    main()
