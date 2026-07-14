import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = f"/tmp/matplotlib-{os.getuid()}"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_attention_scores import (
    capture_self_attn_inputs,
    compute_post_rope_qk,
    compute_score_matrix,
    get_attention_helpers,
    parse_heads,
    read_prompt_rows,
    select_score_window,
)
from utils.model_utils import load_model_and_tokenizer
from utils.prompt_utils import apply_template


ELEMENT_COLUMNS = [
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
]

DELTA_COLUMNS = [
    "prompt_id",
    "source_index",
    "layer",
    "head",
    "row_pos",
    "col_pos",
    "original_value",
    "jailbreak_value",
    "delta",
]

ELEMENT_KEY_COLUMNS = [
    "prompt_id",
    "source_index",
    "layer",
    "head",
    "row_pos",
    "col_pos",
]


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Save and plot strict-lower-triangle values from the antisymmetric "
            "component of post-RoPE, pre-mask, pre-softmax attention scores."
        )
    )
    parser.add_argument("--model", required=True, help="Model path.")
    parser.add_argument(
        "--original_prompts",
        required=True,
        help="Original-prompt JSONL file with a prompt key.",
    )
    parser.add_argument(
        "--jailbreak_prompts",
        required=True,
        help="Jailbreak-prompt JSONL file with a prompt key.",
    )
    parser.add_argument(
        "--heads",
        required=True,
        help="Semicolon-separated layer,head list, e.g. '13,4;3,0'.",
    )
    parser.add_argument(
        "--tail_tokens",
        required=True,
        type=int,
        help="Use exactly the final N tokens from each prompt independently.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for CSV and PNG outputs.")
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Optional number of prompt pairs to process.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt passed to apply_template().",
    )
    parser.add_argument(
        "--hist_bins",
        type=int,
        default=100,
        help="Number of shared histogram bins (default: 100).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print non-prompt shape diagnostics for the first pair.",
    )
    return parser.parse_args()


def explicit_source_id(row):
    if "source_index" in row:
        return row["source_index"]
    if "index" in row:
        return row["index"]
    return None


def resolved_source_index(row, prompt_id):
    source_index = row.get("source_index", row.get("index", prompt_id))
    if isinstance(source_index, bool) or not isinstance(source_index, (int, str)):
        raise ValueError(
            "source_index/index must resolve to an integer or string: "
            f"prompt_id={prompt_id}"
        )
    return source_index


def validate_prompt_rows(rows, condition):
    if not rows:
        raise ValueError(f"{condition}: prompt file contains no prompt rows")

    source_indices = []
    for prompt_id, row in enumerate(rows):
        if "prompt" not in row:
            raise ValueError(f"{condition}: prompt_id={prompt_id} missing required key: prompt")
        if not isinstance(row["prompt"], str):
            raise ValueError(f"{condition}: prompt_id={prompt_id} prompt must be a string")
        source_indices.append(resolved_source_index(row, prompt_id))

    duplicate_ids = [
        source_index
        for source_index, count in Counter(source_indices).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(
            f"{condition}: duplicate source_index/index values detected; "
            f"duplicate_count={len(duplicate_ids)}"
        )


def prepare_prompt_pairs(original_path, jailbreak_path, max_prompts):
    original_rows = read_prompt_rows(original_path)
    jailbreak_rows = read_prompt_rows(jailbreak_path)

    if len(original_rows) != len(jailbreak_rows):
        raise ValueError(
            "original/jailbreak prompt row counts differ: "
            f"original={len(original_rows)}, jailbreak={len(jailbreak_rows)}"
        )

    validate_prompt_rows(original_rows, "original")
    validate_prompt_rows(jailbreak_rows, "jailbreak")

    pairs = []
    for prompt_id, (original_row, jailbreak_row) in enumerate(
        zip(original_rows, jailbreak_rows)
    ):
        original_explicit_id = explicit_source_id(original_row)
        jailbreak_explicit_id = explicit_source_id(jailbreak_row)
        if (
            original_explicit_id is not None
            and jailbreak_explicit_id is not None
            and original_explicit_id != jailbreak_explicit_id
        ):
            raise ValueError(
                "original/jailbreak IDs differ: "
                f"prompt_id={prompt_id}, "
                f"original_source_index={original_explicit_id!r}, "
                f"jailbreak_source_index={jailbreak_explicit_id!r}"
            )

        original_source_index = resolved_source_index(original_row, prompt_id)
        jailbreak_source_index = resolved_source_index(jailbreak_row, prompt_id)
        if original_source_index != jailbreak_source_index:
            raise ValueError(
                "Resolved original/jailbreak source indices differ: "
                f"prompt_id={prompt_id}, "
                f"original_source_index={original_source_index!r}, "
                f"jailbreak_source_index={jailbreak_source_index!r}"
            )
        pairs.append(
            {
                "prompt_id": prompt_id,
                "source_index": original_source_index,
                "original_prompt": original_row["prompt"],
                "jailbreak_prompt": jailbreak_row["prompt"],
            }
        )

    if max_prompts is not None:
        pairs = pairs[:max_prompts]
    if not pairs:
        raise ValueError("No prompt pairs selected")
    return pairs


def extract_element_rows(
    condition,
    prompt_id,
    source_index,
    prompt,
    model,
    tokenizer,
    model_config,
    selected_heads,
    selected_layers,
    tail_tokens,
    system_prompt,
    apply_rotary_pos_emb_fn,
    repeat_kv_fn,
    debug=False,
):
    try:
        with torch.inference_mode():
            templated_prompt = apply_template(
                prompts=[prompt],
                system_prompt=system_prompt,
                tokenizer=tokenizer,
                model_config=model_config,
            )[0]
            inputs = tokenizer(
                templated_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
            seq_len = int(inputs.input_ids.shape[-1])
            if seq_len < tail_tokens:
                raise ValueError(
                    "Sequence shorter than tail window: "
                    f"condition={condition}, prompt_id={prompt_id}, "
                    f"source_index={source_index}, seq_len={seq_len}, "
                    f"tail_tokens={tail_tokens}"
                )

            position_ids = torch.arange(seq_len, device=model.device).unsqueeze(0)
            captured_inputs = capture_self_attn_inputs(model, selected_layers, inputs)

            qk_by_layer = {}
            for layer in sorted(set(selected_layers)):
                self_attn = model.model.layers[layer].self_attn
                qk_by_layer[layer] = compute_post_rope_qk(
                    self_attn=self_attn,
                    hidden_states=captured_inputs[layer],
                    position_ids=position_ids,
                    apply_rotary_pos_emb_fn=apply_rotary_pos_emb_fn,
                    repeat_kv_fn=repeat_kv_fn,
                    model_name=model_config["model_name"],
                )

            output_rows = []
            for layer, head in selected_heads:
                query_states, key_states = qk_by_layer[layer]
                score_matrix = compute_score_matrix(query_states, key_states, head)
                score_window, _, window_size = select_score_window(
                    score_matrix,
                    tail_tokens,
                )
                if window_size != tail_tokens:
                    raise RuntimeError(
                        "Unexpected attention score window size: "
                        f"condition={condition}, prompt_id={prompt_id}, "
                        f"source_index={source_index}, seq_len={seq_len}, "
                        f"tail_tokens={tail_tokens}"
                    )

                asym = (
                    score_window - score_window.transpose(-1, -2)
                ) / 2
                lower_indices = torch.tril_indices(
                    tail_tokens,
                    tail_tokens,
                    offset=-1,
                    device=asym.device,
                )
                asym_lower = asym[
                    lower_indices[0],
                    lower_indices[1],
                ]

                row_positions = lower_indices[0].detach().cpu().tolist()
                col_positions = lower_indices[1].detach().cpu().tolist()
                values = asym_lower.detach().float().cpu().tolist()
                output_rows.extend(
                    {
                        "condition": condition,
                        "prompt_id": prompt_id,
                        "source_index": source_index,
                        "layer": layer,
                        "head": head,
                        "row_pos": row_pos,
                        "col_pos": col_pos,
                        "value": value,
                        "seq_len": seq_len,
                        "window_size": window_size,
                    }
                    for row_pos, col_pos, value in zip(
                        row_positions,
                        col_positions,
                        values,
                    )
                )

            if debug:
                print(
                    "debug: "
                    f"condition={condition}, prompt_id={prompt_id}, "
                    f"source_index={source_index}, seq_len={seq_len}, "
                    f"captured_layers={sorted(captured_inputs)}, "
                    f"rows={len(output_rows)}"
                )
            return output_rows
    except ValueError as exc:
        if str(exc).startswith("Sequence shorter than tail window:"):
            raise
        raise RuntimeError(
            "Failed to process prompt: "
            f"condition={condition}, prompt_id={prompt_id}, "
            f"source_index={source_index}, error_type={type(exc).__name__}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Failed to process prompt: "
            f"condition={condition}, prompt_id={prompt_id}, "
            f"source_index={source_index}, error_type={type(exc).__name__}"
        ) from exc


def element_key(row):
    return tuple(row[column] for column in ELEMENT_KEY_COLUMNS)


def build_delta_rows(element_rows):
    rows_by_condition = {"original": {}, "jailbreak": {}}
    ordered_original_keys = []

    for row in element_rows:
        condition = row["condition"]
        if condition not in rows_by_condition:
            raise ValueError(f"Unexpected condition in element rows: {condition!r}")
        key = element_key(row)
        if key in rows_by_condition[condition]:
            raise ValueError(f"Duplicate element key for condition={condition}")
        rows_by_condition[condition][key] = row
        if condition == "original":
            ordered_original_keys.append(key)

    original_keys = set(rows_by_condition["original"])
    jailbreak_keys = set(rows_by_condition["jailbreak"])
    if original_keys != jailbreak_keys:
        raise ValueError(
            "Original/jailbreak element keys do not match: "
            f"original_only={len(original_keys - jailbreak_keys)}, "
            f"jailbreak_only={len(jailbreak_keys - original_keys)}"
        )
    if len(rows_by_condition["original"]) != len(rows_by_condition["jailbreak"]):
        raise ValueError(
            "Original/jailbreak element counts differ: "
            f"original={len(rows_by_condition['original'])}, "
            f"jailbreak={len(rows_by_condition['jailbreak'])}"
        )

    delta_rows = []
    for key in ordered_original_keys:
        original_value = rows_by_condition["original"][key]["value"]
        jailbreak_value = rows_by_condition["jailbreak"][key]["value"]
        delta_rows.append(
            {
                **dict(zip(ELEMENT_KEY_COLUMNS, key)),
                "original_value": original_value,
                "jailbreak_value": jailbreak_value,
                "delta": jailbreak_value - original_value,
            }
        )
    return delta_rows


def validate_numeric_outputs(
    element_rows,
    delta_rows,
    pairs,
    selected_heads,
    tail_tokens,
):
    elements_per_prompt_head = tail_tokens * (tail_tokens - 1) // 2
    expected_element_rows = (
        len(pairs) * len(selected_heads) * elements_per_prompt_head * 2
    )
    expected_delta_rows = len(pairs) * len(selected_heads) * elements_per_prompt_head
    if len(element_rows) != expected_element_rows:
        raise ValueError(
            "Element row count does not match theory: "
            f"actual={len(element_rows)}, expected={expected_element_rows}"
        )
    if len(delta_rows) != expected_delta_rows:
        raise ValueError(
            "Delta row count does not match theory: "
            f"actual={len(delta_rows)}, expected={expected_delta_rows}"
        )

    element_values = np.asarray([row["value"] for row in element_rows], dtype=np.float64)
    if not np.isfinite(element_values).all():
        raise ValueError("Element values contain NaN or infinity")

    delta_values = np.asarray([row["delta"] for row in delta_rows], dtype=np.float64)
    original_values = np.asarray(
        [row["original_value"] for row in delta_rows],
        dtype=np.float64,
    )
    jailbreak_values = np.asarray(
        [row["jailbreak_value"] for row in delta_rows],
        dtype=np.float64,
    )
    if not (
        np.isfinite(delta_values).all()
        and np.isfinite(original_values).all()
        and np.isfinite(jailbreak_values).all()
    ):
        raise ValueError("Delta CSV values contain NaN or infinity")
    if not np.allclose(delta_values, jailbreak_values - original_values):
        raise ValueError("Delta values do not equal jailbreak_value - original_value")

    if any(row["row_pos"] <= row["col_pos"] for row in element_rows):
        raise ValueError("Element rows contain positions outside the strict lower triangle")
    if any(
        not (
            0 <= row["row_pos"] < tail_tokens
            and 0 <= row["col_pos"] < tail_tokens
        )
        for row in element_rows
    ):
        raise ValueError("Element rows contain positions outside the tail window")

    group_counts = Counter(
        (
            row["condition"],
            row["prompt_id"],
            row["source_index"],
            row["layer"],
            row["head"],
        )
        for row in element_rows
    )
    if len(group_counts) != len(pairs) * len(selected_heads) * 2:
        raise ValueError("Unexpected number of prompt/head/condition groups")
    if any(count != elements_per_prompt_head for count in group_counts.values()):
        raise ValueError(
            "A prompt/head/condition group has an unexpected element count"
        )

    condition_counts = Counter(row["condition"] for row in element_rows)
    if condition_counts["original"] != condition_counts["jailbreak"]:
        raise ValueError(
            "Original/jailbreak element counts differ: "
            f"original={condition_counts['original']}, "
            f"jailbreak={condition_counts['jailbreak']}"
        )

    delta_keys = [element_key(row) for row in delta_rows]
    if len(delta_keys) != len(set(delta_keys)):
        raise ValueError("Delta composite keys are not unique")


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def prompt_texts_from_pairs(pairs):
    return {
        pair[prompt_key]
        for pair in pairs
        for prompt_key in ("original_prompt", "jailbreak_prompt")
        if pair[prompt_key]
    }


def validate_output_row_schemas(element_rows, delta_rows):
    element_columns = set(ELEMENT_COLUMNS)
    delta_columns = set(DELTA_COLUMNS)
    if any(set(row) != element_columns for row in element_rows):
        raise ValueError("Element output rows do not match the prompt-free CSV schema")
    if any(set(row) != delta_columns for row in delta_rows):
        raise ValueError("Delta output rows do not match the prompt-free CSV schema")


def validate_prompt_text_absent(csv_paths, pairs):
    prompt_texts = prompt_texts_from_pairs(pairs)
    for csv_path in csv_paths:
        csv_text = csv_path.read_text(encoding="utf-8")
        if any(prompt_text in csv_text for prompt_text in prompt_texts):
            raise ValueError(f"Prompt text detected in output CSV: {csv_path}")


def histogram_paths(output_dir, layer, head, multiple_heads):
    suffix = f"_layer{layer}_head{head}" if multiple_heads else ""
    return (
        output_dir / f"histogram_original_vs_jailbreak{suffix}.png",
        output_dir / f"histogram_delta{suffix}.png",
    )


def save_histograms(
    delta_rows,
    output_dir,
    selected_heads,
    tail_tokens,
    prompt_count,
    hist_bins,
):
    multiple_heads = len(selected_heads) > 1
    for layer, head in selected_heads:
        head_rows = [
            row
            for row in delta_rows
            if row["layer"] == layer and row["head"] == head
        ]
        original_values = np.asarray(
            [row["original_value"] for row in head_rows],
            dtype=np.float64,
        )
        jailbreak_values = np.asarray(
            [row["jailbreak_value"] for row in head_rows],
            dtype=np.float64,
        )
        delta_values = np.asarray(
            [row["delta"] for row in head_rows],
            dtype=np.float64,
        )
        if not head_rows:
            raise ValueError(f"No delta rows found for layer={layer}, head={head}")

        comparison_path, delta_path = histogram_paths(
            output_dir,
            layer,
            head,
            multiple_heads,
        )
        shared_bin_edges = np.histogram_bin_edges(
            np.concatenate([original_values, jailbreak_values]),
            bins=hist_bins,
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(
            original_values,
            bins=shared_bin_edges,
            density=True,
            alpha=0.55,
            label="original",
        )
        ax.hist(
            jailbreak_values,
            bins=shared_bin_edges,
            density=True,
            alpha=0.55,
            label="jailbreak",
        )
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Antisymmetric strict-lower-triangle value")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Original vs Jailbreak | layer={layer}, head={head} | "
            f"tail_tokens={tail_tokens}, prompts={prompt_count}"
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(comparison_path, dpi=200)
        plt.close(fig)

        max_abs_delta = float(np.max(np.abs(delta_values)))
        if max_abs_delta == 0:
            max_abs_delta = 1.0
        symmetric_bin_edges = np.linspace(
            -max_abs_delta,
            max_abs_delta,
            hist_bins + 1,
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(
            delta_values,
            bins=symmetric_bin_edges,
            density=True,
            alpha=0.75,
            color="#7A5195",
        )
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Delta (jailbreak - original)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Element Delta | layer={layer}, head={head} | "
            f"tail_tokens={tail_tokens}, prompts={prompt_count}"
        )
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(delta_path, dpi=200)
        plt.close(fig)


def print_summary(
    model_name,
    prompt_count,
    selected_heads,
    tail_tokens,
    element_rows,
    delta_rows,
    output_dir,
):
    original_values = np.asarray(
        [row["original_value"] for row in delta_rows],
        dtype=np.float64,
    )
    jailbreak_values = np.asarray(
        [row["jailbreak_value"] for row in delta_rows],
        dtype=np.float64,
    )
    delta_values = np.asarray(
        [row["delta"] for row in delta_rows],
        dtype=np.float64,
    )
    selected_heads_label = ";".join(
        f"{layer},{head}" for layer, head in selected_heads
    )
    elements_per_prompt_head = tail_tokens * (tail_tokens - 1) // 2

    print(f"model name: {model_name}")
    print(f"processed prompt pairs: {prompt_count}")
    print(f"selected heads: {selected_heads_label}")
    print(f"tail_tokens: {tail_tokens}")
    print(f"elements per prompt/head: {elements_per_prompt_head}")
    print(f"element CSV rows: {len(element_rows)}")
    print(f"delta CSV rows: {len(delta_rows)}")
    print(f"original mean: {original_values.mean():.10g}")
    print(f"original median: {np.median(original_values):.10g}")
    print(f"jailbreak mean: {jailbreak_values.mean():.10g}")
    print(f"jailbreak median: {np.median(jailbreak_values):.10g}")
    print(f"delta mean: {delta_values.mean():.10g}")
    print(f"delta median: {np.median(delta_values):.10g}")
    print(f"delta positive rate: {(delta_values > 0).mean():.10g}")
    print(f"output directory: {output_dir}")


def main():
    args = get_args()
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("--max_prompts must be positive when specified")
    if args.tail_tokens < 2:
        raise ValueError("--tail_tokens must be at least 2")
    if args.hist_bins <= 0:
        raise ValueError("--hist_bins must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = prepare_prompt_pairs(
        args.original_prompts,
        args.jailbreak_prompts,
        args.max_prompts,
    )

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    model.eval()
    selected_heads = parse_heads(args.heads, model_config)
    if len(selected_heads) != len(set(selected_heads)):
        raise ValueError("--heads contains duplicate layer,head pairs")
    selected_layers = [layer for layer, _ in selected_heads]
    apply_rotary_pos_emb_fn, repeat_kv_fn = get_attention_helpers(
        model_config["model_name"]
    )

    element_rows = []
    for pair_index, pair in enumerate(tqdm(pairs, desc="prompt pairs")):
        element_rows.extend(
            extract_element_rows(
                condition="original",
                prompt_id=pair["prompt_id"],
                source_index=pair["source_index"],
                prompt=pair["original_prompt"],
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                selected_heads=selected_heads,
                selected_layers=selected_layers,
                tail_tokens=args.tail_tokens,
                system_prompt=args.system_prompt,
                apply_rotary_pos_emb_fn=apply_rotary_pos_emb_fn,
                repeat_kv_fn=repeat_kv_fn,
                debug=args.debug and pair_index == 0,
            )
        )
        element_rows.extend(
            extract_element_rows(
                condition="jailbreak",
                prompt_id=pair["prompt_id"],
                source_index=pair["source_index"],
                prompt=pair["jailbreak_prompt"],
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                selected_heads=selected_heads,
                selected_layers=selected_layers,
                tail_tokens=args.tail_tokens,
                system_prompt=args.system_prompt,
                apply_rotary_pos_emb_fn=apply_rotary_pos_emb_fn,
                repeat_kv_fn=repeat_kv_fn,
                debug=args.debug and pair_index == 0,
            )
        )

    delta_rows = build_delta_rows(element_rows)
    validate_numeric_outputs(
        element_rows=element_rows,
        delta_rows=delta_rows,
        pairs=pairs,
        selected_heads=selected_heads,
        tail_tokens=args.tail_tokens,
    )
    validate_output_row_schemas(element_rows, delta_rows)

    element_csv_path = output_dir / "element_values_selected_heads.csv"
    delta_csv_path = output_dir / "delta_element_values_selected_heads.csv"
    write_csv(element_csv_path, ELEMENT_COLUMNS, element_rows)
    write_csv(delta_csv_path, DELTA_COLUMNS, delta_rows)
    validate_prompt_text_absent([element_csv_path, delta_csv_path], pairs)
    save_histograms(
        delta_rows=delta_rows,
        output_dir=output_dir,
        selected_heads=selected_heads,
        tail_tokens=args.tail_tokens,
        prompt_count=len(pairs),
        hist_bins=args.hist_bins,
    )
    print_summary(
        model_name=model_config["model_name"],
        prompt_count=len(pairs),
        selected_heads=selected_heads,
        tail_tokens=args.tail_tokens,
        element_rows=element_rows,
        delta_rows=delta_rows,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
