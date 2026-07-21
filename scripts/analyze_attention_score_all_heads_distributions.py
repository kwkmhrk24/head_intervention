import argparse
import csv
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import transformers
from numpy.lib.format import open_memmap
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_attention_scores import (
    capture_self_attn_inputs,
    compute_post_rope_qk,
    compute_score_matrix,
    get_attention_helpers,
    read_prompt_rows,
    select_score_window,
)
from utils.model_utils import load_model_and_tokenizer
from utils.prompt_utils import apply_template


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

RAW_TARGET_FILENAMES = [
    *RAW_ARRAY_FILENAMES.values(),
    "prompt_manifest.csv",
    "metadata.json",
    "raw_distribution_summary.csv",
    "per_head_summary.csv",
]

PROMPT_MANIFEST_COLUMNS = [
    "raw_prompt_index",
    "input_row_index",
    "prompt_id",
    "source_index",
    "original_source_index",
    "jailbreak_source_index",
    "original_seq_len",
    "jailbreak_seq_len",
    "tail_tokens",
    "included",
    "exclusion_reason",
]

MEAN_TARGET_FILENAMES = [
    *MEAN_ARRAY_FILENAMES.values(),
    "element_values_all_heads_mean.csv",
    "delta_element_values_all_heads_mean.csv",
    "all_heads_mean_distribution_summary.csv",
    "prompt_level_all_heads_mean_summary.csv",
    "metadata.json",
]

DISTRIBUTION_SUMMARY_COLUMNS = [
    "condition",
    "count",
    "mean",
    "std",
    "min",
    "q01",
    "q05",
    "q25",
    "median",
    "q75",
    "q95",
    "q99",
    "max",
    "positive_rate",
    "negative_rate",
    "zero_rate",
]

PER_HEAD_SUMMARY_COLUMNS = [
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
]

MEAN_ELEMENT_COLUMNS = [
    "condition",
    "prompt_id",
    "source_index",
    "row_pos",
    "col_pos",
    "value",
    "layer_head_count",
    "window_size",
]

MEAN_DELTA_COLUMNS = [
    "prompt_id",
    "source_index",
    "row_pos",
    "col_pos",
    "original_value",
    "jailbreak_value",
    "delta",
    "layer_head_count",
    "window_size",
]

PROMPT_SUMMARY_COLUMNS = [
    "prompt_id",
    "source_index",
    "original_mean",
    "jailbreak_mean",
    "delta_mean",
    "original_median",
    "jailbreak_median",
    "delta_median",
    "delta_positive_rate",
    "original_std",
    "jailbreak_std",
    "delta_std",
    "original_iqr",
    "jailbreak_iqr",
    "delta_iqr",
    "element_count",
]

SUMMARY_QUANTILES = np.asarray([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract strict-lower antisymmetric attention score values for all "
            "layers and heads, and compute their unnormalized all-head mean."
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
        "--tail_tokens",
        required=True,
        type=int,
        help="Use exactly the final N tokens from each prompt independently.",
    )
    parser.add_argument(
        "--raw_output_dir",
        required=True,
        help="Directory for all-head raw NPY arrays and summaries.",
    )
    parser.add_argument(
        "--mean_output_dir",
        required=True,
        help="Directory for all-head mean NPY/CSV outputs and summaries.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Optional number of prompt pairs to process.",
    )
    parser.add_argument(
        "--short_sequence_policy",
        choices=("error", "exclude_pair"),
        default="error",
        help=(
            "How to handle a pair when either sequence is shorter than "
            "--tail_tokens (default: error)."
        ),
    )
    parser.add_argument(
        "--expected_included_prompts",
        type=int,
        default=None,
        help="Optional exact included-pair count required after length filtering.",
    )
    parser.add_argument(
        "--attack_name",
        default=None,
        help="Attack label stored in metadata; inferred from the prompt path if omitted.",
    )
    parser.add_argument(
        "--system_prompt",
        default=None,
        help="Optional system prompt passed to the existing chat-template helper.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of this script's existing target files.",
    )
    return parser.parse_args()


def prepare_line_indexed_prompt_pairs(
    original_path: str,
    jailbreak_path: str,
    max_prompts: int | None,
) -> List[Dict[str, Any]]:
    original_rows = read_prompt_rows(original_path)
    jailbreak_rows = read_prompt_rows(jailbreak_path)
    if len(original_rows) != len(jailbreak_rows):
        raise ValueError(
            "original/jailbreak prompt row counts differ: "
            f"original={len(original_rows)}, jailbreak={len(jailbreak_rows)}"
        )
    if not original_rows:
        raise ValueError("Prompt files contain no prompt rows")

    pairs: List[Dict[str, Any]] = []
    for prompt_id, (original_row, jailbreak_row) in enumerate(
        zip(original_rows, jailbreak_rows)
    ):
        if not isinstance(original_row.get("prompt"), str):
            raise ValueError(
                f"original prompt must be a string: prompt_id={prompt_id}, "
                f"input_file={original_path}"
            )
        if not isinstance(jailbreak_row.get("prompt"), str):
            raise ValueError(
                f"jailbreak prompt must be a string: prompt_id={prompt_id}, "
                f"input_file={jailbreak_path}"
            )
        original_source_index = original_row.get(
            "source_index", original_row.get("index", prompt_id)
        )
        jailbreak_source_index = jailbreak_row.get(
            "source_index", jailbreak_row.get("index", prompt_id)
        )
        if original_source_index != jailbreak_source_index:
            raise ValueError(
                "original/jailbreak source_index mismatch: "
                f"prompt_id={prompt_id}, "
                f"original_source_index={original_source_index}, "
                f"jailbreak_source_index={jailbreak_source_index}"
            )
        pairs.append(
            {
                "prompt_id": prompt_id,
                "input_row_index": prompt_id,
                "source_index": original_source_index,
                "original_source_index": original_source_index,
                "jailbreak_source_index": jailbreak_source_index,
                "original_prompt": original_row["prompt"],
                "jailbreak_prompt": jailbreak_row["prompt"],
            }
        )

    if max_prompts is not None:
        pairs = pairs[:max_prompts]
    if not pairs:
        raise ValueError("No prompt pairs selected")
    source_indices = [pair["source_index"] for pair in pairs]
    try:
        duplicate_count = len(source_indices) - len(set(source_indices))
    except TypeError as exc:
        raise ValueError("source_index values must be scalar/hashable") from exc
    if duplicate_count:
        raise ValueError(f"Duplicate source_index values: count={duplicate_count}")
    return pairs


def apply_short_sequence_policy(
    pairs: Sequence[Dict[str, Any]],
    original_seq_lengths: Sequence[int],
    jailbreak_seq_lengths: Sequence[int],
    tail_tokens: int,
    policy: str,
    expected_included_prompts: int | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if policy not in {"error", "exclude_pair"}:
        raise ValueError(f"Unsupported short sequence policy: {policy}")
    if not (
        len(pairs) == len(original_seq_lengths) == len(jailbreak_seq_lengths)
    ):
        raise ValueError("Prompt pairs and sequence-length arrays must have equal length")

    included_pairs: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for pair, original_seq_len, jailbreak_seq_len in zip(
        pairs, original_seq_lengths, jailbreak_seq_lengths
    ):
        original_seq_len = int(original_seq_len)
        jailbreak_seq_len = int(jailbreak_seq_len)
        original_short = original_seq_len < tail_tokens
        jailbreak_short = jailbreak_seq_len < tail_tokens
        if original_short or jailbreak_short:
            if original_short and jailbreak_short:
                reason = "original_and_jailbreak_shorter_than_tail_tokens"
            elif original_short:
                reason = "original_shorter_than_tail_tokens"
            else:
                reason = "jailbreak_shorter_than_tail_tokens"
            if policy == "error":
                condition = "original" if original_short else "jailbreak"
                seq_len = original_seq_len if original_short else jailbreak_seq_len
                require_tail_window(
                    condition=condition,
                    input_file="paired_prompt_inputs",
                    prompt_id=int(pair["prompt_id"]),
                    source_index=pair["source_index"],
                    seq_len=seq_len,
                    tail_tokens=tail_tokens,
                )
            raw_prompt_index: int | str = ""
            included = False
        else:
            reason = ""
            raw_prompt_index = len(included_pairs)
            included = True
            included_pair = dict(pair)
            included_pair.update(
                {
                    "raw_prompt_index": raw_prompt_index,
                    "original_seq_len": original_seq_len,
                    "jailbreak_seq_len": jailbreak_seq_len,
                }
            )
            included_pairs.append(included_pair)

        manifest_rows.append(
            {
                "raw_prompt_index": raw_prompt_index,
                "input_row_index": int(pair["input_row_index"]),
                "prompt_id": int(pair["prompt_id"]),
                "source_index": pair["source_index"],
                "original_source_index": pair["original_source_index"],
                "jailbreak_source_index": pair["jailbreak_source_index"],
                "original_seq_len": original_seq_len,
                "jailbreak_seq_len": jailbreak_seq_len,
                "tail_tokens": tail_tokens,
                "included": included,
                "exclusion_reason": reason,
            }
        )

    if not included_pairs:
        raise ValueError("No prompt pairs remain after short-sequence filtering")
    if (
        expected_included_prompts is not None
        and len(included_pairs) != expected_included_prompts
    ):
        raise ValueError(
            "Included prompt-pair count differs from expectation: "
            f"actual={len(included_pairs)}, expected={expected_included_prompts}"
        )
    return included_pairs, manifest_rows


def measure_pair_sequence_lengths(
    pairs: Sequence[Dict[str, Any]],
    tokenizer: Any,
    model_config: Dict[str, Any],
    system_prompt: str | None,
) -> Tuple[List[int], List[int]]:
    lengths: Dict[str, List[int]] = {"original": [], "jailbreak": []}
    for pair in tqdm(pairs, desc="tokenizing prompt pairs"):
        for condition in ("original", "jailbreak"):
            templated_prompt = apply_template(
                prompts=[pair[f"{condition}_prompt"]],
                system_prompt=system_prompt,
                tokenizer=tokenizer,
                model_config=model_config,
            )[0]
            encoded = tokenizer(
                templated_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            lengths[condition].append(int(encoded.input_ids.shape[-1]))
    return lengths["original"], lengths["jailbreak"]


def write_prompt_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    write_dict_rows(path, PROMPT_MANIFEST_COLUMNS, rows)


def validate_prompt_axis(
    included_pairs: Sequence[Dict[str, Any]],
    manifest_rows: Sequence[Dict[str, Any]],
    raw_prompt_count: int,
) -> None:
    included_manifest = [row for row in manifest_rows if row["included"]]
    if (
        len(included_pairs) != len(included_manifest)
        or len(included_pairs) != raw_prompt_count
    ):
        raise ValueError(
            "Included pair/manifest/raw prompt counts differ: "
            f"pairs={len(included_pairs)}, manifest={len(included_manifest)}, "
            f"raw={raw_prompt_count}"
        )
    raw_indices = [int(row["raw_prompt_index"]) for row in included_manifest]
    if raw_indices != list(range(raw_prompt_count)):
        raise ValueError("Manifest raw_prompt_index must be contiguous from zero")
    source_indices = [row["source_index"] for row in included_manifest]
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("Included manifest source_index values are not unique")


def strict_lower_positions(window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    rows, columns = np.tril_indices(window_size, k=-1)
    return rows.astype(np.int64), columns.astype(np.int64)


def require_tail_window(
    condition: str,
    input_file: str,
    prompt_id: int,
    source_index: int,
    seq_len: int,
    tail_tokens: int,
) -> None:
    if seq_len < tail_tokens:
        raise ValueError(
            "Sequence shorter than tail window: "
            f"condition={condition}, input_file={input_file}, "
            f"prompt_id={prompt_id}, source_index={source_index}, "
            f"seq_len={seq_len}, tail_tokens={tail_tokens}"
        )


def validate_output_directories(
    raw_output_dir: Path,
    mean_output_dir: Path,
    overwrite: bool,
) -> None:
    if raw_output_dir.resolve() == mean_output_dir.resolve():
        raise ValueError("--raw_output_dir and --mean_output_dir must be different")

    for output_dir, target_names in (
        (raw_output_dir, RAW_TARGET_FILENAMES),
        (mean_output_dir, MEAN_TARGET_FILENAMES),
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = [name for name in target_names if (output_dir / name).exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"Target files already exist in {output_dir}; "
                "use --overwrite to replace them: "
                f"{existing}"
            )
        try:
            with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".write_test_"):
                pass
        except OSError as exc:
            raise OSError(f"Output directory is not writable: {output_dir}") from exc


def mean_over_layer_heads(values: np.ndarray) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError(
            "Expected values with axes (layer, head, position); "
            f"received shape={values.shape}"
        )
    return np.mean(values, axis=(0, 1), dtype=np.float64)


def compute_mean_distributions(
    raw_original: np.ndarray,
    raw_jailbreak: np.ndarray,
    raw_delta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (
        raw_original.shape == raw_jailbreak.shape == raw_delta.shape
        and raw_original.ndim == 3
    ):
        raise ValueError("Raw prompt arrays must share shape (layer, head, position)")
    original_mean = mean_over_layer_heads(raw_original)
    jailbreak_mean = mean_over_layer_heads(raw_jailbreak)
    delta_from_means = jailbreak_mean - original_mean
    delta_mean = mean_over_layer_heads(raw_delta)
    if not np.allclose(delta_from_means, delta_mean, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "All-head mean delta identity failed: mean(jailbreak-original) "
            "!= mean(jailbreak)-mean(original)"
        )
    return original_mean, jailbreak_mean, delta_from_means


def extract_condition_into_memmap(
    *,
    condition: str,
    input_file: str,
    prompt_id: int,
    source_index: int,
    prompt: str,
    model: Any,
    tokenizer: Any,
    model_config: Dict[str, Any],
    system_prompt: str | None,
    tail_tokens: int,
    layers: Sequence[int],
    num_attention_heads: int,
    lower_indices: torch.Tensor,
    destination: np.ndarray,
    apply_rotary_pos_emb_fn: Any,
    repeat_kv_fn: Any,
) -> Tuple[int, np.ndarray]:
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
            require_tail_window(
                condition=condition,
                input_file=input_file,
                prompt_id=prompt_id,
                source_index=source_index,
                seq_len=seq_len,
                tail_tokens=tail_tokens,
            )
            position_ids = torch.arange(seq_len, device=model.device).unsqueeze(0)
            captured_inputs = capture_self_attn_inputs(model, layers, inputs)
            accumulator = np.zeros(lower_indices.shape[1], dtype=np.float64)

            for layer in layers:
                self_attn = model.model.layers[layer].self_attn
                hidden_states = captured_inputs.pop(layer)
                query_states, key_states = compute_post_rope_qk(
                    self_attn=self_attn,
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    apply_rotary_pos_emb_fn=apply_rotary_pos_emb_fn,
                    repeat_kv_fn=repeat_kv_fn,
                    model_name=model_config["model_name"],
                )
                if query_states.shape[1] != num_attention_heads:
                    raise RuntimeError(
                        "Unexpected attention-head count: "
                        f"layer={layer}, tensor_heads={query_states.shape[1]}, "
                        f"configured_heads={num_attention_heads}"
                    )

                for head in range(num_attention_heads):
                    score_matrix = compute_score_matrix(query_states, key_states, head)
                    score_window, _, window_size = select_score_window(
                        score_matrix,
                        tail_tokens,
                    )
                    if window_size != tail_tokens:
                        raise RuntimeError(
                            "Unexpected attention score window size: "
                            f"condition={condition}, input_file={input_file}, "
                            f"prompt_id={prompt_id}, source_index={source_index}, "
                            f"seq_len={seq_len}, tail_tokens={tail_tokens}"
                        )
                    asym = (
                        score_window - score_window.transpose(-1, -2)
                    ) / 2
                    asym_lower = asym[lower_indices[0], lower_indices[1]]
                    if not bool(torch.isfinite(asym_lower).all().item()):
                        raise ValueError(
                            "Non-finite attention values: "
                            f"condition={condition}, input_file={input_file}, "
                            f"prompt_id={prompt_id}, source_index={source_index}, "
                            f"layer={layer}, head={head}"
                        )
                    values = asym_lower.detach().float().cpu().numpy()
                    destination[layer, head, :] = values
                    accumulator += values
                    del score_matrix, score_window, asym, asym_lower, values

                del hidden_states, query_states, key_states

            if captured_inputs:
                raise RuntimeError(
                    f"Unconsumed captured layers: {sorted(captured_inputs)}"
                )
            layer_head_count = len(layers) * num_attention_heads
            return seq_len, accumulator / layer_head_count
    except ValueError as exc:
        if str(exc).startswith("Sequence shorter than tail window:"):
            raise
        raise RuntimeError(
            "Failed to process prompt: "
            f"condition={condition}, input_file={input_file}, "
            f"prompt_id={prompt_id}, source_index={source_index}, "
            f"error_type={type(exc).__name__}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Failed to process prompt: "
            f"condition={condition}, input_file={input_file}, "
            f"prompt_id={prompt_id}, source_index={source_index}, "
            f"error_type={type(exc).__name__}"
        ) from exc


def summarize_distribution(condition: str, values: np.ndarray) -> Dict[str, Any]:
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        raise ValueError(f"Cannot summarize empty distribution: {condition}")
    if not np.isfinite(flat).all():
        raise ValueError(f"Distribution contains NaN or infinity: {condition}")
    quantiles = np.quantile(flat, SUMMARY_QUANTILES)
    return {
        "condition": condition,
        "count": int(flat.size),
        "mean": float(np.mean(flat, dtype=np.float64)),
        "std": float(np.std(flat, dtype=np.float64)),
        "min": float(np.min(flat)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "max": float(np.max(flat)),
        "positive_rate": float(np.mean(flat > 0)),
        "negative_rate": float(np.mean(flat < 0)),
        "zero_rate": float(np.mean(flat == 0)),
    }


def validate_finite_array(name: str, values: np.ndarray) -> None:
    for prompt_index in range(values.shape[0]):
        if not np.isfinite(values[prompt_index]).all():
            raise ValueError(
                f"Output array contains NaN or infinity: name={name}, "
                f"prompt_index={prompt_index}"
            )


def write_dict_rows(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_distribution_summary(
    path: Path,
    original: np.ndarray,
    jailbreak: np.ndarray,
    delta: np.ndarray,
) -> None:
    rows = [
        summarize_distribution("original", original),
        summarize_distribution("jailbreak", jailbreak),
        summarize_distribution("delta", delta),
    ]
    write_dict_rows(path, DISTRIBUTION_SUMMARY_COLUMNS, rows)


def write_per_head_summary(
    path: Path,
    raw_original: np.ndarray,
    raw_jailbreak: np.ndarray,
    raw_delta: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_HEAD_SUMMARY_COLUMNS)
        writer.writeheader()
        for layer in range(raw_original.shape[1]):
            for head in range(raw_original.shape[2]):
                original = np.asarray(raw_original[:, layer, head, :]).reshape(-1)
                jailbreak = np.asarray(raw_jailbreak[:, layer, head, :]).reshape(-1)
                delta = np.asarray(raw_delta[:, layer, head, :]).reshape(-1)
                writer.writerow(
                    {
                        "layer": layer,
                        "head": head,
                        "value_count": int(original.size),
                        "original_mean": float(np.mean(original, dtype=np.float64)),
                        "jailbreak_mean": float(np.mean(jailbreak, dtype=np.float64)),
                        "delta_mean": float(np.mean(delta, dtype=np.float64)),
                        "original_std": float(np.std(original, dtype=np.float64)),
                        "jailbreak_std": float(np.std(jailbreak, dtype=np.float64)),
                        "delta_std": float(np.std(delta, dtype=np.float64)),
                        "original_positive_rate": float(np.mean(original > 0)),
                        "jailbreak_positive_rate": float(np.mean(jailbreak > 0)),
                        "delta_positive_rate": float(np.mean(delta > 0)),
                    }
                )


def write_mean_element_csvs(
    mean_output_dir: Path,
    mean_original: np.ndarray,
    mean_jailbreak: np.ndarray,
    mean_delta: np.ndarray,
    row_positions: np.ndarray,
    col_positions: np.ndarray,
    layer_head_count: int,
    window_size: int,
    included_pairs: Sequence[Dict[str, Any]] | None = None,
) -> None:
    element_path = mean_output_dir / "element_values_all_heads_mean.csv"
    delta_path = mean_output_dir / "delta_element_values_all_heads_mean.csv"
    with element_path.open("w", newline="", encoding="utf-8") as element_file:
        writer = csv.DictWriter(element_file, fieldnames=MEAN_ELEMENT_COLUMNS)
        writer.writeheader()
        for condition, values in (
            ("original", mean_original),
            ("jailbreak", mean_jailbreak),
        ):
            for raw_prompt_index in range(values.shape[0]):
                pair = (
                    included_pairs[raw_prompt_index]
                    if included_pairs is not None
                    else {
                        "prompt_id": raw_prompt_index,
                        "source_index": raw_prompt_index,
                    }
                )
                for position_index, (row_pos, col_pos) in enumerate(
                    zip(row_positions, col_positions)
                ):
                    writer.writerow(
                        {
                            "condition": condition,
                            "prompt_id": pair["prompt_id"],
                            "source_index": pair["source_index"],
                            "row_pos": int(row_pos),
                            "col_pos": int(col_pos),
                            "value": float(values[raw_prompt_index, position_index]),
                            "layer_head_count": layer_head_count,
                            "window_size": window_size,
                        }
                    )

    with delta_path.open("w", newline="", encoding="utf-8") as delta_file:
        writer = csv.DictWriter(delta_file, fieldnames=MEAN_DELTA_COLUMNS)
        writer.writeheader()
        for raw_prompt_index in range(mean_delta.shape[0]):
            pair = (
                included_pairs[raw_prompt_index]
                if included_pairs is not None
                else {
                    "prompt_id": raw_prompt_index,
                    "source_index": raw_prompt_index,
                }
            )
            for position_index, (row_pos, col_pos) in enumerate(
                zip(row_positions, col_positions)
            ):
                writer.writerow(
                    {
                        "prompt_id": pair["prompt_id"],
                        "source_index": pair["source_index"],
                        "row_pos": int(row_pos),
                        "col_pos": int(col_pos),
                        "original_value": float(mean_original[raw_prompt_index, position_index]),
                        "jailbreak_value": float(mean_jailbreak[raw_prompt_index, position_index]),
                        "delta": float(mean_delta[raw_prompt_index, position_index]),
                        "layer_head_count": layer_head_count,
                        "window_size": window_size,
                    }
                )


def interquartile_range(values: np.ndarray) -> float:
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return float(q75 - q25)


def write_prompt_level_summary(
    path: Path,
    mean_original: np.ndarray,
    mean_jailbreak: np.ndarray,
    mean_delta: np.ndarray,
    included_pairs: Sequence[Dict[str, Any]] | None = None,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROMPT_SUMMARY_COLUMNS)
        writer.writeheader()
        for raw_prompt_index in range(mean_original.shape[0]):
            pair = (
                included_pairs[raw_prompt_index]
                if included_pairs is not None
                else {
                    "prompt_id": raw_prompt_index,
                    "source_index": raw_prompt_index,
                }
            )
            original = np.asarray(mean_original[raw_prompt_index])
            jailbreak = np.asarray(mean_jailbreak[raw_prompt_index])
            delta = np.asarray(mean_delta[raw_prompt_index])
            writer.writerow(
                {
                    "prompt_id": pair["prompt_id"],
                    "source_index": pair["source_index"],
                    "original_mean": float(np.mean(original, dtype=np.float64)),
                    "jailbreak_mean": float(np.mean(jailbreak, dtype=np.float64)),
                    "delta_mean": float(np.mean(delta, dtype=np.float64)),
                    "original_median": float(np.median(original)),
                    "jailbreak_median": float(np.median(jailbreak)),
                    "delta_median": float(np.median(delta)),
                    "delta_positive_rate": float(np.mean(delta > 0)),
                    "original_std": float(np.std(original, dtype=np.float64)),
                    "jailbreak_std": float(np.std(jailbreak, dtype=np.float64)),
                    "delta_std": float(np.std(delta, dtype=np.float64)),
                    "original_iqr": interquartile_range(original),
                    "jailbreak_iqr": interquartile_range(jailbreak),
                    "delta_iqr": interquartile_range(delta),
                    "element_count": int(original.size),
                }
            )


def write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_common_metadata(
    *,
    args: argparse.Namespace,
    model_config: Dict[str, Any],
    input_pair_count: int,
    included_pairs: Sequence[Dict[str, Any]],
    manifest_rows: Sequence[Dict[str, Any]],
    tail_tokens: int,
    elements_per_head: int,
    num_layers: int,
    num_attention_heads: int,
    row_positions: np.ndarray,
    col_positions: np.ndarray,
    execution_datetime: str,
    tokenizer: Any,
    model: Any,
    source_commit: str | None,
) -> Dict[str, Any]:
    included_pair_count = len(included_pairs)
    excluded_rows = [row for row in manifest_rows if not row["included"]]
    model_config_object = getattr(model, "config", None)
    num_key_value_heads = getattr(
        model_config_object,
        "num_key_value_heads",
        num_attention_heads,
    )
    head_dim = getattr(model_config_object, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(model_config_object, "hidden_size", None)
        if hidden_size is not None:
            head_dim = int(hidden_size) // num_attention_heads
    attack_name = args.attack_name
    if attack_name is None:
        attack_name = Path(args.jailbreak_prompts).parent.name or None
    return {
        "model": args.model,
        "model_path": args.model,
        "model_name": model_config["model_name"],
        "original_prompts": args.original_prompts,
        "jailbreak_prompts": args.jailbreak_prompts,
        "original_prompts_path": args.original_prompts,
        "jailbreak_prompts_path": args.jailbreak_prompts,
        "attack_name": attack_name,
        "input_pair_count": input_pair_count,
        "included_pair_count": included_pair_count,
        "excluded_pair_count": len(excluded_rows),
        "excluded_source_indices": [row["source_index"] for row in excluded_rows],
        "short_sequence_policy": args.short_sequence_policy,
        "processed_prompt_count": included_pair_count,
        "tail_tokens": tail_tokens,
        "window_size": tail_tokens,
        "elements_per_head": elements_per_head,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": int(num_key_value_heads),
        "head_dim": int(head_dim) if head_dim is not None else None,
        "gqa": int(num_key_value_heads) != num_attention_heads,
        "repeat_kv": int(num_key_value_heads) != num_attention_heads,
        "layer_head_count": num_layers * num_attention_heads,
        "dtype": "float32",
        "score_definition": "S = QK^T / sqrt(d_head)",
        "post_rope": True,
        "pre_mask": True,
        "pre_softmax": True,
        "strict_lower_triangle": "row_pos > col_pos",
        "strict_lower_position_order": "row-major order",
        "strict_lower_row_positions": row_positions.tolist(),
        "strict_lower_col_positions": col_positions.tolist(),
        "prompt_id_definition": "zero-based JSONL input row index",
        "source_index_definition": (
            "source_index, then index from input JSONL; otherwise input row index"
        ),
        "delta_definition": "jailbreak - original at paired prompt/layer/head/position",
        "prompt_manifest_path": "prompt_manifest.csv",
        "system_prompt": args.system_prompt,
        "chat_template": {
            "identifier": "tokenizer.chat_template",
            "tokenizer_class": type(tokenizer).__name__,
            "defined": getattr(tokenizer, "chat_template", None) is not None,
        },
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "source_commit": source_commit,
        "execution_datetime_utc": execution_datetime,
        "cli_arguments": vars(args),
    }


def get_source_commit(repo_root: Path = REPO_ROOT) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = get_args()
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("--max_prompts must be positive when specified")
    if args.tail_tokens < 2:
        raise ValueError("--tail_tokens must be at least 2")
    if (
        args.expected_included_prompts is not None
        and args.expected_included_prompts <= 0
    ):
        raise ValueError("--expected_included_prompts must be positive when specified")

    raw_output_dir = Path(args.raw_output_dir)
    mean_output_dir = Path(args.mean_output_dir)
    validate_output_directories(raw_output_dir, mean_output_dir, args.overwrite)
    input_pairs = prepare_line_indexed_prompt_pairs(
        args.original_prompts,
        args.jailbreak_prompts,
        args.max_prompts,
    )

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    model.eval()
    num_layers = int(model_config["n_layers"])
    num_attention_heads = int(model_config["n_heads"])
    layers = list(range(num_layers))
    layer_head_count = num_layers * num_attention_heads
    apply_rotary_pos_emb_fn, repeat_kv_fn = get_attention_helpers(
        model_config["model_name"]
    )

    original_seq_lengths, jailbreak_seq_lengths = measure_pair_sequence_lengths(
        input_pairs,
        tokenizer,
        model_config,
        args.system_prompt,
    )
    pairs, manifest_rows = apply_short_sequence_policy(
        input_pairs,
        original_seq_lengths,
        jailbreak_seq_lengths,
        args.tail_tokens,
        args.short_sequence_policy,
        args.expected_included_prompts,
    )
    validate_prompt_axis(pairs, manifest_rows, len(pairs))
    write_prompt_manifest(raw_output_dir / "prompt_manifest.csv", manifest_rows)

    row_positions, col_positions = strict_lower_positions(args.tail_tokens)
    elements_per_head = len(row_positions)
    expected_elements = args.tail_tokens * (args.tail_tokens - 1) // 2
    if elements_per_head != expected_elements:
        raise RuntimeError(
            "Strict-lower element count mismatch: "
            f"actual={elements_per_head}, expected={expected_elements}"
        )
    lower_indices = torch.tril_indices(
        args.tail_tokens,
        args.tail_tokens,
        offset=-1,
        device=model.device,
    )

    raw_shape = (
        len(pairs),
        num_layers,
        num_attention_heads,
        elements_per_head,
    )
    mean_shape = (len(pairs), elements_per_head)
    raw_original = open_memmap(
        raw_output_dir / RAW_ARRAY_FILENAMES["original"],
        mode="w+",
        dtype=np.float32,
        shape=raw_shape,
    )
    raw_jailbreak = open_memmap(
        raw_output_dir / RAW_ARRAY_FILENAMES["jailbreak"],
        mode="w+",
        dtype=np.float32,
        shape=raw_shape,
    )
    raw_delta = open_memmap(
        raw_output_dir / RAW_ARRAY_FILENAMES["delta"],
        mode="w+",
        dtype=np.float32,
        shape=raw_shape,
    )
    mean_original = open_memmap(
        mean_output_dir / MEAN_ARRAY_FILENAMES["original"],
        mode="w+",
        dtype=np.float32,
        shape=mean_shape,
    )
    mean_jailbreak = open_memmap(
        mean_output_dir / MEAN_ARRAY_FILENAMES["jailbreak"],
        mode="w+",
        dtype=np.float32,
        shape=mean_shape,
    )
    mean_delta = open_memmap(
        mean_output_dir / MEAN_ARRAY_FILENAMES["delta"],
        mode="w+",
        dtype=np.float32,
        shape=mean_shape,
    )

    for prompt_index, pair in enumerate(tqdm(pairs, desc="prompt pairs")):
        original_seq_len, accumulated_original_mean = extract_condition_into_memmap(
            condition="original",
            input_file=args.original_prompts,
            prompt_id=pair["prompt_id"],
            source_index=pair["source_index"],
            prompt=pair["original_prompt"],
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            system_prompt=args.system_prompt,
            tail_tokens=args.tail_tokens,
            layers=layers,
            num_attention_heads=num_attention_heads,
            lower_indices=lower_indices,
            destination=raw_original[prompt_index],
            apply_rotary_pos_emb_fn=apply_rotary_pos_emb_fn,
            repeat_kv_fn=repeat_kv_fn,
        )
        jailbreak_seq_len, accumulated_jailbreak_mean = extract_condition_into_memmap(
            condition="jailbreak",
            input_file=args.jailbreak_prompts,
            prompt_id=pair["prompt_id"],
            source_index=pair["source_index"],
            prompt=pair["jailbreak_prompt"],
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            system_prompt=args.system_prompt,
            tail_tokens=args.tail_tokens,
            layers=layers,
            num_attention_heads=num_attention_heads,
            lower_indices=lower_indices,
            destination=raw_jailbreak[prompt_index],
            apply_rotary_pos_emb_fn=apply_rotary_pos_emb_fn,
            repeat_kv_fn=repeat_kv_fn,
        )
        if (
            original_seq_len != pair["original_seq_len"]
            or jailbreak_seq_len != pair["jailbreak_seq_len"]
        ):
            raise ValueError(
                "Sequence length changed after prompt filtering: "
                f"prompt_id={pair['prompt_id']}, source_index={pair['source_index']}, "
                f"original={original_seq_len}/{pair['original_seq_len']}, "
                f"jailbreak={jailbreak_seq_len}/{pair['jailbreak_seq_len']}"
            )
        raw_delta[prompt_index] = (
            raw_jailbreak[prompt_index] - raw_original[prompt_index]
        )
        if not np.allclose(
            raw_delta[prompt_index],
            raw_jailbreak[prompt_index] - raw_original[prompt_index],
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(
                f"Raw delta mismatch: prompt_id={pair['prompt_id']}, "
                f"source_index={pair['source_index']}"
            )
        if not np.isfinite(raw_delta[prompt_index]).all():
            raise ValueError(
                "Raw delta contains NaN or infinity: "
                f"prompt_id={pair['prompt_id']}, source_index={pair['source_index']}"
            )
        computed_original, computed_jailbreak, computed_delta = (
            compute_mean_distributions(
                raw_original[prompt_index],
                raw_jailbreak[prompt_index],
                raw_delta[prompt_index],
            )
        )
        if not np.allclose(
            accumulated_original_mean,
            computed_original,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(
                f"Original all-head accumulator mismatch: prompt_id={pair['prompt_id']}"
            )
        if not np.allclose(
            accumulated_jailbreak_mean,
            computed_jailbreak,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(
                f"Jailbreak all-head accumulator mismatch: prompt_id={pair['prompt_id']}"
            )
        mean_original[prompt_index] = computed_original.astype(np.float32)
        mean_jailbreak[prompt_index] = computed_jailbreak.astype(np.float32)
        mean_delta[prompt_index] = computed_delta.astype(np.float32)
        if not np.allclose(
            mean_delta[prompt_index],
            mean_jailbreak[prompt_index] - mean_original[prompt_index],
            rtol=1e-5,
            atol=2e-6,
        ):
            raise ValueError(
                f"Stored all-head mean delta mismatch: prompt_id={pair['prompt_id']}"
            )
        raw_original.flush()
        raw_jailbreak.flush()
        raw_delta.flush()
        mean_original.flush()
        mean_jailbreak.flush()
        mean_delta.flush()

    for name, values in (
        ("raw_original", raw_original),
        ("raw_jailbreak", raw_jailbreak),
        ("raw_delta", raw_delta),
        ("mean_original", mean_original),
        ("mean_jailbreak", mean_jailbreak),
        ("mean_delta", mean_delta),
    ):
        validate_finite_array(name, values)

    write_distribution_summary(
        raw_output_dir / "raw_distribution_summary.csv",
        raw_original,
        raw_jailbreak,
        raw_delta,
    )
    write_per_head_summary(
        raw_output_dir / "per_head_summary.csv",
        raw_original,
        raw_jailbreak,
        raw_delta,
    )
    write_mean_element_csvs(
        mean_output_dir,
        mean_original,
        mean_jailbreak,
        mean_delta,
        row_positions,
        col_positions,
        layer_head_count,
        args.tail_tokens,
        pairs,
    )
    write_distribution_summary(
        mean_output_dir / "all_heads_mean_distribution_summary.csv",
        mean_original,
        mean_jailbreak,
        mean_delta,
    )
    write_prompt_level_summary(
        mean_output_dir / "prompt_level_all_heads_mean_summary.csv",
        mean_original,
        mean_jailbreak,
        mean_delta,
        pairs,
    )

    execution_datetime = datetime.now(timezone.utc).isoformat()
    common_metadata = build_common_metadata(
        args=args,
        model_config=model_config,
        input_pair_count=len(input_pairs),
        included_pairs=pairs,
        manifest_rows=manifest_rows,
        tail_tokens=args.tail_tokens,
        elements_per_head=elements_per_head,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        row_positions=row_positions,
        col_positions=col_positions,
        execution_datetime=execution_datetime,
        tokenizer=tokenizer,
        model=model,
        source_commit=get_source_commit(),
    )
    raw_metadata = {
        **common_metadata,
        "distribution": "all layer/head raw values without head averaging",
        "array_shape": list(raw_shape),
        "array_axes": ["prompt", "layer", "head", "strict_lower_position"],
        "array_files": RAW_ARRAY_FILENAMES,
    }
    mean_metadata = {
        **common_metadata,
        "distribution": "arithmetic mean over all layer/head raw scores",
        "normalization": "none",
        "accumulator_dtype": "float64",
        "array_shape": list(mean_shape),
        "array_axes": ["prompt", "strict_lower_position"],
        "array_files": MEAN_ARRAY_FILENAMES,
        "position_order": "numpy/torch strict-lower row-major order",
    }
    write_metadata(raw_output_dir / "metadata.json", raw_metadata)
    write_metadata(mean_output_dir / "metadata.json", mean_metadata)

    print(f"model name: {model_config['model_name']}")
    print(f"input prompt pairs: {len(input_pairs)}")
    print(f"processed prompt pairs: {len(pairs)}")
    print(f"excluded prompt pairs: {len(input_pairs) - len(pairs)}")
    print(f"tail_tokens: {args.tail_tokens}")
    print(f"elements per prompt/head: {elements_per_head}")
    print(f"layer-head count: {layer_head_count}")
    print(f"raw values per condition: {int(np.prod(raw_shape))}")
    print(f"raw array shape: {raw_shape}")
    print(f"mean values per condition: {int(np.prod(mean_shape))}")
    print(f"mean element CSV rows: {2 * int(np.prod(mean_shape))}")
    print(f"mean delta CSV rows: {int(np.prod(mean_shape))}")
    print(f"raw output directory: {raw_output_dir}")
    print(f"mean output directory: {mean_output_dir}")


if __name__ == "__main__":
    main()
