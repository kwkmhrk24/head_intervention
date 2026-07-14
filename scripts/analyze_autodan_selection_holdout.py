import argparse
import csv
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_llama2_repro_data import read_jsonl, write_jsonl


INTERNAL_HOLDOUT_NOTE = (
    "All 130 AutoDAN pairs were previously used for exploratory analysis. "
    "This split is an internal AutoDAN hold-out validation, not a fully unseen test set."
)

BASELINE_LABEL_NOTE = (
    "Selection uses the pre-existing AutoDAN source success field. These labels "
    "are not recomputed with this repository's current 32-token refusal-substring "
    "ASR evaluation, and the source JSON does not contain response text."
)

SPLIT_FILENAMES = [
    "selection_original_prompts.jsonl",
    "selection_jailbreak_prompts.jsonl",
    "evaluation_original_prompts.jsonl",
    "evaluation_jailbreak_prompts.jsonl",
    "split_manifest.json",
    "selection_indices.json",
    "evaluation_indices.json",
]

ANALYSIS_FILENAMES = [
    "head_ranking.csv",
    "selected_heads.json",
    "bootstrap_head_stability.csv",
    "bootstrap_jaccard.csv",
    "bootstrap_summary.json",
]

RANKING_COLUMNS = [
    "layer",
    "head",
    "count",
    "mean_original_asym_ratio",
    "mean_jailbreak_asym_ratio",
    "mean_delta_asym_ratio",
    "median_delta_asym_ratio",
    "std_delta_asym_ratio",
    "delta_positive_rate",
    "delta_negative_rate",
    "positive_rank",
    "negative_rank",
    "absolute_rank",
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a baseline-success AutoDAN internal hold-out split and rank "
            "Llama2 attention heads using selection-only tail16 asym_ratio metrics."
        )
    )
    parser.add_argument(
        "--original-prompts",
        default="data/derived/llama2/autodan/original_prompts.jsonl",
    )
    parser.add_argument(
        "--jailbreak-prompts",
        default="data/derived/llama2/autodan/jailbreak_prompts.jsonl",
    )
    parser.add_argument(
        "--baseline-labels",
        default="data/raw/llama2/llama2_AutoDAN.json",
        help="JSON list containing index, user prompt, and boolean success fields.",
    )
    parser.add_argument(
        "--original-metrics",
        default=(
            "data/outputs/llama2/analysis/"
            "attention_score_metrics_original130_all_heads_tail16.csv"
        ),
    )
    parser.add_argument(
        "--jailbreak-metrics",
        default=(
            "data/outputs/llama2/analysis/"
            "attention_score_metrics_jailbreak130_all_heads_tail16.csv"
        ),
    )
    parser.add_argument(
        "--split-output-dir",
        default=None,
        help="Default: data/derived/llama2/autodan/holdout_seed{seed}",
    )
    parser.add_argument(
        "--analysis-output-dir",
        default=None,
        help=(
            "Default: data/outputs/llama2/autodan/holdout_seed{seed}/"
            "head_selection_tail{expected_window_size}"
        ),
    )
    parser.add_argument("--selection-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--expected-window-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json_list(path: str) -> List[Dict[str, Any]]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as exc:
        raise ValueError(
            f"Failed to read baseline labels: path={path}, "
            f"error_type={type(exc).__name__}"
        ) from exc
    if not isinstance(value, list) or not value:
        raise ValueError(f"Baseline labels must be a non-empty JSON list: {path}")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Every baseline label record must be a JSON object: {path}")
    return value


def _validate_input_index(row: Dict[str, Any], row_index: int, label: str) -> None:
    for key in ("index", "prompt_id", "source_index"):
        if key in row and row[key] not in (None, ""):
            if isinstance(row[key], bool) or not isinstance(row[key], int):
                raise ValueError(
                    f"{label} {key} must be an integer when present: row={row_index}"
                )
            if row[key] != row_index:
                raise ValueError(
                    f"{label} {key} does not match zero-based JSONL row index: "
                    f"row={row_index}, value={row[key]}"
                )


def validate_prompt_pairs_and_labels(
    original_rows: Sequence[Dict[str, Any]],
    jailbreak_rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int], bool]:
    counts = (len(original_rows), len(jailbreak_rows), len(baseline_rows))
    if len(set(counts)) != 1:
        raise ValueError(
            "Original, jailbreak, and baseline-label counts differ: "
            f"original={counts[0]}, jailbreak={counts[1]}, baseline={counts[2]}"
        )
    if not original_rows:
        raise ValueError("Prompt inputs contain no rows")

    annotated_original: List[Dict[str, Any]] = []
    annotated_jailbreak: List[Dict[str, Any]] = []
    success_indices: List[int] = []
    response_keys = {"response", "output", "generation", "generated_text"}
    response_available = all(
        any(key in row and isinstance(row[key], str) for key in response_keys)
        for row in baseline_rows
    )

    for row_index, (original, jailbreak, baseline) in enumerate(
        zip(original_rows, jailbreak_rows, baseline_rows)
    ):
        _validate_input_index(original, row_index, "original prompt")
        _validate_input_index(jailbreak, row_index, "jailbreak prompt")
        for label, row in (("original", original), ("jailbreak", jailbreak)):
            if not isinstance(row.get("prompt"), str):
                raise ValueError(f"{label} prompt is not a string: row={row_index}")

        missing = [
            key for key in ("index", "user prompt", "success") if key not in baseline
        ]
        if missing:
            raise ValueError(
                f"Baseline label row {row_index} is missing required keys: {missing}"
            )
        if isinstance(baseline["index"], bool) or not isinstance(
            baseline["index"], int
        ):
            raise ValueError(f"Baseline index must be an integer: row={row_index}")
        if baseline["index"] != row_index:
            raise ValueError(
                "Baseline index does not match zero-based prompt row: "
                f"row={row_index}, baseline_index={baseline['index']}"
            )
        if baseline["user prompt"] != jailbreak["prompt"]:
            raise ValueError(
                "Baseline user prompt does not match jailbreak prompt: "
                f"row={row_index}"
            )
        if type(baseline["success"]) is not bool:
            raise ValueError(f"Baseline success must be boolean: row={row_index}")
        if baseline["success"]:
            success_indices.append(row_index)

        common = {
            "index": row_index,
            "prompt_id": row_index,
            "source_index": row_index,
        }
        annotated_original.append({**common, "prompt": original["prompt"]})
        annotated_jailbreak.append({**common, "prompt": jailbreak["prompt"]})

    return (
        annotated_original,
        annotated_jailbreak,
        success_indices,
        response_available,
    )


def build_holdout_split(
    total_count: int,
    success_indices: Sequence[int],
    selection_size: int = 20,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    if total_count <= 0:
        raise ValueError("total_count must be positive")
    if selection_size <= 0:
        raise ValueError("selection_size must be positive")
    if selection_size >= total_count:
        raise ValueError("selection_size must be smaller than total_count")
    if len(set(success_indices)) != len(success_indices):
        raise ValueError("success_indices contains duplicates")
    if any(index < 0 or index >= total_count for index in success_indices):
        raise ValueError("success_indices contains an out-of-range index")
    if len(success_indices) < selection_size:
        raise ValueError(
            "Not enough baseline-success prompts for selection: "
            f"success_count={len(success_indices)}, selection_size={selection_size}"
        )

    selected = sorted(random.Random(seed).sample(list(success_indices), selection_size))
    selected_set = set(selected)
    evaluation = [index for index in range(total_count) if index not in selected_set]
    if selected_set.intersection(evaluation):
        raise RuntimeError("Selection and evaluation indices overlap")
    if selected_set.union(evaluation) != set(range(total_count)):
        raise RuntimeError("Selection/evaluation union does not cover the dataset")
    if not selected_set.issubset(set(success_indices)):
        raise RuntimeError("Selection contains a baseline-failed prompt")
    return selected, evaluation


def subset_rows(rows: Sequence[Dict[str, Any]], indices: Sequence[int]) -> List[Dict[str, Any]]:
    return [dict(rows[index]) for index in indices]


def _parse_csv_int(value: Any, field: str, path: str, line_no: int) -> int:
    text = str(value).strip()
    try:
        parsed = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: invalid integer {field} at CSV line {line_no}: {value!r}"
        ) from exc
    return parsed


def _metric_source_index(row: Dict[str, str], path: str, line_no: int) -> int:
    source_index = row.get("source_index", "").strip()
    if source_index and source_index.casefold() not in {"nan", "none"}:
        return _parse_csv_int(source_index, "source_index", path, line_no)
    return _parse_csv_int(row.get("prompt_id", ""), "prompt_id", path, line_no)


def read_selected_metric_csv(
    path: str,
    selection_indices: Iterable[int],
    expected_window_size: int = 16,
) -> pd.DataFrame:
    selected = set(selection_indices)
    if not selected:
        raise ValueError("selection_indices must not be empty")
    rows: List[Dict[str, Any]] = []
    try:
        handle = Path(path).open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to open metrics CSV: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metrics CSV has no header: {path}")
        available = set(reader.fieldnames)
        required = {"prompt_id", "source_index", "layer", "head", "window_size"}
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"Metrics CSV missing columns {missing}: {path}")
        if "asym_ratio" not in available and "skew_ratio" not in available:
            raise ValueError(
                f"Metrics CSV requires asym_ratio or legacy skew_ratio: {path}"
            )

        for line_no, row in enumerate(reader, start=2):
            source_index = _metric_source_index(row, path, line_no)
            if source_index not in selected:
                continue
            layer = _parse_csv_int(row["layer"], "layer", path, line_no)
            head = _parse_csv_int(row["head"], "head", path, line_no)
            window_size = _parse_csv_int(
                row["window_size"], "window_size", path, line_no
            )
            if window_size != expected_window_size:
                raise ValueError(
                    f"Selected metric row is not tail{expected_window_size}: "
                    f"path={path}, line={line_no}, window_size={window_size}"
                )

            ratio_text = row.get("asym_ratio", "").strip()
            legacy_text = row.get("skew_ratio", "").strip()
            if ratio_text:
                ratio = float(ratio_text)
                if legacy_text and not math.isclose(
                    ratio, float(legacy_text), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"asym_ratio and skew_ratio disagree: path={path}, line={line_no}"
                    )
            elif legacy_text:
                ratio = float(legacy_text)
            else:
                raise ValueError(
                    f"Selected metric row has an empty asym/skew ratio: "
                    f"path={path}, line={line_no}"
                )
            if not math.isfinite(ratio):
                raise ValueError(
                    f"Selected metric ratio is NaN/inf: path={path}, line={line_no}"
                )
            rows.append(
                {
                    "source_index": source_index,
                    "layer": layer,
                    "head": head,
                    "asym_ratio": ratio,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"Metrics CSV contains no selected rows: {path}")
    duplicates = frame.duplicated(["source_index", "layer", "head"])
    if duplicates.any():
        raise ValueError(
            f"Metrics CSV has duplicate selected prompt/head keys: "
            f"path={path}, count={int(duplicates.sum())}"
        )
    present = set(int(value) for value in frame["source_index"].unique())
    if present != selected:
        raise ValueError(
            "Metrics CSV does not contain every selection index: "
            f"path={path}, missing={sorted(selected - present)}, "
            f"unexpected={sorted(present - selected)}"
        )
    return frame


def build_selection_metric_pairs(
    original: pd.DataFrame,
    jailbreak: pd.DataFrame,
    selection_indices: Sequence[int],
    expected_layers: int | None = 32,
    expected_heads_per_layer: int | None = 32,
) -> pd.DataFrame:
    keys = ["source_index", "layer", "head"]
    original_keys = set(original[keys].itertuples(index=False, name=None))
    jailbreak_keys = set(jailbreak[keys].itertuples(index=False, name=None))
    if original_keys != jailbreak_keys:
        raise ValueError(
            "Original/jailbreak selected metric keys differ: "
            f"original_only={len(original_keys - jailbreak_keys)}, "
            f"jailbreak_only={len(jailbreak_keys - original_keys)}"
        )
    merged = original.merge(
        jailbreak,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_original", "_jailbreak"),
    )
    merged = merged.rename(
        columns={
            "asym_ratio_original": "original_asym_ratio",
            "asym_ratio_jailbreak": "jailbreak_asym_ratio",
        }
    )
    merged["delta_asym_ratio"] = (
        merged["jailbreak_asym_ratio"] - merged["original_asym_ratio"]
    )

    head_grid = merged[["layer", "head"]].drop_duplicates().sort_values(
        ["layer", "head"]
    )
    if expected_layers is not None and expected_heads_per_layer is not None:
        expected_grid = {
            (layer, head)
            for layer in range(expected_layers)
            for head in range(expected_heads_per_layer)
        }
        actual_grid = set(head_grid.itertuples(index=False, name=None))
        if actual_grid != expected_grid:
            raise ValueError(
                "Selected metrics do not contain the expected Llama2 head grid: "
                f"expected={len(expected_grid)}, actual={len(actual_grid)}"
            )
    head_count = len(head_grid)
    expected_prompt_count = len(selection_indices)
    per_head_counts = merged.groupby(["layer", "head"]).size()
    per_prompt_counts = merged.groupby("source_index").size()
    if not per_head_counts.eq(expected_prompt_count).all():
        raise ValueError("Every head must contain every selection prompt")
    if not per_prompt_counts.eq(head_count).all():
        raise ValueError("Every selection prompt must contain the same head grid")
    if set(merged["source_index"].unique()) != set(selection_indices):
        raise ValueError("Merged metrics contain an unexpected prompt set")
    return merged.sort_values(keys).reset_index(drop=True)


def _assign_ordinal_rank(
    frame: pd.DataFrame,
    metric: str,
    output_column: str,
    ascending: bool,
    absolute: bool = False,
) -> None:
    values = frame[metric].abs() if absolute else frame[metric]
    temporary = frame.assign(_rank_value=values)
    order = temporary.sort_values(
        ["_rank_value", "layer", "head"],
        ascending=[ascending, True, True],
        kind="mergesort",
    ).index
    ranks = pd.Series(np.arange(1, len(frame) + 1), index=order)
    frame[output_column] = ranks.reindex(frame.index).astype(int)


def aggregate_head_ranking(prompt_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (layer, head), group in prompt_metrics.groupby(["layer", "head"], sort=True):
        delta = group["delta_asym_ratio"]
        rows.append(
            {
                "layer": int(layer),
                "head": int(head),
                "count": int(len(group)),
                "mean_original_asym_ratio": float(group["original_asym_ratio"].mean()),
                "mean_jailbreak_asym_ratio": float(
                    group["jailbreak_asym_ratio"].mean()
                ),
                "mean_delta_asym_ratio": float(delta.mean()),
                "median_delta_asym_ratio": float(delta.median()),
                "std_delta_asym_ratio": float(delta.std(ddof=1)),
                "delta_positive_rate": float((delta > 0).mean()),
                "delta_negative_rate": float((delta < 0).mean()),
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        raise ValueError("Cannot rank an empty prompt metric frame")
    if not np.isfinite(ranking.select_dtypes(include=[np.number])).all().all():
        raise ValueError("Head ranking contains NaN/inf; at least two prompts are required")
    _assign_ordinal_rank(
        ranking, "mean_delta_asym_ratio", "positive_rank", ascending=False
    )
    _assign_ordinal_rank(
        ranking, "mean_delta_asym_ratio", "negative_rank", ascending=True
    )
    _assign_ordinal_rank(
        ranking,
        "mean_delta_asym_ratio",
        "absolute_rank",
        ascending=False,
        absolute=True,
    )
    return ranking[RANKING_COLUMNS].sort_values(["layer", "head"]).reset_index(drop=True)


def select_ranked_heads(
    ranking: pd.DataFrame, top_k: int
) -> Dict[str, pd.DataFrame]:
    if top_k <= 0 or top_k > len(ranking):
        raise ValueError(f"top_k must satisfy 1 <= top_k <= {len(ranking)}")
    return {
        "positive": ranking.sort_values("positive_rank").head(top_k).copy(),
        "negative": ranking.sort_values("negative_rank").head(top_k).copy(),
        "absolute": ranking.sort_values("absolute_rank").head(top_k).copy(),
    }


def _head_set(frame: pd.DataFrame) -> Set[Tuple[int, int]]:
    return {
        (int(row.layer), int(row.head))
        for row in frame[["layer", "head"]].itertuples(index=False)
    }


def _jaccard(first: Set[Tuple[int, int]], second: Set[Tuple[int, int]]) -> float:
    union = first.union(second)
    return float(len(first.intersection(second)) / len(union)) if union else 1.0


def bootstrap_head_stability(
    prompt_metrics: pd.DataFrame,
    base_selected: Dict[str, pd.DataFrame],
    iterations: int = 1000,
    seed: int = 42,
    top_k: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    prompt_indices = sorted(int(value) for value in prompt_metrics["source_index"].unique())
    heads = (
        prompt_metrics[["layer", "head"]]
        .drop_duplicates()
        .sort_values(["layer", "head"])
        .reset_index(drop=True)
    )
    delta = prompt_metrics.pivot(
        index="source_index", columns=["layer", "head"], values="delta_asym_ratio"
    ).reindex(index=prompt_indices, columns=pd.MultiIndex.from_frame(heads))
    if delta.isna().any().any():
        raise ValueError("Bootstrap input is missing a prompt/head delta")
    values = delta.to_numpy(np.float64)
    if values.shape != (len(prompt_indices), len(heads)):
        raise ValueError("Unexpected bootstrap delta matrix shape")
    if top_k <= 0 or top_k > len(heads):
        raise ValueError("Invalid bootstrap top_k")

    layers = heads["layer"].to_numpy(np.int64)
    head_numbers = heads["head"].to_numpy(np.int64)
    counts = {
        condition: np.zeros(len(heads), dtype=np.int64)
        for condition in ("positive", "negative", "absolute")
    }
    base_sets = {condition: _head_set(frame) for condition, frame in base_selected.items()}
    jaccard_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    for iteration in range(iterations):
        sampled = rng.integers(0, len(prompt_indices), size=len(prompt_indices))
        means = values[sampled].mean(axis=0)
        orders = {
            "positive": np.lexsort((head_numbers, layers, -means)),
            "negative": np.lexsort((head_numbers, layers, means)),
            "absolute": np.lexsort((head_numbers, layers, -np.abs(means))),
        }
        selected_sets: Dict[str, Set[Tuple[int, int]]] = {}
        for condition, order in orders.items():
            chosen = order[:top_k]
            counts[condition][chosen] += 1
            selected_sets[condition] = {
                (int(layers[index]), int(head_numbers[index])) for index in chosen
            }
        jaccard_rows.append(
            {
                "iteration": iteration + 1,
                "positive_jaccard": _jaccard(
                    base_sets["positive"], selected_sets["positive"]
                ),
                "negative_jaccard": _jaccard(
                    base_sets["negative"], selected_sets["negative"]
                ),
                "absolute_jaccard": _jaccard(
                    base_sets["absolute"], selected_sets["absolute"]
                ),
            }
        )

    stability = heads.copy()
    for condition in ("positive", "negative", "absolute"):
        stability[f"{condition}_selection_count"] = counts[condition]
        stability[f"{condition}_selection_rate"] = counts[condition] / iterations
        stability[f"base_{condition}_selected"] = [
            (int(layer), int(head)) in base_sets[condition]
            for layer, head in stability[["layer", "head"]].itertuples(index=False)
        ]
    jaccard = pd.DataFrame(jaccard_rows)
    jaccard_summary: Dict[str, Dict[str, float]] = {}
    for condition in ("positive", "negative", "absolute"):
        series = jaccard[f"{condition}_jaccard"]
        jaccard_summary[condition] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    summary = {
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "selection_prompt_count": len(prompt_indices),
        "selection_indices": prompt_indices,
        "top_k": top_k,
        "evaluation_metrics_used": False,
        "jaccard_against_original_selection": jaccard_summary,
        "research_context_note": INTERNAL_HOLDOUT_NOTE,
    }
    return stability, jaccard, summary


def _records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        converted: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (np.integer,)):
                converted[key] = int(value)
            elif isinstance(value, (np.floating,)):
                converted[key] = float(value)
            elif isinstance(value, (np.bool_,)):
                converted[key] = bool(value)
            else:
                converted[key] = value
        records.append(converted)
    return records


def prepare_targets(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Target files already exist; use --overwrite: {existing}"
        )
    for parent in sorted({path.parent for path in paths}):
        parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not determine the current git commit") from exc
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("git rev-parse returned an empty commit")
    return commit


def main() -> None:
    args = get_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive")
    if args.expected_window_size != 16:
        raise ValueError("This experiment is fixed to --expected-window-size 16")

    original_rows = read_jsonl(args.original_prompts)
    jailbreak_rows = read_jsonl(args.jailbreak_prompts)
    baseline_rows = read_json_list(args.baseline_labels)
    (
        annotated_original,
        annotated_jailbreak,
        success_indices,
        response_available,
    ) = validate_prompt_pairs_and_labels(original_rows, jailbreak_rows, baseline_rows)
    selection_indices, evaluation_indices = build_holdout_split(
        len(annotated_original), success_indices, args.selection_size, args.seed
    )

    original_metrics = read_selected_metric_csv(
        args.original_metrics, selection_indices, args.expected_window_size
    )
    jailbreak_metrics = read_selected_metric_csv(
        args.jailbreak_metrics, selection_indices, args.expected_window_size
    )
    prompt_metrics = build_selection_metric_pairs(
        original_metrics, jailbreak_metrics, selection_indices
    )
    ranking = aggregate_head_ranking(prompt_metrics)
    selected = select_ranked_heads(ranking, args.top_k)
    stability, jaccard, bootstrap_summary = bootstrap_head_stability(
        prompt_metrics,
        selected,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
        top_k=args.top_k,
    )

    split_dir = Path(
        args.split_output_dir
        or f"data/derived/llama2/autodan/holdout_seed{args.seed}"
    )
    analysis_dir = Path(
        args.analysis_output_dir
        or (
            f"data/outputs/llama2/autodan/holdout_seed{args.seed}/"
            f"head_selection_tail{args.expected_window_size}"
        )
    )
    split_targets = [split_dir / name for name in SPLIT_FILENAMES]
    analysis_targets = [analysis_dir / name for name in ANALYSIS_FILENAMES]
    prepare_targets([*split_targets, *analysis_targets], args.overwrite)

    write_jsonl(
        split_dir / "selection_original_prompts.jsonl",
        subset_rows(annotated_original, selection_indices),
    )
    write_jsonl(
        split_dir / "selection_jailbreak_prompts.jsonl",
        subset_rows(annotated_jailbreak, selection_indices),
    )
    write_jsonl(
        split_dir / "evaluation_original_prompts.jsonl",
        subset_rows(annotated_original, evaluation_indices),
    )
    write_jsonl(
        split_dir / "evaluation_jailbreak_prompts.jsonl",
        subset_rows(annotated_jailbreak, evaluation_indices),
    )
    write_json(split_dir / "selection_indices.json", selection_indices)
    write_json(split_dir / "evaluation_indices.json", evaluation_indices)

    timestamp = datetime.now(timezone.utc).isoformat()
    manifest = {
        "dataset": "AutoDAN 130 paired prompts",
        "model": "Llama-2-7b-chat-hf",
        "attack": "AutoDAN",
        "total_count": len(annotated_original),
        "baseline_success_count": len(success_indices),
        "selection_size": len(selection_indices),
        "evaluation_size": len(evaluation_indices),
        "seed": args.seed,
        "sampling_algorithm": "sorted(random.Random(seed).sample(success_indices, selection_size))",
        "selection_from_baseline_success_only": True,
        "baseline_success_label_source": {
            "path": args.baseline_labels,
            "field": "success",
            "required_type": "boolean",
            "prompt_match_field": "user prompt",
        },
        "baseline_response_available": response_available,
        "baseline_success_label_note": BASELINE_LABEL_NOTE,
        "selection_indices": selection_indices,
        "evaluation_indices": evaluation_indices,
        "index_definition": "zero-based row index in both original and jailbreak input JSONL files",
        "original_input_paths": {
            "original_prompts": args.original_prompts,
            "jailbreak_prompts": args.jailbreak_prompts,
        },
        "timestamp": timestamp,
        "git_commit": current_git_commit(),
        "research_context_note": INTERNAL_HOLDOUT_NOTE,
    }
    write_json(split_dir / "split_manifest.json", manifest)

    ranking.to_csv(analysis_dir / "head_ranking.csv", index=False)
    condition_keys = {
        "positive": f"positive_top{args.top_k}",
        "negative": f"negative_bottom{args.top_k}",
        "absolute": f"absolute_top{args.top_k}",
    }
    selected_heads_json: Dict[str, Any] = {
        "metric": "mean_delta_asym_ratio",
        "delta_definition": "jailbreak_asym_ratio - original_asym_ratio",
        "asym_ratio_definition": "|| (S - S^T) / 2 ||_F / (||S||_F + epsilon)",
        "window": "tail16",
        "selection_indices": selection_indices,
        "top_k": args.top_k,
        "research_context_note": INTERNAL_HOLDOUT_NOTE,
        "baseline_success_label_note": BASELINE_LABEL_NOTE,
    }
    for condition, key in condition_keys.items():
        selected_heads_json[key] = _records(selected[condition])
    write_json(analysis_dir / "selected_heads.json", selected_heads_json)

    stability.to_csv(analysis_dir / "bootstrap_head_stability.csv", index=False)
    jaccard.to_csv(analysis_dir / "bootstrap_jaccard.csv", index=False)
    bootstrap_summary.update(
        {
            "window": "tail16",
            "metric": "mean_delta_asym_ratio",
            "original_metrics": args.original_metrics,
            "jailbreak_metrics": args.jailbreak_metrics,
            "selected_heads": {
                condition_keys[condition]: _records(frame)
                for condition, frame in selected.items()
            },
            "timestamp": timestamp,
            "git_commit": manifest["git_commit"],
            "baseline_success_label_note": BASELINE_LABEL_NOTE,
        }
    )
    write_json(analysis_dir / "bootstrap_summary.json", bootstrap_summary)

    print(f"baseline successes: {len(success_indices)} / {len(annotated_original)}")
    print(f"baseline label note: {BASELINE_LABEL_NOTE}")
    print(f"selection indices: {selection_indices}")
    print(f"evaluation count: {len(evaluation_indices)}")
    for condition, frame in selected.items():
        labels = [f"L{int(row.layer)}H{int(row.head)}" for row in frame.itertuples()]
        print(f"{condition}: {labels}")
    print(f"split output directory: {split_dir}")
    print(f"analysis output directory: {analysis_dir}")


if __name__ == "__main__":
    main()
