#!/usr/bin/env python3
"""Summarize cross-attack attention patterns without loading a model.

The statistical unit is one layer/head. Cross-model comparisons are limited to
aggregate relationships and normalized layer depth; head identities are never
joined across model families.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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

from scripts.analyze_attention_head_metric_relationships import (
    pearson_correlation,
    spearman_correlation,
    validate_head_grid,
)


METRICS = [
    "delta_asym_ratio",
    "element_delta_mean",
    "element_std_shift",
    "element_positive_rate_shift",
]
RELATIONSHIP_PAIRS = [
    ("delta_asym_ratio", "element_std_shift"),
    ("delta_asym_ratio", "element_delta_mean"),
    ("delta_asym_ratio", "element_positive_rate_shift"),
    ("element_delta_mean", "element_positive_rate_shift"),
]
K_VALUES = (3, 5, 10, 20, 50)
INTERVENTION_K = (3, 5, 10)


def default_paths() -> dict[str, str]:
    root = "data/outputs/attention_score_element_distributions"
    return {
        "llama2_autodan_full": f"{root}/llama2/autodan/head_metric_relationships_tail16_full/head_metrics_merged.csv",
        "llama2_gcg_full": f"{root}/llama2/gcg/head_metric_relationships_tail16_full/head_metrics_merged.csv",
        "mistral_autodan_full": f"{root}/mistral/autodan/head_metric_relationships_tail16_full/head_metrics_merged.csv",
        "mistral_gcg_full": f"{root}/mistral/gcg/head_metric_relationships_tail16_full/head_metrics_merged.csv",
        "llama2_autodan_matched": f"{root}/llama2/autodan_vs_gcg_matched70_tail16/autodan_matched_head_metrics.csv",
        "llama2_gcg_matched": f"{root}/llama2/autodan_vs_gcg_matched70_tail16/gcg_matched_head_metrics.csv",
        "mistral_autodan_matched": f"{root}/mistral/autodan_vs_gcg_matched49_tail16/autodan_matched_head_metrics.csv",
        "mistral_gcg_matched": f"{root}/mistral/autodan_vs_gcg_matched49_tail16/gcg_matched_head_metrics.csv",
        "llama2_matched_metadata": f"{root}/llama2/autodan_vs_gcg_matched70_tail16/metadata.json",
        "mistral_matched_metadata": f"{root}/mistral/autodan_vs_gcg_matched49_tail16/metadata.json",
        "output_dir": f"{root}/cross_model_attention_pattern_summary",
    }


def get_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(
        description="CPU-only cross-model summary of existing attention head metrics."
    )
    for key, value in defaults.items():
        parser.add_argument(f"--{key}", default=value)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".write_test_"):
            pass
    except OSError as exc:
        raise OSError(f"Output directory is not writable: {path}") from exc


def read_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"Failed to read metadata: path={path}, error_type={type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"Metadata must be an object: path={path}")
    return value


def read_head_metrics(path: Path, label: str, scope: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read head metrics: path={path}, error_type={type(exc).__name__}"
        ) from exc
    required = {"layer", "head", *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    if frame.duplicated(["layer", "head"]).any():
        raise ValueError(f"{label} has duplicate layer/head keys")
    numeric = frame[["layer", "head", *METRICS]].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} contains NaN/inf")
    validate_head_grid(frame[["layer", "head"]])
    result = frame.copy()
    if "head_label" not in result:
        result["head_label"] = [
            f"L{int(layer)}H{int(head)}"
            for layer, head in result[["layer", "head"]].itertuples(index=False)
        ]
    if scope == "full":
        identities = [
            ("element_std_shift", "element_jailbreak_std", "element_original_std"),
            (
                "element_positive_rate_shift",
                "element_jailbreak_positive_rate",
                "element_original_positive_rate",
            ),
        ]
    else:
        identities = [
            ("element_std_shift", "jailbreak_element_std", "original_element_std"),
            (
                "element_positive_rate_shift",
                "jailbreak_positive_rate",
                "original_positive_rate",
            ),
        ]
    for derived, jailbreak, original in identities:
        if jailbreak in result and original in result:
            error = np.max(np.abs(result[derived] - (result[jailbreak] - result[original])))
            if error > 1e-6:
                raise ValueError(f"{label} derived metric identity failed: {derived}")
    return result.sort_values(["layer", "head"]).reset_index(drop=True)


def linear_relationship(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("Relationship input contains NaN/inf")
    slope, intercept = np.polyfit(x_values, y_values, 1)
    predicted = slope * x_values + intercept
    residual = float(np.sum((y_values - predicted) ** 2))
    total = float(np.sum((y_values - y_values.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else float("nan")
    return {
        "pearson": pearson_correlation(x_values, y_values),
        "spearman": spearman_correlation(x_values, y_values),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "sign_agreement_rate": float((np.sign(x_values) == np.sign(y_values)).mean()),
        "n": len(x_values),
    }


def quadrant_labels(x: Sequence[float], y: Sequence[float]) -> np.ndarray:
    x_values = np.asarray(x)
    y_values = np.asarray(y)
    labels = np.full(len(x_values), "zero", dtype=object)
    labels[(x_values > 0) & (y_values > 0)] = "Q1"
    labels[(x_values < 0) & (y_values > 0)] = "Q2"
    labels[(x_values < 0) & (y_values < 0)] = "Q3"
    labels[(x_values > 0) & (y_values < 0)] = "Q4"
    return labels


def build_within_condition_outputs(
    conditions: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    relationships: list[dict[str, Any]] = []
    quadrants: list[dict[str, Any]] = []
    for condition_id, spec in conditions.items():
        frame = spec["frame"]
        for metric_x, metric_y in RELATIONSHIP_PAIRS:
            row = {
                "condition_id": condition_id,
                "model": spec["model"],
                "attack": spec["attack"],
                "scope": spec["scope"],
                "metric_x": metric_x,
                "metric_y": metric_y,
            }
            row.update(linear_relationship(frame[metric_x], frame[metric_y]))
            relationships.append(row)
        labels = quadrant_labels(frame.delta_asym_ratio, frame.element_std_shift)
        for quadrant in ("Q1", "Q2", "Q3", "Q4", "zero"):
            count = int(np.sum(labels == quadrant))
            quadrants.append(
                {
                    "condition_id": condition_id,
                    "model": spec["model"],
                    "attack": spec["attack"],
                    "scope": spec["scope"],
                    "quadrant": quadrant,
                    "count": count,
                    "rate": count / len(frame),
                    "n": len(frame),
                }
            )
    return pd.DataFrame(relationships), pd.DataFrame(quadrants)


def ranked(frame: pd.DataFrame, metric: str, direction: str) -> pd.DataFrame:
    if direction not in {"increase", "decrease"}:
        raise ValueError(f"Unsupported direction: {direction}")
    return frame.sort_values(
        [metric, "layer", "head"],
        ascending=[direction == "decrease", True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_topk_shift_summary(
    conditions: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for condition_id, spec in conditions.items():
        for direction in ("increase", "decrease"):
            ordered = ranked(spec["frame"], "delta_asym_ratio", direction)
            for k in K_VALUES:
                subset = ordered.head(k)
                rows.append(
                    {
                        "condition_id": condition_id,
                        "model": spec["model"],
                        "attack": spec["attack"],
                        "scope": spec["scope"],
                        "direction": direction,
                        "k": k,
                        "element_std_shift_mean": subset.element_std_shift.mean(),
                        "element_std_shift_median": subset.element_std_shift.median(),
                        "std_shift_positive_rate": (subset.element_std_shift > 0).mean(),
                        "std_shift_negative_rate": (subset.element_std_shift < 0).mean(),
                        "element_delta_mean_mean": subset.element_delta_mean.mean(),
                        "element_positive_rate_shift_mean": (
                            subset.element_positive_rate_shift.mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def rank_positions(frame: pd.DataFrame, metric: str, direction: str) -> pd.DataFrame:
    ordered = ranked(frame, metric, direction).copy()
    ordered["rank"] = np.arange(1, len(ordered) + 1)
    return ordered[["layer", "head", "rank", metric]]


def build_full_matched_stability(
    full: pd.DataFrame,
    matched: pd.DataFrame,
    model: str,
    attack: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(map(tuple, full[["layer", "head"]].to_numpy())) != set(
        map(tuple, matched[["layer", "head"]].to_numpy())
    ):
        raise ValueError(f"Full/matched head grids differ: model={model}, attack={attack}")
    summary_rows = []
    topk_rows = []
    for metric in METRICS:
        joined = full[["layer", "head", metric]].merge(
            matched[["layer", "head", metric]],
            on=["layer", "head"],
            validate="one_to_one",
            suffixes=("_full", "_matched"),
        )
        relation = linear_relationship(joined[f"{metric}_full"], joined[f"{metric}_matched"])
        for direction in ("increase", "decrease"):
            full_rank = rank_positions(full, metric, direction).rename(
                columns={"rank": "full_rank"}
            )
            matched_rank = rank_positions(matched, metric, direction).rename(
                columns={"rank": "matched_rank"}
            )
            ranks = full_rank[["layer", "head", "full_rank"]].merge(
                matched_rank[["layer", "head", "matched_rank"]],
                on=["layer", "head"],
                validate="one_to_one",
            )
            differences = np.abs(ranks.full_rank - ranks.matched_rank)
            summary_rows.append(
                {
                    "model": model,
                    "attack": attack,
                    "metric": metric,
                    "direction": direction,
                    **relation,
                    "mean_absolute_rank_change": differences.mean(),
                    "median_absolute_rank_change": differences.median(),
                    "max_absolute_rank_change": differences.max(),
                }
            )
            for k in K_VALUES:
                full_set = set(
                    map(tuple, ranked(full, metric, direction).head(k)[["layer", "head"]].to_numpy())
                )
                matched_set = set(
                    map(tuple, ranked(matched, metric, direction).head(k)[["layer", "head"]].to_numpy())
                )
                overlap = full_set & matched_set
                topk_rows.append(
                    {
                        "model": model,
                        "attack": attack,
                        "metric": metric,
                        "direction": direction,
                        "k": k,
                        "overlap_count": len(overlap),
                        "retention_rate": len(overlap) / k,
                        "jaccard": len(overlap) / len(full_set | matched_set),
                        "common_heads": ";".join(
                            f"L{int(layer)}H{int(head)}" for layer, head in sorted(overlap)
                        ),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(topk_rows)


def build_consensus(
    autodan: pd.DataFrame,
    gcg: pd.DataFrame,
    model: str,
    direction: str,
) -> pd.DataFrame:
    auto_rank = rank_positions(autodan, "delta_asym_ratio", direction).rename(
        columns={"rank": "autodan_signed_rank", "delta_asym_ratio": "autodan_delta_asym_ratio"}
    )
    gcg_rank = rank_positions(gcg, "delta_asym_ratio", direction).rename(
        columns={"rank": "gcg_signed_rank", "delta_asym_ratio": "gcg_delta_asym_ratio"}
    )
    result = auto_rank.merge(gcg_rank, on=["layer", "head"], validate="one_to_one")
    if direction == "increase":
        result = result[(result.autodan_delta_asym_ratio > 0) & (result.gcg_delta_asym_ratio > 0)]
    else:
        result = result[(result.autodan_delta_asym_ratio < 0) & (result.gcg_delta_asym_ratio < 0)]
    n = len(autodan)
    result["model"] = model
    result["direction"] = direction
    result["head_label"] = [f"L{l}H{h}" for l, h in result[["layer", "head"]].itertuples(index=False)]
    result["mean_rank"] = (result.autodan_signed_rank + result.gcg_signed_rank) / 2
    result["worst_rank"] = result[["autodan_signed_rank", "gcg_signed_rank"]].max(axis=1)
    result["mean_percentile_rank"] = 1 - (
        (result.autodan_signed_rank - 1 + result.gcg_signed_rank - 1) / (2 * (n - 1))
    )
    result["minimum_absolute_metric_magnitude"] = result[
        ["autodan_delta_asym_ratio", "gcg_delta_asym_ratio"]
    ].abs().min(axis=1)
    result["mean_metric_value"] = result[
        ["autodan_delta_asym_ratio", "gcg_delta_asym_ratio"]
    ].mean(axis=1)
    result["sign_consistency"] = True
    result = result.sort_values(
        ["worst_rank", "mean_rank", "minimum_absolute_metric_magnitude", "layer", "head"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    result.insert(0, "consensus_worst_rank", np.arange(1, len(result) + 1))
    percentile_order = result.sort_values(
        ["mean_percentile_rank", "minimum_absolute_metric_magnitude", "layer", "head"],
        ascending=[False, False, True, True],
    )
    percentile_ranks = pd.Series(
        np.arange(1, len(percentile_order) + 1), index=percentile_order.index
    )
    result["consensus_percentile_rank"] = percentile_ranks.sort_index().to_numpy()
    return result


def build_attack_specific_vs_consensus(
    matched_conditions: Mapping[str, Mapping[str, Any]],
    consensus: Mapping[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for _, spec in matched_conditions.items():
        frame = spec["frame"]
        for direction in ("increase", "decrease"):
            ordered = ranked(frame, "delta_asym_ratio", direction)
            consensus_frame = consensus[(spec["model"], direction)]
            consensus_by_head = consensus_frame.set_index(["layer", "head"])
            for k in INTERVENTION_K + (20,):
                for rank_value, row in enumerate(ordered.head(k).itertuples(index=False), 1):
                    key = (int(row.layer), int(row.head))
                    in_consensus = key in consensus_by_head.index
                    rows.append(
                        {
                            "model": spec["model"],
                            "attack": spec["attack"],
                            "direction": direction,
                            "k": k,
                            "attack_rank": rank_value,
                            "layer": key[0],
                            "head": key[1],
                            "head_label": f"L{key[0]}H{key[1]}",
                            "delta_asym_ratio": row.delta_asym_ratio,
                            "element_std_shift": row.element_std_shift,
                            "element_delta_mean": row.element_delta_mean,
                            "element_positive_rate_shift": row.element_positive_rate_shift,
                            "in_attack_consensus": in_consensus,
                            "consensus_worst_rank": (
                                int(consensus_by_head.loc[key, "consensus_worst_rank"])
                                if in_consensus else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def build_layer_consensus(
    matched_by_model: Mapping[str, Mapping[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for model, attacks in matched_by_model.items():
        layer_count = len(attacks["AutoDAN"].layer.unique())
        summaries = {}
        for attack, frame in attacks.items():
            summaries[attack] = frame.groupby("layer").agg(
                delta_asym_ratio_mean=("delta_asym_ratio", "mean"),
                delta_asym_ratio_median=("delta_asym_ratio", "median"),
                positive_head_rate=("delta_asym_ratio", lambda x: (x > 0).mean()),
                negative_head_rate=("delta_asym_ratio", lambda x: (x < 0).mean()),
                mean_absolute_delta_asym_ratio=("delta_asym_ratio", lambda x: np.abs(x).mean()),
                element_std_shift_mean=("element_std_shift", "mean"),
            )
        joined = summaries["AutoDAN"].join(
            summaries["GCG"], lsuffix="_autodan", rsuffix="_gcg", validate="one_to_one"
        ).reset_index()
        joined.insert(0, "model", model)
        joined["normalized_layer_depth"] = joined.layer / (layer_count - 1)
        joined["delta_asym_ratio_mean_attack_difference"] = (
            joined.delta_asym_ratio_mean_gcg - joined.delta_asym_ratio_mean_autodan
        )
        joined["delta_asym_ratio_mean_sign_agreement"] = (
            np.sign(joined.delta_asym_ratio_mean_autodan)
            == np.sign(joined.delta_asym_ratio_mean_gcg)
        )
        rows.append(joined)
    return pd.concat(rows, ignore_index=True)


def build_hypothesis_evaluations(
    relationships: pd.DataFrame,
    topk: pd.DataFrame,
    matched_attack_overlap: Mapping[str, pd.DataFrame],
    layer_consensus: pd.DataFrame,
) -> pd.DataFrame:
    primary = relationships[
        (relationships.metric_x == "delta_asym_ratio")
        & (relationships.metric_y == "element_std_shift")
    ]
    increase20 = topk[(topk.direction == "increase") & (topk.k == 20)]
    decrease20 = topk[(topk.direction == "decrease") & (topk.k == 20)]
    h1_rate = float(increase20.std_shift_positive_rate.mean())
    h2_rate = float(decrease20.std_shift_negative_rate.mean())
    increase_overlap = {
        model: int(
            overlap[
                (overlap.metric == "delta_asym_ratio")
                & (overlap.direction == "increase")
                & (overlap.k == 10)
            ].overlap_count.iloc[0]
        )
        for model, overlap in matched_attack_overlap.items()
    }
    decrease_overlap = {
        model: int(
            overlap[
                (overlap.metric == "delta_asym_ratio")
                & (overlap.direction == "decrease")
                & (overlap.k == 10)
            ].overlap_count.iloc[0]
        )
        for model, overlap in matched_attack_overlap.items()
    }
    layer_rates = layer_consensus.groupby("model")[
        "delta_asym_ratio_mean_sign_agreement"
    ].mean()
    head_matched = primary[primary.scope == "matched"].groupby("model").spearman.mean()
    rows = [
        {
            "hypothesis": "H1",
            "statement": "delta_asym_ratio increase corresponds to distribution expansion",
            "judgment": "support" if h1_rate >= 0.7 else "partial support" if h1_rate >= 0.5 else "not supported",
            "evidence": f"mean Top20 std-shift positive rate={h1_rate:.3f}; mean Spearman={primary.spearman.mean():.3f}",
        },
        {
            "hypothesis": "H2",
            "statement": "delta_asym_ratio decrease corresponds to distribution contraction",
            "judgment": "support" if h2_rate >= 0.7 else "partial support" if h2_rate >= 0.5 else "not supported",
            "evidence": f"mean Top20 std-shift negative rate={h2_rate:.3f}; mean Spearman={primary.spearman.mean():.3f}",
        },
        {
            "hypothesis": "H3",
            "statement": "extreme increase heads are attack-dependent",
            "judgment": "support" if max(increase_overlap.values()) <= 2 else "partial support",
            "evidence": f"matched increase Top10 overlap={increase_overlap}",
        },
        {
            "hypothesis": "H4",
            "statement": "decrease-side stability is model-dependent",
            "judgment": "support" if len(set(decrease_overlap.values())) > 1 else "not supported",
            "evidence": f"matched decrease Top10 overlap={decrease_overlap}",
        },
        {
            "hypothesis": "H5",
            "statement": "layer regions agree even when individual heads agree weakly",
            "judgment": "support" if all(layer_rates > head_matched) else "partial support",
            "evidence": f"layer sign agreement={layer_rates.to_dict()}; matched head Spearman={head_matched.to_dict()}",
        },
    ]
    return pd.DataFrame(rows)


def build_cross_model_pattern_comparison(
    conditions: Mapping[str, Mapping[str, Any]],
    consensus: Mapping[tuple[str, str], pd.DataFrame],
    layer_consensus: pd.DataFrame,
    hypotheses: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in ("Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.2"):
        slug = "llama2" if model.startswith("Llama") else "mistral"
        for scope in ("full", "matched"):
            auto = conditions[f"{slug}_autodan_{scope}"]["frame"]
            gcg = conditions[f"{slug}_gcg_{scope}"]["frame"]
            joined = auto[["layer", "head", "delta_asym_ratio"]].merge(
                gcg[["layer", "head", "delta_asym_ratio"]],
                on=["layer", "head"],
                validate="one_to_one",
                suffixes=("_autodan", "_gcg"),
            )
            rows.append(
                {
                    "record_type": "attack_head_relationship",
                    "model": model,
                    "scope": scope,
                    "metric": "delta_asym_ratio",
                    "pearson": pearson_correlation(
                        joined.delta_asym_ratio_autodan,
                        joined.delta_asym_ratio_gcg,
                    ),
                    "spearman": spearman_correlation(
                        joined.delta_asym_ratio_autodan,
                        joined.delta_asym_ratio_gcg,
                    ),
                    "sign_agreement_rate": float(
                        (
                            np.sign(joined.delta_asym_ratio_autodan)
                            == np.sign(joined.delta_asym_ratio_gcg)
                        ).mean()
                    ),
                    "evidence": "one observation per layer/head within the same model",
                }
            )
        layers = layer_consensus[layer_consensus.model == model]
        rows.append(
            {
                "record_type": "matched_layer_relationship",
                "model": model,
                "scope": "matched",
                "metric": "layer_mean_delta_asym_ratio",
                "pearson": pearson_correlation(
                    layers.delta_asym_ratio_mean_autodan,
                    layers.delta_asym_ratio_mean_gcg,
                ),
                "spearman": spearman_correlation(
                    layers.delta_asym_ratio_mean_autodan,
                    layers.delta_asym_ratio_mean_gcg,
                ),
                "sign_agreement_rate": float(
                    layers.delta_asym_ratio_mean_sign_agreement.mean()
                ),
                "evidence": "normalized layer depth; head identities not compared across models",
            }
        )
        for direction in ("increase", "decrease"):
            count = len(consensus[(model, direction)])
            rows.append(
                {
                    "record_type": "attack_consistent_head_prevalence",
                    "model": model,
                    "scope": "matched",
                    "metric": f"delta_asym_ratio_{direction}",
                    "value": count / len(
                        conditions[f"{slug}_autodan_matched"]["frame"]
                    ),
                    "count": count,
                    "evidence": "same-sign heads across AutoDAN and GCG",
                }
            )
    result = pd.DataFrame(rows)
    hypothesis_rows = hypotheses.copy()
    hypothesis_rows.insert(0, "record_type", "hypothesis_evaluation")
    return pd.concat([result, hypothesis_rows], ignore_index=True, sort=False)


def build_intervention_candidates(
    matched_conditions: Mapping[str, Mapping[str, Any]],
    consensus: Mapping[tuple[str, str], pd.DataFrame],
    full_autodan_by_model: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for _, spec in matched_conditions.items():
        for direction, candidate_type in (
            ("increase", "attack_specific_increase"),
            ("decrease", "attack_specific_decrease"),
        ):
            ordered = ranked(spec["frame"], "delta_asym_ratio", direction)
            for k in INTERVENTION_K:
                for rank_value, row in enumerate(ordered.head(k).itertuples(index=False), 1):
                    rows.append(
                        candidate_row(spec["model"], candidate_type, spec["attack"], direction, k, row, rank_value, "attack-specific matched signed rank")
                    )
    for (model, direction), frame in consensus.items():
        candidate_type = f"attack_consistent_{direction}"
        for k in INTERVENTION_K:
            for row in frame.head(k).itertuples(index=False):
                rows.append(
                    {
                        "model": model,
                        "candidate_type": candidate_type,
                        "source_attack": "AutoDAN+GCG",
                        "direction": direction,
                        "k": k,
                        "layer": int(row.layer),
                        "head": int(row.head),
                        "rank": int(row.consensus_worst_rank),
                        "selection_metric": "delta_asym_ratio consensus worst rank",
                        "selection_value": float(row.mean_metric_value),
                        "attack_consistency": True,
                        "notes": f"worst_rank={int(row.worst_rank)}; mean_percentile={row.mean_percentile_rank:.6f}",
                    }
                )
    for model, full_autodan in full_autodan_by_model.items():
        if "aie_rank" in full_autodan:
            aie = full_autodan.sort_values(["aie_rank", "layer", "head"])
            for k in INTERVENTION_K:
                for row in aie.head(k).itertuples(index=False):
                    rows.append(
                        {
                            "model": model,
                            "candidate_type": "existing_aie",
                            "source_attack": "AutoDAN",
                            "direction": "signed_aie_high",
                            "k": k,
                            "layer": int(row.layer),
                            "head": int(row.head),
                            "rank": int(row.aie_rank),
                            "selection_metric": "aie",
                            "selection_value": float(row.aie),
                            "attack_consistency": False,
                            "notes": "AIE is unavailable for the corresponding GCG condition",
                        }
                    )
        if "paper_head" in full_autodan:
            paper = full_autodan[
                full_autodan.paper_head.astype(bool)
            ].sort_values(["layer", "head"])
            for k in INTERVENTION_K:
                for rank_value, row in enumerate(
                    paper.head(k).itertuples(index=False), 1
                ):
                    rows.append(
                        {
                            "model": model,
                            "candidate_type": "paper_heads",
                            "source_attack": "paper",
                            "direction": "existing",
                            "k": k,
                            "layer": int(row.layer),
                            "head": int(row.head),
                            "rank": rank_value,
                            "selection_metric": "paper_head",
                            "selection_value": 1.0,
                            "attack_consistency": False,
                            "notes": "Only five predefined paper heads are available",
                        }
                    )
    return pd.DataFrame(rows)


def candidate_row(
    model: str,
    candidate_type: str,
    attack: str,
    direction: str,
    k: int,
    row: Any,
    rank_value: int,
    notes: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "candidate_type": candidate_type,
        "source_attack": attack,
        "direction": direction,
        "k": k,
        "layer": int(row.layer),
        "head": int(row.head),
        "rank": rank_value,
        "selection_metric": "delta_asym_ratio",
        "selection_value": float(row.delta_asym_ratio),
        "attack_consistency": False,
        "notes": notes,
    }


def save_plots(
    output_dir: Path,
    conditions: Mapping[str, Mapping[str, Any]],
    relationships: pd.DataFrame,
    quadrants: pd.DataFrame,
    topk: pd.DataFrame,
    stability: pd.DataFrame,
    consensus: Mapping[tuple[str, str], pd.DataFrame],
    layer_summary: pd.DataFrame,
    dpi: int,
) -> list[Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    colors = {"AutoDAN": "tab:blue", "GCG": "tab:orange"}
    for scope in ("full", "matched"):
        specs = [(key, value) for key, value in conditions.items() if value["scope"] == scope]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        try:
            for axis, (condition_id, spec) in zip(axes.flat, specs):
                frame = spec["frame"]
                axis.scatter(frame.delta_asym_ratio, frame.element_std_shift, s=10, alpha=0.45)
                axis.axhline(0, color="black", linewidth=0.7, linestyle="--")
                axis.axvline(0, color="black", linewidth=0.7, linestyle="--")
                rel = relationships[
                    (relationships.condition_id == condition_id)
                    & (relationships.metric_x == "delta_asym_ratio")
                    & (relationships.metric_y == "element_std_shift")
                ].iloc[0]
                axis.set_title(f"{spec['model']} / {spec['attack']}\nr={rel.pearson:.3f}, rho={rel.spearman:.3f}")
                axis.set_xlabel("delta_asym_ratio")
                axis.set_ylabel("element_std_shift")
                axis.grid(alpha=0.15)
            fig.suptitle(f"{scope.title()} head relationships (exploratory)")
            fig.tight_layout()
            path = plot_dir / f"{scope}_delta_asym_ratio_vs_element_std_shift.png"
            fig.savefig(path, dpi=dpi)
            paths.append(path)
        finally:
            plt.close(fig)
    pivot = quadrants.pivot(index="condition_id", columns="quadrant", values="rate").fillna(0)
    fig, axis = plt.subplots(figsize=(12, 6))
    try:
        bottom = np.zeros(len(pivot))
        for quadrant in ("Q1", "Q2", "Q3", "Q4", "zero"):
            values = pivot.get(quadrant, pd.Series(0, index=pivot.index)).to_numpy()
            axis.bar(pivot.index, values, bottom=bottom, label=quadrant)
            bottom += values
        axis.set_ylabel("Head rate")
        axis.set_title("delta_asym_ratio / element_std_shift quadrants")
        axis.tick_params(axis="x", rotation=45)
        axis.legend(ncol=5)
        fig.tight_layout()
        path = plot_dir / "quadrant_rate_stacked.png"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    try:
        for axis, direction in zip(axes, ("increase", "decrease")):
            subset = topk[(topk.scope == "matched") & (topk.direction == direction)]
            rate_column = "std_shift_positive_rate" if direction == "increase" else "std_shift_negative_rate"
            for condition_id, group in subset.groupby("condition_id"):
                axis.plot(group.k, group[rate_column], marker="o", label=condition_id)
            axis.set_title(f"{direction}: expected std-shift sign rate")
            axis.set_xlabel("signed Top-k")
            axis.set_ylabel("Rate")
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.2)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        path = plot_dir / "topk_std_shift_sign_rates.png"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    finally:
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(11, 6))
    try:
        subset = stability[(stability.metric == "delta_asym_ratio") & (stability.direction == "increase")]
        labels = subset.model + " / " + subset.attack
        axis.bar(labels, subset.spearman)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Full vs matched Spearman")
        axis.set_title("Full/matched delta_asym_ratio rank stability")
        axis.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = plot_dir / "full_vs_matched_rank_stability.png"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    finally:
        plt.close(fig)
    for model_slug, model_name in (("llama2", "Llama-2-7b-chat-hf"), ("mistral", "Mistral-7B-Instruct-v0.2")):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        try:
            for axis, direction in zip(axes, ("increase", "decrease")):
                frame = consensus[(model_name, direction)].head(20).iloc[::-1]
                axis.barh(frame.head_label, frame.minimum_absolute_metric_magnitude)
                axis.set_title(f"{direction} consensus Top20")
                axis.set_xlabel("Minimum |delta_asym_ratio| across attacks")
            fig.suptitle(model_name)
            fig.tight_layout()
            path = plot_dir / f"{model_slug}_attack_consistent_top20.png"
            fig.savefig(path, dpi=dpi)
            paths.append(path)
        finally:
            plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    try:
        for axis, model in zip(axes, ("Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.2")):
            subset = layer_summary[layer_summary.model == model]
            axis.plot(subset.normalized_layer_depth, subset.delta_asym_ratio_mean_autodan, label="AutoDAN")
            axis.plot(subset.normalized_layer_depth, subset.delta_asym_ratio_mean_gcg, label="GCG")
            axis.axhline(0, color="black", linewidth=0.7, linestyle="--")
            axis.set_title(model)
            axis.set_xlabel("Normalized layer depth")
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Layer mean delta_asym_ratio")
        axes[1].legend()
        fig.tight_layout()
        path = plot_dir / "normalized_layer_depth_profile.png"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    finally:
        plt.close(fig)
    heat = relationships.pivot_table(
        index="condition_id", columns=["metric_x", "metric_y"], values="spearman"
    )
    fig, axis = plt.subplots(figsize=(11, 6))
    try:
        image = axis.imshow(heat.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        axis.set_yticks(range(len(heat.index)), heat.index)
        labels = [f"{x}\nvs {y}" for x, y in heat.columns]
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        fig.colorbar(image, ax=axis, label="Spearman rho")
        axis.set_title("Within-condition metric relationship heatmap")
        fig.tight_layout()
        path = plot_dir / "metric_correlation_heatmap.png"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    finally:
        plt.close(fig)
    return paths


def current_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> None:
    args = get_args()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    model_names = {
        "llama2": "Llama-2-7b-chat-hf",
        "mistral": "Mistral-7B-Instruct-v0.2",
    }
    conditions: dict[str, dict[str, Any]] = {}
    input_paths: dict[str, str] = {}
    for model_slug, model_name in model_names.items():
        for attack_slug, attack_name in (("autodan", "AutoDAN"), ("gcg", "GCG")):
            for scope in ("full", "matched"):
                key = f"{model_slug}_{attack_slug}_{scope}"
                path = Path(getattr(args, key))
                input_paths[key] = str(path)
                conditions[key] = {
                    "model": model_name,
                    "attack": attack_name,
                    "scope": scope,
                    "frame": read_head_metrics(path, key, scope),
                }
    metadata_paths = {
        "llama2": Path(args.llama2_matched_metadata),
        "mistral": Path(args.mistral_matched_metadata),
    }
    metadata_inputs = {model: read_metadata(path) for model, path in metadata_paths.items()}
    definitions = [json.dumps(meta.get("metric_definitions"), sort_keys=True) for meta in metadata_inputs.values()]
    if len(set(definitions)) != 1:
        raise ValueError("Matched metric definitions differ between models")
    if {int(meta.get("std_ddof", -1)) for meta in metadata_inputs.values()} != {0}:
        raise ValueError("Matched std ddof must be 0")
    if {int(meta.get("tail_tokens", -1)) for meta in metadata_inputs.values()} != {16}:
        raise ValueError("Expected tail16 metadata")
    relationships, quadrants = build_within_condition_outputs(conditions)
    topk = build_topk_shift_summary(conditions)
    stability_parts = []
    stability_topk_parts = []
    for model_slug, model_name in model_names.items():
        for attack_slug, attack_name in (("autodan", "AutoDAN"), ("gcg", "GCG")):
            summary, overlap = build_full_matched_stability(
                conditions[f"{model_slug}_{attack_slug}_full"]["frame"],
                conditions[f"{model_slug}_{attack_slug}_matched"]["frame"],
                model_name,
                attack_name,
            )
            stability_parts.append(summary)
            stability_topk_parts.append(overlap)
    stability = pd.concat(stability_parts, ignore_index=True)
    stability_topk = pd.concat(stability_topk_parts, ignore_index=True)
    matched_by_model = {
        model_name: {
            "AutoDAN": conditions[f"{model_slug}_autodan_matched"]["frame"],
            "GCG": conditions[f"{model_slug}_gcg_matched"]["frame"],
        }
        for model_slug, model_name in model_names.items()
    }
    consensus: dict[tuple[str, str], pd.DataFrame] = {}
    for model_name, attacks in matched_by_model.items():
        for direction in ("increase", "decrease"):
            frame = build_consensus(attacks["AutoDAN"], attacks["GCG"], model_name, direction)
            consensus[(model_name, direction)] = frame
            slug = "llama2" if model_name.startswith("Llama") else "mistral"
            frame.to_csv(output_dir / f"{slug}_attack_consistent_{direction}_heads.csv", index=False)
    matched_specs = {
        key: value for key, value in conditions.items() if value["scope"] == "matched"
    }
    attack_specific = build_attack_specific_vs_consensus(matched_specs, consensus)
    layer_consensus = build_layer_consensus(matched_by_model)
    from scripts.analyze_matched_attention_head_metrics import build_topk_overlap
    matched_attack_overlap = {
        model: build_topk_overlap(attacks["AutoDAN"], attacks["GCG"])
        for model, attacks in matched_by_model.items()
    }
    hypotheses = build_hypothesis_evaluations(
        relationships, topk, matched_attack_overlap, layer_consensus
    )
    cross_model_patterns = build_cross_model_pattern_comparison(
        conditions, consensus, layer_consensus, hypotheses
    )
    intervention = build_intervention_candidates(
        matched_specs,
        consensus,
        {
            model_name: conditions[f"{model_slug}_autodan_full"]["frame"]
            for model_slug, model_name in model_names.items()
        },
    )
    relationships.to_csv(output_dir / "within_condition_metric_relationships.csv", index=False)
    quadrants.to_csv(output_dir / "delta_asym_ratio_std_quadrants.csv", index=False)
    topk.to_csv(output_dir / "topk_distribution_shift_summary.csv", index=False)
    stability.to_csv(output_dir / "full_vs_matched_head_stability.csv", index=False)
    stability_topk.to_csv(output_dir / "full_vs_matched_topk_stability.csv", index=False)
    attack_specific.to_csv(output_dir / "attack_specific_vs_consensus_heads.csv", index=False)
    layer_consensus.to_csv(output_dir / "layer_consensus_summary.csv", index=False)
    cross_model_patterns.to_csv(
        output_dir / "cross_model_pattern_comparison.csv", index=False
    )
    intervention.to_csv(output_dir / "intervention_candidate_head_sets.csv", index=False)
    pngs = save_plots(
        output_dir,
        conditions,
        relationships,
        quadrants,
        topk,
        stability,
        consensus,
        layer_consensus,
        args.dpi,
    )
    if len(pngs) != 9 or any(not path.is_file() or path.stat().st_size == 0 for path in pngs):
        raise ValueError("Expected nine non-empty plots")
    elapsed = time.perf_counter() - started
    audit = {
        key: {
            "rows": len(spec["frame"]),
            "layers": int(spec["frame"].layer.nunique()),
            "heads_per_layer": int(spec["frame"]["head"].nunique()),
            "prompt_count": int(spec["frame"].element_value_count.iloc[0] // 120),
            "tail_tokens": 16,
            "elements_per_head": 120,
        }
        for key, spec in conditions.items()
    }
    metadata = {
        "analysis_type": "cross_model_attention_pattern_summary",
        "exploratory": True,
        "held_out": False,
        "models": list(model_names.values()),
        "attacks": ["AutoDAN", "GCG"],
        "statistical_unit": "layer/head; 1024 heads per condition",
        "cross_model_head_identity_comparison": False,
        "cross_model_comparison_basis": [
            "metric relationships",
            "quadrant rates",
            "signed Top-k patterns",
            "normalized layer depth",
        ],
        "metric_definitions": metadata_inputs["llama2"].get("metric_definitions"),
        "std_ddof": 0,
        "consensus_primary_ranking": "worst signed rank ascending, then mean rank",
        "consensus_secondary_ranking": "mean percentile rank descending",
        "random_head_sets": {
            "generated": False,
            "required_k": [3, 5, 10],
            "sampling_rule": "uniform without replacement over the complete model-specific layer/head grid; use a recorded seed and exclude selected comparison heads within each draw",
        },
        "aie_availability": {
            "Llama-2 AutoDAN": "available in the merged input and included",
            "Llama-2 GCG": "unavailable",
            "Mistral AutoDAN": "no comparable AIE columns in the merged input; a separate no-target-space artifact was not mixed into this candidate table",
            "Mistral GCG": "unavailable",
        },
        "input_audit": audit,
        "input_paths": input_paths,
        "matched_metadata_paths": {key: str(path) for key, path in metadata_paths.items()},
        "hypothesis_evaluations": hypotheses.to_dict("records"),
        "limitations": [
            "exploratory post-hoc internal analysis",
            "not a held-out evaluation",
            "does not measure intervention efficacy or causal importance",
            "attack success definitions/composition are not harmonized",
            "Llama-2 and Mistral head identities are not compared directly",
        ],
        "source_commit": current_commit(),
        "cli_args": vars(args),
        "execution_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "execution_time_seconds": elapsed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"conditions: {len(conditions)}")
    print(f"relationship rows: {len(relationships)}")
    print(f"quadrant rows: {len(quadrants)}")
    print(f"intervention candidate rows: {len(intervention)}")
    print(f"plots: {len(pngs)}")
    print(f"execution time seconds: {elapsed:.3f}")
    print(f"output directory: {output_dir}")


if __name__ == "__main__":
    main()
