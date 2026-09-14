#!/usr/bin/env python3
"""Explain frozen GCG Top-500 first-token effects on GCG and AutoDAN."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_gcg_first_token_reference_ablation import (  # noqa: E402
    load_model_and_tokenizer,
)
from scripts.evaluate_gcg_position_element_topk_intervention import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MODEL,
    load_selection,
    validate_prompt_runtime_invariants,
)
from utils.asym_score_intervention import (  # noqa: E402
    temporarily_disable_default_dynamic_cache,
)
from utils.experiment_runtime import git_value, seed_everything  # noqa: E402
from utils.gcg_position_element_topk_intervention import (  # noqa: E402
    patch_llama_position_element_attention,
    selection_diagnostics,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    llama_model_config_summary,
    read_json,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.gcg_wasserstein_reference_evaluation import (  # noqa: E402
    ensure_output_absent,
)
from utils.reference_asym_score_intervention import (  # noqa: E402
    load_validated_reference_artifacts,
    validate_loaded_llama_model,
)
from scripts.diagnose_gcg_reference_factorization import (  # noqa: E402
    EXPECTED_SCORE_ARTIFACT_SHA256,
    EXPECTED_SELECTION_MANIFEST_SHA256,
    EXPECTED_TOP5,
)


DEFAULT_GCG_RUN = (
    "data/outputs/gcg_wasserstein_reference_intervention/llama2/"
    "independent100/evaluation_20260911T064911Z"
)
DEFAULT_AUTODAN_RUN = (
    "data/outputs/gcg_position_element_topk/autodan130_transfer/"
    "full130_20260914T085700Z"
)
DATASETS = {
    "GCG": {
        "group": "independent100",
        "count": 100,
        "indices": list(range(220, 320)),
        "response_changed": 23,
        "argmax_changed": 23,
        "newly_defended": 15,
    },
    "AutoDAN": {
        "group": "autodan130",
        "count": 130,
        "indices": list(range(130)),
        "response_changed": 0,
        "argmax_changed": 0,
        "newly_defended": 0,
    },
}
K = 500
ALPHA = 1.0
INTERVENTION_TIMING = "first_token_only"
TIE_ABSOLUTE_TOLERANCE = 1e-6
TIE_RELATIVE_TOLERANCE = 2**-10
TIE_RULE = (
    "abs(a-b) <= absolute_tolerance + relative_tolerance * "
    "max(abs(a), abs(b)); both the saved defense top1-top2 margin and the "
    "current max-vs-saved-token gap must satisfy the rule"
)
SUMMARY_STATISTICS = (
    "mean",
    "median",
    "std",
    "min",
    "q25",
    "q75",
    "q95",
    "max",
)
ATTENTION_METRICS = (
    "selected_abs_before_reference_mean",
    "selected_abs_before_reference_median",
    "selected_abs_before_reference_max",
    "selected_abs_before_reference_l1",
    "selected_abs_before_reference_l2",
    "selected_abs_after_before_mean",
    "selected_abs_after_before_median",
    "selected_abs_after_before_max",
    "selected_abs_after_before_l1",
    "selected_abs_after_before_l2",
    "selected_abs_after_reference_mean",
    "selected_abs_after_reference_max",
)
LOGIT_METRICS = (
    "max_absolute_logit_difference",
    "logit_l1_distance",
    "logit_l2_distance",
    "relative_logit_l2_distance",
    "kl_baseline_to_defense",
    "baseline_top1_logit",
    "baseline_top2_logit",
    "baseline_top1_top2_margin",
    "defense_rank_of_baseline_top1",
    "top5_token_overlap_count",
    "top5_token_overlap_rate",
    "max_abs_logit_difference_to_baseline_margin_ratio",
    "baseline_top1_decision_headroom_after_defense",
    "top1_changed",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("cpu-validate", "run"), default="cpu-validate"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--gcg-run", default=DEFAULT_GCG_RUN)
    parser.add_argument("--autodan-run", default=DEFAULT_AUTODAN_RUN)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        help="Leading rows per dataset for implementation smoke validation only.",
    )
    return parser.parse_args(argv)


def _jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid JSON at line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}: line {line_number} is not an object")
        rows.append(row)
    return rows


def _sha256_json(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_existing_run(run_dir: Path, *, expected_group: str) -> dict:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    artifact_manifest_path = run_dir / "artifact_manifest.json"
    manifest = read_json(manifest_path)
    artifact_manifest = read_json(artifact_manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"source run is not complete: {run_dir}")
    if manifest.get("alpha") != ALPHA:
        raise ValueError(f"source run alpha mismatch: {run_dir}")
    if manifest.get("intervention_timing") != INTERVENTION_TIMING:
        raise ValueError(f"source run intervention timing mismatch: {run_dir}")
    if manifest.get("dataset_groups") != [expected_group]:
        raise ValueError(f"source run dataset group mismatch: {run_dir}")
    if manifest.get("artifact_verification", {}).get(
        "selection_manifest_sha256"
    ) != EXPECTED_SELECTION_MANIFEST_SHA256:
        raise ValueError(f"source run selection artifact mismatch: {run_dir}")
    if manifest.get("artifact_verification", {}).get(
        "score_artifact_sha256"
    ) != EXPECTED_SCORE_ARTIFACT_SHA256:
        raise ValueError(f"source run score artifact mismatch: {run_dir}")
    if [row.get("k") for row in manifest.get("selection", {}).get("conditions", [])] != [K]:
        raise ValueError(f"source run is not frozen to Top-500: {run_dir}")
    actual_artifact_manifest_hash = sha256_file(artifact_manifest_path)
    if manifest.get("artifact_manifest_sha256") != actual_artifact_manifest_hash:
        raise ValueError(f"source artifact manifest hash mismatch: {run_dir}")
    artifacts = artifact_manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"source artifact manifest is empty: {run_dir}")
    for record in artifacts:
        path = run_dir / record["relative_path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"source artifact hash mismatch: {path}")
    return {
        "run_dir": str(run_dir),
        "run_manifest_sha256": sha256_file(manifest_path),
        "artifact_manifest_sha256": actual_artifact_manifest_hash,
        "verified_artifact_count": len(artifacts),
        "evaluator_source_sha256": manifest.get("evaluator_source_sha256"),
        "execution_git_head": manifest.get("execution_git_head"),
        "model": manifest.get("model"),
        "source_score_hashes": manifest.get("source_score_hashes"),
    }


def load_existing_records(
    run_dir: Path, *, dataset: str, limit: int | None
) -> tuple[list[dict], dict]:
    specification = DATASETS[dataset]
    group = specification["group"]
    baseline_path = run_dir / f"{group}_baseline.jsonl"
    defense_path = run_dir / group / "k_500" / "prompt_results.jsonl"
    baseline = _jsonl(baseline_path)
    defense = _jsonl(defense_path)
    expected_count = specification["count"]
    if len(baseline) != expected_count or len(defense) != expected_count:
        raise ValueError(f"{dataset} source record count mismatch")
    expected_indices = specification["indices"]
    if [row.get("dataset_index") for row in baseline] != expected_indices:
        raise ValueError(f"{dataset} baseline indices mismatch")
    if [row.get("dataset_index") for row in defense] != expected_indices:
        raise ValueError(f"{dataset} defense indices mismatch")

    prepared = []
    for baseline_row, defense_row in zip(baseline, defense):
        index = baseline_row["dataset_index"]
        if baseline_row.get("prompt_id") != defense_row.get("prompt_id"):
            raise ValueError(f"{dataset} prompt ID mismatch at index={index}")
        if baseline_row.get("prompt") != defense_row.get("prompt"):
            raise ValueError(f"{dataset} prompt mismatch at index={index}")
        input_ids = baseline_row.get("input_ids")
        if input_ids != defense_row.get("input_ids"):
            raise ValueError(f"{dataset} exact input IDs mismatch at index={index}")
        if (
            not isinstance(input_ids, list)
            or not input_ids
            or any(type(token) is not int for token in input_ids)
            or len(input_ids) != baseline_row.get("prompt_length")
            or len(input_ids) != defense_row.get("prompt_length")
        ):
            raise ValueError(f"{dataset} invalid saved input IDs at index={index}")
        if baseline_row.get("generation_failed") is not False:
            raise ValueError(f"{dataset} saved baseline failed at index={index}")
        if defense_row.get("generation_failed") is not False:
            raise ValueError(f"{dataset} saved defense failed at index={index}")
        if defense_row.get("k") != K or defense_row.get("alpha") != ALPHA:
            raise ValueError(f"{dataset} saved defense condition mismatch at index={index}")
        if defense_row.get("intervention_timing") != INTERVENTION_TIMING:
            raise ValueError(f"{dataset} saved timing mismatch at index={index}")
        if not baseline_row.get("response_token_ids"):
            raise ValueError(f"{dataset} saved baseline tokens missing at index={index}")
        if not defense_row.get("response_token_ids"):
            raise ValueError(f"{dataset} saved defense tokens missing at index={index}")
        prepared.append(
            {
                "dataset": dataset,
                "dataset_group": group,
                "dataset_index": index,
                "prompt_id": baseline_row["prompt_id"],
                "input_ids": input_ids,
                "prompt_length": len(input_ids),
                "existing_baseline_first_token": baseline_row["response_token_ids"][0],
                "existing_defense_first_token": defense_row["response_token_ids"][0],
                "existing_baseline_success": baseline_row["success"],
                "existing_defense_success": defense_row["success"],
                "existing_response_changed": not defense_row["response_equal"],
                "existing_first_token_changed": not defense_row[
                    "first_token_argmax_equal"
                ],
                "existing_transition": defense_row["transition"],
                "existing_first_token_logit_metrics": defense_row[
                    "first_token_logit_metrics"
                ],
            }
        )

    response_changed = sum(row["existing_response_changed"] for row in prepared)
    argmax_changed = sum(row["existing_first_token_changed"] for row in prepared)
    newly_defended = sum(
        row["existing_transition"] == "newly_defended" for row in prepared
    )
    if response_changed != specification["response_changed"]:
        raise ValueError(f"{dataset} saved response-change count mismatch")
    if argmax_changed != specification["argmax_changed"]:
        raise ValueError(f"{dataset} saved argmax-change count mismatch")
    if newly_defended != specification["newly_defended"]:
        raise ValueError(f"{dataset} saved newly-defended count mismatch")

    full_input_hash = _sha256_json([row["input_ids"] for row in prepared])
    if limit is not None:
        if type(limit) is not int or not 1 <= limit <= min(
            item["count"] for item in DATASETS.values()
        ):
            raise ValueError("--limit-per-dataset must be in [1, 100]")
        prepared = prepared[:limit]
    return prepared, {
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": sha256_file(baseline_path),
        "defense_path": str(defense_path.resolve()),
        "defense_sha256": sha256_file(defense_path),
        "full_record_count": expected_count,
        "analyzed_record_count": len(prepared),
        "indices": expected_indices,
        "exact_input_ids_verified": True,
        "full_input_ids_sha256": full_input_hash,
        "saved_response_changed_count": response_changed,
        "saved_first_token_changed_count": argmax_changed,
        "saved_newly_defended_count": newly_defended,
    }


def validate_inputs(args) -> tuple[object, dict, dict[str, list[dict]], dict]:
    model_path = Path(args.model).resolve()
    selection, verification, score_metadata, score_paths = load_selection(
        Path(args.artifact_dir).resolve(), k_values=(K,), model_path=model_path
    )
    source_runs = {}
    datasets = {}
    dataset_sources = {}
    for dataset, run_value in (
        ("GCG", args.gcg_run),
        ("AutoDAN", args.autodan_run),
    ):
        run_dir = Path(run_value).resolve()
        source_runs[dataset] = verify_existing_run(
            run_dir, expected_group=DATASETS[dataset]["group"]
        )
        if Path(source_runs[dataset]["model"]).resolve() != model_path:
            raise ValueError(f"{dataset} source run model mismatch")
        datasets[dataset], dataset_sources[dataset] = load_existing_records(
            run_dir, dataset=dataset, limit=args.limit_per_dataset
        )
    if source_runs["GCG"]["source_score_hashes"] != source_runs["AutoDAN"][
        "source_score_hashes"
    ]:
        raise ValueError("GCG/AutoDAN source score hashes differ")
    current_source_hashes = {
        name: sha256_file(path) for name, path in score_paths.items()
    }
    if source_runs["GCG"]["source_score_hashes"] != current_source_hashes:
        raise ValueError("source runs do not match current frozen score artifacts")
    frozen = {
        "k": K,
        "alpha": ALPHA,
        "intervention_timing": INTERVENTION_TIMING,
        "reference": "Selection20 original mean",
        "selection_or_reference_from_analysis_data": False,
        "artifact_verification": verification,
        "source_score_hashes": current_source_hashes,
        "score_metadata": score_metadata,
        "selection": selection_diagnostics(selection),
        "top500_identity_sha256": _sha256_json(
            [
                {
                    key: row[key]
                    for key in ("rank", "layer", "head", "position", "flat_index")
                }
                for row in selection.topk_records(K)
            ]
        ),
    }
    sources = {
        "source_runs": source_runs,
        "datasets": dataset_sources,
    }
    return selection, frozen, datasets, sources


def distribution(values: list[float | int | bool]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary values must be a non-empty finite vector")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def vector_magnitudes(values: torch.Tensor, *, prefix: str) -> dict:
    values = values.double()
    if values.ndim != 1 or values.numel() != K:
        raise ValueError(f"{prefix} must contain exactly {K} values")
    if not bool(torch.isfinite(values).all().item()) or bool((values < 0).any().item()):
        raise ValueError(f"{prefix} must contain finite non-negative values")
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_median": float(values.median().item()),
        f"{prefix}_max": float(values.max().item()),
        f"{prefix}_l1": float(values.sum().item()),
        f"{prefix}_l2": float(torch.linalg.vector_norm(values).item()),
    }


def compute_logit_diagnostics(
    baseline: torch.Tensor, defense: torch.Tensor
) -> dict:
    baseline = baseline.double().cpu()
    defense = defense.double().cpu()
    if baseline.ndim != 1 or defense.shape != baseline.shape:
        raise ValueError("logit vectors must be matching rank-1 tensors")
    if not bool(torch.isfinite(baseline).all().item()) or not bool(
        torch.isfinite(defense).all().item()
    ):
        raise ValueError("logit vectors contain NaN or infinity")
    delta = defense - baseline
    baseline_top = torch.topk(baseline, 5)
    defense_top = torch.topk(defense, 5)
    baseline_top1 = int(baseline_top.indices[0].item())
    defense_top1 = int(defense_top.indices[0].item())
    baseline_margin = float((baseline_top.values[0] - baseline_top.values[1]).item())
    if baseline_margin <= 0:
        raise ValueError("baseline top-1 margin must be positive")
    defense_order = torch.argsort(defense, descending=True, stable=True)
    rank = int((defense_order == baseline_top1).nonzero(as_tuple=False)[0].item()) + 1
    other = defense.clone()
    other[baseline_top1] = -torch.inf
    decision_headroom = float(
        (defense[baseline_top1] - other.max()).item()
    )
    log_p = torch.log_softmax(baseline, dim=0)
    log_q = torch.log_softmax(defense, dim=0)
    kl = float(torch.sum(torch.exp(log_p) * (log_p - log_q)).item())
    if kl < 0 and kl > -1e-12:
        kl = 0.0
    baseline_norm = float(torch.linalg.vector_norm(baseline).item())
    if baseline_norm == 0:
        raise ValueError("baseline logit norm is zero")
    result = {
        "max_absolute_logit_difference": float(delta.abs().max().item()),
        "logit_l1_distance": float(delta.abs().sum().item()),
        "logit_l2_distance": float(torch.linalg.vector_norm(delta).item()),
        "relative_logit_l2_distance": float(
            torch.linalg.vector_norm(delta).item() / baseline_norm
        ),
        "kl_baseline_to_defense": kl,
        "baseline_top1_token_id": baseline_top1,
        "defense_top1_token_id": defense_top1,
        "baseline_top1_logit": float(baseline_top.values[0].item()),
        "baseline_top2_logit": float(baseline_top.values[1].item()),
        "baseline_top1_top2_margin": baseline_margin,
        "defense_rank_of_baseline_top1": rank,
        "top5_token_overlap_count": len(
            set(baseline_top.indices.tolist()) & set(defense_top.indices.tolist())
        ),
        "top5_token_overlap_rate": len(
            set(baseline_top.indices.tolist()) & set(defense_top.indices.tolist())
        )
        / 5,
        "top1_changed": baseline_top1 != defense_top1,
        "max_abs_logit_difference_to_baseline_margin_ratio": float(
            delta.abs().max().item() / baseline_margin
        ),
        "baseline_top1_decision_headroom_after_defense": decision_headroom,
    }
    if not all(
        math.isfinite(float(value))
        for value in result.values()
        if type(value) in (int, float)
    ):
        raise ValueError("logit diagnostics contain NaN or infinity")
    return result


def numerical_tie_tolerance(first: float, second: float) -> float:
    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError("tie comparison values must be finite")
    return TIE_ABSOLUTE_TOLERANCE + TIE_RELATIVE_TOLERANCE * max(
        abs(first), abs(second)
    )


def validate_defense_argmax_reproduction(
    *, row: dict, defense_logits: torch.Tensor
) -> dict:
    """Require a strict match or a predeclared two-sided numerical tie."""

    expected_token = int(row["existing_defense_first_token"])
    observed_token = int(torch.argmax(defense_logits).item())
    if observed_token == expected_token:
        return {
            "match_type": "strict",
            "expected_token_id": expected_token,
            "observed_token_id": observed_token,
        }

    saved = row["existing_first_token_logit_metrics"]
    saved_margin = float(saved["intervention_top1_top2_margin"])
    saved_scale = float(saved["intervention_logit_at_baseline_top1"])
    saved_margin_tolerance = numerical_tie_tolerance(saved_scale, saved_scale)
    current_max = float(defense_logits.max().item())
    expected_logit = float(defense_logits[expected_token].item())
    current_gap = current_max - expected_logit
    current_gap_tolerance = numerical_tie_tolerance(current_max, expected_logit)
    equivalent_mask = (current_max - defense_logits) <= current_gap_tolerance
    equivalent_tokens = equivalent_mask.nonzero(as_tuple=False).flatten().tolist()
    saved_is_tie = 0.0 <= saved_margin <= saved_margin_tolerance
    expected_is_current_tie = (
        expected_token in equivalent_tokens
        and 0.0 <= current_gap <= current_gap_tolerance
    )
    if not saved_is_tie or not expected_is_current_tie:
        raise RuntimeError(
            "defense first-token argmax reproduction mismatch outside tie rule: "
            f"dataset={row['dataset']} index={row['dataset_index']} "
            f"expected={expected_token} observed={observed_token} "
            f"saved_margin={saved_margin} current_gap={current_gap}"
        )
    return {
        "match_type": "tie_allowed",
        "dataset": row["dataset"],
        "dataset_index": row["dataset_index"],
        "prompt_id": row["prompt_id"],
        "expected_token_id": expected_token,
        "observed_token_id": observed_token,
        "saved_top1_top2_margin": saved_margin,
        "saved_margin_tolerance": saved_margin_tolerance,
        "current_max_logit": current_max,
        "current_expected_token_logit": expected_logit,
        "current_max_expected_token_gap": current_gap,
        "current_gap_tolerance": current_gap_tolerance,
        "current_numerically_equivalent_top_token_ids": equivalent_tokens,
        "current_numerically_equivalent_top_token_count": len(equivalent_tokens),
    }


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty row group")
    metrics = ATTENTION_METRICS + LOGIT_METRICS
    return {
        "prompt_count": len(rows),
        "metrics": {
            metric: distribution([row[metric] for row in rows])
            for metric in metrics
        },
    }


def build_comparison_summary(rows: list[dict], *, full: bool) -> dict:
    gcg = [row for row in rows if row["dataset"] == "GCG"]
    autodan = [row for row in rows if row["dataset"] == "AutoDAN"]
    groups = {
        "GCG_all": gcg,
        "AutoDAN_all": autodan,
        "GCG_response_changed": [
            row for row in gcg if row["existing_response_changed"]
        ],
        "GCG_response_unchanged": [
            row for row in gcg if not row["existing_response_changed"]
        ],
        "GCG_newly_defended": [
            row for row in gcg if row["existing_transition"] == "newly_defended"
        ],
        "GCG_response_changed_label_unchanged": [
            row
            for row in gcg
            if row["existing_response_changed"]
            and row["existing_baseline_success"] == row["existing_defense_success"]
        ],
    }
    if full:
        expected_counts = {
            "GCG_all": 100,
            "AutoDAN_all": 130,
            "GCG_response_changed": 23,
            "GCG_response_unchanged": 77,
            "GCG_newly_defended": 15,
            "GCG_response_changed_label_unchanged": 8,
        }
        observed = {name: len(items) for name, items in groups.items()}
        if observed != expected_counts:
            raise ValueError(f"analysis subgroup counts mismatch: {observed}")
    summaries = {
        name: summarize_rows(items) for name, items in groups.items() if items
    }
    key_metrics = (
        "selected_abs_before_reference_mean",
        "selected_abs_after_before_mean",
        "selected_abs_before_reference_l1",
        "selected_abs_after_before_l1",
        "max_absolute_logit_difference",
        "logit_l2_distance",
        "relative_logit_l2_distance",
        "kl_baseline_to_defense",
        "baseline_top1_top2_margin",
        "max_abs_logit_difference_to_baseline_margin_ratio",
        "top1_changed",
    )
    return {
        "schema_version": 1,
        "analysis_role": "post_hoc_explanatory_not_method_selection_or_tuning",
        "summary_statistics": list(SUMMARY_STATISTICS),
        "standard_deviation": "sample standard deviation (ddof=1)",
        "groups": summaries,
        "key_gcg_autodan_comparison": {
            metric: {
                "GCG": summaries["GCG_all"]["metrics"][metric],
                "AutoDAN": summaries["AutoDAN_all"]["metrics"][metric],
            }
            for metric in key_metrics
        },
    }


def first_forward_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(), temporarily_disable_default_dynamic_cache(model):
        output = model(
            input_ids=input_ids.to(model.device),
            use_cache=False,
            return_dict=True,
        )
    logits = output.logits[0, -1].float().detach().cpu()
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("first-forward logits contain NaN or infinity")
    return logits


def run_analysis(args, selection, frozen, datasets, sources) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for analysis; refusing CPU model load")
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    output_dir = ensure_output_absent(args.output_dir).resolve()
    model_path = Path(args.model).resolve()
    audited_reference = load_validated_reference_artifacts(
        Path(args.artifact_dir).resolve(),
        expected_heads=EXPECTED_TOP5,
        expected_selection_manifest_sha256=EXPECTED_SELECTION_MANIFEST_SHA256,
        expected_score_artifact_sha256=EXPECTED_SCORE_ARTIFACT_SHA256,
        expected_model_path=model_path,
    )
    seed_everything(42)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    created_at = utc_now()
    model, tokenizer = load_model_and_tokenizer(model_path)
    model_validation = validate_loaded_llama_model(model, audited_reference)
    top500 = list(selection.topk_records(K))
    top500_identity = [
        (int(row["layer"]), int(row["head"]), int(row["position"]))
        for row in top500
    ]

    source_rows = [row for dataset in ("GCG", "AutoDAN") for row in datasets[dataset]]
    baseline_logits = {}
    baseline_started = time.perf_counter()
    for position, row in enumerate(source_rows, 1):
        input_ids = torch.tensor([row["input_ids"]], dtype=torch.long)
        logits = first_forward_logits(model, input_ids)
        top1 = int(torch.argmax(logits).item())
        if top1 != row["existing_baseline_first_token"]:
            raise RuntimeError(
                "baseline first-token argmax reproduction mismatch: "
                f"dataset={row['dataset']} index={row['dataset_index']}"
            )
        baseline_logits[(row["dataset"], row["dataset_index"])] = logits
        print(
            f"[baseline {position}/{len(source_rows)}] "
            f"{row['dataset']} index={row['dataset_index']}",
            flush=True,
        )
    torch.cuda.synchronize()
    baseline_seconds = time.perf_counter() - baseline_started

    per_prompt = []
    defense_reproduction_records = []
    defense_started = time.perf_counter()
    with patch_llama_position_element_attention(
        model,
        selection=selection,
        k=K,
        alpha=ALPHA,
        intervention_timing=INTERVENTION_TIMING,
        capture_selected_values=True,
    ) as controller:
        for position, row in enumerate(source_rows, 1):
            controller.clear()
            controller.set_prompt_context(prompt_length=row["prompt_length"])
            input_ids = torch.tensor([row["input_ids"]], dtype=torch.long)
            defense_logits = first_forward_logits(model, input_ids)
            diagnostics = controller.summary()
            validate_prompt_runtime_invariants(diagnostics, k=K)
            trace = controller.selected_value_trace()
            if len(trace) != K:
                raise RuntimeError("selected-element trace count mismatch")
            by_identity = {
                (item["layer"], item["head"], item["position"]): item
                for item in trace
            }
            if len(by_identity) != K or set(by_identity) != set(top500_identity):
                raise RuntimeError("selected-element trace identities mismatch")
            ordered = [by_identity[identity] for identity in top500_identity]
            before = torch.tensor(
                [item["score_before"] for item in ordered], dtype=torch.float64
            )
            reference = torch.tensor(
                [item["reference"] for item in ordered], dtype=torch.float64
            )
            constructed_after = torch.tensor(
                [item["score_after_float32"] for item in ordered],
                dtype=torch.float64,
            )
            after = torch.tensor(
                [item["score_after"] for item in ordered], dtype=torch.float64
            )
            abs_before_reference = (before - reference).abs()
            abs_after_before = (after - before).abs()
            abs_after_reference = (after - reference).abs()
            constructed_reference_error = (constructed_after - reference).abs()
            intended_magnitude_error = (
                (constructed_after - before).abs() - abs_before_reference
            ).abs()
            attention_summary = {
                **vector_magnitudes(
                    abs_before_reference,
                    prefix="selected_abs_before_reference",
                ),
                **vector_magnitudes(
                    abs_after_before,
                    prefix="selected_abs_after_before",
                ),
                "selected_abs_after_reference_mean": float(
                    abs_after_reference.mean().item()
                ),
                "selected_abs_after_reference_max": float(
                    abs_after_reference.max().item()
                ),
                "alpha1_constructed_after_reference_max_error": float(
                    constructed_reference_error.max().item()
                ),
                "alpha1_intended_magnitude_identity_max_error": float(
                    intended_magnitude_error.max().item()
                ),
            }
            baseline = baseline_logits[(row["dataset"], row["dataset_index"])]
            logit_summary = compute_logit_diagnostics(baseline, defense_logits)
            if logit_summary["baseline_top1_token_id"] != row[
                "existing_baseline_first_token"
            ]:
                raise RuntimeError("baseline top-1 changed during analysis")
            reproduction = validate_defense_argmax_reproduction(
                row=row, defense_logits=defense_logits
            )
            defense_reproduction_records.append(reproduction)
            current_strict_top1_changed = logit_summary.pop("top1_changed")
            logit_summary["top1_changed"] = row["existing_first_token_changed"]
            if not all(
                math.isfinite(value)
                for item in ordered
                for value in (
                    item["score_before"],
                    item["reference"],
                    item["score_after_float32"],
                    item["score_after"],
                )
            ):
                raise RuntimeError("selected-value trace contains NaN or infinity")
            per_prompt.append(
                {
                    **row,
                    "input_ids_sha256": _sha256_json(row["input_ids"]),
                    "selected_element_count": K,
                    "defense_argmax_reproduction": reproduction,
                    "current_strict_top1_changed": current_strict_top1_changed,
                    **attention_summary,
                    **logit_summary,
                    "runtime_diagnostics": diagnostics,
                    "selected_element_values_rank_order": {
                        "score_before": before.tolist(),
                        "reference": reference.tolist(),
                        "score_after_float32": constructed_after.tolist(),
                        "score_after": after.tolist(),
                        "abs_score_before_reference": abs_before_reference.tolist(),
                        "abs_score_after_before": abs_after_before.tolist(),
                    },
                }
            )
            print(
                f"[defense {position}/{len(source_rows)}] "
                f"{row['dataset']} index={row['dataset_index']} "
                f"top1_changed={logit_summary['top1_changed']} "
                f"reproduction={reproduction['match_type']}",
                flush=True,
            )
    torch.cuda.synchronize()
    defense_seconds = time.perf_counter() - defense_started

    full = args.limit_per_dataset is None
    comparison = build_comparison_summary(per_prompt, full=full)
    maximum_constructed_error = max(
        row["alpha1_constructed_after_reference_max_error"] for row in per_prompt
    )
    maximum_identity_error = max(
        row["alpha1_intended_magnitude_identity_max_error"] for row in per_prompt
    )
    if maximum_constructed_error > 1e-5 or maximum_identity_error > 1e-5:
        raise RuntimeError("alpha=1 constructed intervention identity failed")
    tie_allowed_records = [
        record
        for record in defense_reproduction_records
        if record["match_type"] == "tie_allowed"
    ]
    strict_match_count = sum(
        record["match_type"] == "strict"
        for record in defense_reproduction_records
    )

    output_dir.mkdir(parents=True)
    per_prompt_path = output_dir / "per_prompt_diagnostics.jsonl"
    comparison_path = output_dir / "comparison_summary.json"
    run_manifest_path = output_dir / "run_manifest.json"
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    write_jsonl(per_prompt_path, per_prompt)
    write_json(comparison_path, comparison)
    torch.cuda.synchronize()
    completed = utc_now()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "analysis_role": "post_hoc_explanatory_not_method_selection_or_tuning",
        "created_at": created_at,
        "execution_git_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "execution_git_head": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "execution_git_status_porcelain": git_value(
            REPO_ROOT, "status", "--porcelain"
        ).splitlines(),
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        "controller_source_sha256": sha256_file(
            REPO_ROOT / "utils/gcg_position_element_topk_intervention.py"
        ),
        "model": str(model_path),
        "model_config": llama_model_config_summary(model),
        "model_validation": model_validation,
        "tokenizer": {
            "name_or_path": tokenizer.name_or_path,
            "class": type(tokenizer).__name__,
        },
        "gpu": torch.cuda.get_device_name(0),
        "generation_reexecuted": False,
        "forward_passes_per_prompt": {"baseline": 1, "defense": 1},
        "use_cache": False,
        "seed": 42,
        "limit_per_dataset": args.limit_per_dataset,
        "frozen_defense": frozen,
        "sources": sources,
        "top500_rank_order": [dict(row) for row in top500],
        "validation": {
            "exact_saved_input_ids_used": True,
            "baseline_argmax_reproduced_count": len(per_prompt),
            "defense_argmax_reproduced_count": len(per_prompt),
            "defense_argmax_strict_match_count": strict_match_count,
            "defense_argmax_tie_allowed_match_count": len(tie_allowed_records),
            "defense_argmax_tie_allowed_records": tie_allowed_records,
            "defense_argmax_tie_rule": {
                "rule": TIE_RULE,
                "absolute_tolerance": TIE_ABSOLUTE_TOLERANCE,
                "relative_tolerance": TIE_RELATIVE_TOLERANCE,
                "relative_tolerance_basis": "torch.float16 machine epsilon",
                "non_tie_mismatch_policy": "fail_fast",
            },
            "runtime_invariant_pass_count": len(per_prompt),
            "finite_prompt_count": len(per_prompt),
            "alpha1_constructed_after_reference_max_error": maximum_constructed_error,
            "alpha1_intended_magnitude_identity_max_error": maximum_identity_error,
        },
        "runtime": {
            "baseline_seconds": baseline_seconds,
            "defense_seconds": defense_seconds,
            "total_seconds_including_model_load": time.perf_counter() - started,
            "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_vram_mib": torch.cuda.max_memory_reserved() / (1024**2),
        },
        "completed_at": completed,
    }
    write_json(run_manifest_path, manifest)
    write_json(
        artifact_manifest_path,
        {
            "schema_version": 1,
            "created_at": completed,
            "artifacts": [
                {
                    "relative_path": path.name,
                    "sha256": sha256_file(path),
                }
                for path in (per_prompt_path, comparison_path, run_manifest_path)
            ],
        },
    )
    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "prompt_count": len(per_prompt),
        "comparison_summary_sha256": sha256_file(comparison_path),
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        "runtime": manifest["runtime"],
    }


def main(argv=None):
    args = parse_args(argv)
    selection, frozen, datasets, sources = validate_inputs(args)
    if args.stage == "cpu-validate":
        result = {
            "status": "cpu_validation_passed",
            "gpu_used": False,
            "artifacts_written": False,
            "frozen_defense": frozen,
            "sources": sources,
            "analyzed_counts": {
                name: len(rows) for name, rows in datasets.items()
            },
        }
    else:
        result = run_analysis(args, selection, frozen, datasets, sources)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
