#!/usr/bin/env python3
"""Compare frozen GCG and AutoDAN attention-to-logit decision propagation."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_gcg_position_element_topk_intervention import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MODEL,
    EXPECTED_AUTODAN_SHA256,
    format_autodan_records,
    load_model_and_tokenizer,
    load_selection,
    validate_prompt_runtime_invariants,
)
from scripts.run_autodan_position_element_selection_ablation import (  # noqa: E402
    ALPHA,
    EXPECTED_SELECTION_INDICES,
    INTERVENTION_TIMING,
    K,
    MODEL_REPOSITORY_HEAD,
    OFFICIAL_FORMAT,
    first_forward_logits,
    load_autodan_selection_artifacts,
    model_repository_head,
    verify_artifact_manifest,
)
from utils.experiment_runtime import git_value, seed_everything  # noqa: E402
from utils.gcg_position_element_topk_intervention import (  # noqa: E402
    patch_llama_position_element_attention,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    TAIL_TOKENS,
    llama_model_config_summary,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.gcg_wasserstein_reference_evaluation import (  # noqa: E402
    ensure_output_absent,
)

GCG_RUN = REPO_ROOT / (
    "data/outputs/gcg_wasserstein_reference_intervention/llama2/"
    "independent100/evaluation_20260911T064911Z"
)
GCG_MECHANISM = REPO_ROOT / (
    "data/outputs/gcg_position_element_topk/attack_transfer_mechanism/"
    "full230_tieaware_20260914T092000Z"
)
AUTODAN_SELECTION = REPO_ROOT / (
    "data/outputs/gcg_position_element_topk/autodan_selection20/"
    "official_fastchat_34062e9_20260914T163205Z"
)
AUTODAN_RUN = REPO_ROOT / (
    "data/outputs/gcg_position_element_topk/autodan_selection_reference_ablation/"
    "evaluation110_official_fastchat_34062e9_20260914T163404Z"
)
AUTODAN_RAW = REPO_ROOT / "data/raw/llama2/llama2_AutoDAN.json"
EXPECTED_SOURCE_HASHES = {
    GCG_RUN / "artifact_manifest.json":
        "6ef7ee37c8e3c0540840237e9c21efb4225bfb9cd312fee9574089dbf7a8b1ba",
    GCG_MECHANISM / "artifact_manifest.json":
        "1958750adb5a577678d63eb7964b6ddb36d525123e328b8c923fbc92ce328a65",
    AUTODAN_SELECTION / "artifact_manifest.json":
        "a32918903c5d82ba6303d55912aa8dc6dbbcfa69aa38f6bbdfb20c7309f9bd0a",
    AUTODAN_RUN / "artifact_manifest.json":
        "685e247d5b82d6c6bd2c9cdbf80845c792782ef98f70e46378944c3ad2c3988e",
}
METRICS = (
    "baseline_margin", "max_absolute_logit_difference", "logit_l2",
    "kl_baseline_to_intervention", "baseline_top1_delta_logit",
    "intervention_residual_margin", "boundary_progress",
    "selected_score_change_mean_abs", "selected_score_change_rms",
    "selected_score_change_max_abs", "selected_score_change_l2",
    "selected_reference_before_mean_abs",
    "logit_l2_per_selected_score_change_l2",
    "max_logit_change_per_selected_score_change_l2",
)
RELATIONS = (
    ("baseline_margin", "max_absolute_logit_difference"),
    ("baseline_margin", "boundary_progress"),
    ("selected_score_change_l2", "logit_l2"),
)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu-preflight", "run"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gcg-artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    if args.stage == "run" and not args.output_dir:
        parser.error("--output-dir is required for --stage run")
    return args


def verify_listed_artifacts(directory: Path) -> dict:
    manifest = read_json(directory / "artifact_manifest.json")
    records = manifest.get("artifacts", manifest.get("files"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"invalid artifact manifest: {directory}")
    for record in records:
        path = directory / record["relative_path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"artifact hash mismatch: {path}")
    return manifest


def source_hashes() -> dict:
    paths = (
        "scripts/diagnose_gcg_autodan_logit_propagation.py",
        "scripts/analyze_top500_attack_transfer_mechanism.py",
        "scripts/run_autodan_position_element_selection_ablation.py",
        "scripts/evaluate_gcg_position_element_topk_intervention.py",
        "utils/gcg_position_element_topk_intervention.py",
    )
    return {
        "entrypoint_sha256": sha256_file(REPO_ROOT / paths[0]),
        "dependency_source_sha256": {
            path: sha256_file(REPO_ROOT / path) for path in paths[1:]
        },
    }


def preflight(args) -> dict:
    if K != 500 or ALPHA != 1.0 or INTERVENTION_TIMING != "first_token_only":
        raise ValueError("frozen intervention constants changed")
    if TAIL_TOKENS != 16 or OFFICIAL_FORMAT != "official_fastchat_34062e9":
        raise ValueError("frozen tail or AutoDAN formatting changed")
    model_path = Path(args.model).resolve()
    if model_repository_head(model_path) != MODEL_REPOSITORY_HEAD:
        raise ValueError("model revision mismatch")
    for path, digest in EXPECTED_SOURCE_HASHES.items():
        if sha256_file(path) != digest:
            raise ValueError(f"fixed source hash mismatch: {path}")
    if sha256_file(AUTODAN_RAW) != EXPECTED_AUTODAN_SHA256:
        raise ValueError("AutoDAN raw source hash mismatch")
    for directory in (GCG_RUN, GCG_MECHANISM, AUTODAN_RUN):
        verify_listed_artifacts(directory)
    verify_artifact_manifest(AUTODAN_SELECTION)

    gcg, gcg_verification, _, gcg_score_paths = load_selection(
        Path(args.gcg_artifact_dir).resolve(),
        k_values=(500,), model_path=model_path,
    )
    autodan, split, autodan_metadata = load_autodan_selection_artifacts(
        AUTODAN_SELECTION
    )
    if len(gcg.topk_records(K)) != K or len(autodan.topk_records(K)) != K:
        raise ValueError("Top-500 artifact is incomplete")
    if split["selection_indices"] != EXPECTED_SELECTION_INDICES:
        raise ValueError("AutoDAN Selection20 differs from frozen split")
    evaluation_indices = split["evaluation_indices"]
    if evaluation_indices != [
        index for index in range(130) if index not in EXPECTED_SELECTION_INDICES
    ] or len(evaluation_indices) != 110 or 10 not in evaluation_indices:
        raise ValueError("AutoDAN evaluation110 or index10 retention mismatch")

    gm = read_json(GCG_RUN / "run_manifest.json")
    mm = read_json(GCG_MECHANISM / "run_manifest.json")
    am = read_json(AUTODAN_RUN / "run_manifest.json")
    if (
        gm["alpha"] != ALPHA
        or gm["intervention_timing"] != INTERVENTION_TIMING
        or gm["selection"]["tail_tokens"] != TAIL_TOKENS
        or gm["selection"]["k_values"] != [K]
        or gm["model"] != str(model_path)
    ):
        raise ValueError("GCG frozen conditions mismatch")
    if (
        mm["frozen_defense"]["k"] != K
        or mm["frozen_defense"]["alpha"] != ALPHA
        or mm["frozen_defense"]["intervention_timing"] != INTERVENTION_TIMING
        or mm["model"] != str(model_path)
        or mm["sources"]["source_runs"]["GCG"]["artifact_manifest_sha256"]
        != EXPECTED_SOURCE_HASHES[GCG_RUN / "artifact_manifest.json"]
    ):
        raise ValueError("GCG mechanism reuse conditions mismatch")
    if (
        am["format"] != OFFICIAL_FORMAT
        or am["evaluation_indices"] != evaluation_indices
        or am["conditions"]["D"]["elements"] != "AutoDAN"
        or am["conditions"]["D"]["reference"] != "AutoDAN"
        or am["model"] != str(model_path)
    ):
        raise ValueError("AutoDAN condition D provenance mismatch")
    if any(value != 110 for value in am["validation"]["condition_input_ids_match_count"].values()) or any(
        value != 110 for value in am["validation"]["runtime_invariant_pass_count"].values()
    ) or any(value != 0 for value in am["validation"]["generation_failure_count"].values()):
        raise ValueError("AutoDAN saved runtime validation failed")
    if am["generation_settings"] != {
        "torch_dtype": "torch.float16", "attention_implementation": "eager",
        "model_eval": True, "do_sample": False, "temperature": None,
        "top_p": None, "max_new_tokens": 32, "use_cache": False,
        "batch_size": 1, "seed": 42,
    }:
        raise ValueError("AutoDAN generation settings mismatch")

    gb = read_jsonl(GCG_RUN / "independent100_baseline.jsonl")
    gd = read_jsonl(GCG_RUN / "independent100/k_500/prompt_results.jsonl")
    mechanism = [
        row for row in read_jsonl(GCG_MECHANISM / "per_prompt_diagnostics.jsonl")
        if row["dataset"] == "GCG"
    ]
    ab = read_jsonl(AUTODAN_RUN / "evaluation110_baseline.jsonl")
    ad = read_jsonl(AUTODAN_RUN / "condition_d_results.jsonl")
    if any(len(rows) != count for rows, count in (
        (gb, 100), (gd, 100), (mechanism, 100), (ab, 110), (ad, 110)
    )):
        raise ValueError("frozen prompt count mismatch")
    if sum(row["success"] for row in gb) != 53 or sum(row["success"] for row in gd) != 38:
        raise ValueError("GCG saved ASR mismatch")
    if sum(row["success"] for row in ab) != 29 or sum(row["success"] for row in ad) != 29:
        raise ValueError("AutoDAN saved ASR mismatch")
    if sum(not row["first_token_argmax_equal"] for row in gd) != 23:
        raise ValueError("GCG argmax transition mismatch")
    if sum(not row["logit_diagnostics"]["argmax_equal"] for row in ad) != 0:
        raise ValueError("AutoDAN argmax transition mismatch")

    gcg_identity = [
        (int(row["layer"]), int(row["head"]), int(row["position"]))
        for row in gcg.topk_records(K)
    ]
    for position, (baseline, defense, old) in enumerate(zip(gb, gd, mechanism)):
        if not (
            baseline["dataset_index"] == defense["dataset_index"] == old["dataset_index"]
            == 220 + position
            and baseline["input_ids"] == defense["input_ids"] == old["input_ids"]
            and baseline["success"] == defense["baseline_success"]
            == old["existing_baseline_success"]
            and defense["success"] == old["existing_defense_success"]
            and defense["first_token_argmax_equal"]
            == (not old["existing_first_token_changed"])
            and defense["k"] == K and defense["alpha"] == ALPHA
            and defense["intervention_timing"] == INTERVENTION_TIMING
        ):
            raise ValueError(f"GCG frozen row mismatch at position={position}")
        validate_prompt_runtime_invariants(defense["diagnostics"], k=K)
        validate_prompt_runtime_invariants(old["runtime_diagnostics"], k=K)
        trace = old["selected_element_values_rank_order"]
        if any(len(trace[key]) != K for key in (
            "score_before", "reference", "score_after"
        )):
            raise ValueError("GCG selected-value trace incomplete")
        if old["selected_element_count"] != K:
            raise ValueError("GCG selected element count mismatch")
    if [
        (int(item["layer"]), int(item["head"]), int(item["position"]))
        for item in mm["top500_rank_order"]
    ] != gcg_identity:
        raise ValueError("GCG selected identities differ from reused mechanism")

    raw = json.loads(AUTODAN_RAW.read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    formatted = format_autodan_records(
        [{"prompt": item["user prompt"], "dataset_index": index}
         for index, item in enumerate(raw)],
        tokenizer, prompt_format=OFFICIAL_FORMAT,
    )
    for position, (baseline, defense) in enumerate(zip(ab, ad)):
        index = evaluation_indices[position]
        expected = formatted[index]
        ids = expected["input_ids"][0].tolist()
        if not (
            baseline["dataset_index"] == defense["dataset_index"] == index
            and baseline["input_ids"] == defense["input_ids"] == ids
            and baseline["formatted_prompt"] == defense["formatted_prompt"]
            == expected["formatted_prompt"]
            and baseline["success"] == defense["baseline_success"]
            and defense["k"] == K and defense["alpha"] == ALPHA
            and defense["intervention_timing"] == INTERVENTION_TIMING
            and defense["elements_source"] == "AutoDAN"
            and defense["reference_source"] == "AutoDAN"
            and not defense["generation_failed"]
        ):
            raise ValueError(f"AutoDAN frozen row mismatch at index={index}")
        validate_prompt_runtime_invariants(defense["runtime_diagnostics"], k=K)
    return {
        "model_path": model_path, "gcg_selection": gcg,
        "autodan_selection": autodan, "gcg_baseline": gb,
        "gcg_defense": gd, "gcg_mechanism": mechanism,
        "autodan_baseline": ab, "autodan_defense": ad,
        "gcg_artifact_verification": gcg_verification,
        "gcg_score_hashes": {name: sha256_file(path) for name, path in gcg_score_paths.items()},
        "autodan_score_metadata": autodan_metadata,
        "source_hashes": {str(path.relative_to(REPO_ROOT)): digest
                          for path, digest in EXPECTED_SOURCE_HASHES.items()},
        "gcg_identity": gcg_identity,
        "autodan_identity": [
            (int(row["layer"]), int(row["head"]), int(row["position"]))
            for row in autodan.topk_records(K)
        ],
    }


def decision_metrics(baseline: torch.Tensor, intervention: torch.Tensor) -> dict:
    baseline = baseline.double().cpu()
    intervention = intervention.double().cpu()
    if baseline.ndim != 1 or intervention.shape != baseline.shape or not (
        bool(torch.isfinite(baseline).all()) and bool(torch.isfinite(intervention).all())
    ):
        raise ValueError("logits must be matching finite vectors")
    top = torch.topk(baseline, 2)
    top1 = int(top.indices[0])
    margin = float(top.values[0] - top.values[1])
    if margin <= 0:
        raise ValueError("baseline margin must be positive")
    competitor = intervention.clone()
    competitor[top1] = -torch.inf
    residual = float(intervention[top1] - competitor.max())
    delta = intervention - baseline
    log_p = torch.log_softmax(baseline, dim=0)
    log_q = torch.log_softmax(intervention, dim=0)
    kl = float(torch.sum(torch.exp(log_p) * (log_p - log_q)))
    if kl < 0 and kl > -1e-12:
        kl = 0.0
    result = {
        "baseline_top1_token_id": top1,
        "baseline_top2_token_id": int(top.indices[1]),
        "intervention_top1_token_id": int(torch.argmax(intervention)),
        "argmax_changed": int(torch.argmax(intervention)) != top1,
        "baseline_margin": margin,
        "max_absolute_logit_difference": float(delta.abs().max()),
        "logit_l2": float(torch.linalg.vector_norm(delta)),
        "kl_baseline_to_intervention": kl,
        "baseline_top1_delta_logit": float(delta[top1]),
        "intervention_residual_margin": residual,
        "boundary_progress": 1.0 - residual / margin,
    }
    if not all(math.isfinite(value) for value in result.values() if type(value) is float):
        raise ValueError("decision metrics contain NaN or infinity")
    return result


def selected_score_metrics(before, reference, after, logit_l2, max_logit) -> dict:
    vectors = [np.asarray(values, dtype=np.float64) for values in (before, reference, after)]
    if any(vector.shape != (K,) or not np.isfinite(vector).all() for vector in vectors):
        raise ValueError("selected-score vectors must contain 500 finite elements")
    delta = vectors[2] - vectors[0]
    score_l2 = float(np.linalg.norm(delta))
    if score_l2 == 0:
        raise ValueError("selected-score-change L2 is zero; propagation ratio undefined")
    result = {
        "selected_score_change_mean_abs": float(np.mean(np.abs(delta))),
        "selected_score_change_rms": float(np.sqrt(np.mean(delta * delta))),
        "selected_score_change_max_abs": float(np.max(np.abs(delta))),
        "selected_score_change_l2": score_l2,
        "selected_reference_before_mean_abs": float(np.mean(np.abs(vectors[1] - vectors[0]))),
        "logit_l2_per_selected_score_change_l2": float(logit_l2 / score_l2),
        "max_logit_change_per_selected_score_change_l2": float(max_logit / score_l2),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("selected-score metrics contain NaN or infinity")
    return result


def distribution(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary requires a non-empty finite vector")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = array[rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))].mean(axis=1)
    return {
        "count": int(len(array)), "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "q25": float(np.quantile(array, .25)), "q75": float(np.quantile(array, .75)),
        "min": float(np.min(array)), "max": float(np.max(array)),
        "bootstrap_mean_95_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
    }


def correlation(rows, left, right) -> dict:
    x = np.asarray([row[left] for row in rows], dtype=np.float64)
    y = np.asarray([row[right] for row in rows], dtype=np.float64)
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return {"status": "undefined", "reason": "fewer than two finite observations"}
    if np.std(x) == 0 or np.std(y) == 0:
        return {"status": "undefined", "reason": "constant vector"}
    return {"status": "defined", "pearson": float(np.corrcoef(x, y)[0, 1])}


def summarize(rows) -> dict:
    if not rows:
        raise ValueError("cannot summarize empty group")
    return {
        "prompt_count": len(rows),
        "argmax_changed_count": sum(row["argmax_changed"] for row in rows),
        "metrics": {metric: distribution([row[metric] for row in rows]) for metric in METRICS},
        "relations": {
            f"{left}_vs_{right}": correlation(rows, left, right)
            for left, right in RELATIONS
        },
    }


def mean_difference_ci(gcg_rows, autodan_rows, metric) -> dict:
    gcg = np.asarray([row[metric] for row in gcg_rows], dtype=np.float64)
    autodan = np.asarray([row[metric] for row in autodan_rows], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gb = gcg[rng.integers(0, len(gcg), size=(BOOTSTRAP_REPLICATES, len(gcg)))].mean(axis=1)
    ab = autodan[rng.integers(0, len(autodan), size=(BOOTSTRAP_REPLICATES, len(autodan)))].mean(axis=1)
    diff = ab - gb
    return {
        "mean_auto_minus_gcg": float(np.mean(autodan) - np.mean(gcg)),
        "bootstrap_95_ci": [float(np.quantile(diff, .025)), float(np.quantile(diff, .975))],
    }


def classify_direction(result, direction) -> str:
    lower, upper = result["bootstrap_95_ci"]
    if direction == "higher":
        return "supported" if lower > 0 else "not supported" if upper < 0 else "inconclusive"
    return "supported" if upper < 0 else "not supported" if lower > 0 else "inconclusive"


def comparison(gcg_rows, autodan_rows) -> dict:
    differences = {
        metric: mean_difference_ci(gcg_rows, autodan_rows, metric)
        for metric in (
            "baseline_margin", "selected_score_change_l2",
            "logit_l2_per_selected_score_change_l2", "logit_l2",
            "boundary_progress", "max_absolute_logit_difference",
        )
    }
    progress = classify_direction(differences["boundary_progress"], "lower")
    logit = classify_direction(differences["logit_l2"], "higher")
    h4 = (
        "supported" if progress == "supported" and logit == "supported"
        else "not supported" if progress == "not supported" or logit == "not supported"
        else "inconclusive"
    )
    return {
        "role": "post_hoc_diagnostic_not_causal_or_method_selection",
        "primary_group": "all frozen prompts",
        "mean_differences_auto_minus_gcg": differences,
        "hypotheses": {
            "H1_larger_AutoDAN_baseline_margin":
                classify_direction(differences["baseline_margin"], "higher"),
            "H2_smaller_AutoDAN_selected_attention_change":
                classify_direction(differences["selected_score_change_l2"], "lower"),
            "H3_weaker_AutoDAN_attention_to_logit_propagation":
                classify_direction(differences["logit_l2_per_selected_score_change_l2"], "lower"),
            "H4_sufficient_logit_change_but_weaker_boundary_progress": h4,
        },
        "hypothesis_rule": (
            "All-prompt AutoDAN-minus-GCG bootstrap mean-difference 95% CI excluding "
            "zero in the stated direction supports H1-H3; excluding zero in the "
            "opposite direction is not supported, otherwise inconclusive. H4 requires "
            "both logit L2 higher and boundary progress lower CIs; an opposite CI for "
            "either component is not supported, otherwise inconclusive."
        ),
    }


def run(args, frozen) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU diagnosis cannot run")
    output_dir = ensure_output_absent(Path(args.output_dir)).resolve()
    seed_everything(42)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(frozen["model_path"])
    if (
        model.config._attn_implementation != "eager"
        or model.training or next(model.parameters()).dtype != torch.float16
        or llama_model_config_summary(model) != read_json(AUTODAN_RUN / "run_manifest.json")["model_config"]
    ):
        raise ValueError("loaded model differs from frozen FP16 eager eval model")
    if tokenizer.__class__.__name__ != "LlamaTokenizerFast":
        raise ValueError("tokenizer differs from frozen LlamaTokenizerFast")
    rows = []
    # The GCG full230 mechanism trace already contains intervention logits and scores.
    # Only top2 token identity was absent, so replay one baseline forward per prompt.
    for position, (baseline, defense, old) in enumerate(zip(
        frozen["gcg_baseline"], frozen["gcg_defense"], frozen["gcg_mechanism"]
    ), 1):
        ids = torch.tensor([baseline["input_ids"]], dtype=torch.long)
        logits = first_forward_logits(model, ids)
        top = torch.topk(logits.double(), 2)
        margin = float(top.values[0] - top.values[1])
        if (
            int(top.indices[0]) != old["baseline_top1_token_id"]
            or margin != old["baseline_top1_top2_margin"]
        ):
            raise RuntimeError(f"GCG baseline logit reproduction failed at index={baseline['dataset_index']}")
        metrics = {
            "baseline_top1_token_id": int(top.indices[0]),
            "baseline_top2_token_id": int(top.indices[1]),
            "intervention_top1_token_id": int(old["defense_top1_token_id"]),
            "argmax_changed": bool(old["existing_first_token_changed"]),
            "baseline_margin": margin,
            "max_absolute_logit_difference": float(old["max_absolute_logit_difference"]),
            "logit_l2": float(old["logit_l2_distance"]),
            "kl_baseline_to_intervention": float(old["kl_baseline_to_defense"]),
            "baseline_top1_delta_logit": float(
                old["existing_first_token_logit_metrics"]["top1_logit_change"]
            ),
            "intervention_residual_margin": float(old["baseline_top1_decision_headroom_after_defense"]),
        }
        metrics["boundary_progress"] = 1.0 - metrics["intervention_residual_margin"] / margin
        trace = old["selected_element_values_rank_order"]
        metrics.update(selected_score_metrics(
            trace["score_before"], trace["reference"], trace["score_after"],
            metrics["logit_l2"], metrics["max_absolute_logit_difference"],
        ))
        rows.append({
            "dataset": "GCG", "dataset_index": baseline["dataset_index"],
            "prompt_id": baseline["prompt_id"], "input_ids": baseline["input_ids"],
            "baseline_attack_success": baseline["success"],
            "intervention_attack_success": defense["success"],
            "metric_source": "reused_full230_mechanism_except_baseline_top2_token",
            "selected_element_values_rank_order": {
                key: trace[key] for key in (
                    "score_before", "reference", "score_after_float32", "score_after"
                )
            },
            **metrics,
        })
        print(f"[GCG baseline {position}/100] index={baseline['dataset_index']}", flush=True)

    autodan_baseline_logits = []
    for position, baseline in enumerate(frozen["autodan_baseline"], 1):
        ids = torch.tensor([baseline["input_ids"]], dtype=torch.long)
        logits = first_forward_logits(model, ids)
        if int(torch.argmax(logits)) != baseline["response_token_ids"][0]:
            raise RuntimeError(
                f"AutoDAN baseline top1 reproduction failed at index={baseline['dataset_index']}"
            )
        autodan_baseline_logits.append(logits)
        print(f"[AutoDAN baseline {position}/110] index={baseline['dataset_index']}", flush=True)

    top500_identity = set(frozen["autodan_identity"])
    with patch_llama_position_element_attention(
        model, selection=frozen["autodan_selection"], k=K, alpha=ALPHA,
        intervention_timing=INTERVENTION_TIMING, capture_selected_values=True,
    ) as controller:
        for position, (baseline, defense, baseline_logits) in enumerate(zip(
            frozen["autodan_baseline"], frozen["autodan_defense"],
            autodan_baseline_logits,
        ), 1):
            ids = torch.tensor([baseline["input_ids"]], dtype=torch.long)
            controller.clear()
            controller.set_prompt_context(prompt_length=baseline["prompt_length"])
            intervention_logits = first_forward_logits(model, ids)
            runtime = controller.summary()
            validate_prompt_runtime_invariants(runtime, k=K)
            trace = controller.selected_value_trace()
            if len(trace) != K or {
                (item["layer"], item["head"], item["position"]) for item in trace
            } != top500_identity:
                raise RuntimeError("AutoDAN selected-value identities mismatch")
            metrics = decision_metrics(baseline_logits, intervention_logits)
            saved = defense["logit_diagnostics"]
            if (
                metrics["baseline_top1_token_id"] != saved["baseline_top1_token_id"]
                or metrics["intervention_top1_token_id"] != saved["condition_top1_token_id"]
                or metrics["baseline_margin"] != saved["baseline_top1_top2_margin"]
                or metrics["max_absolute_logit_difference"] != saved["max_absolute_logit_difference"]
                or abs(metrics["logit_l2"] - saved["logit_l2_distance"]) > 1e-6
                or abs(metrics["kl_baseline_to_intervention"] - saved["kl_baseline_to_condition"]) > 1e-8
            ):
                raise RuntimeError(f"AutoDAN condition D logit reproduction failed at index={baseline['dataset_index']}")
            by_identity = {
                (item["layer"], item["head"], item["position"]): item
                for item in trace
            }
            ordered = [by_identity[identity] for identity in frozen["autodan_identity"]]
            metrics.update(selected_score_metrics(
                [item["score_before"] for item in ordered],
                [item["reference"] for item in ordered],
                [item["score_after"] for item in ordered],
                metrics["logit_l2"], metrics["max_absolute_logit_difference"],
            ))
            rows.append({
                "dataset": "AutoDAN", "dataset_index": baseline["dataset_index"],
                "prompt_id": baseline["prompt_id"], "input_ids": baseline["input_ids"],
                "baseline_attack_success": baseline["success"],
                "intervention_attack_success": defense["success"],
                "metric_source": "new_first_forward_and_selected_value_trace",
                "runtime_diagnostics": runtime, **metrics,
                "selected_element_values_rank_order": {
                    key: [item[key] for item in ordered]
                    for key in (
                        "score_before", "reference", "score_after_float32", "score_after"
                    )
                },
            })
            print(f"[AutoDAN diagnostic {position}/110] index={baseline['dataset_index']}", flush=True)

    gcg_rows = [row for row in rows if row["dataset"] == "GCG"]
    auto_rows = [row for row in rows if row["dataset"] == "AutoDAN"]
    if sum(row["argmax_changed"] for row in gcg_rows) != 23 or sum(
        row["argmax_changed"] for row in auto_rows
    ) != 0:
        raise RuntimeError("argmax changes differ from frozen outcomes")
    gcg_summary = {
        "all_prompts": summarize(gcg_rows),
        "baseline_attack_success": summarize([row for row in gcg_rows if row["baseline_attack_success"]]),
        "argmax_changed": summarize([row for row in gcg_rows if row["argmax_changed"]]),
        "argmax_unchanged": summarize([row for row in gcg_rows if not row["argmax_changed"]]),
    }
    autodan_summary = {
        "all_prompts": summarize(auto_rows),
        "baseline_attack_success": summarize([row for row in auto_rows if row["baseline_attack_success"]]),
        "sensitivity_excluding_index10": summarize([row for row in auto_rows if row["dataset_index"] != 10]),
    }
    comparison_summary = comparison(gcg_rows, auto_rows)
    comparison_summary["baseline_attack_success_comparison"] = {
        metric: mean_difference_ci(
            [row for row in gcg_rows if row["baseline_attack_success"]],
            [row for row in auto_rows if row["baseline_attack_success"]],
            metric,
        ) for metric in ("baseline_margin", "selected_score_change_l2",
                         "logit_l2_per_selected_score_change_l2", "logit_l2",
                         "boundary_progress")
    }
    runtime = {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    manifest = {
        "schema_version": 1, "status": "complete", "created_at": utc_now(),
        "analysis_role": "post_hoc_diagnostic_no_tuning",
        "execution_git_head": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "execution_git_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "execution_git_status_porcelain": git_value(REPO_ROOT, "status", "--porcelain").splitlines(),
        **source_hashes(),
        "model": str(frozen["model_path"]),
        "model_revision": MODEL_REPOSITORY_HEAD,
        "model_config": llama_model_config_summary(model),
        "model_dtype": "float16", "attention_implementation": "eager",
        "tokenizer_class": tokenizer.__class__.__name__,
        "gcg_prompt_format": "frozen independent100 saved formatted input IDs",
        "autodan_prompt_format": OFFICIAL_FORMAT,
        "k": K, "alpha": ALPHA, "intervention_timing": INTERVENTION_TIMING,
        "tail_tokens": TAIL_TOKENS, "use_cache": False, "seed": 42,
        "source_artifact_manifest_sha256": frozen["source_hashes"],
        "gcg_score_hashes": frozen["gcg_score_hashes"],
        "gcg_artifact_verification": frozen["gcg_artifact_verification"],
        "autodan_score_metadata": frozen["autodan_score_metadata"],
        "metric_definitions": {
            "baseline_margin": "baseline top1 logit minus baseline top2 logit; must be positive",
            "intervention_residual_margin": "intervention logit at baseline top1 minus maximum intervention logit over all other tokens",
            "boundary_progress": "1 - intervention_residual_margin / baseline_margin; 1 means boundary reached, greater than 1 means strict reversal",
            "selected_score_change": "realized FP16-cast antisymmetric lower-triangle score_after minus score_before at frozen selected 500 elements",
            "propagation_ratio": "logit change magnitude divided by selected-score-change L2; zero denominator stops run",
            "kl": "KL(softmax(baseline logits) || softmax(intervention logits))",
            "std": "sample standard deviation (ddof=1)",
            "bootstrap": f"{BOOTSTRAP_REPLICATES} within-group resamples with replacement, seed {BOOTSTRAP_SEED}; percentile 95% CI for means and AutoDAN-minus-GCG mean differences",
            "relations": "Pearson product-moment correlation of finite raw metric values; undefined for constant vectors",
            "hypothesis_classification": comparison_summary["hypothesis_rule"],
        },
        "reuse": {
            "GCG": "full230 mechanism selected-value and logit diagnostics; frozen independent100 responses and input IDs",
            "AutoDAN": "evaluation110 baseline/condition D responses and input IDs; new first-forward logits and selected-value trace",
        },
        "validation": {
            "exact_input_ids": True, "fixed_artifact_hashes": True,
            "gcg_baseline_count": 100, "autodan_baseline_count": 110,
            "autodan_runtime_invariant_pass_count": 110,
            "gcg_existing_runtime_invariant_pass_count": 100,
            "finite_metrics": True, "zero_denominator_count": 0,
        },
        "runtime": runtime,
    }
    write_jsonl(output_dir / "per_prompt_diagnostics.jsonl", rows)
    write_json(output_dir / "gcg_summary.json", gcg_summary)
    write_json(output_dir / "autodan_summary.json", autodan_summary)
    write_json(output_dir / "comparison_summary.json", comparison_summary)
    write_json(output_dir / "run_manifest.json", manifest)
    artifact_files = (
        "per_prompt_diagnostics.jsonl", "gcg_summary.json", "autodan_summary.json",
        "comparison_summary.json", "run_manifest.json",
    )
    write_json(output_dir / "artifact_manifest.json", {
        "schema_version": 1,
        "files": [
            {"relative_path": name, "sha256": sha256_file(output_dir / name)}
            for name in artifact_files
        ],
    })
    return {
        "output_dir": str(output_dir), "runtime": runtime,
        "hypotheses": comparison_summary["hypotheses"],
        "gcg_summary": gcg_summary["all_prompts"],
        "autodan_summary": autodan_summary["all_prompts"],
        "artifact_manifest_sha256": sha256_file(output_dir / "artifact_manifest.json"),
    }


def main(argv=None):
    args = parse_args(argv)
    frozen = preflight(args)
    if args.stage == "cpu-preflight":
        print(json.dumps({
            "status": "cpu_preflight_passed", "gpu_used": False,
            "GCG_count": len(frozen["gcg_baseline"]),
            "AutoDAN_count": len(frozen["autodan_baseline"]),
            "fixed_source_hashes": frozen["source_hashes"],
        }, indent=2))
    else:
        print(json.dumps(run(args, frozen), indent=2))


if __name__ == "__main__":
    main()
