#!/usr/bin/env python3
"""Describe frozen independent100 selection positions; never generate or intervene."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate_gcg_dynamic_asym_independent100 as evaluation  # noqa: E402
from scripts.diagnose_gcg_dynamic_asym_selection import tensor_sha256  # noqa: E402
from utils.artifact_io import (  # noqa: E402
    ensure_output_directory_absent, read_json, read_jsonl, sha256_file,
    utc_now, write_json, write_jsonl,
)
from utils.experiment_runtime import git_state, git_value, seed_everything  # noqa: E402
from utils.gcg_wasserstein_reference import strict_lower_indices  # noqa: E402


REGIONS = ("Early→Early", "Late→Early", "Late→Late")
METRICS = (
    "candidate_count", "selected_count", "candidate_share", "selected_share",
    "selection_rate", "enrichment",
)
BINS = 10
DEFAULT_PRIOR_RUN = (
    "data/outputs/gcg_dynamic_asym_replacement_modes/llama2/independent100/"
    "raw_s_zero/evaluation_20261001T050701Z"
)
SOURCE_PATHS = (
    "scripts/analyze_gcg_dynamic_asym_selected_positions.py",
    "scripts/evaluate_gcg_dynamic_asym_independent100.py",
    "scripts/evaluate_gcg_first_token_reference_ablation.py",
    "utils/gcg_dynamic_asym_selection_diagnostics.py",
    "utils/gcg_dynamic_asym_stage1.py",
    "utils/gcg_dynamic_asym_stage2.py",
    "utils/gcg_dynamic_asym_intervention.py",
    "utils/gcg_wasserstein_reference.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu-preflight", "run"), default="cpu-preflight")
    parser.add_argument("--prior-run", default=DEFAULT_PRIOR_RUN)
    parser.add_argument("--coordinates-from", help="Replay a completed analysis on CPU, without model loading.")
    parser.add_argument("--output-dir")
    parser.add_argument("--sandbox-exempt-gpu", action="store_true")
    parser.add_argument("--dataset", choices=("independent100", "autodan110-reference-comparison"),
                        default="independent100")
    parser.add_argument("--autodan-original-reference")
    parser.add_argument("--gcg-analysis", default=(
        "data/outputs/gcg_dynamic_asym_selected_positions/llama2/independent100/"
        "analysis_20261001T125944Z"))
    return parser.parse_args(argv)


def count_metrics(candidate: int, selected: int, total_candidate: int, total_selected: int):
    if min(candidate, selected) < 0 or selected > candidate:
        raise ValueError("selected/candidate counts are invalid")
    if total_candidate <= 0 or total_selected <= 0:
        raise ValueError("prompt totals must be positive")
    if candidate == 0:
        return {"candidate_count": 0, "selected_count": 0,
                **dict.fromkeys(METRICS[2:], None)}
    candidate_share = candidate / total_candidate
    selected_share = selected / total_selected
    return {
        "candidate_count": candidate, "selected_count": selected,
        "candidate_share": candidate_share, "selected_share": selected_share,
        "selection_rate": selected / candidate,
        "enrichment": selected_share / candidate_share,
    }


def position_metrics(prompt_length, selected_indices, *, layer_count=32, head_count=32, bins=BINS):
    """Decode the existing collector's layer/head/strict-lower flat identities."""
    if type(prompt_length) is not int or prompt_length < 2:
        raise ValueError("prompt_length must be an integer >= 2")
    if min(layer_count - 1, head_count, bins) <= 0:
        raise ValueError("invalid analysis dimensions")
    if (not isinstance(selected_indices, torch.Tensor) or selected_indices.ndim != 1
            or selected_indices.dtype != torch.int64 or selected_indices.device.type != "cpu"):
        raise ValueError("selected_indices must be rank-1 CPU int64")
    lower = strict_lower_indices(prompt_length)
    elements = lower.shape[1]
    if (not selected_indices.numel() or selected_indices.unique().numel() != selected_indices.numel()
            or int(selected_indices.min()) < head_count * elements
            or int(selected_indices.max()) >= layer_count * head_count * elements):
        raise ValueError("selected indices are empty, duplicated, out of bounds or include Layer 0")
    layer_head = selected_indices // elements
    layers = layer_head // head_count
    heads = layer_head % head_count
    selected_rows, selected_cols = lower[:, selected_indices % elements]
    coordinates = torch.stack((layers, heads, selected_rows, selected_cols), dim=1).tolist()

    def region_ids(rows, cols):
        # Integer comparisons give the exact u=i/n, v=j/n boundary, including odd n.
        return torch.where(2 * rows < prompt_length, 0,
                           torch.where(2 * cols < prompt_length, 1, 2))

    def bin_ids(rows, cols):
        return (rows * bins // prompt_length) * bins + cols * bins // prompt_length

    multiplicity = (layer_count - 1) * head_count
    candidate_regions = torch.bincount(region_ids(*lower), minlength=3) * multiplicity
    selected_regions = torch.bincount(region_ids(selected_rows, selected_cols), minlength=3)
    candidate_bins = torch.bincount(bin_ids(*lower), minlength=bins * bins) * multiplicity
    selected_bins = torch.bincount(bin_ids(selected_rows, selected_cols), minlength=bins * bins)
    total_candidate = multiplicity * elements
    total_selected = selected_indices.numel()
    if (int(candidate_regions.sum()) != total_candidate or int(candidate_bins.sum()) != total_candidate
            or int(selected_regions.sum()) != total_selected or int(selected_bins.sum()) != total_selected):
        raise AssertionError("position count conservation failed")
    regions = {name: count_metrics(int(c), int(s), total_candidate, total_selected)
               for name, c, s in zip(REGIONS, candidate_regions, selected_regions)}
    position_bins = [
        {"query_bin": index // bins, "key_bin": index % bins,
         **count_metrics(int(c), int(s), total_candidate, total_selected)}
        for index, (c, s) in enumerate(zip(candidate_bins, selected_bins))
    ]
    return {
        "prompt_length": prompt_length, "total_candidate_count": total_candidate,
        "total_selected_count": total_selected, "regions": regions, "position_bins": position_bins,
        "layer_counts": torch.bincount(layers, minlength=layer_count).tolist(),
    }, coordinates


def metric_summary(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"valid_prompt_count": 0, **dict.fromkeys(("mean", "median", "std", "min", "max"), None)}
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite metric")
    return {
        "valid_prompt_count": len(values), "mean": statistics.fmean(values),
        "median": statistics.median(values), "std": statistics.pstdev(values),
        "min": min(values), "max": max(values),
    }


def aggregate_positions(rows, *, bins=BINS):
    if not rows or len({row["dataset_index"] for row in rows}) != len(rows):
        raise ValueError("analysis rows are empty or duplicated")
    total_candidate = sum(row["total_candidate_count"] for row in rows)
    total_selected = sum(row["total_selected_count"] for row in rows)

    def aggregate_groups(groups):
        return {
            "prompt_equal": {key: metric_summary([group[key] for group in groups]) for key in METRICS},
            "enrichment_gt_1_prompt_count": sum(
                group["enrichment"] is not None and group["enrichment"] > 1 for group in groups
            ),
            "pooled_reference_only": count_metrics(
                sum(group["candidate_count"] for group in groups),
                sum(group["selected_count"] for group in groups), total_candidate, total_selected,
            ),
        }

    regions = {name: aggregate_groups([row["regions"][name] for row in rows]) for name in REGIONS}
    position_bins = []
    for index in range(bins * bins):
        groups = [row["position_bins"][index] for row in rows]
        position_bins.append({"query_bin": index // bins, "key_bin": index % bins,
                              **aggregate_groups(groups)})
    return {
        "prompt_count": len(rows), "total_candidate_count": total_candidate,
        "total_selected_count": total_selected,
        "selected_count_statistics": metric_summary([row["total_selected_count"] for row in rows]),
        "aggregation": "per-prompt metric first, equal-weight mean across prompts",
        "std_definition": "population_ddof_0",
        "empty_candidate_policy": "null in JSON/CSV; masked in heatmaps; exclude undefined metrics from means",
        "regions": regions, "position_bins": position_bins,
    }


def load_inputs(args):
    frozen = evaluation.parse_args([])
    provenance, baseline = evaluation.validate_frozen_inputs(frozen)
    prior_path = Path(args.prior_run).resolve()
    evaluation.verify_artifact_manifest(prior_path)
    prior_manifest = read_json(prior_path / "run_manifest.json")
    expected = {
        "dataset": "independent100", "component": "antisymmetric",
        "selection_direction": "absolute", "intervention_target_sign": "all",
        "selection": evaluation.fixed_selection(), "alpha": 1.0, "layer0_included": False,
        "cap_rate": 0.005, "timing": "first_prefill_only",
        "position_scope": "full_prompt_strict_lower_i_gt_j",
        "reference_source": "gcg-selection20-original",
    }
    if prior_manifest.get("status") != "complete" or prior_manifest["provenance"] != provenance:
        raise ValueError("prior run provenance mismatch")
    for key, value in expected.items():
        if prior_manifest["scientific_conditions"].get(key) != value:
            raise ValueError(f"prior selection condition mismatch: {key}")
    prior = read_jsonl(prior_path / "prompt_results.jsonl")
    if (len(prior) != 100 or {row["dataset_index"] for row in prior} != set(range(220, 320))
            or sum(row["selected_count"] for row in prior) != 91531
            or min(row["selected_count"] for row in prior) != 660
            or max(row["selected_count"] for row in prior) != 1755):
        raise ValueError("prior independent100 identities/counts mismatch")
    return frozen, provenance, baseline, {row["dataset_index"]: row for row in prior}, prior_path


def collect_positions(frozen, baseline, prior):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA required for attention collection; no CPU fallback")
    seed_everything(evaluation.SEED)
    model, tokenizer = evaluation.load_model_and_tokenizer(Path(frozen.model).resolve())
    _, reference = evaluation.load_requested_reference(frozen)
    records = evaluation.load_independent_evaluation_records(
        config_path=frozen.independent_config, input_manifest_path=frozen.independent_input_manifest,
        attack_run_dir=frozen.independent_attack_run_dir, adapter_path=frozen.independent_adapter,
    )
    records = evaluation.format_records(records, tokenizer)
    evaluation.validate_formatted_evaluation_identity(records)
    baseline_by_index = {row["dataset_index"]: row for row in baseline}
    coordinate_rows = []
    for position, record in enumerate(records, 1):
        index = record["dataset_index"]
        if evaluation.serialized_input_ids(record["input_ids"]) != baseline_by_index[index]["input_ids"]:
            raise RuntimeError("formatted input differs from reused baseline")
        with evaluation.patch_llama_abs_z_collection(
            model, layer_count=evaluation.NUM_LAYERS, head_count=evaluation.NUM_HEADS,
            component="antisymmetric",
        ) as collector:
            collector.set_prompt_context(prompt_length=record["prompt_length"],
                                         excluded_selection_position=0, mu=reference.mu, sigma=reference.sigma)
            # Exactly one unmodified prefill, no LM head, generate(), or replacement controller.
            with torch.inference_mode():
                model.model(input_ids=record["input_ids"].to(model.device), use_cache=False, return_dict=True)
            signed_z, collection = collector.result(signed=True)
        plan = evaluation.build_stage2_selection_plan(
            signed_z.abs(), selection=evaluation.fixed_selection(),
            layer_count=evaluation.NUM_LAYERS, head_count=evaluation.NUM_HEADS,
            elements_per_head=collection["strict_lower_elements_per_head"],
            layer0_included=False, signed_z=signed_z, selection_direction="absolute",
        )
        plan = evaluation.restrict_absolute_plan_to_intervention_sign(
            plan, signed_z, intervention_target_sign="all", layer_count=evaluation.NUM_LAYERS,
            head_count=evaluation.NUM_HEADS, elements_per_head=collection["strict_lower_elements_per_head"],
        )
        if (not collection["all_finite"] or plan["selection_shortfall_count"] != 0
                or plan["cap_activated"] or plan["selected_count"] > plan["cap_count"]):
            raise RuntimeError("selection collection invariant failed")
        old = prior[index]
        for key in ("selected_count", "selected_z", "positive_selected_count", "negative_selected_count", "cap_count"):
            if plan[key] != old[key]:
                raise RuntimeError(f"prior selection mismatch: index={index}, {key}")
        if (record["prompt_length"] != old["prompt_length"]
                or plan["layer_counts"] != old["diagnostics"]["planned_layer_counts"]):
            raise RuntimeError("prior prompt length/Layer selection mismatch")
        coordinate_rows.append({
            "dataset_index": index, "prompt_id": record["prompt_id"],
            "prompt_length": record["prompt_length"],
            "selected_indices": plan["selected_indices"].tolist(),
            "selected_indices_sha256": tensor_sha256(plan["selected_indices"]),
            "selected_z": plan["selected_z"], "layer_counts": plan["layer_counts"],
        })
        print(f"[collection {position}/100] index={index} selected={plan['selected_count']}", flush=True)
    return coordinate_rows


def analyze_coordinates(coordinate_rows, prior):
    if len(coordinate_rows) != 100 or {row["dataset_index"] for row in coordinate_rows} != set(prior):
        raise ValueError("coordinate artifact must cover independent100 exactly once")
    metrics_rows, decoded_rows = [], []
    for row in coordinate_rows:
        index = row["dataset_index"]
        indices = torch.tensor(row["selected_indices"], dtype=torch.int64)
        if tensor_sha256(indices) != row["selected_indices_sha256"]:
            raise ValueError("selected identity hash mismatch")
        metrics, coordinates = position_metrics(row["prompt_length"], indices)
        old = prior[index]
        if (row["prompt_length"] != old["prompt_length"]
                or metrics["total_selected_count"] != old["selected_count"]
                or metrics["layer_counts"] != old["diagnostics"]["planned_layer_counts"]
                or row["layer_counts"] != metrics["layer_counts"]
                or row["selected_z"] != old["selected_z"]):
            raise ValueError("coordinate/prior artifact mismatch")
        metrics_rows.append({"dataset_index": index, "prompt_id": row["prompt_id"], **metrics})
        decoded_rows.append({**row, "coordinate_order": ["layer", "head", "i", "j"],
                             "selected_coordinates": coordinates})
    return metrics_rows, decoded_rows


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(path, position_bins, metric, provenance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    values = np.full((BINS, BINS), np.nan)
    for row in position_bins:
        value = row["prompt_equal"][metric]["mean"]
        if value is not None:
            values[row["query_bin"], row["key_bin"]] = value * (100 if metric != "enrichment" else 1)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(np.ma.masked_invalid(values), origin="upper", extent=(0, 1, 1, 0),
                      vmin=0, cmap=cmap, interpolation="nearest")
    label = {"selected_share": "Selected share (%)", "enrichment": "Enrichment (x)",
             "selection_rate": "Selection rate (%)"}[metric]
    fig.colorbar(image, ax=ax, label=label)
    for i in range(BINS):
        for j in range(BINS):
            if math.isfinite(values[i, j]):
                ax.text((j + 0.5) / BINS, (i + 0.5) / BINS, f"{values[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if values[i, j] < np.nanmax(values) * 0.6 else "black")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Key relative position j/n (Early → Late)")
    ax.set_ylabel("Query relative position i/n (Early → Late)")
    ax.set_title(f"{provenance.get('dataset', 'GCG independent100')}: {label}\n"
                 "Prompt-equal, Layers 1–31; gray = no candidates")
    fig.tight_layout()
    fig.savefig(path, dpi=180, metadata={"Provenance": json.dumps(provenance, ensure_ascii=False)})
    plt.close(fig)


def save_analysis(output, metrics_rows, coordinate_rows, provenance, *,
                  expected_total=91531, expected_prompt_count=100,
                  validation=None, coordinate_filename="selected_coordinates.jsonl"):
    summary = {"provenance": provenance, **aggregate_positions(metrics_rows), "validation": {
        "prompt_count": expected_prompt_count, "prior_counts_and_layer_counts_match": True,
        "prior_selected_z_match": True, "region_and_bin_counts_conserved": True,
        "layer0_selected_count": 0, "nonfinite_metric_count": 0,
        "selection_shortfall_count": 0, "cap_activation_count": 0, "cap_violation_count": 0,
        "intervention_applied": False, "response_generation_performed": False,
    }}
    if validation:
        summary["validation"].update(validation)
    if (summary["prompt_count"] != expected_prompt_count
            or summary["total_selected_count"] != expected_total):
        raise AssertionError("selected total/prompt count mismatch")
    # No non-standard JSON NaN/Infinity; absent candidate metrics are null.
    json.dumps(summary, allow_nan=False)
    json.dumps(metrics_rows, allow_nan=False)
    ensure_output_directory_absent(output)
    output.mkdir(parents=True)
    write_json(output / "summary.json", summary)
    write_jsonl(output / "per_prompt_region_metrics.jsonl", [
        {"provenance": provenance, **row} for row in metrics_rows
    ])
    write_jsonl(output / coordinate_filename, [
        {"provenance": provenance, **row} for row in coordinate_rows
    ])
    provenance_json = json.dumps(provenance, ensure_ascii=False)
    region_csv = []
    for region, values in summary["regions"].items():
        for metric, stats in values["prompt_equal"].items():
            region_csv.append({"region": region, "metric": metric, **stats,
                               "enrichment_gt_1_prompt_count": values["enrichment_gt_1_prompt_count"],
                               "provenance": provenance_json})
    write_csv(output / "region_summary.csv", region_csv)
    bin_csv = []
    for row in summary["position_bins"]:
        flat = {"query_bin": row["query_bin"], "key_bin": row["key_bin"],
                "query_lower": row["query_bin"] / BINS, "query_upper": (row["query_bin"] + 1) / BINS,
                "key_lower": row["key_bin"] / BINS, "key_upper": (row["key_bin"] + 1) / BINS,
                "enrichment_gt_1_prompt_count": row["enrichment_gt_1_prompt_count"]}
        for metric, stats in row["prompt_equal"].items():
            flat.update({f"{metric}_{stat}": value for stat, value in stats.items()})
        flat.update({f"pooled_{key}": value for key, value in row["pooled_reference_only"].items()})
        bin_csv.append({**flat, "provenance": provenance_json})
    write_csv(output / "position_bin_10x10.csv", bin_csv)
    for metric in ("selected_share", "enrichment", "selection_rate"):
        plot_heatmap(output / f"{metric}_heatmap.png", summary["position_bins"], metric, provenance)
    manifest = {"status": "complete", "created_at": utc_now(), "provenance": provenance}
    manifest["files"] = {path.name: sha256_file(path) for path in sorted(output.iterdir())}
    write_json(output / "run_manifest.json", manifest)
    write_json(output / "artifact_manifest.json", {"files": [
        {"path": path.name, "sha256": sha256_file(path)} for path in sorted(output.iterdir())
    ]})
    return summary


def main(argv=None):
    args = parse_args(argv)
    if args.dataset == "autodan110-reference-comparison":
        from utils.autodan_dynamic_asym_position_comparison import run_comparison
        return run_comparison(args)
    if args.stage == "cpu-preflight" and (args.output_dir or args.sandbox_exempt_gpu):
        raise ValueError("cpu-preflight cannot write outputs or claim GPU execution")
    if args.coordinates_from and args.sandbox_exempt_gpu:
        raise ValueError("CPU coordinate replay must not claim GPU execution")
    output = None
    if args.stage == "run":
        if not args.output_dir:
            raise ValueError("--output-dir is required")
        output = ensure_output_directory_absent(Path(args.output_dir).resolve())
    frozen, inputs, baseline, prior, prior_path = load_inputs(args)
    replay_rows = None
    replay_manifest = None
    if args.coordinates_from:
        replay_path = Path(args.coordinates_from).resolve()
        evaluation.verify_artifact_manifest(replay_path)
        replay_manifest = read_json(replay_path / "run_manifest.json")
        if replay_manifest.get("status") != "complete" or replay_manifest["provenance"]["inputs"] != inputs:
            raise ValueError("coordinate replay provenance mismatch")
        replay_rows = read_jsonl(replay_path / "selected_coordinates.jsonl")
        analyze_coordinates(replay_rows, prior)
    print(json.dumps({"status": "cpu_preflight_passed", "prompt_count": len(prior),
                      "selected_total": sum(row["selected_count"] for row in prior.values()),
                      "reference_sha256": inputs["reference_sha256"],
                      "attention_collection_required": replay_rows is None,
                      "intervention_or_response_generation": False}, indent=2), flush=True)
    if args.stage == "cpu-preflight":
        return
    coordinate_rows = replay_rows if replay_rows is not None else collect_positions(frozen, baseline, prior)
    metrics_rows, decoded_rows = analyze_coordinates(coordinate_rows, prior)
    head, branch = git_state(REPO_ROOT)
    provenance = {
        "git_head": head, "git_branch": branch,
        "dirty_status": git_value(REPO_ROOT, "status", "--porcelain").splitlines(),
        "source_code_sha256": {path: sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS},
        "dataset": "GCG independent100", "prompt_count": 100, "dataset_indices": list(range(220, 320)),
        "inputs": inputs, "reference_path": inputs["reference"], "reference_sha256": inputs["reference_sha256"],
        "prior_run": str(prior_path), "prior_manifest_sha256": sha256_file(prior_path / "run_manifest.json"),
        "selection": evaluation.fixed_selection(), "selection_direction": "absolute",
        "intervention_target": "all", "component": "antisymmetric", "layer0_included": False,
        "candidate_definition": "Layers 1–31 × all32 Heads × full-prompt strict lower i>j",
        "cap_rate": 0.005, "cap_denominator": "all32 Layers; unchanged existing selection utility",
        "bin_count": BINS, "normalization": {"u": "i/n", "v": "j/n", "bin": "floor(10*i/n), floor(10*j/n)"},
        "region_definition": {"Early→Early": "u<0.5,v<0.5", "Late→Early": "u>=0.5,v<0.5",
                              "Late→Late": "u>=0.5,v>=0.5"},
        "aggregation": "prompt-equal; pooled metrics are supplementary only", "std_definition": "population_ddof_0",
        "seed": evaluation.SEED, "model": frozen.model, "model_dtype": "float16", "attention": "eager",
        "gpu_attention_collection": replay_rows is None, "sandbox_exempt_gpu": args.sandbox_exempt_gpu,
        "gpu": torch.cuda.get_device_name(0) if replay_rows is None else None,
        "collection_forward_count": 100 if replay_rows is None else 0,
        "intervention_applied": False, "response_generation_performed": False,
        "coordinate_source": replay_manifest["provenance"] if replay_manifest else "fresh unmodified prefill collection",
    }
    summary = save_analysis(output, metrics_rows, decoded_rows, provenance)
    print(json.dumps({"output_dir": str(output), "selected_total": summary["total_selected_count"],
                      "regions": summary["regions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
