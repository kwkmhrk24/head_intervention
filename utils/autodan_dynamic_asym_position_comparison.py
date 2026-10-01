"""AutoDAN reference comparison for the existing selection-position runner.

Only original Selection20 builds C; only evaluation110 is analyzed. B and C
observe the same unmodified prefill, with the existing z-score/selection code.
"""

import json
from pathlib import Path

import torch

from scripts import analyze_gcg_dynamic_asym_selected_positions as analysis
from scripts import prepare_gcg_dynamic_asym_reference as builder
from utils.artifact_io import (
    ensure_output_directory_absent, read_json, read_jsonl, sha256_file,
    sha256_text, write_json,
)
from utils.experiment_runtime import git_state, git_value, seed_everything
from utils.gcg_dynamic_asym_intervention import load_dynamic_reference
from utils.gcg_dynamic_asym_selection_diagnostics import patch_llama_multi_reference_z_collection


DEFAULT_PRIOR = (
    "data/outputs/gcg_dynamic_asym_replacement_modes/llama2/autodan130/"
    "raw_s_zero/evaluation_20261001T062025Z"
)
CONDITIONS = {"B": "gcg_original_ref", "C": "autodan_original_ref"}


def validate_original_reference(path, originals, split):
    path = Path(path).resolve()
    reference = load_dynamic_reference(path)
    metadata = reference.metadata
    manifest = read_json(path.parent / builder.MANIFEST_FILE)
    expected_records = [{key: row[key] for key in ("dataset_index", "original_prompt_sha256")}
                        for row in originals]
    if (
        metadata["reference_condition"] != builder.AUTODAN_ORIGINAL_CONDITION
        or metadata["original_prompt_records"] != expected_records
        or metadata["selection_dataset_indices"] != split["selection_indices"]
        or metadata["evaluation_dataset_indices"] != split["evaluation_indices"]
        or metadata["original_input_sha256"] != sha256_file(builder.DEFAULT_AUTODAN_ORIGINAL_INPUT)
        or metadata["selection_manifest_sha256"] != builder.EXPECTED_AUTODAN_SELECTION_MANIFEST_SHA256
        or manifest["reference_sha256"] != sha256_file(path)
        or manifest["jailbreak_reference_data_used"] is not False
        or [row["dataset_index"] for row in manifest["prompt_records"]] != split["selection_indices"]
        or [row["prompt_sha256"] for row in manifest["prompt_records"]]
        != [row["original_prompt_sha256"] for row in originals]
    ):
        raise ValueError("C reference must use exactly the audited corresponding original20")
    return reference


def audit_inputs(args):
    evaluation = analysis.evaluation
    originals, split = builder.load_autodan_selection20_records(
        builder.DEFAULT_AUTODAN_SELECTION_MANIFEST,
        original_input=builder.DEFAULT_AUTODAN_ORIGINAL_INPUT,
    )
    frozen = evaluation.parse_args(["--dataset", "autodan130", "--autodan-split", "evaluation110"])
    builder.validate_autodan_original_model(frozen.model)
    inputs, baseline = evaluation.validate_frozen_inputs(frozen)
    baseline = evaluation.restrict_autodan_baseline(frozen, baseline)
    if [row["dataset_index"] for row in baseline] != split["evaluation_indices"]:
        raise ValueError("baseline/evaluation110 identity mismatch")
    prior_path = Path(DEFAULT_PRIOR).resolve()
    evaluation.verify_artifact_manifest(prior_path)
    manifest = read_json(prior_path / "run_manifest.json")
    expected = {
        "dataset": "autodan130", "autodan_split": "full130", "component": "antisymmetric",
        "selection_direction": "absolute", "intervention_target_sign": "all",
        "selection": evaluation.fixed_selection(), "layer0_included": False,
        "position_scope": "full_prompt_strict_lower_i_gt_j", "cap_rate": 0.005,
        "reference_source": "gcg-selection20-original",
    }
    if manifest["status"] != "complete" or manifest["provenance"] != inputs:
        raise ValueError("prior AutoDAN provenance mismatch")
    for key, value in expected.items():
        if manifest["scientific_conditions"].get(key) != value:
            raise ValueError(f"prior AutoDAN selection condition mismatch: {key}")
    prior = {row["dataset_index"]: row for row in read_jsonl(prior_path / "prompt_results.jsonl")
             if row["dataset_index"] in split["evaluation_indices"]}
    if set(prior) != set(split["evaluation_indices"]):
        raise ValueError("prior evaluation110 incomplete")
    gcg_path = Path(args.gcg_analysis).resolve()
    evaluation.verify_artifact_manifest(gcg_path)
    gcg = read_json(gcg_path / "summary.json")
    if (gcg["prompt_count"] != 100 or gcg["total_selected_count"] != 91531
            or gcg["provenance"]["reference_sha256"] != inputs["reference_sha256"]
            or gcg["provenance"]["selection"] != evaluation.fixed_selection()):
        raise ValueError("existing GCG A analysis condition mismatch")
    references = {"B": load_dynamic_reference(inputs["reference"])}
    if args.autodan_original_reference:
        references["C"] = validate_original_reference(args.autodan_original_reference, originals, split)
    return frozen, inputs, baseline, originals, split, prior, prior_path, gcg, references


def selection_plan(signed_z, collection):
    evaluation = analysis.evaluation
    plan = evaluation.build_stage2_selection_plan(
        signed_z.abs(), selection=evaluation.fixed_selection(), layer_count=32, head_count=32,
        elements_per_head=collection["strict_lower_elements_per_head"],
        layer0_included=False, signed_z=signed_z, selection_direction="absolute",
    )
    plan = evaluation.restrict_absolute_plan_to_intervention_sign(
        plan, signed_z, intervention_target_sign="all", layer_count=32, head_count=32,
        elements_per_head=collection["strict_lower_elements_per_head"],
    )
    if (not collection["all_finite"] or plan["selection_shortfall_count"]
            or plan["cap_activated"] or plan["selected_count"] > plan["cap_count"]
            or plan["layer_counts"][0]):
        raise RuntimeError("selection invariant failed")
    return plan


def compare_selected_positions(b_rows, c_rows):
    if (not b_rows or [row["dataset_index"] for row in b_rows] != [row["dataset_index"] for row in c_rows]
            or len({row["dataset_index"] for row in b_rows}) != len(b_rows)):
        raise ValueError("B/C indices differ or are duplicated")
    comparisons = []
    for b, c in zip(b_rows, c_rows):
        if b["prompt_length"] != c["prompt_length"]:
            raise ValueError("B/C prompt lengths differ")
        b_set, c_set = set(b["selected_indices"]), set(c["selected_indices"])
        if not b_set or not c_set or len(b_set) != len(b["selected_indices"]) or len(c_set) != len(c["selected_indices"]):
            raise ValueError("empty or duplicated selection identities")
        intersection, union = len(b_set & c_set), len(b_set | c_set)
        comparisons.append({
            "dataset_index": b["dataset_index"], "B_selected_count": len(b_set),
            "C_selected_count": len(c_set), "C_minus_B_selected_count": len(c_set) - len(b_set),
            "intersection_count": intersection, "union_count": union,
            "B_only_count": len(b_set - c_set), "C_only_count": len(c_set - b_set),
            "overlap_fraction_of_B": intersection / len(b_set), "jaccard": intersection / union,
        })
    return {
        "prompt_count": len(comparisons), "per_prompt": comparisons,
        "prompt_equal_overlap_fraction_of_B": analysis.metric_summary([r["overlap_fraction_of_B"] for r in comparisons]),
        "prompt_equal_jaccard": analysis.metric_summary([r["jaccard"] for r in comparisons]),
        "intersection_total": sum(r["intersection_count"] for r in comparisons),
        "pooled_jaccard_supplementary": sum(r["intersection_count"] for r in comparisons) / sum(r["union_count"] for r in comparisons),
    }


def run_comparison(args):
    if args.coordinates_from:
        raise ValueError("GCG coordinate replay cannot substitute AutoDAN evaluation110")
    if args.stage == "cpu-preflight" and (args.output_dir or args.sandbox_exempt_gpu):
        raise ValueError("CPU preflight cannot write outputs or claim GPU execution")
    output = None
    if args.stage == "run":
        if not args.output_dir or not args.autodan_original_reference:
            raise ValueError("run requires --output-dir and --autodan-original-reference")
        output = ensure_output_directory_absent(Path(args.output_dir).resolve())
    frozen, inputs, baseline, originals, split, prior, prior_path, gcg, references = audit_inputs(args)
    print({"status": "cpu_preflight_passed", "original_reference_prompt_count": len(originals),
           "evaluation_prompt_count": len(prior), "split_overlap_count": 0,
           "B_expected_selected_total": sum(row["selected_count"] for row in prior.values()),
           "C_reference_validated": "C" in references, "generation_or_intervention": False}, flush=True)
    if args.stage == "cpu-preflight":
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA required; no CPU fallback")
    evaluation = analysis.evaluation
    seed_everything(42)
    model, tokenizer = evaluation.load_model_and_tokenizer(Path(frozen.model).resolve())
    raw_records, _ = evaluation.load_autodan_transfer_records(Path(frozen.autodan_input))
    records = evaluation.format_autodan_records(
        [r for r in raw_records if r["dataset_index"] in prior], tokenizer,
        prompt_format=builder.AUTODAN_OFFICIAL_FORMAT,
    )
    if [r["dataset_index"] for r in records] != split["evaluation_indices"]:
        raise RuntimeError("formatted evaluation110 identity mismatch")
    baseline_by_index = {r["dataset_index"]: r for r in baseline}
    rows = {name: [] for name in CONDITIONS}
    coordinate_rows = {name: [] for name in CONDITIONS}
    for position, record in enumerate(records, 1):
        index = record["dataset_index"]
        if (evaluation.serialized_input_ids(record["input_ids"]) != baseline_by_index[index]["input_ids"]
                or record["formatted_prompt"] != baseline_by_index[index]["formatted_prompt"]):
            raise RuntimeError("official formatted tokens differ from baseline")
        with patch_llama_multi_reference_z_collection(
            model, references=references, prompt_length=record["prompt_length"],
        ) as collectors:
            with torch.inference_mode():
                model.model(input_ids=record["input_ids"].to(model.device), use_cache=False, return_dict=True)
            collected = {name: collector.result(signed=True) for name, collector in collectors.items()}
        for name, (signed_z, collection) in collected.items():
            plan = selection_plan(signed_z, collection)
            old = prior[index]
            if name == "B":
                for key in ("selected_count", "selected_z", "positive_selected_count", "negative_selected_count", "cap_count"):
                    if plan[key] != old[key]:
                        raise RuntimeError(f"B prior selection mismatch: index={index}, {key}")
                if (record["prompt_length"] != old["prompt_length"]
                        or plan["layer_counts"] != old["diagnostics"]["planned_layer_counts"]):
                    raise RuntimeError("B prior length/Layer selection mismatch")
            metrics, coordinates = analysis.position_metrics(record["prompt_length"], plan["selected_indices"])
            if metrics["layer_counts"] != plan["layer_counts"] or plan["selected_count"] != old["selected_count"]:
                raise RuntimeError("B/C proportional selected count/Layer mismatch")
            rows[name].append({"dataset_index": index, "prompt_id": record["prompt_id"], **metrics})
            coordinate_rows[name].append({
                "dataset_index": index, "prompt_id": record["prompt_id"], "prompt_length": record["prompt_length"],
                "prompt_sha256": sha256_text(record["prompt"]),
                "formatted_prompt_sha256": sha256_text(record["formatted_prompt"]),
                "input_ids_sha256": analysis.tensor_sha256(record["input_ids"]),
                "selected_indices": plan["selected_indices"].tolist(),
                "selected_indices_sha256": analysis.tensor_sha256(plan["selected_indices"]),
                "coordinate_order": ["layer", "head", "i", "j"], "selected_coordinates": coordinates,
                "selected_z": plan["selected_z"], "layer_counts": plan["layer_counts"], "collection": collection,
            })
        print(f"[shared prefill {position}/110] index={index} selected={old['selected_count']}", flush=True)
    head, branch = git_state(analysis.REPO_ROOT)
    common = {
        "git_head": head, "git_branch": branch,
        "dirty_status": git_value(analysis.REPO_ROOT, "status", "--porcelain").splitlines(),
        "source_code_sha256": {p: sha256_file(analysis.REPO_ROOT / p) for p in (*analysis.SOURCE_PATHS,
            "utils/autodan_dynamic_asym_position_comparison.py", "scripts/prepare_gcg_dynamic_asym_reference.py")},
        "dataset": "AutoDAN evaluation110", "prompt_count": 110, "dataset_indices": split["evaluation_indices"],
        "reference_selection_indices": split["selection_indices"],
        "split_manifest": str(Path(builder.DEFAULT_AUTODAN_SELECTION_MANIFEST).resolve()),
        "split_manifest_sha256": builder.EXPECTED_AUTODAN_SELECTION_MANIFEST_SHA256,
        "dataset_sources": split["source_provenance"], "inputs": inputs,
        "prompt_format": builder.AUTODAN_OFFICIAL_FORMAT, "seed": 42,
        "selection": evaluation.fixed_selection(), "selection_direction": "absolute", "intervention_target": "all",
        "component": "antisymmetric", "layer0_included": False, "cap_rate": 0.005,
        "candidate_definition": "Layers 1–31 × all32 Heads × full-prompt strict lower i>j",
        "cap_denominator": "all32 Layers; unchanged existing selection utility",
        "bin_count": 10, "normalization": {"u": "i/n", "v": "j/n", "bin": "floor(10*i/n), floor(10*j/n)"},
        "region_definition": gcg["provenance"]["region_definition"],
        "aggregation": "prompt-equal; pooled metrics are supplementary only", "std_definition": "population_ddof_0",
        "model": frozen.model, "model_repository_head": builder.MODEL_REPOSITORY_HEAD,
        "model_dtype": "float16", "attention": "eager",
        "gpu": torch.cuda.get_device_name(0), "sandbox_exempt_gpu": args.sandbox_exempt_gpu,
        "collection_forward_count_shared_B_C": 110, "reference_collection_forward_count_C": 20,
        "intervention_applied": False, "response_generation_performed": False,
        "prior_run": str(prior_path), "prior_manifest_sha256": sha256_file(prior_path / "run_manifest.json"),
        "A_analysis": str(Path(args.gcg_analysis).resolve()),
        "A_summary_sha256": sha256_file(Path(args.gcg_analysis) / "summary.json"),
    }
    summaries = {"A": gcg}
    for name, dirname in CONDITIONS.items():
        reference_path = inputs["reference"] if name == "B" else args.autodan_original_reference
        provenance = {**common, "condition": name, "reference_path": str(Path(reference_path).resolve()),
                      "reference_sha256": sha256_file(reference_path), "reference_metadata": references[name].metadata}
        summaries[name] = analysis.save_analysis(
            output / dirname, rows[name], coordinate_rows[name], provenance,
            expected_total=sum(r["selected_count"] for r in prior.values()), expected_prompt_count=110,
            coordinate_filename="selected_positions.jsonl",
            validation={"prior_counts_and_layer_counts_match": True if name == "B" else None,
                        "prior_selected_z_match": True if name == "B" else None,
                        "split_overlap_count": 0, "official_baseline_input_ids_match": True,
                        "shared_prefill_input_B_C_identical": True, "reference_jailbreak_data_used": False},
        )
    overlap = compare_selected_positions(coordinate_rows["B"], coordinate_rows["C"])
    table = [{"region": region, **{name: summary["regions"][region]["prompt_equal"]["enrichment"]["mean"]
                                  for name, summary in summaries.items()}} for region in analysis.REGIONS]
    bin_diff = [{"query_bin": b["query_bin"], "key_bin": b["key_bin"], **{
        f"C_minus_B_{metric}": (c["prompt_equal"][metric]["mean"] - b["prompt_equal"][metric]["mean"]
                                if b["prompt_equal"][metric]["mean"] is not None else None)
        for metric in ("selected_share", "enrichment", "selection_rate")}}
        for b, c in zip(summaries["B"]["position_bins"], summaries["C"]["position_bins"])]
    comparison = {
        "provenance": common, "condition_directories": CONDITIONS, "enrichment_comparison": table,
        "late_early_enrichment_gt_1": {name: {"count": s["regions"][analysis.REGIONS[1]]["enrichment_gt_1_prompt_count"],
                                             "prompt_count": s["prompt_count"]} for name, s in summaries.items()},
        "selected_positions_overlap_B_C": overlap, "position_bins_C_minus_B": bin_diff,
        "validation": {"B_C_prompt_indices_identical": True, "B_C_selected_counts_identical": True,
                       "split_overlap_count": 0, "GCG_prefill_count": 0, "total_GPU_prefill_count": 130},
    }
    json.dumps(comparison, allow_nan=False)
    write_json(output / "reference_comparison_summary.json", comparison)
    provenance_json = json.dumps(common, ensure_ascii=False)
    analysis.write_csv(output / "reference_comparison.csv", [{**r, "provenance": provenance_json} for r in table])
    analysis.write_csv(output / "position_bin_difference.csv", [{**r, "provenance": provenance_json} for r in bin_diff])
    write_json(output / "artifact_manifest.json", {"files": [
        {"path": str(p.relative_to(output)), "sha256": sha256_file(p)} for p in sorted(output.rglob("*")) if p.is_file()
    ]})
    print({"output_dir": str(output), "enrichment_comparison": table,
           "overlap_mean": overlap["prompt_equal_overlap_fraction_of_B"]["mean"],
           "jaccard_mean": overlap["prompt_equal_jaccard"]["mean"]}, flush=True)
