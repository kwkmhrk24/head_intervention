#!/usr/bin/env python3
"""Compare fixed and candidate-proportional Top-N on Selection20 originals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_gcg_dynamic_asym_reference import (  # noqa: E402
    DEFAULT_MODEL,
    load_eager_model,
    tokenize_original_prompt,
)
from utils.artifact_io import (  # noqa: E402
    ensure_output_directory_absent,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)
from utils.experiment_runtime import git_state, git_value  # noqa: E402
from utils.gcg_dynamic_asym_intervention import (  # noqa: E402
    NUMERICAL_SIGMA_FLOOR,
    load_selection20_original_prompts,
    validate_dynamic_reference_payload,
)
from utils.gcg_dynamic_asym_selection_diagnostics import (  # noqa: E402
    FIXED_TOP_N,
    TOTAL_SELECTION_BUDGET,
    add_count_summaries,
    candidate_count,
    compute_leave_one_out_statistics,
    concentration_summary,
    distribution_summary,
    patch_llama_abs_z_collection,
    proportional_top_n_allocation,
    selection_count_summary,
    stable_top_indices,
    zero_count_summary,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    llama_model_config_summary,
)


DEFAULT_REFERENCE_ARTIFACT = (
    "data/outputs/gcg_dynamic_asym_reference/llama2/selection20_original/"
    "reference_20260916T094521Z/"
    "selection20_original_full_prompt_asym_reference.pt"
)
EXPECTED_REFERENCE_SHA256 = (
    "ba8f3dd92fea60ccd3cb2b9b880e9c99eaabc8ee6f6d54527673ad85092f19f2"
)
DIAGNOSTICS_FILE = "selection_diagnostics.json"
RUN_MANIFEST_FILE = "run_manifest.json"
ARTIFACT_MANIFEST_FILE = "artifact_manifest.json"
SOURCE_CODE_PATHS = (
    "scripts/diagnose_gcg_dynamic_asym_selection.py",
    "utils/gcg_dynamic_asym_selection_diagnostics.py",
    "utils/gcg_dynamic_asym_intervention.py",
    "utils/reference_asym_score_intervention.py",
    "scripts/prepare_gcg_dynamic_asym_reference.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("cpu-preflight", "run"),
        default="cpu-preflight",
    )
    parser.add_argument("--reference-artifact", default=DEFAULT_REFERENCE_ARTIFACT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--sandbox-exempt-gpu",
        action="store_true",
        help="Record that GPU execution required sandbox-exempt permissions.",
    )
    return parser.parse_args(argv)


def validate_request(args) -> Path | None:
    if args.stage == "cpu-preflight":
        if args.output_dir is not None:
            raise ValueError("cpu-preflight does not write an output directory")
        if args.sandbox_exempt_gpu:
            raise ValueError("cpu-preflight must not claim GPU execution")
        return None
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    return ensure_output_directory_absent(Path(args.output_dir).resolve())


def tensor_sha256(value: torch.Tensor) -> str:
    if value.device.type != "cpu" or not value.is_contiguous():
        value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def runtime_provenance() -> dict:
    git_head, git_branch = git_state(REPO_ROOT)
    return {
        "git_head": git_head,
        "git_branch": git_branch,
        "dirty_status": git_value(REPO_ROOT, "status", "--porcelain").splitlines(),
        "source_code_sha256": {
            relative_path: sha256_file(REPO_ROOT / relative_path)
            for relative_path in SOURCE_CODE_PATHS
        },
    }


def load_fixed_inputs(reference_artifact_path: str | Path) -> dict:
    reference_path = Path(reference_artifact_path).resolve()
    if sha256_file(reference_path) != EXPECTED_REFERENCE_SHA256:
        raise ValueError("fixed dynamic Asym reference SHA-256 mismatch")
    payload = torch.load(reference_path, map_location="cpu", weights_only=True)
    validate_dynamic_reference_payload(payload)
    metadata = payload["metadata"]
    if metadata["jailbreak_reference_data_used"] is not False:
        raise ValueError("reference metadata indicates jailbreak data use")

    reference_manifest_path = reference_path.parent / "reference_manifest.json"
    reference_manifest = read_json(reference_manifest_path)
    if reference_manifest.get("reference_sha256") != EXPECTED_REFERENCE_SHA256:
        raise ValueError("reference manifest SHA-256 declaration mismatch")
    if reference_manifest.get("jailbreak_reference_data_used") is not False:
        raise ValueError("reference manifest indicates jailbreak data use")
    if reference_manifest.get("tau") is not None:
        raise ValueError("reference manifest unexpectedly fixes tau")

    records, selection_manifest = load_selection20_original_prompts(
        split_manifest_path=metadata["selection_manifest"],
        original_input_path=metadata["original_input"],
    )
    if len(records) != SELECTION_COUNT:
        raise ValueError("Selection20 original record count mismatch")
    prompt_records = reference_manifest.get("prompt_records")
    if not isinstance(prompt_records, list) or len(prompt_records) != SELECTION_COUNT:
        raise ValueError("reference manifest prompt records are incomplete")
    lengths = []
    for position, (record, prompt_record) in enumerate(zip(records, prompt_records)):
        expected_identity = (
            position,
            record["row_position"],
            record["dataset_index"],
            record["original_prompt_sha256"],
        )
        observed_identity = (
            prompt_record.get("selection_position"),
            prompt_record.get("row_position"),
            prompt_record.get("dataset_index"),
            prompt_record.get("original_prompt_sha256"),
        )
        if observed_identity != expected_identity:
            raise ValueError("reference prompt identity mismatch")
        length = prompt_record.get("prompt_length")
        if type(length) is not int or length < 2:
            raise ValueError("reference prompt length is invalid")
        expected_elements = length * (length - 1) // 2
        if prompt_record.get("strict_lower_element_count_per_head") != expected_elements:
            raise ValueError("reference strict-lower count mismatch")
        lengths.append(length)

    candidate_counts = tuple(candidate_count(length) for length in lengths)
    proportional_counts = proportional_top_n_allocation(candidate_counts)
    if sum(proportional_counts) != TOTAL_SELECTION_BUDGET:
        raise AssertionError("proportional allocation total mismatch")
    return {
        "reference_path": reference_path,
        "reference_manifest_path": reference_manifest_path,
        "reference_payload": payload,
        "reference_manifest": reference_manifest,
        "selection_manifest": selection_manifest,
        "records": records,
        "prompt_lengths": tuple(lengths),
        "candidate_counts": candidate_counts,
        "proportional_counts": proportional_counts,
    }


def preflight_summary(inputs: dict, args) -> dict:
    return {
        "stage": args.stage,
        "diagnostic_role": "selection20_original_leave_one_out_diagnostic",
        "reference_artifact": str(inputs["reference_path"]),
        "reference_artifact_sha256": sha256_file(inputs["reference_path"]),
        "reference_manifest_sha256": sha256_file(
            inputs["reference_manifest_path"]
        ),
        "selection_count": len(inputs["records"]),
        "selection_dataset_indices": [
            record["dataset_index"] for record in inputs["records"]
        ],
        "prompt_lengths": list(inputs["prompt_lengths"]),
        "candidate_counts": list(inputs["candidate_counts"]),
        "fixed_top_n": FIXED_TOP_N,
        "proportional_top_n": list(inputs["proportional_counts"]),
        "proportional_total": sum(inputs["proportional_counts"]),
        "jailbreak_data_used": False,
        "independent100_used": False,
        "autodan_used": False,
        "intervention_applied": False,
        "generation_or_asr_evaluation": False,
        "tau": None,
    }


def rule_summary(
    *,
    abs_z: torch.Tensor,
    selected_indices: torch.Tensor,
    candidate_total: int,
    elements_per_head: int,
) -> tuple[dict, dict]:
    counts = selection_count_summary(
        selected_indices,
        layer_count=NUM_LAYERS,
        head_count=NUM_HEADS,
        elements_per_head=elements_per_head,
    )
    selected_values = abs_z[selected_indices]
    layer_head_flat = [
        value
        for layer_row in counts["layer_head_counts"]
        for value in layer_row
    ]
    return (
        {
            "selected_count": int(selected_indices.numel()),
            "selected_fraction": int(selected_indices.numel()) / candidate_total,
            "minimum_selected_abs_z": float(selected_values.min().item()),
            "selected_abs_z": distribution_summary(selected_values),
            **counts,
            "head_concentration": concentration_summary(
                counts["head_counts"], labels=tuple(range(NUM_HEADS))
            ),
            "layer_head_concentration": concentration_summary(
                layer_head_flat,
                labels=tuple(
                    (layer, head)
                    for layer in range(NUM_LAYERS)
                    for head in range(NUM_HEADS)
                ),
            ),
        },
        counts,
    )


def run_diagnostic(model, tokenizer, inputs: dict) -> dict:
    payload = inputs["reference_payload"]
    loo_references = compute_leave_one_out_statistics(
        payload["prompt_means"], payload["prompt_mean_squares"]
    )
    if len(loo_references) != SELECTION_COUNT:
        raise AssertionError("LOO reference count mismatch")

    per_prompt = []
    all_abs_z = []
    all_fixed_selected = []
    all_proportional_selected = []
    fixed_counts = zero_count_summary(NUM_LAYERS, NUM_HEADS)
    proportional_counts = zero_count_summary(NUM_LAYERS, NUM_HEADS)
    total_overlap = 0
    total_union = 0
    with patch_llama_abs_z_collection(
        model,
        layer_count=NUM_LAYERS,
        head_count=NUM_HEADS,
    ) as collector:
        for position, record in enumerate(inputs["records"]):
            input_ids = tokenize_original_prompt(tokenizer, record["prompt"])
            prompt_length = int(input_ids.shape[-1])
            if prompt_length != inputs["prompt_lengths"][position]:
                raise RuntimeError("runtime prompt length differs from reference manifest")
            loo = loo_references[position]
            if (
                loo["excluded_selection_position"] != position
                or position in loo["included_selection_positions"]
                or loo["reference_prompt_count"] != SELECTION_COUNT - 1
                or loo["self_excluded"] is not True
            ):
                raise RuntimeError("leave-one-out self-exclusion invariant failed")
            collector.clear()
            collector.set_prompt_context(
                prompt_length=prompt_length,
                excluded_selection_position=position,
                mu=loo["mu"],
                sigma=loo["sigma"],
            )
            with torch.inference_mode():
                model.model(
                    input_ids=input_ids.to(model.device),
                    use_cache=False,
                    return_dict=True,
                )
            abs_z, runtime = collector.result()
            expected_candidates = inputs["candidate_counts"][position]
            if abs_z.numel() != expected_candidates:
                raise RuntimeError("runtime candidate count mismatch")
            proportional_n = inputs["proportional_counts"][position]
            fixed_indices = stable_top_indices(abs_z, FIXED_TOP_N)
            proportional_indices = stable_top_indices(abs_z, proportional_n)
            fixed_set = set(fixed_indices.tolist())
            proportional_set = set(proportional_indices.tolist())
            overlap = len(fixed_set & proportional_set)
            union = len(fixed_set | proportional_set)
            fixed, prompt_fixed_counts = rule_summary(
                abs_z=abs_z,
                selected_indices=fixed_indices,
                candidate_total=expected_candidates,
                elements_per_head=runtime["strict_lower_elements_per_head"],
            )
            proportional, prompt_proportional_counts = rule_summary(
                abs_z=abs_z,
                selected_indices=proportional_indices,
                candidate_total=expected_candidates,
                elements_per_head=runtime["strict_lower_elements_per_head"],
            )
            fixed_counts = add_count_summaries(fixed_counts, prompt_fixed_counts)
            proportional_counts = add_count_summaries(
                proportional_counts, prompt_proportional_counts
            )
            all_abs_z.append(abs_z)
            all_fixed_selected.append(abs_z[fixed_indices])
            all_proportional_selected.append(abs_z[proportional_indices])
            total_overlap += overlap
            total_union += union
            per_prompt.append(
                {
                    "selection_position": position,
                    "row_position": record["row_position"],
                    "dataset_index": record["dataset_index"],
                    "prompt_id": record["prompt_id"],
                    "original_prompt_sha256": record["original_prompt_sha256"],
                    "prompt_length": prompt_length,
                    "candidate_count": expected_candidates,
                    "abs_z_distribution": distribution_summary(abs_z),
                    "fixed_top500": fixed,
                    "proportional_top_n": {
                        "n": proportional_n,
                        **proportional,
                    },
                    "overlap_count": overlap,
                    "union_count": union,
                    "jaccard": overlap / union,
                    "loo": {
                        "excluded_selection_position": position,
                        "included_selection_positions": list(
                            loo["included_selection_positions"]
                        ),
                        "reference_prompt_count": loo["reference_prompt_count"],
                        "self_excluded": loo["self_excluded"],
                        "mu_sha256": tensor_sha256(loo["mu"]),
                        "sigma_sha256": tensor_sha256(loo["sigma"]),
                        "clamped_negative_variance_count": loo[
                            "clamped_negative_variance_count"
                        ],
                        "sigma_floor_head_count": loo["sigma_floor_head_count"],
                    },
                    "runtime": runtime,
                }
            )
            print(
                f"[{position + 1}/{SELECTION_COUNT}] "
                f"dataset_index={record['dataset_index']} L={prompt_length} "
                f"C={expected_candidates} N={proportional_n} "
                f"overlap={overlap}",
                flush=True,
            )

    all_abs_z_tensor = torch.cat(all_abs_z)
    fixed_selected_tensor = torch.cat(all_fixed_selected)
    proportional_selected_tensor = torch.cat(all_proportional_selected)
    expected_total_candidates = sum(inputs["candidate_counts"])
    if all_abs_z_tensor.numel() != expected_total_candidates:
        raise AssertionError("overall candidate count mismatch")
    if fixed_selected_tensor.numel() != TOTAL_SELECTION_BUDGET:
        raise AssertionError("overall fixed selection count mismatch")
    if proportional_selected_tensor.numel() != TOTAL_SELECTION_BUDGET:
        raise AssertionError("overall proportional selection count mismatch")

    def overall_rule(counts: dict, selected: torch.Tensor) -> dict:
        layer_head_flat = [
            value for row in counts["layer_head_counts"] for value in row
        ]
        return {
            "selected_count": int(selected.numel()),
            "selected_fraction": int(selected.numel()) / expected_total_candidates,
            "minimum_selected_abs_z": float(selected.min().item()),
            "selected_abs_z": distribution_summary(selected),
            **counts,
            "head_concentration": concentration_summary(
                counts["head_counts"], labels=tuple(range(NUM_HEADS))
            ),
            "layer_head_concentration": concentration_summary(
                layer_head_flat,
                labels=tuple(
                    (layer, head)
                    for layer in range(NUM_LAYERS)
                    for head in range(NUM_HEADS)
                ),
            ),
        }

    overall = {
        "prompt_count": SELECTION_COUNT,
        "candidate_count": expected_total_candidates,
        "abs_z_distribution": distribution_summary(all_abs_z_tensor),
        "fixed_top500": overall_rule(fixed_counts, fixed_selected_tensor),
        "proportional_top_n": {
            "allocation": list(inputs["proportional_counts"]),
            **overall_rule(proportional_counts, proportional_selected_tensor),
        },
        "overlap_count": total_overlap,
        "union_count": total_union,
        "jaccard": total_overlap / total_union,
        "loo_self_exclusion_verified_count": sum(
            row["loo"]["self_excluded"] for row in per_prompt
        ),
        "clamped_negative_variance_count": sum(
            row["loo"]["clamped_negative_variance_count"] for row in per_prompt
        ),
        "sigma_floor_head_count": sum(
            row["loo"]["sigma_floor_head_count"] for row in per_prompt
        ),
        "pre_score_nonfinite_count": sum(
            row["runtime"]["pre_score_nonfinite_count"] for row in per_prompt
        ),
        "z_nonfinite_count": sum(
            row["runtime"]["z_nonfinite_count"] for row in per_prompt
        ),
    }
    return {
        "schema_version": 1,
        "diagnostic_role": "selection20_original_leave_one_out_diagnostic",
        "scientific_conditions": {
            "dataset": "Selection20 original only",
            "loo_reference_prompt_count": SELECTION_COUNT - 1,
            "score_capture_position": (
                "post-RoPE, after repeat_kv, immediately after "
                "QK^T/sqrt(head_dim), pre-mask, pre-softmax"
            ),
            "position_scope": "full_prompt_strict_lower_i_gt_j",
            "component": "A_asym=(S-S^T)/2",
            "fixed_rule": "per-prompt abs_z Top-500",
            "proportional_rule": (
                "total 10000 allocated proportional to candidate count by floor, "
                "then largest integer remainder; ties by selection position"
            ),
            "candidate_ranking_tie_break": "flat candidate identity ascending",
            "jailbreak_data_used": False,
            "independent100_used": False,
            "autodan_used": False,
            "intervention_applied": False,
            "generation_or_asr_evaluation": False,
            "tau": None,
        },
        "per_prompt": per_prompt,
        "overall": overall,
    }


def main(argv=None):
    args = parse_args(argv)
    output_dir = validate_request(args)
    inputs = load_fixed_inputs(args.reference_artifact)
    preflight = preflight_summary(inputs, args)
    print(json.dumps(preflight, indent=2))
    if args.stage == "cpu-preflight":
        print("CPU preflight complete; no files created.")
        return

    provenance = runtime_provenance()
    model, tokenizer = load_eager_model(args.model)
    model_summary = llama_model_config_summary(model)
    diagnostics = run_diagnostic(model, tokenizer, inputs)
    diagnostics.update(
        {
            "created_at": utc_now(),
            "reference_artifact": str(inputs["reference_path"]),
            "reference_artifact_sha256": EXPECTED_REFERENCE_SHA256,
        }
    )

    ensure_output_directory_absent(output_dir)
    output_dir.mkdir(parents=True)
    diagnostics_path = output_dir / DIAGNOSTICS_FILE
    write_json(diagnostics_path, diagnostics)
    run_manifest = {
        **preflight,
        "created_at": diagnostics["created_at"],
        **provenance,
        "sandbox_exempt_gpu": args.sandbox_exempt_gpu,
        "model_path": str(Path(args.model).resolve()),
        "model_config": model_summary,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "reference_manifest": str(inputs["reference_manifest_path"]),
        "reference_manifest_sha256": sha256_file(
            inputs["reference_manifest_path"]
        ),
        "selection_manifest_sha256": inputs["reference_payload"]["metadata"][
            "selection_manifest_sha256"
        ],
        "original_input_sha256": inputs["reference_payload"]["metadata"][
            "original_input_sha256"
        ],
        "diagnostics_file": DIAGNOSTICS_FILE,
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "raw_per_element_tensor_saved": False,
    }
    run_manifest_path = output_dir / RUN_MANIFEST_FILE
    write_json(run_manifest_path, run_manifest)
    artifact_manifest = {
        "schema_version": 1,
        "created_at": diagnostics["created_at"],
        "files": [
            {
                "path": DIAGNOSTICS_FILE,
                "sha256": sha256_file(diagnostics_path),
            },
            {
                "path": RUN_MANIFEST_FILE,
                "sha256": sha256_file(run_manifest_path),
            },
        ],
    }
    artifact_manifest_path = output_dir / ARTIFACT_MANIFEST_FILE
    write_json(artifact_manifest_path, artifact_manifest)
    overall = diagnostics["overall"]
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "diagnostics_sha256": sha256_file(diagnostics_path),
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
                "candidate_count": overall["candidate_count"],
                "fixed_selected_count": overall["fixed_top500"][
                    "selected_count"
                ],
                "proportional_selected_count": overall["proportional_top_n"][
                    "selected_count"
                ],
                "overlap_count": overall["overlap_count"],
                "jaccard": overall["jaccard"],
                "pre_score_nonfinite_count": overall[
                    "pre_score_nonfinite_count"
                ],
                "z_nonfinite_count": overall["z_nonfinite_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
