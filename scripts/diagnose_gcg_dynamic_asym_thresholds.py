#!/usr/bin/env python3
"""Diagnose fixed z-score thresholds on Selection20 original LOO scores."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_gcg_dynamic_asym_selection import (  # noqa: E402
    DEFAULT_REFERENCE_ARTIFACT,
    EXPECTED_REFERENCE_SHA256,
    load_fixed_inputs,
    tensor_sha256,
)
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
from utils.gcg_dynamic_asym_selection_diagnostics import (  # noqa: E402
    FIXED_TOP_N,
    TOTAL_SELECTION_BUDGET,
    add_count_summaries,
    compute_leave_one_out_statistics,
    concentration_summary,
    distribution_summary,
    patch_llama_abs_z_collection,
    prompt_selected_count_summary,
    threshold_selection_summary,
    zero_count_summary,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    llama_model_config_summary,
)


TAUS = (2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.5)
DEFAULT_SELECTION_DIAGNOSTICS = (
    "data/outputs/gcg_dynamic_asym_selection_diagnostics/llama2/"
    "selection20_original/diagnostic_20260916T120053Z/selection_diagnostics.json"
)
EXPECTED_SELECTION_DIAGNOSTICS_SHA256 = (
    "1422e0b9538f3c76a4b14ed90eb8fd41fbc79ec9cef8b4ba101f0e2ea91fc0fd"
)
DIAGNOSTICS_FILE = "threshold_selection_diagnostics.json"
RUN_MANIFEST_FILE = "run_manifest.json"
ARTIFACT_MANIFEST_FILE = "artifact_manifest.json"
SOURCE_CODE_PATHS = (
    "scripts/diagnose_gcg_dynamic_asym_thresholds.py",
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
    parser.add_argument(
        "--selection-diagnostics",
        default=DEFAULT_SELECTION_DIAGNOSTICS,
        help="Immutable Fixed/proportional comparison artifact.",
    )
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


def _require_baseline_conditions(diagnostics: dict) -> None:
    conditions = diagnostics.get("scientific_conditions")
    required = {
        "dataset": "Selection20 original only",
        "loo_reference_prompt_count": SELECTION_COUNT - 1,
        "position_scope": "full_prompt_strict_lower_i_gt_j",
        "component": "A_asym=(S-S^T)/2",
        "jailbreak_data_used": False,
        "independent100_used": False,
        "autodan_used": False,
        "intervention_applied": False,
        "generation_or_asr_evaluation": False,
        "tau": None,
    }
    if not isinstance(conditions, dict):
        raise ValueError("selection diagnostic scientific conditions are missing")
    for key, expected in required.items():
        if conditions.get(key) != expected:
            raise ValueError(f"selection diagnostic condition mismatch: {key}")


def load_selection_baseline(path: str | Path, inputs: dict) -> dict:
    baseline_path = Path(path).resolve()
    if sha256_file(baseline_path) != EXPECTED_SELECTION_DIAGNOSTICS_SHA256:
        raise ValueError("fixed selection diagnostic SHA-256 mismatch")
    diagnostics = read_json(baseline_path)
    if diagnostics.get("reference_artifact_sha256") != EXPECTED_REFERENCE_SHA256:
        raise ValueError("selection diagnostic reference SHA-256 mismatch")
    _require_baseline_conditions(diagnostics)
    prompt_rows = diagnostics.get("per_prompt")
    if not isinstance(prompt_rows, list) or len(prompt_rows) != SELECTION_COUNT:
        raise ValueError("selection diagnostic prompt records are incomplete")
    for position, (record, row) in enumerate(zip(inputs["records"], prompt_rows)):
        observed = (
            row.get("selection_position"),
            row.get("row_position"),
            row.get("dataset_index"),
            row.get("original_prompt_sha256"),
            row.get("prompt_length"),
            row.get("candidate_count"),
            row.get("fixed_top500", {}).get("selected_count"),
            row.get("proportional_top_n", {}).get("n"),
        )
        expected = (
            position,
            record["row_position"],
            record["dataset_index"],
            record["original_prompt_sha256"],
            inputs["prompt_lengths"][position],
            inputs["candidate_counts"][position],
            FIXED_TOP_N,
            inputs["proportional_counts"][position],
        )
        if observed != expected:
            raise ValueError("selection diagnostic prompt identity/count mismatch")
    overall = diagnostics.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("selection diagnostic overall summary is missing")
    if overall.get("candidate_count") != sum(inputs["candidate_counts"]):
        raise ValueError("selection diagnostic candidate total mismatch")
    for name in ("fixed_top500", "proportional_top_n"):
        if overall.get(name, {}).get("selected_count") != TOTAL_SELECTION_BUDGET:
            raise ValueError(f"selection diagnostic {name} budget mismatch")
    for name in (
        "clamped_negative_variance_count",
        "sigma_floor_head_count",
        "pre_score_nonfinite_count",
        "z_nonfinite_count",
    ):
        if overall.get(name) != 0:
            raise ValueError(f"selection diagnostic numerical anomaly: {name}")
    return {"path": baseline_path, "payload": diagnostics}


def comparison_baselines(baseline: dict, inputs: dict) -> dict:
    overall = baseline["payload"]["overall"]
    fixed = dict(overall["fixed_top500"])
    fixed["prompt_selected_count"] = prompt_selected_count_summary(
        [FIXED_TOP_N] * SELECTION_COUNT
    )
    fixed["distance_of_mean_from_500"] = 0.0
    proportional_values = list(inputs["proportional_counts"])
    proportional = dict(overall["proportional_top_n"])
    proportional["prompt_selected_counts"] = proportional_values
    proportional["prompt_selected_count"] = prompt_selected_count_summary(
        proportional_values
    )
    proportional["distance_of_mean_from_500"] = (
        proportional["prompt_selected_count"]["mean"] - FIXED_TOP_N
    )
    return {
        "source_artifact": str(baseline["path"]),
        "source_artifact_sha256": EXPECTED_SELECTION_DIAGNOSTICS_SHA256,
        "fixed_top500": fixed,
        "proportional_top_n": proportional,
    }


def preflight_summary(inputs: dict, baseline: dict, args) -> dict:
    return {
        "stage": args.stage,
        "diagnostic_role": "selection20_original_leave_one_out_z_threshold_diagnostic",
        "reference_artifact": str(inputs["reference_path"]),
        "reference_artifact_sha256": EXPECTED_REFERENCE_SHA256,
        "selection_diagnostics": str(baseline["path"]),
        "selection_diagnostics_sha256": EXPECTED_SELECTION_DIAGNOSTICS_SHA256,
        "selection_count": len(inputs["records"]),
        "prompt_lengths": list(inputs["prompt_lengths"]),
        "candidate_counts": list(inputs["candidate_counts"]),
        "candidate_total": sum(inputs["candidate_counts"]),
        "taus": list(TAUS),
        "threshold_rule": "abs_z > tau",
        "std_definition": "population_ddof_0",
        "fixed_top500_total": TOTAL_SELECTION_BUDGET,
        "proportional_top_n": list(inputs["proportional_counts"]),
        "proportional_total": sum(inputs["proportional_counts"]),
        "jailbreak_data_used": False,
        "independent100_used": False,
        "autodan_used": False,
        "intervention_applied": False,
        "generation_or_asr_evaluation": False,
        "final_tau_selected": False,
        "alpha_used": False,
    }


def _assert_distribution_matches(observed: dict, expected: dict) -> None:
    if observed.keys() != expected.keys():
        raise RuntimeError("runtime abs_z distribution fields differ from baseline")
    for key in observed:
        left = observed[key]
        right = expected[key]
        if isinstance(left, int):
            matches = left == right
        else:
            matches = math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-7)
        if not matches:
            raise RuntimeError(
                f"runtime abs_z distribution differs from baseline at {key}"
            )


def run_threshold_diagnostic(model, tokenizer, inputs: dict, baseline: dict) -> dict:
    payload = inputs["reference_payload"]
    loo_references = compute_leave_one_out_statistics(
        payload["prompt_means"], payload["prompt_mean_squares"]
    )
    if len(loo_references) != SELECTION_COUNT:
        raise AssertionError("LOO reference count mismatch")

    aggregate_counts = {
        tau: zero_count_summary(NUM_LAYERS, NUM_HEADS) for tau in TAUS
    }
    aggregate_selected_values = {tau: [] for tau in TAUS}
    prompt_selected_counts = {tau: [] for tau in TAUS}
    per_prompt = []
    baseline_rows = baseline["payload"]["per_prompt"]
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
            abs_z_distribution = distribution_summary(abs_z)
            _assert_distribution_matches(
                abs_z_distribution, baseline_rows[position]["abs_z_distribution"]
            )

            threshold_rows = []
            previous_count = expected_candidates + 1
            for tau in TAUS:
                summary, counts, selected_values = threshold_selection_summary(
                    abs_z,
                    tau=tau,
                    layer_count=NUM_LAYERS,
                    head_count=NUM_HEADS,
                    elements_per_head=runtime["strict_lower_elements_per_head"],
                )
                selected_count = summary["selected_count"]
                if selected_count > previous_count:
                    raise AssertionError("threshold counts are not monotone in tau")
                previous_count = selected_count
                aggregate_counts[tau] = add_count_summaries(
                    aggregate_counts[tau], counts
                )
                aggregate_selected_values[tau].append(selected_values)
                prompt_selected_counts[tau].append(selected_count)
                threshold_rows.append(summary)
            per_prompt.append(
                {
                    "selection_position": position,
                    "row_position": record["row_position"],
                    "dataset_index": record["dataset_index"],
                    "prompt_id": record["prompt_id"],
                    "original_prompt_sha256": record["original_prompt_sha256"],
                    "prompt_length": prompt_length,
                    "candidate_count": expected_candidates,
                    "abs_z_distribution": abs_z_distribution,
                    "thresholds": threshold_rows,
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
            count_text = ", ".join(
                f"{tau:.1f}:{prompt_selected_counts[tau][-1]}" for tau in TAUS
            )
            print(
                f"[{position + 1}/{SELECTION_COUNT}] "
                f"dataset_index={record['dataset_index']} L={prompt_length} "
                f"threshold_counts={count_text}",
                flush=True,
            )

    candidate_total = sum(inputs["candidate_counts"])
    thresholds = []
    previous_total = candidate_total + 1
    for tau in TAUS:
        counts = aggregate_counts[tau]
        selected_count_values = prompt_selected_counts[tau]
        selected_total = sum(selected_count_values)
        if selected_total > previous_total:
            raise AssertionError("overall threshold counts are not monotone in tau")
        previous_total = selected_total
        if sum(counts["layer_counts"]) != selected_total:
            raise AssertionError("threshold layer count total mismatch")
        selected_values = torch.cat(aggregate_selected_values[tau])
        if selected_values.numel() != selected_total:
            raise AssertionError("threshold selected value total mismatch")
        layer_head_flat = [
            value for row in counts["layer_head_counts"] for value in row
        ]
        prompt_summary = prompt_selected_count_summary(selected_count_values)
        if selected_total:
            selected_distribution = distribution_summary(selected_values)
            head_concentration = concentration_summary(
                counts["head_counts"], labels=tuple(range(NUM_HEADS))
            )
            layer_head_concentration = concentration_summary(
                layer_head_flat,
                labels=tuple(
                    (layer, head)
                    for layer in range(NUM_LAYERS)
                    for head in range(NUM_HEADS)
                ),
            )
        else:
            selected_distribution = None
            head_concentration = None
            layer_head_concentration = None
        thresholds.append(
            {
                "tau": tau,
                "selection_rule": "abs_z > tau",
                "selected_count": selected_total,
                "selected_fraction": selected_total / candidate_total,
                "prompt_selected_counts": selected_count_values,
                "prompt_selected_count": prompt_summary,
                "distance_of_mean_from_500": prompt_summary["mean"] - FIXED_TOP_N,
                "absolute_distance_of_mean_from_500": abs(
                    prompt_summary["mean"] - FIXED_TOP_N
                ),
                "selected_abs_z": selected_distribution,
                **counts,
                "head_concentration": head_concentration,
                "layer_head_concentration": layer_head_concentration,
            }
        )

    return {
        "schema_version": 1,
        "diagnostic_role": "selection20_original_leave_one_out_z_threshold_diagnostic",
        "scientific_conditions": {
            "dataset": "Selection20 original only",
            "loo_reference_prompt_count": SELECTION_COUNT - 1,
            "score_capture_position": (
                "post-RoPE, after repeat_kv, immediately after "
                "QK^T/sqrt(head_dim), pre-mask, pre-softmax"
            ),
            "position_scope": "full_prompt_strict_lower_i_gt_j",
            "component": "A_asym=(S-S^T)/2",
            "threshold_rule": "abs_z > tau",
            "taus": list(TAUS),
            "final_tau_selected": False,
            "jailbreak_data_used": False,
            "independent100_used": False,
            "autodan_used": False,
            "intervention_applied": False,
            "generation_or_asr_evaluation": False,
            "alpha_used": False,
        },
        "candidate_count": candidate_total,
        "comparison_baselines": comparison_baselines(baseline, inputs),
        "per_prompt": per_prompt,
        "thresholds": thresholds,
        "numerical_diagnostics": {
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
        },
    }


def main(argv=None):
    args = parse_args(argv)
    output_dir = validate_request(args)
    inputs = load_fixed_inputs(args.reference_artifact)
    baseline = load_selection_baseline(args.selection_diagnostics, inputs)
    preflight = preflight_summary(inputs, baseline, args)
    print(json.dumps(preflight, indent=2))
    if args.stage == "cpu-preflight":
        print("CPU preflight complete; no files created.")
        return

    provenance = runtime_provenance()
    model, tokenizer = load_eager_model(args.model)
    model_summary = llama_model_config_summary(model)
    diagnostics = run_threshold_diagnostic(model, tokenizer, inputs, baseline)
    diagnostics.update(
        {
            "created_at": utc_now(),
            "reference_artifact": str(inputs["reference_path"]),
            "reference_artifact_sha256": EXPECTED_REFERENCE_SHA256,
            "selection_diagnostics": str(baseline["path"]),
            "selection_diagnostics_sha256": EXPECTED_SELECTION_DIAGNOSTICS_SHA256,
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
            {"path": DIAGNOSTICS_FILE, "sha256": sha256_file(diagnostics_path)},
            {"path": RUN_MANIFEST_FILE, "sha256": sha256_file(run_manifest_path)},
        ],
    }
    artifact_manifest_path = output_dir / ARTIFACT_MANIFEST_FILE
    write_json(artifact_manifest_path, artifact_manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "diagnostics_sha256": sha256_file(diagnostics_path),
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
                "thresholds": [
                    {
                        "tau": row["tau"],
                        "selected_count": row["selected_count"],
                        "selected_fraction": row["selected_fraction"],
                        "prompt_selected_count": row["prompt_selected_count"],
                    }
                    for row in diagnostics["thresholds"]
                ],
                "numerical_diagnostics": diagnostics["numerical_diagnostics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
