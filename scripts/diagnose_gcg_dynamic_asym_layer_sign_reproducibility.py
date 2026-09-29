#!/usr/bin/env python3
"""Compare absolute Top-N sign-split Layer distributions across frozen GCG sets."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_gcg_dynamic_asym_selection import tensor_sha256  # noqa: E402
from scripts.evaluate_gcg_dynamic_asym_independent100 import fixed_selection  # noqa: E402
from scripts.evaluate_gcg_dynamic_asym_stage1 import (  # noqa: E402
    DEFAULT_ADVBENCH,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BASELINE_RESULTS,
    DEFAULT_HOLDOUT_RUN,
    DEFAULT_JAILBREAK_INPUT,
    DEFAULT_MODEL,
    DEFAULT_RAW_ARCHIVE,
    DEFAULT_REFERENCE,
    DEFAULT_SELECTION_RUN,
    load_development_inputs,
    validate_and_load_reference,
    validate_runtime_records,
)
from scripts.evaluate_gcg_first_token_reference_ablation import (  # noqa: E402
    load_model_and_tokenizer,
)
from utils.artifact_io import (  # noqa: E402
    ensure_output_directory_absent,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.experiment_runtime import git_state, git_value, seed_everything  # noqa: E402
from utils.gcg_dynamic_asym_selection_diagnostics import (  # noqa: E402
    patch_llama_abs_z_collection,
)
from utils.gcg_dynamic_asym_stage2 import (  # noqa: E402
    build_stage2_selection_plan,
    restrict_absolute_plan_to_intervention_sign,
)
from utils.gcg_wasserstein_reference import NUM_HEADS, NUM_LAYERS  # noqa: E402


SEED = 42
DEVELOPMENT_STAGE2_PLANS = Path(
    "data/outputs/gcg_dynamic_asym_stage2/llama2/development68/"
    "stage2_20260916T133246Z/selection_plans.jsonl"
)
EXPECTED_DEVELOPMENT_STAGE2_PLANS_SHA256 = (
    "10cc4e490bee2a77941368604b2f2eee042ad08097d5d9b633ea3992aba357ed"
)
INDEPENDENT_ARTIFACTS = {
    "positive": Path(
        "data/outputs/gcg_dynamic_asym_absolute_sign_ablation/llama2/gcg_ref/"
        "absolute_positive_only_0p1/evaluation_20260927T154921Z/"
        "prompt_results.jsonl"
    ),
    "negative": Path(
        "data/outputs/gcg_dynamic_asym_absolute_sign_ablation/llama2/gcg_ref/"
        "absolute_negative_only_0p1/evaluation_20260927T154921Z/"
        "prompt_results.jsonl"
    ),
}
EXPECTED_INDEPENDENT_SHA256 = {
    "positive": "92c6da4c5a8947fc2193b77ef7ef8cc97c54026d0947ae70d55cbc36aecb46f3",
    "negative": "7083aec43735c5619c1ddd6c83f4690ee5f7b6f9d89d392189c0b1241d4bebed",
}
SOURCE_PATHS = (
    "scripts/diagnose_gcg_dynamic_asym_layer_sign_reproducibility.py",
    "utils/gcg_dynamic_asym_stage2.py",
    "utils/gcg_dynamic_asym_selection_diagnostics.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu-preflight", "run"), default="cpu-preflight")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--selection-run", default=DEFAULT_SELECTION_RUN)
    parser.add_argument("--holdout-run", default=DEFAULT_HOLDOUT_RUN)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--raw-archive", default=DEFAULT_RAW_ARCHIVE)
    parser.add_argument("--advbench", default=DEFAULT_ADVBENCH)
    parser.add_argument("--output-dir")
    parser.add_argument("--sandbox-exempt-gpu", action="store_true")
    return parser.parse_args(argv)


def validate_request(args) -> Path | None:
    if args.stage == "cpu-preflight":
        if args.output_dir is not None or args.sandbox_exempt_gpu:
            raise ValueError("cpu-preflight cannot write output or claim GPU execution")
        return None
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    return ensure_output_directory_absent(Path(args.output_dir).resolve())


def mean_prompt_normalized_distribution(rows: list[dict], count_key: str) -> list[float]:
    distributions = []
    for row in rows:
        counts = row[count_key]
        if len(counts) != NUM_LAYERS or any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Layer counts are invalid")
        total = sum(counts)
        if total <= 0:
            raise ValueError("Layer counts must contain a selected element")
        distributions.append([value / total for value in counts])
    return [statistics.fmean(row[layer] for row in distributions) for layer in range(NUM_LAYERS)]


def pearson_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson inputs must have matching non-trivial lengths")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Pearson input is constant")
    return numerator / (left_norm * right_norm)


def distribution_comparison(left: list[float], right: list[float]) -> dict:
    if len(left) != NUM_LAYERS or len(right) != NUM_LAYERS:
        raise ValueError("Layer distributions must have 32 entries")
    midpoint = [(a + b) / 2 for a, b in zip(left, right)]

    def kl_divergence(values, reference):
        return sum(
            value * math.log2(value / target)
            for value, target in zip(values, reference)
            if value > 0
        )

    return {
        "pearson_layers_1_31": pearson_correlation(left[1:], right[1:]),
        "total_variation": sum(abs(a - b) for a, b in zip(left, right)) / 2,
        "jensen_shannon_bits": (
            kl_divergence(left, midpoint) + kl_divergence(right, midpoint)
        )
        / 2,
    }


def distribution_summary(distribution: list[float]) -> dict:
    ranking = sorted(range(NUM_LAYERS), key=lambda layer: (-distribution[layer], layer))
    return {
        "mean_fraction_by_layer": distribution,
        "top5_layers": [
            {"layer": layer, "fraction": distribution[layer]} for layer in ranking[:5]
        ],
        "layer_bands": {
            "layers_1_5": sum(distribution[1:6]),
            "layers_6_15": sum(distribution[6:16]),
            "layers_16_31": sum(distribution[16:32]),
        },
    }


def load_independent_rows() -> tuple[dict[str, list[dict]], dict]:
    loaded = {}
    hashes = {}
    for sign, relative_path in INDEPENDENT_ARTIFACTS.items():
        path = (REPO_ROOT / relative_path).resolve()
        observed = sha256_file(path)
        if observed != EXPECTED_INDEPENDENT_SHA256[sign]:
            raise ValueError(f"independent100 {sign} artifact SHA-256 mismatch")
        rows = read_jsonl(path)
        expected_target = sign
        if (
            len(rows) != 100
            or {int(row["dataset_index"]) for row in rows} != set(range(220, 320))
            or any(row.get("intervention_target_sign") != expected_target for row in rows)
        ):
            raise ValueError(f"independent100 {sign} artifact condition mismatch")
        loaded[sign] = [
            {
                "dataset_index": int(row["dataset_index"]),
                "prompt_id": row["prompt_id"],
                "layer_counts": row["diagnostics"]["planned_layer_counts"],
            }
            for row in rows
        ]
        hashes[sign] = observed
    return loaded, hashes


def load_expected_development_plan_hashes() -> dict[tuple[str, int], str]:
    path = (REPO_ROOT / DEVELOPMENT_STAGE2_PLANS).resolve()
    if sha256_file(path) != EXPECTED_DEVELOPMENT_STAGE2_PLANS_SHA256:
        raise ValueError("development68 Stage 2 selection plans SHA-256 mismatch")
    selection = fixed_selection(0.001)
    expected = {}
    for row in read_jsonl(path):
        if row["selection"] == selection and row["layer0_included"] is False:
            key = (row["dataset_group"], int(row["dataset_index"]))
            if key in expected:
                raise ValueError("duplicate development68 absolute selection plan")
            expected[key] = row["selected_indices_sha256"]
    if len(expected) != 68:
        raise ValueError("development68 absolute selection plan count mismatch")
    return expected


def preflight(args) -> tuple[dict, object, dict, dict]:
    inputs = load_development_inputs(args)
    reference_path, reference = validate_and_load_reference(args.reference)
    independent, independent_hashes = load_independent_rows()
    expected_plan_hashes = load_expected_development_plan_hashes()
    development_indices = {
        int(row["dataset_index"])
        for records in inputs["groups"].values()
        for row in records
    }
    independent_indices = {row["dataset_index"] for row in independent["positive"]}
    if development_indices & independent_indices:
        raise ValueError("development68 overlaps independent100 dataset indices")
    development_prompt_ids = {
        row["prompt_id"] for records in inputs["groups"].values() for row in records
    }
    independent_prompt_ids = {row["prompt_id"] for row in independent["positive"]}
    if development_prompt_ids & independent_prompt_ids:
        raise ValueError("development68 overlaps independent100 prompt IDs")
    payload = {
        "status": "cpu_preflight_passed",
        "diagnostic_role": "selection_structure_reproducibility_only",
        "development68_prompt_count": len(development_indices),
        "development68_groups": {name: len(rows) for name, rows in inputs["groups"].items()},
        "independent100_prompt_count": len(independent_indices),
        "dataset_index_overlap_count": 0,
        "prompt_id_overlap_count": 0,
        "reference": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "selection": fixed_selection(0.001),
        "selection_direction": "absolute_then_sign_partition",
        "layer0_included": False,
        "generation_used": False,
        "intervention_used": False,
        "independent_artifact_sha256": independent_hashes,
        "development_stage2_selection_plans_sha256": EXPECTED_DEVELOPMENT_STAGE2_PLANS_SHA256,
    }
    return inputs, reference, independent, {"payload": payload, "plan_hashes": expected_plan_hashes}


def run(args, output_dir: Path, inputs, reference, independent, audit) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA is required for prefill score reconstruction")
    seed_everything(SEED)
    model, tokenizer = load_model_and_tokenizer(Path(args.model).resolve())
    groups = validate_runtime_records(inputs["groups"], tokenizer)
    selection = fixed_selection(0.001)
    rows = []
    with patch_llama_abs_z_collection(
        model,
        layer_count=NUM_LAYERS,
        head_count=NUM_HEADS,
        component="antisymmetric",
    ) as collector:
        for group, records in groups.items():
            for position, record in enumerate(records, 1):
                collector.clear()
                collector.set_prompt_context(
                    prompt_length=record["prompt_length"],
                    excluded_selection_position=record["row_position"],
                    mu=reference.mu,
                    sigma=reference.sigma,
                )
                with torch.inference_mode():
                    model.model(
                        input_ids=record["input_ids"].to(model.device),
                        use_cache=False,
                        return_dict=True,
                    )
                signed_z, runtime = collector.result(signed=True)
                absolute = build_stage2_selection_plan(
                    signed_z.abs(),
                    signed_z=signed_z,
                    selection_direction="absolute",
                    selection=selection,
                    layer_count=NUM_LAYERS,
                    head_count=NUM_HEADS,
                    elements_per_head=runtime["strict_lower_elements_per_head"],
                    layer0_included=False,
                )
                key = (group, int(record["dataset_index"]))
                if tensor_sha256(absolute["selected_indices"]) != audit["plan_hashes"][key]:
                    raise RuntimeError("reconstructed absolute selection differs from Stage 2")
                subsets = {
                    sign: restrict_absolute_plan_to_intervention_sign(
                        absolute,
                        signed_z,
                        intervention_target_sign=sign,
                        layer_count=NUM_LAYERS,
                        head_count=NUM_HEADS,
                        elements_per_head=runtime["strict_lower_elements_per_head"],
                    )
                    for sign in ("positive", "negative")
                }
                if (
                    subsets["positive"]["selected_count"]
                    + subsets["negative"]["selected_count"]
                    != absolute["selected_count"]
                    or subsets["positive"]["layer_counts"][0] != 0
                    or subsets["negative"]["layer_counts"][0] != 0
                    or runtime["all_finite"] is not True
                ):
                    raise RuntimeError("sign-partition runtime invariant failed")
                rows.append(
                    {
                        "dataset_group": group,
                        "dataset_index": int(record["dataset_index"]),
                        "prompt_id": record["prompt_id"],
                        "prompt_length": record["prompt_length"],
                        "absolute_selected_count": absolute["selected_count"],
                        "positive_selected_count": subsets["positive"]["selected_count"],
                        "negative_selected_count": subsets["negative"]["selected_count"],
                        "positive_layer_counts": subsets["positive"]["layer_counts"],
                        "negative_layer_counts": subsets["negative"]["layer_counts"],
                        "absolute_selected_indices_sha256": tensor_sha256(
                            absolute["selected_indices"]
                        ),
                        "all_finite": True,
                    }
                )
                print(
                    f"[development68 {group} {position}/{len(records)}] "
                    f"index={record['dataset_index']} absolute={absolute['selected_count']} "
                    f"positive={subsets['positive']['selected_count']} "
                    f"negative={subsets['negative']['selected_count']}",
                    flush=True,
                )

    development_distributions = {
        sign: mean_prompt_normalized_distribution(rows, f"{sign}_layer_counts")
        for sign in ("positive", "negative")
    }
    independent_distributions = {
        sign: mean_prompt_normalized_distribution(independent[sign], "layer_counts")
        for sign in ("positive", "negative")
    }
    summary = {
        "schema_version": 1,
        **audit["payload"],
        "status": "complete",
        "development68": {
            sign: distribution_summary(development_distributions[sign])
            for sign in ("positive", "negative")
        },
        "independent100": {
            sign: distribution_summary(independent_distributions[sign])
            for sign in ("positive", "negative")
        },
        "development68_vs_independent100": {
            sign: distribution_comparison(
                development_distributions[sign], independent_distributions[sign]
            )
            for sign in ("positive", "negative")
        },
        "positive_vs_negative": {
            "development68": distribution_comparison(
                development_distributions["positive"],
                development_distributions["negative"],
            ),
            "independent100": distribution_comparison(
                independent_distributions["positive"],
                independent_distributions["negative"],
            ),
        },
    }
    ensure_output_directory_absent(output_dir)
    output_dir.mkdir(parents=True)
    rows_path = output_dir / "prompt_layer_counts.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(rows_path, rows)
    write_json(summary_path, summary)
    head, branch = git_state(REPO_ROOT)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": utc_now(),
        "git_head": head,
        "git_branch": branch,
        "dirty_status": git_value(REPO_ROOT, "status", "--porcelain").splitlines(),
        "sandbox_exempt_gpu": args.sandbox_exempt_gpu,
        "gpu": torch.cuda.get_device_name(0),
        "generation_used": False,
        "intervention_used": False,
        "source_code_sha256": {
            path: sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS
        },
        "files": {
            rows_path.name: sha256_file(rows_path),
            summary_path.name: sha256_file(summary_path),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "output_dir": str(output_dir),
        "summary_sha256": sha256_file(summary_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "summary": summary,
    }


def main(argv=None):
    args = parse_args(argv)
    output_dir = validate_request(args)
    inputs, reference, independent, audit = preflight(args)
    print(json.dumps(audit["payload"], ensure_ascii=False, indent=2))
    if args.stage == "cpu-preflight":
        return
    print(
        json.dumps(
            run(args, output_dir, inputs, reference, independent, audit),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
