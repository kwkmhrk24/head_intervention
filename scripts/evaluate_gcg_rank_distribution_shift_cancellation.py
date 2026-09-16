#!/usr/bin/env python3
"""CPU-validate or later evaluate Selection20 rank shift cancellation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers.models.llama import modeling_llama


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_gcg_reference_factorization import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BASELINE_RESULTS,
    DEFAULT_HOLDOUT_RUN,
    DEFAULT_JAILBREAK_INPUT,
    DEFAULT_MODEL,
    DEFAULT_SELECTION_RUN,
    GENERATION_SETTINGS,
    format_records,
    load_model_and_tokenizer,
)
from scripts.evaluate_gcg_position_element_topk_intervention import (  # noqa: E402
    finish_cuda_measurement,
    parse_cli_alpha,
    run_baseline,
    serialized_input_ids,
    start_cuda_measurement,
)
from scripts.evaluate_gcg_rank_element_topk_intervention import (  # noqa: E402
    run_condition,
    validate_prompt_runtime_invariants,
)
from scripts.prepare_gcg_rank_element_topk import load_source_scores  # noqa: E402
from utils.experiment_runtime import git_value, seed_everything  # noqa: E402
from utils.gcg_first_token_reference_ablation import (  # noqa: E402
    load_selection_development_records,
)
from utils.gcg_first_token_holdout import (  # noqa: E402
    EVALUATION_ROLE,
    EXCLUDED_AMBIGUOUS_INDICES,
    HOLDOUT_COUNT,
    HOLDOUT_INDICES,
    MANIFEST_RECORD_FIELDS,
    _records_sha256,
)
from utils.gcg_rank_distribution_shift_cancellation import (  # noqa: E402
    ALL_RANK_COUNT,
    ALPHA_DEFAULT,
    METHOD,
    RankDistributionShiftCancellationController,
    load_shift_artifact,
    offline_shift_crossing_diagnostics,
    patch_llama_shift_cancellation_attention,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    llama_model_config_summary,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.gcg_wasserstein_reference_evaluation import ensure_output_absent  # noqa: E402
from utils.reference_asym_score_intervention import SUPPORTED_TRANSFORMERS_VERSION  # noqa: E402


INTERVENTION_TIMING = "first_token_only"
DATASET_GROUPS = ("selection20", "holdout48")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu-validate", "run"), default="cpu-validate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--shift-artifact-dir", required=True)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--selection-run", default=DEFAULT_SELECTION_RUN)
    parser.add_argument("--holdout-run", default=DEFAULT_HOLDOUT_RUN)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--alpha", type=parse_cli_alpha, default=ALPHA_DEFAULT)
    parser.add_argument(
        "--dataset-groups", nargs="+", choices=DATASET_GROUPS,
        default=list(DATASET_GROUPS),
    )
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def validate_dataset_groups(groups) -> tuple[str, ...]:
    names = tuple(groups)
    if not names or len(names) != len(set(names)) or not set(names) <= set(DATASET_GROUPS):
        raise ValueError("dataset groups must be distinct Selection20/Holdout48 groups")
    return names


def load_holdout_development_records(holdout_run: Path) -> tuple[list[dict], dict]:
    """Use the fixed, audited 48-row manifest without reading AdvBench 220..319.

    The original validator reads and hashes the entire AdvBench file.  This
    evaluator verifies the pre-baseline manifest, its record hashes, the fixed
    included indices, and saved baseline alignment instead.  It never opens
    AdvBench or the raw GCG archive.
    """

    manifest_path = holdout_run / "holdout_manifest.json"
    manifest = read_json(manifest_path)
    audited_run = read_json(holdout_run / "run_manifest.json")
    required_flags = {
        "evaluation_role": EVALUATION_ROLE,
        "fully_unseen_across_all_prior_research": False,
        "unused_for_current_gcg_head_selection": True,
        "unused_for_current_reference_construction": True,
        "unused_for_current_gcg_evaluation": True,
        "original_behaviors_previously_used_in_autodan_research": True,
    }
    for field, expected in required_flags.items():
        if manifest.get(field) is not expected and manifest.get(field) != expected:
            raise ValueError(f"holdout manifest metadata mismatch: {field}")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 52:
        raise ValueError("holdout manifest must contain 52 fixed records")
    if manifest.get("records_sha256") != _records_sha256(records):
        raise ValueError("holdout manifest record hash mismatch")
    if (
        audited_run.get("status") != "complete"
        or audited_run.get("holdout_manifest_sha256_fixed_before_baseline")
        != sha256_file(manifest_path)
        or audited_run.get("holdout_manifest_records_sha256")
        != manifest["records_sha256"]
        or audited_run.get("included_dataset_indices") != list(HOLDOUT_INDICES)
        or audited_run.get("excluded_dataset_indices")
        != list(EXCLUDED_AMBIGUOUS_INDICES)
        or audited_run.get("included_count") != HOLDOUT_COUNT
        or audited_run.get("raw_artifact_sha256") != manifest.get("raw_artifact_sha256")
        or audited_run.get("advbench_sha256") != manifest.get("advbench_sha256")
    ):
        raise ValueError("saved Holdout48 audit provenance mismatch")
    runtime = []
    excluded = []
    for record in records:
        if not isinstance(record, dict) or set(MANIFEST_RECORD_FIELDS) - set(record):
            raise ValueError("holdout manifest record fields mismatch")
        index = record.get("dataset_index")
        if type(index) is not int or not 0 <= index < 50:
            raise ValueError("holdout record index outside fixed 0..49")
        if type(record.get("raw_file_position")) is not int or record["raw_file_position"] < 0:
            raise ValueError("holdout raw file position is invalid")
        if record.get("raw_archive_index") != index:
            raise ValueError("holdout raw archive index mismatch")
        original = record.get("original_prompt")
        prompt = record.get("GCG_prompt")
        if (
            not isinstance(original, str) or not original
            or not isinstance(prompt, str) or not prompt.startswith(original)
            or sha256_text(original) != record.get("original_prompt_sha256")
            or sha256_text(prompt) != record.get("GCG_prompt_sha256")
            or record.get("raw_artifact_sha256") != manifest["raw_artifact_sha256"]
        ):
            raise ValueError("holdout manifest prompt/hash mismatch")
        if record.get("included") is True:
            if index in EXCLUDED_AMBIGUOUS_INDICES or record.get("exclusion_reason") is not None:
                raise ValueError("holdout included/excluded index mismatch")
            runtime.append({
                "prompt_id": f"llama:gcg-holdout:{index}",
                "row_position": len(runtime),
                "dataset_index": index,
                "raw_file_position": record["raw_file_position"],
                "raw_archive_index": index,
                "prompt": prompt,
                "GCG_prompt_sha256": record["GCG_prompt_sha256"],
                "original_prompt_sha256": record["original_prompt_sha256"],
            })
        else:
            if index not in EXCLUDED_AMBIGUOUS_INDICES or not record.get("exclusion_reason"):
                raise ValueError("unexpected excluded holdout record")
            excluded.append(index)
    if (
        tuple(row["dataset_index"] for row in runtime) != HOLDOUT_INDICES
        or sorted(excluded) != [10, 10, 13, 13]
    ):
        raise ValueError("holdout manifest fixed 48-row split mismatch")
    baseline_path = holdout_run / "baseline/prompt_results.jsonl"
    saved_baselines = read_jsonl(baseline_path)
    by_index = {row.get("dataset_index"): row for row in saved_baselines}
    if len(saved_baselines) != HOLDOUT_COUNT or len(by_index) != HOLDOUT_COUNT:
        raise ValueError("saved Holdout48 baseline count/identity mismatch")
    for row in runtime:
        saved = by_index[row["dataset_index"]]
        if (
            saved.get("prompt_id") != row["prompt_id"]
            or saved.get("row_position") != row["row_position"]
            or saved.get("prompt") != row["prompt"]
            or saved.get("generation_failed") is not False
            or type(saved.get("success")) is not bool
            or not isinstance(saved.get("response_token_ids"), list)
        ):
            raise ValueError("saved Holdout48 baseline alignment mismatch")
        row.update(
            baseline_response=saved["response"],
            baseline_response_token_ids=saved["response_token_ids"],
            baseline_success=saved["success"],
            baseline_generation_failed=False,
            baseline_response_token_count=saved["response_token_count"],
        )
    return runtime, {
        "holdout_manifest_sha256": sha256_file(manifest_path),
        "holdout_records_sha256": manifest["records_sha256"],
        "saved_baseline_sha256": sha256_file(baseline_path),
        "holdout_prompt_count": HOLDOUT_COUNT,
        "advbench_or_raw_archive_opened": False,
    }


def load_development_groups(args, config: dict) -> tuple[dict[str, list[dict]], dict]:
    """Load only requested, audited development groups; D stays artifact-only."""

    requested = validate_dataset_groups(args.dataset_groups)
    groups = {}
    provenance = {}
    if "selection20" in requested:
        split_path = Path(args.artifact_dir).resolve() / "split_manifest.json"
        if sha256_file(split_path) != config["source_selection_manifest_sha256"]:
            raise ValueError("Selection20 split/source mismatch")
        selection, _ = load_selection_development_records(
            split_manifest_path=split_path,
            jailbreak_input=Path(args.jailbreak_input).resolve(),
            baseline_results=Path(args.baseline_results).resolve(),
        )
        saved_path = Path(args.selection_run).resolve() / "baseline_selection_results.jsonl"
        saved_rows = read_jsonl(saved_path)
        by_index = {row.get("dataset_index"): row for row in saved_rows}
        if len(selection) != SELECTION_COUNT or len(saved_rows) != SELECTION_COUNT or len(by_index) != SELECTION_COUNT:
            raise ValueError("Selection20 saved baseline count/identity mismatch")
        for row in selection:
            saved = by_index[row["dataset_index"]]
            if (
                saved.get("prompt") != row["prompt"]
                or saved.get("response") != row["baseline_response"]
                or saved.get("success") is not row["baseline_success"]
                or saved.get("generation_failed") is not False
            ):
                raise ValueError("Selection20 saved baseline alignment mismatch")
            row["baseline_response_token_ids"] = saved["response_token_ids"]
        groups["selection20"] = selection
        provenance["selection20"] = {
            "split_manifest_sha256": sha256_file(split_path),
            "saved_baseline_sha256": sha256_file(saved_path),
            "prompt_count": len(selection),
        }
    if "holdout48" in requested:
        holdout, provenance["holdout48"] = load_holdout_development_records(
            Path(args.holdout_run).resolve()
        )
        groups["holdout48"] = holdout
    return {name: groups[name] for name in requested}, provenance


def validate_shift_runtime(diagnostics: dict, *, k: int = ALL_RANK_COUNT) -> None:
    if diagnostics.get("alpha") == 0.0:
        if (
            diagnostics.get("selected_rank_intervention_count") != 0
            or diagnostics.get("intervention_call_count") != 0
            or diagnostics.get("selected_value_crossing_count") != 0
            or diagnostics.get("later_forward_intervention_count") != 0
            or diagnostics.get("alpha_zero_bypass_count")
            != diagnostics.get("attention_forward_call_count")
            or diagnostics.get("alpha_zero_bypass_count", 0) < NUM_LAYERS
        ):
            raise RuntimeError("alpha=0 runtime must be a complete no-op")
        return
    validate_prompt_runtime_invariants(diagnostics, k=k)


def cpu_runtime_smoke(mean_shift: torch.Tensor, *, alpha: float) -> dict:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    controller = RankDistributionShiftCancellationController(
        mean_shift=mean_shift, alpha=alpha, intervention_timing=INTERVENTION_TIMING
    )
    controller.set_prompt_context(prompt_length=20)
    for layer in range(NUM_LAYERS):
        scores = torch.randn((1, NUM_HEADS, 20, 20), generator=generator)
        before = scores.clone()
        result = controller.apply(layer, scores)
        if alpha == 0.0:
            if result is not scores or not torch.equal(result, before):
                raise RuntimeError("alpha=0 CPU smoke changed raw scores")
        elif result is scores or torch.equal(result, before):
            raise RuntimeError(f"CPU smoke did not modify layer {layer}")
    later = torch.randn((1, NUM_HEADS, 21, 21), generator=generator)
    if controller.apply(0, later) is not later:
        raise RuntimeError("CPU smoke intervened after the first token")
    diagnostics = controller.summary()
    if alpha == 0.0:
        if diagnostics["alpha_zero_bypass_count"] != NUM_LAYERS + 1:
            raise RuntimeError("alpha=0 CPU smoke bypass count mismatch")
    else:
        validate_shift_runtime(diagnostics)
    return diagnostics


def load_validated_shift(args):
    shift_dir = Path(args.shift_artifact_dir).resolve()
    source_dir = Path(args.artifact_dir).resolve()
    model_path = Path(args.model).resolve()
    mean_shift, config, verification = load_shift_artifact(shift_dir)
    if Path(config.get("model", "")).resolve() != model_path:
        raise ValueError("shift artifact model mismatch")
    if Path(config.get("source_artifact_dir", "")).resolve() != source_dir:
        raise ValueError("shift artifact source directory mismatch")
    original, gcg, metadata, source_verification, paths = load_source_scores(
        source_dir, model_path=model_path
    )
    for field, expected in (
        ("source_selection_manifest_sha256", source_verification["selection_manifest_sha256"]),
        ("source_score_artifact_sha256", source_verification["score_artifact_sha256"]),
        ("original_score_sha256", sha256_file(paths["original"])),
        ("jailbreak_score_sha256", sha256_file(paths["jailbreak"])),
    ):
        if config.get(field) != expected:
            raise ValueError(f"shift/source provenance mismatch: {field}")
    if config.get("model_config") != metadata["model_config"]:
        raise ValueError("shift/source model config mismatch")
    return mean_shift, config, verification, source_verification, original, gcg


def cpu_validation_payload(args) -> dict:
    requested = validate_dataset_groups(args.dataset_groups)
    mean_shift, config, verification, source_verification, original, gcg = (
        load_validated_shift(args)
    )
    groups, split_provenance = load_development_groups(args, config)
    smoke = cpu_runtime_smoke(mean_shift, alpha=args.alpha)
    return {
        "status": "cpu_validation_passed",
        "method": METHOD,
        "alpha": args.alpha,
        "model_loaded": False,
        "gpu_used": False,
        "artifacts_written": False,
        "dataset_groups": list(requested),
        "development_prompt_counts": {name: len(rows) for name, rows in groups.items()},
        "development_split_provenance": split_provenance,
        "independent100_used": False,
        "holdout48_loaded_for_preflight": "holdout48" in requested,
        "holdout48_used_for_shift": False,
        "gpu_evaluation_performed": False,
        "shift_shape": list(mean_shift.shape),
        "shift_artifact_verification": verification,
        "source_artifact_verified": source_verification["verified"],
        "scientific_conditions": config["scientific_conditions"],
        "offline_selection20_crossing_diagnostics": {
            "original": offline_shift_crossing_diagnostics(original, mean_shift, alpha=args.alpha),
            "gcg": offline_shift_crossing_diagnostics(gcg, mean_shift, alpha=args.alpha),
        },
        "cpu_runtime_smoke": smoke,
    }


def validate_loaded_model(model, config: dict) -> dict:
    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError("Transformers version mismatch")
    summary = llama_model_config_summary(model)
    if summary != config["model_config"]:
        raise ValueError("loaded model config does not match shift source")
    if len(model.model.layers) != NUM_LAYERS or model.config._attn_implementation != "eager":
        raise ValueError("loaded model layer count/eager attention mismatch")
    if model.config.use_cache is not False or next(model.parameters()).dtype != torch.float16:
        raise ValueError("loaded model cache/dtype mismatch")
    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        if (
            type(attention) is not modeling_llama.LlamaAttention
            or attention.layer_idx != layer_index
            or attention.num_heads != NUM_HEADS
        ):
            raise ValueError(f"loaded model attention mismatch at layer {layer_index}")
    return {"transformers_version": transformers.__version__, "model_config": summary,
            "attention_implementation": "eager", "model_parameter_dtype": "torch.float16"}


def _shift_patcher(model, *, selection, k, alpha, intervention_timing):
    if k != ALL_RANK_COUNT or intervention_timing != INTERVENTION_TIMING:
        raise ValueError("shift cancellation requires all ranks and first_token_only")
    return patch_llama_shift_cancellation_attention(model, mean_shift=selection, alpha=alpha)


def write_group_results(
    output_dir: Path, group_name: str, baseline: list[dict], rows: list[dict],
    summary: dict, baseline_runtime: dict,
) -> tuple[dict, list[Path]]:
    """Save baseline and intervention metrics under each development group."""

    if group_name not in DATASET_GROUPS or not baseline or len(rows) != len(baseline):
        raise ValueError("development group result count mismatch")
    if summary.get("dataset_group") != group_name or summary.get("prompt_count") != len(rows):
        raise ValueError("development group summary mismatch")
    if summary.get("input_ids_match_count") != len(rows):
        raise ValueError("baseline/intervention input ID match count mismatch")
    expected_interventions = (
        0 if summary.get("alpha") == 0.0 else len(rows) * ALL_RANK_COUNT
    )
    if summary.get("selected_rank_intervention_count") != expected_interventions:
        raise ValueError("all-rank intervention count mismatch")
    for row in rows:
        if (
            row.get("dataset_group") != group_name
            or row.get("input_ids_match_baseline") is not True
            or not isinstance(row.get("first_token_logit_metrics"), dict)
            or not isinstance(row.get("diagnostics"), dict)
            or row.get("transition") not in (
                "newly_defended", "still_attack_success", "newly_worsened", "still_defended"
            )
        ):
            raise ValueError("development prompt result lacks required metrics")
    attack_success = sum(bool(row["success"]) for row in baseline)
    baseline_summary = {
        "dataset_group": group_name,
        "prompt_count": len(baseline),
        "attack_success_count": attack_success,
        "asr": attack_success / len(baseline),
        "generation_failure_count": 0,
        "input_ids_saved_count": len(baseline),
        **baseline_runtime,
    }
    summary = dict(summary, runtime_invariants_passed=True)
    paths = [
        output_dir / f"{group_name}_baseline.jsonl",
        output_dir / f"{group_name}_baseline_summary.json",
        output_dir / f"{group_name}_prompt_results.jsonl",
        output_dir / f"{group_name}_summary.json",
    ]
    write_jsonl(paths[0], [
        {key: serialized_input_ids(value) if key == "input_ids" else value
         for key, value in row.items() if key != "first_logits"}
        for row in baseline
    ])
    write_json(paths[1], baseline_summary)
    write_jsonl(paths[2], rows)
    write_json(paths[3], summary)
    return baseline_summary, paths


def run_gpu(args) -> dict:
    requested = validate_dataset_groups(args.dataset_groups)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --stage run")
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    output_dir = ensure_output_absent(args.output_dir).resolve()
    mean_shift, config, verification, source_verification, _, _ = load_validated_shift(args)
    groups, split_provenance = load_development_groups(args, config)
    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(Path(args.model).resolve())
    model_validation = validate_loaded_model(model, config)
    groups = {name: format_records(records, tokenizer) for name, records in groups.items()}
    output_dir.mkdir(parents=True)
    condition = {
        "name": "rank_shift_cancellation_v1",
        "method": METHOD,
        "k": ALL_RANK_COUNT,
        "alpha": args.alpha,
    }
    baseline_summaries = []
    condition_summaries = []
    paths = []
    for group_name, records in groups.items():
        baseline_started = start_cuda_measurement()
        baseline = run_baseline(
            records, model=model, tokenizer=tokenizer, label=f"{group_name} baseline"
        )
        baseline_runtime = finish_cuda_measurement(baseline_started)
        baseline_by_index = {row["dataset_index"]: row for row in baseline}
        condition_started = start_cuda_measurement()
        rows, summary = run_condition(
            records, baseline_by_index, model=model, tokenizer=tokenizer,
            selection=mean_shift, condition=condition, label=group_name,
            patcher=_shift_patcher, runtime_validator=validate_shift_runtime,
        )
        summary.update(finish_cuda_measurement(condition_started))
        baseline_summary, group_paths = write_group_results(
            output_dir, group_name, baseline, rows, summary, baseline_runtime
        )
        baseline_summaries.append(baseline_summary)
        condition_summaries.append(dict(summary, runtime_invariants_passed=True))
        paths.extend(group_paths)
    result = {
        "status": "complete",
        "dataset_groups": list(groups),
        "baselines": baseline_summaries,
        "conditions": condition_summaries,
    }
    evaluation_summary_path = output_dir / "evaluation_summary.json"
    write_json(evaluation_summary_path, result)
    paths.append(evaluation_summary_path)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": utc_now(),
        "execution_git_head": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "execution_git_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
        "shift_implementation_source_sha256": sha256_file(
            REPO_ROOT / "utils/gcg_rank_distribution_shift_cancellation.py"
        ),
        "rank_runtime_source_sha256": sha256_file(
            REPO_ROOT / "utils/gcg_rank_element_topk_intervention.py"
        ),
        "model": str(Path(args.model).resolve()),
        "model_validation": model_validation,
        "gpu": torch.cuda.get_device_name(0),
        "shift_artifact_verification": verification,
        "source_artifact_verification": source_verification,
        "selection_manifest_sha256": config["source_selection_manifest_sha256"],
        "shift_source": "Selection20 mean_shift artifact only",
        "holdout48_used_for_shift": False,
        "development_split_provenance": split_provenance,
        "dataset_groups": list(groups),
        "development_prompt_counts": {name: len(rows) for name, rows in groups.items()},
        "independent100_used": False,
        "holdout48_used": "holdout48" in groups,
        "condition": condition,
        "intervention_timing": INTERVENTION_TIMING,
        "generation_settings": GENERATION_SETTINGS,
    }
    run_manifest_path = output_dir / "run_manifest.json"
    write_json(run_manifest_path, manifest)
    paths.append(run_manifest_path)
    write_json(output_dir / "artifact_manifest.json", {
        "schema_version": 1,
        "created_at": utc_now(),
        "artifacts": [
            {"relative_path": str(path.relative_to(output_dir)), "sha256": sha256_file(path)}
            for path in paths
        ],
    })
    return dict(result, output_dir=str(output_dir))


def main(argv=None):
    args = parse_args(argv)
    result = cpu_validation_payload(args) if args.stage == "cpu-validate" else run_gpu(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
