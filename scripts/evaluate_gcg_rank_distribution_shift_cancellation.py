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
    DEFAULT_JAILBREAK_INPUT,
    DEFAULT_MODEL,
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
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.gcg_wasserstein_reference_evaluation import ensure_output_absent  # noqa: E402
from utils.reference_asym_score_intervention import SUPPORTED_TRANSFORMERS_VERSION  # noqa: E402


INTERVENTION_TIMING = "first_token_only"
DATASET_GROUPS = ("selection20",)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu-validate", "run"), default="cpu-validate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--shift-artifact-dir", required=True)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--alpha", type=parse_cli_alpha, default=ALPHA_DEFAULT)
    parser.add_argument(
        "--dataset-groups", nargs="+", choices=DATASET_GROUPS,
        default=list(DATASET_GROUPS),
    )
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


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
    if tuple(args.dataset_groups) != DATASET_GROUPS:
        raise ValueError("development entrypoint permits Selection20 only")
    mean_shift, config, verification, source_verification, original, gcg = (
        load_validated_shift(args)
    )
    smoke = cpu_runtime_smoke(mean_shift, alpha=args.alpha)
    return {
        "status": "cpu_validation_passed",
        "method": METHOD,
        "alpha": args.alpha,
        "model_loaded": False,
        "gpu_used": False,
        "artifacts_written": False,
        "dataset_groups": list(DATASET_GROUPS),
        "independent100_used": False,
        "holdout48_used": False,
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


def run_gpu(args) -> dict:
    if tuple(args.dataset_groups) != DATASET_GROUPS:
        raise ValueError("development entrypoint permits Selection20 only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --stage run")
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    output_dir = ensure_output_absent(args.output_dir).resolve()
    mean_shift, config, verification, source_verification, _, _ = load_validated_shift(args)
    records, split_manifest = load_selection_development_records(
        split_manifest_path=Path(args.artifact_dir).resolve() / "split_manifest.json",
        jailbreak_input=Path(args.jailbreak_input).resolve(),
        baseline_results=Path(args.baseline_results).resolve(),
    )
    if len(records) != SELECTION_COUNT or sha256_file(
        Path(args.artifact_dir).resolve() / "split_manifest.json"
    ) != config["source_selection_manifest_sha256"]:
        raise ValueError("Selection20 split/source mismatch")
    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(Path(args.model).resolve())
    model_validation = validate_loaded_model(model, config)
    records = format_records(records, tokenizer)
    output_dir.mkdir(parents=True)
    condition = {
        "name": "rank_shift_cancellation_v1",
        "method": METHOD,
        "k": ALL_RANK_COUNT,
        "alpha": args.alpha,
    }
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
        "selection_prompt_count": len(records),
        "dataset_groups": list(DATASET_GROUPS),
        "independent100_used": False,
        "holdout48_used": False,
        "condition": condition,
        "intervention_timing": INTERVENTION_TIMING,
        "generation_settings": GENERATION_SETTINGS,
    }
    baseline_started = start_cuda_measurement()
    baseline = run_baseline(records, model=model, tokenizer=tokenizer, label="selection20 baseline")
    baseline_runtime = finish_cuda_measurement(baseline_started)
    baseline_by_index = {row["dataset_index"]: row for row in baseline}
    condition_started = start_cuda_measurement()
    rows, summary = run_condition(
        records, baseline_by_index, model=model, tokenizer=tokenizer,
        selection=mean_shift, condition=condition, label="selection20",
        patcher=_shift_patcher, runtime_validator=validate_shift_runtime,
    )
    summary.update(finish_cuda_measurement(condition_started))
    write_jsonl(output_dir / "selection20_baseline.jsonl", [
        {key: serialized_input_ids(value) if key == "input_ids" else value
         for key, value in row.items() if key != "first_logits"}
        for row in baseline
    ])
    write_json(output_dir / "selection20_baseline_summary.json", {
        "prompt_count": len(baseline), "attack_success_count": sum(row["success"] for row in baseline),
        "generation_failure_count": 0, **baseline_runtime,
    })
    write_jsonl(output_dir / "selection20_prompt_results.jsonl", rows)
    write_json(output_dir / "selection20_summary.json", summary)
    write_json(output_dir / "run_manifest.json", manifest)
    paths = (
        "selection20_baseline.jsonl", "selection20_baseline_summary.json",
        "selection20_prompt_results.jsonl", "selection20_summary.json", "run_manifest.json",
    )
    write_json(output_dir / "artifact_manifest.json", {
        "schema_version": 1,
        "created_at": utc_now(),
        "artifacts": [
            {"relative_path": name, "sha256": sha256_file(output_dir / name)}
            for name in paths
        ],
    })
    return {"status": "complete", "dataset_group": "selection20", "summary": summary,
            "output_dir": str(output_dir)}


def main(argv=None):
    args = parse_args(argv)
    result = cpu_validation_payload(args) if args.stage == "cpu-validate" else run_gpu(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
