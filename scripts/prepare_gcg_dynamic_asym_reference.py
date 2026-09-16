#!/usr/bin/env python3
"""Build Selection20-original full-prompt moments for dynamic Asym defense.

The default invocation is a read-only CPU provenance preflight.  Actual score
collection is explicit via ``--stage build-reference`` and requires an eager
Llama model on CUDA.  Existing output directories are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.artifact_io import (  # noqa: E402
    ensure_output_directory_absent,
    sha256_file,
    utc_now,
    write_json,
)
from utils.experiment_runtime import git_state, git_value  # noqa: E402
from utils.gcg_dynamic_asym_intervention import (  # noqa: E402
    REFERENCE_SCHEMA_VERSION,
    compute_equal_prompt_reference_statistics,
    load_selection20_original_prompts,
    patch_llama_full_prompt_moment_collection,
    validate_dynamic_reference_payload,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    llama_model_config_summary,
    save_torch_artifact,
)
from utils.prompt_utils import apply_template  # noqa: E402


DEFAULT_MODEL = "/mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf"
DEFAULT_ORIGINAL_INPUT = "data/derived/llama2/gcg/original_prompts.jsonl"
DEFAULT_SELECTION_MANIFEST = (
    "data/outputs/gcg_wasserstein_reference/llama2/seed42_selection20/"
    "codex_gpu_20260802T061114Z/split_manifest.json"
)
REFERENCE_FILE = "selection20_original_full_prompt_asym_reference.pt"
MANIFEST_FILE = "reference_manifest.json"
SCORE_CAPTURE_POSITION = (
    "post-RoPE, after repeat_kv, immediately after QK^T/sqrt(head_dim), "
    "pre-mask, pre-softmax"
)
SOURCE_CODE_PATHS = (
    "scripts/prepare_gcg_dynamic_asym_reference.py",
    "utils/gcg_dynamic_asym_intervention.py",
    "utils/reference_asym_score_intervention.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("cpu-preflight", "build-reference"),
        default="cpu-preflight",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--original-input", default=DEFAULT_ORIGINAL_INPUT)
    parser.add_argument("--selection-manifest", default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument(
        "--output-dir",
        help="Required only for --stage build-reference; must not already exist.",
    )
    return parser.parse_args(argv)


def validate_request(args) -> Path | None:
    if args.stage == "cpu-preflight":
        if args.output_dir is not None:
            raise ValueError("cpu-preflight does not write an output directory")
        return None
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage build-reference")
    return ensure_output_directory_absent(Path(args.output_dir).resolve())


def load_eager_model(model_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the full-prompt reference")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama import modeling_llama

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model.eval()
    model.config.use_cache = False
    if model.config.num_hidden_layers != NUM_LAYERS:
        raise ValueError(f"expected {NUM_LAYERS} model layers")
    if model.config.num_attention_heads != NUM_HEADS:
        raise ValueError(f"expected {NUM_HEADS} attention heads")
    if model.config.num_key_value_heads != NUM_HEADS:
        raise ValueError(f"expected {NUM_HEADS} key/value heads")
    for layer_index, layer in enumerate(model.model.layers):
        if type(layer.self_attn) is not modeling_llama.LlamaAttention:
            raise TypeError(
                f"expected eager LlamaAttention at layer={layer_index}, "
                f"found={type(layer.self_attn).__name__}"
            )
    return model, tokenizer


def tokenize_original_prompt(tokenizer, prompt: str) -> torch.Tensor:
    formatted = apply_template(
        [prompt],
        system_prompt=None,
        tokenizer=tokenizer,
        model_config={"model_name": "Llama"},
    )[0]
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=False,
    )
    if inputs.input_ids.ndim != 2 or inputs.input_ids.shape[0] != 1:
        raise ValueError("reference extraction requires one unpadded prompt")
    if inputs.input_ids.dtype != torch.long:
        raise ValueError("reference extraction input IDs must be torch.long")
    if inputs.input_ids.shape[-1] < 2:
        raise ValueError("reference extraction prompt must contain at least two tokens")
    return inputs.input_ids


def collect_reference_moments(model, tokenizer, records):
    prompt_means = torch.empty(
        (SELECTION_COUNT, NUM_LAYERS, NUM_HEADS), dtype=torch.float64
    )
    prompt_mean_squares = torch.empty_like(prompt_means)
    prompt_records = []
    with patch_llama_full_prompt_moment_collection(model) as collector:
        for position, record in enumerate(records):
            input_ids = tokenize_original_prompt(tokenizer, record["prompt"])
            prompt_length = int(input_ids.shape[-1])
            collector.clear()
            collector.set_prompt_context(prompt_length=prompt_length)
            with torch.inference_mode():
                model.model(
                    input_ids=input_ids.to(model.device),
                    use_cache=False,
                    return_dict=True,
                )
            means, mean_squares, element_count = collector.result()
            prompt_means[position].copy_(means)
            prompt_mean_squares[position].copy_(mean_squares)
            prompt_records.append(
                {
                    "selection_position": position,
                    "row_position": record["row_position"],
                    "dataset_index": record["dataset_index"],
                    "prompt_id": record["prompt_id"],
                    "original_prompt_sha256": record["original_prompt_sha256"],
                    "prompt_length": prompt_length,
                    "strict_lower_element_count_per_head": element_count,
                }
            )
            print(
                f"[{position + 1}/{SELECTION_COUNT}] "
                f"dataset_index={record['dataset_index']} length={prompt_length}",
                flush=True,
            )
    return prompt_means, prompt_mean_squares, prompt_records


def preflight_summary(records, selection_manifest, args) -> dict:
    return {
        "stage": args.stage,
        "reference_condition": "selection20_original_only",
        "selection_count": len(records),
        "selection_row_positions": [row["row_position"] for row in records],
        "selection_dataset_indices": [row["dataset_index"] for row in records],
        "original_prompt_hashes_verified": True,
        "original_input": str(Path(args.original_input).resolve()),
        "original_input_sha256": selection_manifest["original_input_sha256"],
        "selection_manifest": str(Path(args.selection_manifest).resolve()),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "position_scope": "full_prompt_strict_lower_i_gt_j",
        "prompt_weighting": "equal_weight_per_prompt",
        "jailbreak_reference_data_used": False,
        "tau": None,
    }


def runtime_provenance(repo_root: str | Path = REPO_ROOT) -> dict:
    """Capture Git state and exact source bytes used by a reference run."""

    repo_root = Path(repo_root).resolve()
    git_head, git_branch = git_state(repo_root)
    return {
        "git_head": git_head,
        "git_branch": git_branch,
        "dirty_status": git_value(repo_root, "status", "--porcelain").splitlines(),
        "source_code_sha256": {
            relative_path: sha256_file(repo_root / relative_path)
            for relative_path in SOURCE_CODE_PATHS
        },
    }


def main(argv=None):
    args = parse_args(argv)
    output_dir = validate_request(args)
    records, selection_manifest = load_selection20_original_prompts(
        split_manifest_path=args.selection_manifest,
        original_input_path=args.original_input,
    )
    summary = preflight_summary(records, selection_manifest, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.stage == "cpu-preflight":
        print("CPU provenance preflight complete; no files created.")
        return

    provenance = runtime_provenance()
    model, tokenizer = load_eager_model(args.model)
    model_summary = llama_model_config_summary(model)
    prompt_means, prompt_mean_squares, prompt_records = collect_reference_moments(
        model, tokenizer, records
    )
    statistics = compute_equal_prompt_reference_statistics(
        prompt_means,
        prompt_mean_squares,
    )
    metadata = {
        "reference_condition": "selection20_original_only",
        "prompt_weighting": "equal_weight_per_prompt",
        "position_scope": "full_prompt_strict_lower_i_gt_j",
        "component": "A_asym=(S-S^T)/2",
        "prompt_count": SELECTION_COUNT,
        "mu_definition": "mean_p(mean_strict_lower(A_asym_p))",
        "mean_q_definition": "mean_p(mean_strict_lower(A_asym_p^2))",
        "variance_definition": "mean_q-mu^2",
        "sigma_definition": "sqrt(max(variance,0))",
        "negative_variance_policy": (
            "clamp only within 64*float64_epsilon*max(1,abs(mean_q),mu^2); "
            "fail otherwise"
        ),
        "runtime_sigma_policy": (
            "preserve stored sigma; denominator-only floor at sqrt(float32.tiny)"
        ),
        "score_capture_position": SCORE_CAPTURE_POSITION,
        "score_component_calculation_dtype": "float32",
        "moment_reduction_dtype": "float64",
        "stored_statistic_dtype": "float32",
        "attention_implementation": "eager",
        "model_path": str(Path(args.model).resolve()),
        "model_config": model_summary,
        "selection_manifest": str(Path(args.selection_manifest).resolve()),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "original_input": str(Path(args.original_input).resolve()),
        "original_input_sha256": selection_manifest["original_input_sha256"],
        "selection_row_positions": selection_manifest["selection_row_positions"],
        "selection_dataset_indices": selection_manifest["selection_dataset_indices"],
        "jailbreak_reference_data_used": False,
        "git_head": provenance["git_head"],
        "git_branch": provenance["git_branch"],
        "created_at": utc_now(),
    }
    payload = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "mu": statistics["mu"],
        "sigma": statistics["sigma"],
        "variance": statistics["variance"],
        "mean_q": statistics["mean_q"],
        "prompt_means": statistics["prompt_means"],
        "prompt_mean_squares": statistics["prompt_mean_squares"],
        "metadata": metadata,
    }
    validate_dynamic_reference_payload(payload)

    # Re-check immediately before the first write.  Existing experiment
    # directories and artifacts remain immutable.
    ensure_output_directory_absent(output_dir)
    output_dir.mkdir(parents=True)
    reference_path = output_dir / REFERENCE_FILE
    save_torch_artifact(reference_path, payload)
    manifest_payload = {
        **summary,
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "created_at": metadata["created_at"],
        "git_head": provenance["git_head"],
        "git_branch": provenance["git_branch"],
        "dirty_status": provenance["dirty_status"],
        "source_code_sha256": provenance["source_code_sha256"],
        "model_path": metadata["model_path"],
        "model_config": model_summary,
        "reference_file": REFERENCE_FILE,
        "reference_sha256": sha256_file(reference_path),
        "mu_shape": list(statistics["mu"].shape),
        "sigma_shape": list(statistics["sigma"].shape),
        "clamped_negative_variance_count": statistics[
            "clamped_negative_variance_count"
        ],
        "sigma_floor_head_count": statistics["sigma_floor_head_count"],
        "prompt_records": prompt_records,
        "tau": None,
    }
    write_json(output_dir / MANIFEST_FILE, manifest_payload)
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))
    print(f"reference artifacts written: {output_dir}")


if __name__ == "__main__":
    main()
