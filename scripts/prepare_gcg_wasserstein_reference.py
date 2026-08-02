#!/usr/bin/env python3
"""Create the fixed GCG split, eager score, Wasserstein, and reference artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.gcg_wasserstein_reference import (  # noqa: E402
    CORE_ARTIFACT_FILES,
    NUM_HEADS,
    NUM_LAYERS,
    TAIL_TOKENS,
    build_split_manifest,
    composite_score_sha256,
    compute_head_wasserstein_statistics,
    compute_original_reference,
    ensure_output_directory_absent,
    extract_selection_scores,
    git_state,
    llama_model_config_summary,
    read_jsonl,
    save_torch_artifact,
    selection_records,
    sha256_file,
    strict_lower_indices,
    utc_now,
    validate_reference_payload,
    verify_artifact_directory,
    write_declared_artifact_manifest,
    write_head_ranking_csv,
    write_json,
    write_per_prompt_wasserstein_csv,
    write_split_artifacts,
)


DEFAULT_MODEL = (
    "/mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf"
)
DEFAULT_ORIGINAL = "data/derived/llama2/gcg/original_prompts.jsonl"
DEFAULT_JAILBREAK = "data/derived/llama2/gcg/jailbreak_prompts.jsonl"
DEFAULT_BASELINE = (
    "data/outputs/multidataset_asym_score_intervention/"
    "beta05_full_20260722_041303/llama_gcg/no_intervention/prompt_results.jsonl"
)
DEFAULT_BASELINE_MANIFEST = (
    "data/outputs/multidataset_asym_score_intervention/"
    "beta05_full_20260722_041303/llama_gcg/run_manifest.json"
)
SCORE_CAPTURE_POSITION = (
    "post-RoPE, after repeat_kv, immediately after QK^T/sqrt(head_dim), "
    "pre-mask, pre-softmax"
)
SCORE_CALCULATION_DTYPE = (
    "torch.float16 Q/K matmul and scaling; S_asym computed in torch.float16"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--original-input", default=DEFAULT_ORIGINAL)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE)
    parser.add_argument(
        "--baseline-run-manifest",
        default=DEFAULT_BASELINE_MANIFEST,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the fixed split without creating files or loading a model.",
    )
    return parser.parse_args(argv)


def print_split_summary(manifest: dict, *, dry_run: bool) -> None:
    payload = {
        "dry_run": dry_run,
        "selection_row_positions": manifest["selection_row_positions"],
        "selection_dataset_indices": manifest["selection_dataset_indices"],
        "selection_count": manifest["selection_count"],
        "evaluation_count": manifest["evaluation_count"],
        "baseline_success_count_all": manifest["baseline_success_count_all"],
        "baseline_success_count_selection": manifest[
            "baseline_success_count_selection"
        ],
        "baseline_success_count_evaluation": manifest[
            "baseline_success_count_evaluation"
        ],
        "baseline_asr_all": manifest["baseline_asr_all"],
        "baseline_asr_evaluation": manifest["baseline_asr_evaluation"],
        "sampling_algorithm": manifest["sampling_algorithm"],
        "git_head": manifest["git_head"],
        "git_branch": manifest["git_branch"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def load_eager_model(model_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for score extraction")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama import modeling_llama

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model.eval()
    model.config.use_cache = False
    if model.config.num_hidden_layers != NUM_LAYERS:
        raise ValueError("expected a 32-layer Llama model")
    if model.config.num_attention_heads != NUM_HEADS:
        raise ValueError("expected 32 Llama attention heads")
    if model.config.num_key_value_heads != NUM_HEADS:
        raise ValueError("expected 32 Llama key/value heads")
    for layer_index, layer in enumerate(model.model.layers):
        if type(layer.self_attn) is not modeling_llama.LlamaAttention:
            raise TypeError(
                f"expected eager LlamaAttention at layer={layer_index}, "
                f"found={type(layer.self_attn).__name__}"
            )
    return model, tokenizer


def main(argv=None):
    args = parse_args(argv)
    output_dir = ensure_output_directory_absent(args.output_dir)
    manifest = build_split_manifest(
        original_input=args.original_input,
        jailbreak_input=args.jailbreak_input,
        baseline_results=args.baseline_results,
        baseline_run_manifest=args.baseline_run_manifest,
        repo_root=REPO_ROOT,
    )
    print_split_summary(manifest, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry run complete; no files created: {output_dir}")
        return

    original_rows = read_jsonl(args.original_input)
    jailbreak_rows = read_jsonl(args.jailbreak_input)
    selected = selection_records(manifest)
    model, tokenizer = load_eager_model(args.model)
    model_config = llama_model_config_summary(model)
    original_values, jailbreak_values, sequence_lengths = extract_selection_scores(
        model=model,
        tokenizer=tokenizer,
        original_rows=original_rows,
        jailbreak_rows=jailbreak_rows,
        selected_records=selected,
        progress=print,
    )
    distances, ranking = compute_head_wasserstein_statistics(
        original_values,
        jailbreak_values,
    )
    reference = compute_original_reference(original_values)
    lower_indices = strict_lower_indices(TAIL_TOKENS, device="cpu")

    # Re-check immediately before the first write to avoid accidental reuse.
    ensure_output_directory_absent(output_dir)
    output_dir.mkdir(parents=True)
    write_split_artifacts(manifest, output_dir)
    split_sha256 = sha256_file(output_dir / "split_manifest.json")

    common_score_metadata = {
        "model_path": str(Path(args.model).resolve()),
        "model_config": model_config,
        "torch_dtype": "torch.float16",
        "attention_implementation": "eager",
        "batch_size": 1,
        "use_cache": False,
        "do_sample": False,
        "system_prompt": None,
        "chat_template": "tokenizer.apply_chat_template(user-only, tokenize=False)",
        "tokenize_add_special_tokens": False,
        "score_capture_position": SCORE_CAPTURE_POSITION,
        "score_calculation_dtype": SCORE_CALCULATION_DTYPE,
        "stored_value_dtype": "torch.float32",
        "tail_tokens": TAIL_TOKENS,
        "strict_lower_position_order": "torch.tril_indices(16, 16, offset=-1)",
        "selection_manifest_sha256": split_sha256,
        "selection_row_positions": manifest["selection_row_positions"],
        "selection_dataset_indices": manifest["selection_dataset_indices"],
    }
    save_torch_artifact(
        output_dir / "selection_original_scores.pt",
        {
            "condition": "original",
            "reference_name": "original reference",
            "values": original_values,
            "metadata": common_score_metadata,
        },
    )
    save_torch_artifact(
        output_dir / "selection_jailbreak_scores.pt",
        {
            "condition": "jailbreak",
            "values": jailbreak_values,
            "metadata": common_score_metadata,
        },
    )
    original_score_sha256 = sha256_file(
        output_dir / "selection_original_scores.pt"
    )
    jailbreak_score_sha256 = sha256_file(
        output_dir / "selection_jailbreak_scores.pt"
    )
    score_artifact_sha256 = composite_score_sha256(
        original_score_sha256,
        jailbreak_score_sha256,
    )
    score_manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        **common_score_metadata,
        "score_shape": list(original_values.shape),
        "original_score_file": "selection_original_scores.pt",
        "original_score_sha256": original_score_sha256,
        "jailbreak_score_file": "selection_jailbreak_scores.pt",
        "jailbreak_score_sha256": jailbreak_score_sha256,
        "score_artifact_sha256_definition": (
            "sha256(original_score_sha256 + ':' + jailbreak_score_sha256)"
        ),
        "score_artifact_sha256": score_artifact_sha256,
        "sequence_lengths": sequence_lengths,
    }
    write_json(output_dir / "score_extraction_manifest.json", score_manifest)

    save_torch_artifact(
        output_dir / "per_prompt_wasserstein.pt",
        {
            "values": distances,
            "axes": ["selection_prompt", "layer", "head"],
            "selection_row_positions": manifest["selection_row_positions"],
            "selection_dataset_indices": manifest["selection_dataset_indices"],
            "selection_manifest_sha256": split_sha256,
            "score_artifact_sha256": score_artifact_sha256,
            "method": (
                "equal-weight 1D Wasserstein: mean absolute difference "
                "between independently sorted 120-element vectors"
            ),
        },
    )
    write_per_prompt_wasserstein_csv(
        output_dir / "per_prompt_wasserstein.csv",
        distances,
        selected,
    )
    write_head_ranking_csv(
        output_dir / "head_wasserstein_ranking.csv",
        ranking,
    )
    top5 = {
        "selection_metric": "mean equal-weight 1D Wasserstein distance",
        "ranking_order": [
            "mean Wasserstein descending",
            "layer ascending",
            "head ascending",
        ],
        "selection_count": len(selected),
        "selection_manifest_sha256": split_sha256,
        "score_artifact_sha256": score_artifact_sha256,
        "heads": [dict(row) for row in ranking[:5]],
    }
    write_json(output_dir / "top5_heads.json", top5)

    reference_metadata = {
        "reference_name": "original reference",
        "alternative_name": "non-jailbreak reference",
        "tail_tokens": TAIL_TOKENS,
        "selection_row_positions": manifest["selection_row_positions"],
        "selection_dataset_indices": manifest["selection_dataset_indices"],
        "selection_manifest_sha256": split_sha256,
        "score_artifact_sha256": score_artifact_sha256,
        "model_path": str(Path(args.model).resolve()),
        "model_config": model_config,
        "torch_dtype": "torch.float16",
        "attention_implementation": "eager",
        "score_capture_position": SCORE_CAPTURE_POSITION,
        "score_calculation_dtype": SCORE_CALCULATION_DTYPE,
        "stored_value_dtype": "torch.float32",
        "std_definition": "sample standard deviation with correction=1",
        "sem_definition": "sample standard deviation / sqrt(20)",
    }
    reference_payload = {
        **reference,
        "tril_rows": lower_indices[0].to(torch.int64),
        "tril_cols": lower_indices[1].to(torch.int64),
        "metadata": reference_metadata,
    }
    validate_reference_payload(reference_payload)
    save_torch_artifact(
        output_dir / "original_reference.pt",
        reference_payload,
    )
    reference_summary = {
        "reference_name": "original reference",
        "alternative_name": "non-jailbreak reference",
        "mean_shape": list(reference["mean"].shape),
        "std_shape": list(reference["std"].shape),
        "sem_shape": list(reference["sem"].shape),
        "dtype": str(reference["mean"].dtype),
        "finite": all(
            bool(torch.isfinite(reference[name]).all().item())
            for name in ("mean", "std", "sem")
        ),
        "reference_mean_tensor_summary": {
            "mean": float(reference["mean"].mean().item()),
            "std": float(reference["mean"].std(correction=1).item()),
            "min": float(reference["mean"].min().item()),
            "max": float(reference["mean"].max().item()),
        },
        "selection_manifest_sha256": split_sha256,
        "score_artifact_sha256": score_artifact_sha256,
        "metadata": reference_metadata,
    }
    write_json(
        output_dir / "original_reference_summary.json",
        reference_summary,
    )

    git_head, git_branch = git_state(REPO_ROOT)
    write_declared_artifact_manifest(
        output_dir=output_dir,
        file_names=CORE_ARTIFACT_FILES,
        metadata={
            "experiment": "llama2_gcg_wasserstein_original_reference",
            "git_head": git_head,
            "git_branch": git_branch,
            "selection_manifest_sha256": split_sha256,
            "score_artifact_sha256": score_artifact_sha256,
        },
    )
    verification = verify_artifact_directory(output_dir)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    print(f"artifacts written: {output_dir}")


if __name__ == "__main__":
    main()
