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
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    utc_now,
    write_json,
)
from utils.experiment_runtime import git_state, git_value, seed_everything  # noqa: E402
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
from scripts.evaluate_gcg_position_element_topk_intervention import (  # noqa: E402
    format_autodan_records,
)
from scripts.run_autodan_position_element_selection_ablation import (  # noqa: E402
    EXPECTED_SELECTION_INDICES as AUTODAN_SELECTION_INDICES,
    OFFICIAL_FORMAT as AUTODAN_OFFICIAL_FORMAT,
    parse_args as parse_autodan_args,
    validate_sources as validate_autodan_sources,
    model_repository_head,
    MODEL_REPOSITORY_HEAD,
)


DEFAULT_MODEL = "/mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf"
DEFAULT_ORIGINAL_INPUT = "data/derived/llama2/gcg/original_prompts.jsonl"
DEFAULT_SELECTION_MANIFEST = (
    "data/outputs/gcg_wasserstein_reference/llama2/seed42_selection20/"
    "codex_gpu_20260802T061114Z/split_manifest.json"
)
REFERENCE_FILE = "selection20_original_full_prompt_asym_reference.pt"
AUTODAN_REFERENCE_FILE = "autodan_selection20_official_full_prompt_asym_reference.pt"
AUTODAN_ORIGINAL_REFERENCE_FILE = "autodan_selection20_original_full_prompt_asym_reference.pt"
DEFAULT_AUTODAN_ORIGINAL_INPUT = "data/derived/llama2/autodan/original_prompts.jsonl"
AUTODAN_ORIGINAL_CONDITION = "autodan_selection20_corresponding_originals_only"
GCG_SYM_REFERENCE_FILE = "selection20_original_full_prompt_sym_reference.pt"
AUTODAN_SYM_REFERENCE_FILE = "autodan_selection20_official_full_prompt_sym_reference.pt"
MANIFEST_FILE = "reference_manifest.json"
DEFAULT_AUTODAN_SELECTION_MANIFEST = (
    "data/outputs/gcg_position_element_topk/autodan_selection20/"
    "official_fastchat_34062e9_20260914T163205Z/selection_manifest.json"
)
EXPECTED_AUTODAN_SELECTION_MANIFEST_SHA256 = (
    "494c6beba4f6367ff353c0716ae986734b7bcbdeb823a342fc78ba8fb9741318"
)
SCORE_CAPTURE_POSITION = (
    "post-RoPE, after repeat_kv, immediately after QK^T/sqrt(head_dim), "
    "pre-mask, pre-softmax"
)
SOURCE_CODE_PATHS = (
    "scripts/prepare_gcg_dynamic_asym_reference.py",
    "scripts/evaluate_gcg_position_element_topk_intervention.py",
    "scripts/run_autodan_position_element_selection_ablation.py",
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
    parser.add_argument(
        "--component",
        choices=("antisymmetric", "symmetric"),
        default="antisymmetric",
    )
    parser.add_argument(
        "--source",
        choices=("gcg-selection20-original", "autodan-selection20-official", "autodan-selection20-original"),
        default="gcg-selection20-original",
    )
    parser.add_argument("--original-input", default=DEFAULT_ORIGINAL_INPUT)
    parser.add_argument("--autodan-original-input", default=DEFAULT_AUTODAN_ORIGINAL_INPUT)
    parser.add_argument("--sandbox-exempt-gpu", action="store_true")
    parser.add_argument("--selection-manifest", default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument(
        "--autodan-selection-manifest",
        default=DEFAULT_AUTODAN_SELECTION_MANIFEST,
    )
    parser.add_argument(
        "--output-dir",
        help="Required only for --stage build-reference; must not already exist.",
    )
    return parser.parse_args(argv)


def validate_request(args) -> Path | None:
    if args.source == "autodan-selection20-original" and args.component != "antisymmetric":
        raise ValueError("AutoDAN corresponding-original reference is antisymmetric only")
    if args.stage == "cpu-preflight":
        if args.output_dir is not None or args.sandbox_exempt_gpu:
            raise ValueError("cpu-preflight does not write an output directory")
        return None
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage build-reference")
    return ensure_output_directory_absent(Path(args.output_dir).resolve())


def validate_autodan_original_model(model_path):
    revision = model_repository_head(Path(model_path).resolve())
    if revision != MODEL_REPOSITORY_HEAD:
        raise ValueError("AutoDAN model repository revision mismatch")
    return revision


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


def collect_reference_moments(model, tokenizer, records, *, component="antisymmetric"):
    prompt_means = torch.empty(
        (SELECTION_COUNT, NUM_LAYERS, NUM_HEADS), dtype=torch.float64
    )
    prompt_mean_squares = torch.empty_like(prompt_means)
    prompt_records = []
    with patch_llama_full_prompt_moment_collection(
        model, component=component
    ) as collector:
        for position, record in enumerate(records):
            input_ids = record.get("input_ids")
            if input_ids is None:
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
                    "prompt_sha256": record.get(
                        "original_prompt_sha256",
                        record.get("autodan_prompt_sha256"),
                    ),
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


def load_autodan_selection20_records(selection_manifest_path, tokenizer=None, *, original_input=None):
    selection_path = Path(selection_manifest_path).resolve()
    if sha256_file(selection_path) != EXPECTED_AUTODAN_SELECTION_MANIFEST_SHA256:
        raise ValueError("frozen AutoDAN Selection20 manifest SHA-256 mismatch")
    saved = read_json(selection_path)
    if (
        saved.get("selection_indices") != AUTODAN_SELECTION_INDICES
        or saved.get("selection_count") != SELECTION_COUNT
        or saved.get("evaluation_count") != 110
        or saved.get("format") != AUTODAN_OFFICIAL_FORMAT
    ):
        raise ValueError("frozen AutoDAN Selection20 manifest mismatch")
    autodan_args = parse_autodan_args(["--stage", "cpu-validate"])
    raw_rows, advbench_rows, _, split = validate_autodan_sources(
        autodan_args, tokenizer=tokenizer
    )
    if split["selection_indices"] != AUTODAN_SELECTION_INDICES:
        raise ValueError("revalidated AutoDAN Selection20 indices mismatch")
    if (split["evaluation_indices"] != saved.get("evaluation_indices")
            or set(AUTODAN_SELECTION_INDICES) & set(split["evaluation_indices"])):
        raise ValueError("AutoDAN split overlap or evaluation index mismatch")
    if original_input is not None:
        original_path = Path(original_input).resolve()
        sources = saved["source_provenance"]
        if sha256_file(original_path) != sources["derived_original_sha256"]:
            raise ValueError("AutoDAN corresponding original source hash mismatch")
        for key in ("advbench_sha256", "derived_original_sha256", "autodan_sha256"):
            if split["source_provenance"][key] != sources[key]:
                raise ValueError(f"AutoDAN source provenance mismatch: {key}")
        originals = read_jsonl(original_path)
        if len(originals) != 130:
            raise ValueError("AutoDAN original source must contain 130 aligned prompts")
        records = []
        for index in AUTODAN_SELECTION_INDICES:
            prompt = originals[index]["prompt"]
            digest = sha256_text(prompt)
            if (prompt != advbench_rows[index]["prompt"]
                    or digest != saved["records"][index]["original_prompt_sha256"]
                    or prompt == raw_rows[index]["user prompt"]):
                raise ValueError(f"original/jailbreak pairing mismatch at index={index}")
            records.append({"dataset_index": index, "row_position": index,
                            "prompt_id": f"autodan_original_{index}", "prompt": prompt,
                            "original_prompt_sha256": digest})
        if tokenizer is not None:
            records = format_autodan_records(records, tokenizer, prompt_format=AUTODAN_OFFICIAL_FORMAT)
        return records, saved
    records = [
        {
            "dataset_index": index,
            "row_position": index,
            "prompt_id": f"autodan_{index}",
            "prompt": raw_rows[index]["user prompt"],
            "autodan_prompt_sha256": sha256_text(raw_rows[index]["user prompt"]),
        }
        for index in AUTODAN_SELECTION_INDICES
    ]
    if tokenizer is not None:
        records = format_autodan_records(
            records, tokenizer, prompt_format=AUTODAN_OFFICIAL_FORMAT
        )
    return records, saved


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
        "component": args.component,
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
    is_autodan_original = args.source == "autodan-selection20-original"
    if is_autodan_original:
        validate_autodan_original_model(args.model)
    is_autodan = args.source.startswith("autodan-selection20-")
    if is_autodan:
        records, selection_manifest = load_autodan_selection20_records(
            args.autodan_selection_manifest,
            original_input=args.autodan_original_input if is_autodan_original else None,
        )
        summary = {
            "stage": args.stage,
            "reference_condition": (AUTODAN_ORIGINAL_CONDITION if is_autodan_original
                                    else "autodan_selection20_official_jailbreak_only"),
            "selection_count": len(records),
            "selection_dataset_indices": AUTODAN_SELECTION_INDICES,
            "selection_manifest": str(Path(args.autodan_selection_manifest).resolve()),
            "selection_manifest_sha256": EXPECTED_AUTODAN_SELECTION_MANIFEST_SHA256,
            "prompt_format": AUTODAN_OFFICIAL_FORMAT,
            "position_scope": "full_prompt_strict_lower_i_gt_j",
            "prompt_weighting": "equal_weight_per_prompt",
            "jailbreak_reference_data_used": not is_autodan_original,
            "tau": None,
            "component": args.component,
        }
        if is_autodan_original:
            summary.update({
                "reference_source_description": "AutoDAN Selection20 corresponding originals only",
                "original_input": str(Path(args.autodan_original_input).resolve()),
                "original_input_sha256": sha256_file(args.autodan_original_input),
                "original_prompt_hashes_verified": True,
                "original_prompt_records": [
                    {key: row[key] for key in ("dataset_index", "original_prompt_sha256")}
                    for row in records
                ],
                "evaluation_dataset_indices": selection_manifest["evaluation_indices"],
                "reference_evaluation_overlap_count": 0,
                "source_provenance": selection_manifest["source_provenance"],
                "model_repository_head": MODEL_REPOSITORY_HEAD,
            })
    else:
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
    if is_autodan_original:
        seed_everything(42)
    model, tokenizer = load_eager_model(args.model)
    if is_autodan:
        records, selection_manifest = load_autodan_selection20_records(
            args.autodan_selection_manifest, tokenizer,
            original_input=args.autodan_original_input if is_autodan_original else None,
        )
    model_summary = llama_model_config_summary(model)
    prompt_means, prompt_mean_squares, prompt_records = collect_reference_moments(
        model, tokenizer, records, component=args.component
    )
    statistics = compute_equal_prompt_reference_statistics(
        prompt_means,
        prompt_mean_squares,
    )
    metadata = {
        "reference_condition": (
            (AUTODAN_ORIGINAL_CONDITION if is_autodan_original else "autodan_selection20_official_jailbreak_only")
            if is_autodan
            else "selection20_original_only"
        ),
        "prompt_weighting": "equal_weight_per_prompt",
        "position_scope": "full_prompt_strict_lower_i_gt_j",
        "component": (
            "A_asym=(S-S^T)/2"
            if args.component == "antisymmetric"
            else "A_sym=(S+S^T)/2"
        ),
        "prompt_count": SELECTION_COUNT,
        "mu_definition": f"mean_p(mean_strict_lower(A_{args.component}_p))",
        "mean_q_definition": f"mean_p(mean_strict_lower(A_{args.component}_p^2))",
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
        "selection_manifest": str(
            Path(
                args.autodan_selection_manifest if is_autodan else args.selection_manifest
            ).resolve()
        ),
        "selection_manifest_sha256": (
            EXPECTED_AUTODAN_SELECTION_MANIFEST_SHA256
            if is_autodan
            else sha256_file(args.selection_manifest)
        ),
        "selection_row_positions": (
            AUTODAN_SELECTION_INDICES
            if is_autodan
            else selection_manifest["selection_row_positions"]
        ),
        "selection_dataset_indices": (
            AUTODAN_SELECTION_INDICES
            if is_autodan
            else selection_manifest["selection_dataset_indices"]
        ),
        "jailbreak_reference_data_used": is_autodan and not is_autodan_original,
        "git_head": provenance["git_head"],
        "git_branch": provenance["git_branch"],
        "created_at": utc_now(),
    }
    if is_autodan:
        metadata["prompt_format"] = AUTODAN_OFFICIAL_FORMAT
        if is_autodan_original:
            metadata.update({key: summary[key] for key in (
                "reference_source_description", "original_input", "original_input_sha256",
                "original_prompt_hashes_verified", "original_prompt_records",
                "evaluation_dataset_indices", "reference_evaluation_overlap_count",
                "source_provenance",
                "model_repository_head",
            )})
            metadata["sandbox_exempt_gpu"] = args.sandbox_exempt_gpu
            metadata["reference_collection_forward_count"] = SELECTION_COUNT
            metadata["seed"] = 42
            metadata["gpu"] = torch.cuda.get_device_name(0)
    else:
        metadata.update(
            {
                "original_input": str(Path(args.original_input).resolve()),
                "original_input_sha256": selection_manifest["original_input_sha256"],
            }
        )
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
    reference_path = output_dir / (
        (AUTODAN_ORIGINAL_REFERENCE_FILE if is_autodan_original else
         (AUTODAN_REFERENCE_FILE if is_autodan else REFERENCE_FILE))
        if args.component == "antisymmetric"
        else (AUTODAN_SYM_REFERENCE_FILE if is_autodan else GCG_SYM_REFERENCE_FILE)
    )
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
        "reference_file": reference_path.name,
        "reference_sha256": sha256_file(reference_path),
        "mu_shape": list(statistics["mu"].shape),
        "sigma_shape": list(statistics["sigma"].shape),
        "clamped_negative_variance_count": statistics[
            "clamped_negative_variance_count"
        ],
        "sigma_floor_head_count": statistics["sigma_floor_head_count"],
        "prompt_records": prompt_records,
        "prompt_length": {
            "min": min(row["prompt_length"] for row in prompt_records),
            "mean": sum(row["prompt_length"] for row in prompt_records)
            / len(prompt_records),
            "max": max(row["prompt_length"] for row in prompt_records),
        },
        "tau": None,
    }
    if is_autodan_original:
        manifest_payload.update({"sandbox_exempt_gpu": args.sandbox_exempt_gpu,
                                 "reference_collection_forward_count": SELECTION_COUNT})
    write_json(output_dir / MANIFEST_FILE, manifest_payload)
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))
    print(f"reference artifacts written: {output_dir}")


if __name__ == "__main__":
    main()
