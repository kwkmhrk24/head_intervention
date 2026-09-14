#!/usr/bin/env python3
"""Validate or run GCG Position Element Top-k reference intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_gcg_reference_factorization import (  # noqa: E402
    DEFAULT_ADVBENCH,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BASELINE_RESULTS,
    DEFAULT_HOLDOUT_RUN,
    DEFAULT_JAILBREAK_INPUT,
    DEFAULT_MODEL,
    DEFAULT_RAW_ARCHIVE,
    DEFAULT_SELECTION_ORIGINAL,
    DEFAULT_SELECTION_RUN,
    EXPECTED_SCORE_ARTIFACT_SHA256,
    EXPECTED_SELECTION_MANIFEST_SHA256,
    EXPECTED_TOP5,
    _load_group_inputs,
    format_records,
    generate_with_first_logits,
    load_model_and_tokenizer,
)
from utils.experiment_runtime import git_value, seed_everything  # noqa: E402
from utils.harmbench_gcg_independent_test import (  # noqa: E402
    load_fast_tokenizer as load_independent_fast_tokenizer,
    load_independent_evaluation_records,
    validate_fixed_config as validate_independent_config,
    validate_formatted_evaluation_identity,
    validate_harmbench_revision as validate_independent_harmbench_revision,
    validate_input_manifest as validate_independent_input_manifest,
    validate_model_revision as validate_independent_model_revision,
    verify_all_configured_sources as verify_independent_sources,
)
from utils.gcg_position_element_topk_intervention import (  # noqa: E402
    TOP_K_VALUES,
    build_position_element_selection,
    patch_llama_position_element_attention,
    selection_diagnostics,
    validate_top_k_values,
)
from utils.gcg_reference_intervention_diagnostics import (  # noqa: E402
    compare_token_sequences,
    compute_logit_step_metrics,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    llama_model_config_summary,
    read_json,
    sha256_file,
    utc_now,
    verify_artifact_directory,
    write_json,
    write_jsonl,
)
from utils.gcg_wasserstein_reference_evaluation import (  # noqa: E402
    classify_transition,
    ensure_output_absent,
)
from utils.reference_asym_score_intervention import (  # noqa: E402
    load_validated_reference_artifacts,
    validate_alpha,
    validate_loaded_llama_model,
)


DATASET_GROUPS = ("selection20", "holdout48", "independent100", "autodan130")
DEFAULT_DATASET_GROUPS = ("selection20", "holdout48")
DEFAULT_ALPHA = 1.0
DEFAULT_AUTODAN_INPUT = "data/raw/llama2/llama2_AutoDAN.json"
EXPECTED_AUTODAN_SHA256 = (
    "589add579b5110939c323f5c2de2984823f6d1afd684b9aa4fb112f7109198fb"
)
AUTODAN_RECORD_COUNT = 130
AUTODAN_PROMPT_FORMATS = ("current", "official_fastchat_34062e9")
DEFAULT_AUTODAN_PROMPT_FORMAT = "current"
AUTODAN_OFFICIAL_REPOSITORY_COMMIT = (
    "34062e964185693e81a6775b4f0d00bfd7507612"
)
AUTODAN_OFFICIAL_FASTCHAT_VERSION = "0.2.20"
AUTODAN_OFFICIAL_PREFIX = "[INST] <<SYS>>\n\n<</SYS>>\n\n"
AUTODAN_OFFICIAL_SUFFIX = "[/INST]"
DEFAULT_INDEPENDENT_CONFIG = (
    "configs/gcg_independent_test_v1/attack_config.json"
)
DEFAULT_INDEPENDENT_INPUT_MANIFEST = (
    "configs/gcg_independent_test_v1/input_manifest.json"
)
DEFAULT_INDEPENDENT_ADAPTER = (
    "scripts/generate_harmbench_gcg_independent_test.py"
)

REQUIRED_RUNTIME_INVARIANTS = {
    "first_forward_detected": True,
    "first_forward_intervened": True,
    "later_forward_intervention_count": 0,
    "short_sequence_skip_count": 0,
    "all_scores_finite": True,
    "max_unselected_element_change": 0.0,
    "max_outside_tail_change": 0.0,
    "max_diagonal_change_float32": 0.0,
    "max_constructed_delta_antisymmetry_error_float32": 0.0,
}


def serialized_input_ids(input_ids: torch.Tensor) -> list[int]:
    """Return one prompt's exact token IDs in a JSON-safe representation."""

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise TypeError("input_ids must be a rank-2 torch.Tensor")
    if input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain exactly one non-empty prompt")
    return [int(value) for value in input_ids[0].detach().cpu().tolist()]


def start_cuda_measurement() -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def finish_cuda_measurement(started_at: float) -> dict:
    torch.cuda.synchronize()
    return {
        "elapsed_seconds": time.perf_counter() - started_at,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved() / (1024**2),
    }


def parse_cli_alpha(value: str) -> float:
    """Convert argparse text to float before applying shared validation."""

    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("alpha must be a real number") from error
    try:
        return validate_alpha(numeric)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("cpu-validate", "run"),
        default="cpu-validate",
        help="cpu-validate is read-only and never loads a model.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--selection-run", default=DEFAULT_SELECTION_RUN)
    parser.add_argument("--holdout-run", default=DEFAULT_HOLDOUT_RUN)
    parser.add_argument("--selection-original", default=DEFAULT_SELECTION_ORIGINAL)
    parser.add_argument("--jailbreak-input", default=DEFAULT_JAILBREAK_INPUT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--raw-archive", default=DEFAULT_RAW_ARCHIVE)
    parser.add_argument("--advbench", default=DEFAULT_ADVBENCH)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(TOP_K_VALUES),
    )
    parser.add_argument("--alpha", type=parse_cli_alpha, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--dataset-groups",
        nargs="+",
        choices=DATASET_GROUPS,
        default=list(DEFAULT_DATASET_GROUPS),
    )
    parser.add_argument("--autodan-input", default=DEFAULT_AUTODAN_INPUT)
    parser.add_argument(
        "--autodan-prompt-format",
        choices=AUTODAN_PROMPT_FORMATS,
        default=DEFAULT_AUTODAN_PROMPT_FORMAT,
        help=(
            "AutoDAN input formatting. The official option reproduces the "
            "pinned AutoDAN FastChat llama-2 prompt with an empty SYS block."
        ),
    )
    parser.add_argument(
        "--autodan-smoke-count",
        type=int,
        help=(
            "Evaluate only the first N validated AutoDAN rows for a GPU smoke "
            "test. Omit for the frozen full 130-row transfer test."
        ),
    )
    parser.add_argument("--independent-config", default=DEFAULT_INDEPENDENT_CONFIG)
    parser.add_argument(
        "--independent-input-manifest",
        default=DEFAULT_INDEPENDENT_INPUT_MANIFEST,
    )
    parser.add_argument("--independent-attack-run-dir")
    parser.add_argument(
        "--independent-adapter", default=DEFAULT_INDEPENDENT_ADAPTER
    )
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def load_autodan_transfer_records(
    path: Path, *, smoke_count: int | None = None
) -> tuple[list[dict], dict]:
    """Load the immutable 130-row AutoDAN transfer set in source order."""

    path = path.resolve()
    if sha256_file(path) != EXPECTED_AUTODAN_SHA256:
        raise ValueError("AutoDAN source artifact hash mismatch")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid AutoDAN JSON: {path}") from error
    if not isinstance(rows, list) or len(rows) != AUTODAN_RECORD_COUNT:
        raise ValueError(
            f"AutoDAN source must contain exactly {AUTODAN_RECORD_COUNT} rows"
        )
    expected_indices = list(range(AUTODAN_RECORD_COUNT))
    actual_indices = [row.get("index") for row in rows if isinstance(row, dict)]
    if actual_indices != expected_indices:
        raise ValueError("AutoDAN indices must be the ordered integers 0..129")

    prepared = []
    prompt_hashes = set()
    stored_success_count = 0
    for row_position, row in enumerate(rows):
        prompt = row.get("user prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"invalid AutoDAN prompt at index={row_position}")
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_hash in prompt_hashes:
            raise ValueError(f"duplicate AutoDAN prompt at index={row_position}")
        prompt_hashes.add(prompt_hash)
        if row.get("method") != "AutoDAN":
            raise ValueError(f"AutoDAN method mismatch at index={row_position}")
        if row.get("model") != "llama2-7B-chat-hf":
            raise ValueError(f"AutoDAN model label mismatch at index={row_position}")
        if row.get("system prompt") != "":
            raise ValueError(f"AutoDAN system prompt mismatch at index={row_position}")
        if type(row.get("success")) is not bool:
            raise ValueError(
                f"stored AutoDAN success must be bool at index={row_position}"
            )
        stored_success_count += int(row["success"])
        prepared.append(
            {
                "prompt_id": f"autodan_{row_position}",
                "row_position": row_position,
                "dataset_index": row_position,
                "split": "held_out_attack_transfer_test",
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "stored_source_success": row["success"],
                "stored_source_loss": row.get("loss"),
            }
        )
    if stored_success_count != 84:
        raise ValueError("stored AutoDAN success count mismatch")
    if smoke_count is not None:
        if type(smoke_count) is not int or not 1 <= smoke_count <= len(prepared):
            raise ValueError("--autodan-smoke-count must be in [1, 130]")
        prepared = prepared[:smoke_count]
    provenance = {
        "role": "held_out_attack_transfer_test_not_newly_generated_independent_test",
        "source_path": str(path),
        "source_sha256": EXPECTED_AUTODAN_SHA256,
        "source_record_count": AUTODAN_RECORD_COUNT,
        "source_indices": expected_indices,
        "source_index_unique_count": AUTODAN_RECORD_COUNT,
        "stored_source_success_count_auxiliary_only": stored_success_count,
        "evaluated_record_count": len(prepared),
        "smoke_count": smoke_count,
        "prompt_hashes_unique": True,
        "selection_or_reference_from_autodan": False,
    }
    return prepared, provenance


def autodan_prompt_format_provenance(prompt_format: str) -> dict:
    """Describe the frozen AutoDAN formatting choice without loading a model."""

    if prompt_format not in AUTODAN_PROMPT_FORMATS:
        raise ValueError(f"unknown AutoDAN prompt format: {prompt_format!r}")
    if prompt_format == "current":
        return {
            "name": prompt_format,
            "implementation": "existing Llama tokenizer chat template",
            "system_prompt": None,
            "tokenize_add_special_tokens": False,
        }
    return {
        "name": prompt_format,
        "autodan_repository_commit": AUTODAN_OFFICIAL_REPOSITORY_COMMIT,
        "fastchat_version_used_for_literal_verification": (
            AUTODAN_OFFICIAL_FASTCHAT_VERSION
        ),
        "fastchat_template": "llama-2",
        "formatted_text_formula": (
            "[INST] <<SYS>>\\n\\n<</SYS>>\\n\\n{prompt}[/INST]"
        ),
        "prompt_text_preserved_without_strip": True,
        "empty_system_block": True,
        "assistant_boundary": AUTODAN_OFFICIAL_SUFFIX,
        "tokenize_add_special_tokens": True,
        "bos_added_by_tokenizer": True,
    }


def format_autodan_records(
    records: list[dict], tokenizer, *, prompt_format: str
) -> list[dict]:
    """Format AutoDAN records with the current or pinned official template."""

    autodan_prompt_format_provenance(prompt_format)
    if prompt_format == "current":
        return format_records(records, tokenizer)

    output = []
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        raise ValueError("official AutoDAN formatting requires a tokenizer BOS token")
    for record in records:
        formatted_prompt = (
            AUTODAN_OFFICIAL_PREFIX
            + record["prompt"]
            + AUTODAN_OFFICIAL_SUFFIX
        )
        input_ids = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids
        prompt_length = int(input_ids.shape[-1])
        if prompt_length < 16:
            raise ValueError(
                "AutoDAN prompt is shorter than 16 tokens: "
                f"dataset_index={record['dataset_index']}"
            )
        if int(input_ids[0, 0].item()) != int(bos_token_id):
            raise ValueError("official AutoDAN tokenization did not prepend BOS")
        output.append(
            {
                **record,
                "formatted_prompt": formatted_prompt,
                "prompt_length": prompt_length,
                "input_ids": input_ids,
            }
        )
    return output


def validate_autodan_format_request(args) -> None:
    """Reject AutoDAN-only options when the AutoDAN group is absent."""

    has_autodan = "autodan130" in args.dataset_groups
    if args.autodan_smoke_count is not None and not has_autodan:
        raise ValueError("--autodan-smoke-count requires --dataset-groups autodan130")
    if args.autodan_prompt_format != DEFAULT_AUTODAN_PROMPT_FORMAT and not has_autodan:
        raise ValueError("--autodan-prompt-format requires --dataset-groups autodan130")


def load_selection(artifact_dir: Path, *, k_values, model_path: Path):
    """Load audited selection20 tensors and construct selection in memory."""

    verification = verify_artifact_directory(artifact_dir)
    if verification.get("verified") is not True:
        raise ValueError("source artifact directory verification failed")
    if verification.get("selection_manifest_sha256") != (
        EXPECTED_SELECTION_MANIFEST_SHA256
    ):
        raise ValueError("selection manifest hash mismatch")
    if verification.get("score_artifact_sha256") != EXPECTED_SCORE_ARTIFACT_SHA256:
        raise ValueError("score artifact hash mismatch")
    paths = {
        "original": artifact_dir / "selection_original_scores.pt",
        "jailbreak": artifact_dir / "selection_jailbreak_scores.pt",
    }
    payloads = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in paths.items()
    }
    if payloads["original"].get("condition") != "original":
        raise ValueError("original score artifact condition mismatch")
    if payloads["jailbreak"].get("condition") != "jailbreak":
        raise ValueError("jailbreak score artifact condition mismatch")
    original_metadata = payloads["original"].get("metadata")
    jailbreak_metadata = payloads["jailbreak"].get("metadata")
    if not isinstance(original_metadata, dict) or original_metadata != jailbreak_metadata:
        raise ValueError("original/jailbreak score metadata mismatch")
    if Path(original_metadata.get("model_path", "")).resolve() != model_path:
        raise ValueError("selection score model path mismatch")
    if original_metadata.get("selection_manifest_sha256") != (
        EXPECTED_SELECTION_MANIFEST_SHA256
    ):
        raise ValueError("selection score split hash mismatch")
    selection = build_position_element_selection(
        payloads["original"].get("values"),
        payloads["jailbreak"].get("values"),
        k_values=validate_top_k_values(k_values),
    )
    return selection, verification, original_metadata, paths


def cpu_validation_payload(args) -> dict:
    validate_autodan_format_request(args)
    artifact_dir = Path(args.artifact_dir).resolve()
    model_path = Path(args.model).resolve()
    selection, verification, metadata, paths = load_selection(
        artifact_dir,
        k_values=args.k_values,
        model_path=model_path,
    )
    result = {
        "status": "cpu_validation_passed",
        "model_loaded": False,
        "gpu_used": False,
        "artifacts_written": False,
        "artifact_dir": str(artifact_dir),
        "source_artifact_verified": verification["verified"],
        "selection_manifest_sha256": verification["selection_manifest_sha256"],
        "score_artifact_sha256": verification["score_artifact_sha256"],
        "original_score_sha256": sha256_file(paths["original"]),
        "jailbreak_score_sha256": sha256_file(paths["jailbreak"]),
        "score_metadata": {
            key: metadata[key]
            for key in (
                "tail_tokens",
                "attention_implementation",
                "score_capture_position",
                "stored_value_dtype",
                "selection_row_positions",
                "selection_dataset_indices",
            )
        },
        "reference": {
            "construction": "mean over selection20 original S_asym values only",
            "shape": list(selection.original_mean_reference.shape),
            "dtype": str(selection.original_mean_reference.dtype),
            "device": selection.original_mean_reference.device.type,
            "finite": bool(
                torch.isfinite(selection.original_mean_reference).all().item()
            ),
        },
        "selection": selection_diagnostics(selection),
        "top_elements": [dict(row) for row in selection.ranking[:10]],
    }
    if "independent100" in args.dataset_groups:
        if tuple(args.k_values) != (500,) or args.alpha != 1.0:
            raise ValueError(
                "independent100 is frozen to k-values=500 and alpha=1.0"
            )
        independent_config = read_json(Path(args.independent_config).resolve())
        validate_independent_config(independent_config)
        if Path(independent_config["model"]["local_path"]).resolve() != model_path:
            raise ValueError("independent100 model path differs from frozen config")
        verify_independent_sources(independent_config)
        validate_independent_harmbench_revision(independent_config)
        validate_independent_model_revision(independent_config)
        independent_records = validate_independent_input_manifest(
            read_json(Path(args.independent_input_manifest).resolve()),
            independent_config,
        )
        independent_validation = {
            "input_manifest_verified": True,
            "record_count": len(independent_records),
            "attack_outputs_verified": False,
        }
        if args.independent_attack_run_dir:
            tokenizer = load_independent_fast_tokenizer(independent_config)
            completed = load_independent_evaluation_records(
                config_path=Path(args.independent_config).resolve(),
                input_manifest_path=Path(
                    args.independent_input_manifest
                ).resolve(),
                attack_run_dir=Path(args.independent_attack_run_dir).resolve(),
                adapter_path=Path(args.independent_adapter).resolve(),
                tokenizer=tokenizer,
            )
            completed = format_records(completed, tokenizer)
            validate_formatted_evaluation_identity(completed)
            independent_validation["attack_outputs_verified"] = True
            independent_validation["evaluation_input_ids_verified"] = True
        result["independent100"] = independent_validation
    if "autodan130" in args.dataset_groups:
        if tuple(args.k_values) != (500,) or args.alpha != 1.0:
            raise ValueError("autodan130 is frozen to k-values=500 and alpha=1.0")
        _, autodan_provenance = load_autodan_transfer_records(
            Path(args.autodan_input), smoke_count=args.autodan_smoke_count
        )
        autodan_provenance["prompt_format"] = autodan_prompt_format_provenance(
            args.autodan_prompt_format
        )
        result["autodan130"] = autodan_provenance
    return result


def validate_prompt_runtime_invariants(diagnostics: dict, *, k: int) -> None:
    """Fail fast when a prompt violates the first-token runtime contract."""

    if diagnostics.get("selected_element_intervention_count") != k:
        raise RuntimeError(
            "runtime invariant failed: selected_element_intervention_count "
            f"!= k ({diagnostics.get('selected_element_intervention_count')} != {k})"
        )
    for field, expected in REQUIRED_RUNTIME_INVARIANTS.items():
        observed = diagnostics.get(field)
        matches = observed is expected if isinstance(expected, bool) else (
            type(observed) in (int, float) and observed == expected
        )
        if not matches:
            raise RuntimeError(
                f"runtime invariant failed: {field}={observed!r}, "
                f"expected={expected!r}"
            )


def validate_first_token_outcome_invariant(
    *, response_equal: bool, first_token_argmax_equal: bool
) -> None:
    """A first-token-only edit can change output iff token-0 argmax changes."""

    if response_equal is not first_token_argmax_equal:
        raise RuntimeError(
            "first-token-only invariant failed: "
            "response_equal != first_token_argmax_equal"
        )


def run_baseline(
    records,
    *,
    model,
    tokenizer,
    label: str,
    require_saved_reproduction: bool = True,
) -> list[dict]:
    rows = []
    for position, record in enumerate(records, 1):
        generation = generate_with_first_logits(model, tokenizer, record["input_ids"])
        if generation["generation_failed"]:
            raise RuntimeError(
                f"baseline generation failed: dataset_index={record['dataset_index']}"
            )
        if require_saved_reproduction and (
            generation["response"] != record["baseline_response"]
            or generation["success"] is not record["baseline_success"]
        ):
            raise RuntimeError(
                f"saved baseline reproduction mismatch: "
                f"dataset_index={record['dataset_index']}"
            )
        rows.append({**record, **generation})
        print(
            f"[{label} {position}/{len(records)}] "
            f"dataset_index={record['dataset_index']}",
            flush=True,
        )
    return rows


def run_condition(
    records,
    baseline_by_index,
    *,
    model,
    tokenizer,
    selection,
    k: int,
    alpha: float,
    label: str,
):
    rows = []
    prompt_diagnostics = []
    with patch_llama_position_element_attention(
        model,
        selection=selection,
        k=k,
        alpha=alpha,
        intervention_timing="first_token_only",
    ) as controller:
        for position, record in enumerate(records, 1):
            controller.clear()
            controller.set_prompt_context(prompt_length=record["prompt_length"])
            baseline = baseline_by_index[record["dataset_index"]]
            if not torch.equal(record["input_ids"], baseline["input_ids"]):
                raise RuntimeError(
                    "baseline/defense input ID mismatch: "
                    f"dataset_index={record['dataset_index']}"
                )
            generation = generate_with_first_logits(
                model, tokenizer, record["input_ids"]
            )
            if generation["generation_failed"]:
                raise RuntimeError(
                    f"intervention generation failed: k={k} "
                    f"dataset_index={record['dataset_index']}"
                )
            diagnostics = controller.summary()
            validate_prompt_runtime_invariants(diagnostics, k=k)
            comparison = compare_token_sequences(
                baseline["response_token_ids"], generation["response_token_ids"]
            )
            logit_metrics = compute_logit_step_metrics(
                baseline["first_logits"], generation["first_logits"], step=0
            )
            validate_first_token_outcome_invariant(
                response_equal=bool(comparison["response_equal"]),
                first_token_argmax_equal=bool(logit_metrics["argmax_equal"]),
            )
            rows.append(
                {
                    "dataset_group": label,
                    "prompt_id": record["prompt_id"],
                    "row_position": record["row_position"],
                    "dataset_index": record["dataset_index"],
                    "prompt": record["prompt"],
                    "prompt_length": record["prompt_length"],
                    "input_ids": serialized_input_ids(record["input_ids"]),
                    "input_ids_match_baseline": True,
                    "k": k,
                    "alpha": alpha,
                    "intervention_timing": "first_token_only",
                    "baseline_response": baseline["response"],
                    "baseline_success": baseline["success"],
                    **{
                        key: value
                        for key, value in generation.items()
                        if key != "first_logits"
                    },
                    **comparison,
                    "first_token_argmax_equal": logit_metrics["argmax_equal"],
                    "first_token_logit_metrics": logit_metrics,
                    "transition": classify_transition(
                        baseline["success"], generation["success"]
                    ),
                    "diagnostics": diagnostics,
                }
            )
            prompt_diagnostics.append(diagnostics)
            print(
                f"[{label} k={k} {position}/{len(records)}] "
                f"dataset_index={record['dataset_index']} "
                f"success={generation['success']}",
                flush=True,
            )
    summary = {
        "dataset_group": label,
        "k": k,
        "alpha": alpha,
        "prompt_count": len(rows),
        "attack_success_count": sum(bool(row["success"]) for row in rows),
        "asr": sum(bool(row["success"]) for row in rows) / len(rows),
        "newly_defended_count": sum(
            row["transition"] == "newly_defended" for row in rows
        ),
        "still_attack_success_count": sum(
            row["transition"] == "still_attack_success" for row in rows
        ),
        "newly_worsened_count": sum(
            row["transition"] == "newly_worsened" for row in rows
        ),
        "still_defended_count": sum(
            row["transition"] == "still_defended" for row in rows
        ),
        "response_changed_count": sum(not row["response_equal"] for row in rows),
        "first_token_argmax_changed_count": sum(
            not row["first_token_argmax_equal"] for row in rows
        ),
        "input_ids_match_count": sum(
            bool(row["input_ids_match_baseline"]) for row in rows
        ),
        "first_forward_detected_count": sum(
            item["first_forward_detected"] is True for item in prompt_diagnostics
        ),
        "first_forward_intervened_count": sum(
            item["first_forward_intervened"] is True for item in prompt_diagnostics
        ),
        "selected_element_intervention_count": sum(
            item["selected_element_intervention_count"]
            for item in prompt_diagnostics
        ),
        "later_forward_intervention_count": sum(
            item["later_forward_intervention_count"] for item in prompt_diagnostics
        ),
        "short_sequence_skip_count": sum(
            item["short_sequence_skip_count"] for item in prompt_diagnostics
        ),
        "all_scores_finite": all(
            item["all_scores_finite"] is True for item in prompt_diagnostics
        ),
        "max_unselected_element_change": max(
            item["max_unselected_element_change"] for item in prompt_diagnostics
        ),
        "max_outside_tail_change": max(
            item["max_outside_tail_change"] for item in prompt_diagnostics
        ),
        "max_constructed_delta_antisymmetry_error_float32": max(
            item["max_constructed_delta_antisymmetry_error_float32"]
            for item in prompt_diagnostics
        ),
        "max_diagonal_change_float32": max(
            item["max_diagonal_change_float32"] for item in prompt_diagnostics
        ),
        "max_symmetric_component_error_float32": max(
            item["max_symmetric_component_error_float32"]
            for item in prompt_diagnostics
        ),
        "first_token_outcome_invariant_pass_count": len(rows),
        "generation_failure_count": 0,
    }
    return rows, summary


def run_gpu(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --stage run; refusing CPU model load")
    if not args.output_dir:
        raise ValueError("--output-dir is required for --stage run")
    if len(args.dataset_groups) != len(set(args.dataset_groups)):
        raise ValueError("--dataset-groups must not contain duplicates")
    if "independent100" in args.dataset_groups and (
        tuple(args.k_values) != (500,) or args.alpha != 1.0
    ):
        raise ValueError("independent100 is frozen to k-values=500 and alpha=1.0")
    if "independent100" in args.dataset_groups and not args.independent_attack_run_dir:
        raise ValueError(
            "--independent-attack-run-dir is required for independent100"
        )
    if "autodan130" in args.dataset_groups and (
        tuple(args.k_values) != (500,) or args.alpha != 1.0
    ):
        raise ValueError("autodan130 is frozen to k-values=500 and alpha=1.0")
    validate_autodan_format_request(args)
    output_dir = ensure_output_absent(args.output_dir).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    model_path = Path(args.model).resolve()
    selection, verification, metadata, paths = load_selection(
        artifact_dir,
        k_values=args.k_values,
        model_path=model_path,
    )
    groups = {}
    legacy_groups = [
        name for name in args.dataset_groups if name in DEFAULT_DATASET_GROUPS
    ]
    if legacy_groups:
        # Reuse the existing strict group alignment/audit path. Paired-original
        # values returned by it are not used by this deployable defense.
        loaded, _, _, _, _ = _load_group_inputs(args)
        groups.update({name: loaded[name] for name in legacy_groups})
    if "independent100" in args.dataset_groups:
        independent_config = read_json(Path(args.independent_config).resolve())
        validate_independent_config(independent_config)
        if Path(independent_config["model"]["local_path"]).resolve() != model_path:
            raise ValueError("independent100 model path differs from frozen config")
        verify_independent_sources(independent_config)
        validate_independent_harmbench_revision(independent_config)
        validate_independent_model_revision(independent_config)
        groups["independent100"] = load_independent_evaluation_records(
            config_path=Path(args.independent_config).resolve(),
            input_manifest_path=Path(
                args.independent_input_manifest
            ).resolve(),
            attack_run_dir=Path(args.independent_attack_run_dir).resolve(),
            adapter_path=Path(args.independent_adapter).resolve(),
        )
    autodan_provenance = None
    if "autodan130" in args.dataset_groups:
        groups["autodan130"], autodan_provenance = load_autodan_transfer_records(
            Path(args.autodan_input), smoke_count=args.autodan_smoke_count
        )
        autodan_provenance["prompt_format"] = autodan_prompt_format_provenance(
            args.autodan_prompt_format
        )

    audited_reference = load_validated_reference_artifacts(
        artifact_dir,
        expected_heads=EXPECTED_TOP5,
        expected_selection_manifest_sha256=EXPECTED_SELECTION_MANIFEST_SHA256,
        expected_score_artifact_sha256=EXPECTED_SCORE_ARTIFACT_SHA256,
        expected_model_path=model_path,
    )
    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(model_path)
    model_validation = validate_loaded_llama_model(model, audited_reference)
    groups = {
        name: (
            format_autodan_records(
                records,
                tokenizer,
                prompt_format=args.autodan_prompt_format,
            )
            if name == "autodan130"
            else format_records(records, tokenizer)
        )
        for name, records in groups.items()
    }
    if "independent100" in groups:
        validate_formatted_evaluation_identity(groups["independent100"])

    independent_provenance = None
    if "independent100" in groups:
        attack_run_dir = Path(args.independent_attack_run_dir).resolve()
        attack_final_manifest = attack_run_dir / "final_manifest.json"
        attack_run_manifest = attack_run_dir / "run_manifest.json"
        frozen = read_json(attack_final_manifest)
        for key, expected in (
            ("status", "complete_validated"),
            ("complete_count", 100),
            ("missing_count", 0),
            ("duplicate_count", 0),
            ("processing_failure_history_count", 0),
        ):
            if frozen.get(key) != expected:
                raise ValueError(
                    f"independent final manifest {key} mismatch: "
                    f"{frozen.get(key)!r} != {expected!r}"
                )
        independent_provenance = {
            "attack_run_dir": str(attack_run_dir),
            "attack_run_manifest_sha256": sha256_file(attack_run_manifest),
            "attack_final_manifest_sha256": sha256_file(attack_final_manifest),
            "attack_config_sha256": sha256_file(
                Path(args.independent_config).resolve()
            ),
            "attack_input_manifest_sha256": sha256_file(
                Path(args.independent_input_manifest).resolve()
            ),
            "attack_adapter_sha256": sha256_file(
                Path(args.independent_adapter).resolve()
            ),
            "frozen_prompt_count": len(groups["independent100"]),
        }

    output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at": utc_now(),
        "execution_git_head": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "execution_git_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "execution_git_status_porcelain": git_value(
            REPO_ROOT, "status", "--porcelain"
        ).splitlines(),
        "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
        "model": str(model_path),
        "model_config": llama_model_config_summary(model),
        "model_validation": model_validation,
        "gpu": torch.cuda.get_device_name(0),
        "artifact_dir": str(artifact_dir),
        "artifact_verification": verification,
        "source_score_hashes": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "score_metadata": metadata,
        "selection": selection_diagnostics(selection),
        "top_elements_by_k": {
            str(k): [dict(row) for row in selection.topk_records(k)]
            for k in selection.k_values
        },
        "dataset_groups": list(groups),
        "alpha": args.alpha,
        "intervention_timing": "first_token_only",
        "generation_settings": {
            "torch_dtype": "torch.float16",
            "interpolation_dtype": "torch.float32",
            "attention_implementation": "eager",
            "use_cache": False,
            "batch_size": 1,
            "do_sample": False,
            "max_new_tokens": 32,
            "seed": 42,
        },
        "independent_test_provenance": independent_provenance,
        "autodan_transfer_provenance": autodan_provenance,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    summaries = []
    baseline_summaries = []
    artifact_paths = []
    for group_name, records in groups.items():
        baseline_started = start_cuda_measurement()
        baseline = run_baseline(
            records,
            model=model,
            tokenizer=tokenizer,
            label=f"{group_name} baseline",
            require_saved_reproduction=group_name not in (
                "independent100",
                "autodan130",
            ),
        )
        baseline_runtime = finish_cuda_measurement(baseline_started)
        baseline_by_index = {row["dataset_index"]: row for row in baseline}
        baseline_path = output_dir / f"{group_name}_baseline.jsonl"
        write_jsonl(
            baseline_path,
            [
                {
                    key: (
                        serialized_input_ids(value)
                        if key == "input_ids"
                        else value
                    )
                    for key, value in row.items()
                    if key != "first_logits"
                }
                for row in baseline
            ],
        )
        artifact_paths.append(baseline_path)
        baseline_summary = {
            "dataset_group": group_name,
            "prompt_count": len(baseline),
            "attack_success_count": sum(bool(row["success"]) for row in baseline),
            "asr": sum(bool(row["success"]) for row in baseline) / len(baseline),
            "generation_failure_count": 0,
            "input_ids_saved_count": len(baseline),
            **baseline_runtime,
        }
        baseline_summary_path = output_dir / f"{group_name}_baseline_summary.json"
        write_json(baseline_summary_path, baseline_summary)
        artifact_paths.append(baseline_summary_path)
        baseline_summaries.append(baseline_summary)
        for k in selection.k_values:
            condition_started = start_cuda_measurement()
            rows, summary = run_condition(
                records,
                baseline_by_index,
                model=model,
                tokenizer=tokenizer,
                selection=selection,
                k=k,
                alpha=args.alpha,
                label=group_name,
            )
            summary.update(finish_cuda_measurement(condition_started))
            condition_dir = output_dir / group_name / f"k_{k}"
            condition_dir.mkdir(parents=True)
            prompt_results_path = condition_dir / "prompt_results.jsonl"
            condition_summary_path = condition_dir / "summary.json"
            write_jsonl(prompt_results_path, rows)
            write_json(condition_summary_path, summary)
            artifact_paths.extend([prompt_results_path, condition_summary_path])
            summaries.append(summary)
    result = {
        "status": "complete",
        "completed_at": utc_now(),
        "baselines": baseline_summaries,
        "conditions": summaries,
    }
    evaluation_summary_path = output_dir / "evaluation_summary.json"
    write_json(evaluation_summary_path, result)
    artifact_paths.append(evaluation_summary_path)
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    write_json(
        artifact_manifest_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "artifacts": [
                {
                    "relative_path": str(path.relative_to(output_dir)),
                    "sha256": sha256_file(path),
                }
                for path in artifact_paths
            ],
        },
    )
    manifest_path = output_dir / "run_manifest.json"
    saved_manifest = read_json(manifest_path)
    if saved_manifest.get("status") != "running":
        raise ValueError("run manifest is not in running state")
    saved_manifest["status"] = "complete"
    saved_manifest["completed_at"] = result["completed_at"]
    saved_manifest["artifact_manifest_sha256"] = sha256_file(
        artifact_manifest_path
    )
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary run manifest already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(saved_manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "cpu-validate":
        result = cpu_validation_payload(args)
    else:
        result = run_gpu(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
