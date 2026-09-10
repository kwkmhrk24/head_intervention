#!/usr/bin/env python3
"""Run staged GPU validation for development-only AdvBench index 68."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
from accelerate.utils import find_executable_batch_size
from torch.nn import CrossEntropyLoss


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_gcg_position_element_topk_intervention import (  # noqa: E402
    format_records,
    load_model_and_tokenizer,
    load_selection,
    run_baseline,
    run_condition,
)
from scripts.generate_harmbench_gcg_independent_test import (  # noqa: E402
    DEFAULT_CONFIG,
    _early_stop_completion,
    _load_harmbench,
    validate_defense_artifact,
)
from utils.artifact_io import read_json, sha256_file, sha256_text, utc_now  # noqa: E402
from utils.harmbench_gcg_independent_test import (  # noqa: E402
    atomic_write_json_exclusive,
    build_initial_prompt_layout,
    canonical_json_sha256,
    filter_aligned_candidates,
    load_fast_tokenizer,
    load_or_skip_completed_example,
    sample_control_isolated,
    save_completed_example,
    save_failure_attempt,
    scientific_fingerprint,
    validate_fixed_config,
    validate_formatted_evaluation_identity,
    validate_harmbench_revision,
    validate_model_revision,
    verify_all_configured_sources,
)


DEFAULT_MANIFEST = (
    REPO_ROOT / "configs/gcg_development_index68_smoke_v1/manifest.json"
)
DEFAULT_DEFENSE_ARTIFACT = (
    REPO_ROOT
    / "data/outputs/gcg_wasserstein_reference/llama2/seed42_selection20/"
    "codex_gpu_20260802T061114Z"
)
EXPECTED_MANIFEST_ID = "gcg_development_index68_smoke_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("short", "cache-compare", "replay-compare", "full", "evaluate"),
        required=True,
    )
    parser.add_argument("--label", choices=("short_a", "short_b"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--defense-artifact", default=str(DEFAULT_DEFENSE_ARTIFACT))
    return parser.parse_args(argv)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _token_ids(tokenizer, text: str) -> list[int]:
    return [
        int(value)
        for value in tokenizer(text, add_special_tokens=False)["input_ids"]
    ]


def _gpu_info() -> dict:
    line = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    index, name, uuid, driver, memory_mib = [part.strip() for part in line.split(",")]
    return {
        "index": int(index),
        "name": name,
        "uuid": uuid,
        "driver_version": driver,
        "total_memory_mib": int(memory_mib),
        "torch_cuda_runtime": torch.version.cuda,
    }


def _load_development_record(config: dict, manifest: dict) -> dict:
    behavior_path = Path(config["dataset"]["behavior_source"])
    target_path = Path(config["dataset"]["target_source"])
    with behavior_path.open(newline="", encoding="utf-8") as handle:
        behavior_rows = list(csv.DictReader(handle))
    with target_path.open(encoding="utf-8") as handle:
        targets = json.load(handle)
    index = manifest["dataset_index"]
    source = behavior_rows[index]
    behavior_id = source["BehaviorID"]
    behavior = source["Behavior"]
    target = targets[behavior_id]
    if behavior_id != manifest["behavior_id"]:
        raise ValueError("development BehaviorID mismatch")
    if sha256_text(behavior) != manifest["behavior_sha256"]:
        raise ValueError("development behavior hash mismatch")
    if sha256_text(target) != manifest["target_sha256"]:
        raise ValueError("development target hash mismatch")
    return {
        "row_position": 0,
        "dataset_index": index,
        "behavior_id": behavior_id,
        "behavior": behavior,
        "target": target,
        "behavior_sha256": sha256_text(behavior),
        "target_sha256": sha256_text(target),
        "record_sha256": canonical_json_sha256(
            {
                "dataset_index": index,
                "behavior_id": behavior_id,
                "behavior": behavior,
                "target": target,
            }
        ),
        "example_seed": manifest["example_seed"],
    }


def validate_inputs(args) -> tuple[dict, dict, dict, dict]:
    config_path = Path(args.config).resolve()
    manifest_path = Path(args.manifest).resolve()
    config = read_json(config_path)
    manifest = read_json(manifest_path)
    validate_fixed_config(config)
    expected = {
        "schema_version": 1,
        "manifest_id": EXPECTED_MANIFEST_ID,
        "evaluation_role": "development_smoke",
        "independent_test_data_used": False,
        "base_attack_config_sha256": sha256_file(config_path),
        "dataset_index": 68,
        "behavior_id": "advbench_69",
        "base_seed": 42,
        "example_seed": 110,
        "short_updates": 5,
        "full_max_updates": 500,
        "attack_intervention": "none",
    }
    for key, value in expected.items():
        if manifest.get(key) != value or type(manifest.get(key)) is not type(value):
            raise ValueError(f"development manifest mismatch: {key}")
    if manifest.get("defense") != {
        "k": 500,
        "alpha": 1.0,
        "intervention_timing": "first_token_only",
    }:
        raise ValueError("development defense condition mismatch")
    if manifest.get("pass_criteria_exclude") != [
        "attack_success",
        "ASR_improvement",
    ]:
        raise ValueError("development pass criteria changed")
    source_hashes = verify_all_configured_sources(config)
    harmbench = validate_harmbench_revision(config)
    model = validate_model_revision(config)
    defense = validate_defense_artifact(args.defense_artifact)
    record = _load_development_record(config, manifest)
    return config, manifest, record, {
        "source_hashes": source_hashes,
        "harmbench": harmbench,
        "model": model,
        "defense": defense,
    }


def _run_fingerprint(args) -> dict:
    return scientific_fingerprint(
        Path(args.config).resolve(),
        Path(args.manifest).resolve(),
        Path(__file__).resolve(),
    )


def _ensure_run_manifest(args, record: dict, provenance: dict) -> dict:
    run_dir = Path(args.run_dir).resolve()
    fingerprint = _run_fingerprint(args)
    payload = {
        "schema_version": 1,
        "evaluation_role": "development_smoke",
        "independent_test_data_used": False,
        "dataset_index": 68,
        "behavior_id": "advbench_69",
        "example_seed": 110,
        **fingerprint,
        "record_sha256": record["record_sha256"],
        "gpu": _gpu_info(),
        "torch_version": torch.__version__,
        "created_at": utc_now(),
        "research_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "harmbench_head": provenance["harmbench"]["head"],
        "model_revision": provenance["model"]["head"],
        "defense_selection_manifest_sha256": provenance["defense"][
            "selection_manifest_sha256"
        ],
        "defense_score_artifact_sha256": provenance["defense"][
            "score_artifact_sha256"
        ],
    }
    path = run_dir / "run_manifest.json"
    if path.exists():
        existing = read_json(path)
        for key in (
            "schema_version",
            "evaluation_role",
            "independent_test_data_used",
            "dataset_index",
            "behavior_id",
            "example_seed",
            "config_sha256",
            "input_manifest_sha256",
            "adapter_sha256",
            "adapter_utility_sha256",
            "record_sha256",
            "research_head",
            "harmbench_head",
            "model_revision",
            "defense_selection_manifest_sha256",
            "defense_score_artifact_sha256",
        ):
            if existing.get(key) != payload.get(key):
                raise ValueError(f"existing development run mismatch: {key}")
        return existing
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("non-empty run directory lacks a run manifest")
    atomic_write_json_exclusive(path, payload)
    return payload


def _load_attack_method(config: dict):
    GCG, nonascii_fn, checker = _load_harmbench(config)
    optimization = config["optimization"]
    method = GCG(
        target_model={
            "model_name_or_path": config["model"]["local_path"],
            "dtype": "float16",
            "device_map": "auto",
            "use_fast_tokenizer": True,
            "local_files_only": True,
        },
        targets_path=config["dataset"]["target_source"],
        num_steps=500,
        adv_string_init=optimization["initial_control"],
        allow_non_ascii=False,
        search_width=512,
        use_prefix_cache=True,
        eval_steps=50,
        eval_with_check_refusal=True,
        check_refusal_min_loss=0.05,
        early_stopping=False,
    )
    if type(method.tokenizer).__name__ != "LlamaTokenizerFast":
        raise TypeError("development run did not load LlamaTokenizerFast")
    return method, nonascii_fn, checker


def _prepare(method, record: dict, config: dict) -> dict:
    tokenizer = method.tokenizer
    initial_ids = _token_ids(tokenizer, config["optimization"]["initial_control"])
    base = build_initial_prompt_layout(
        tokenizer,
        behavior=record["behavior"],
        target=record["target"],
        control_ids=initial_ids,
    )
    embed = method.model.get_input_embeddings()
    device = method.model.device
    prefix_ids = torch.tensor([base.prefix_ids], dtype=torch.long, device=device)
    after_ids = torch.tensor([base.after_control_ids], dtype=torch.long, device=device)
    target_ids = torch.tensor([base.target_ids], dtype=torch.long, device=device)
    vocab_ids = torch.arange(embed.weight.shape[0], dtype=torch.long, device=device)
    vocab_embeds = embed(vocab_ids)
    with torch.no_grad():
        prefix_cache = method.model(
            inputs_embeds=embed(prefix_ids), use_cache=True
        ).past_key_values
    return {
        "base": base,
        "embed": embed,
        "prefix_ids": prefix_ids,
        "after_ids": after_ids,
        "target_ids": target_ids,
        "prefix_embeds": embed(prefix_ids),
        "after_embeds": embed(after_ids),
        "target_embeds": embed(target_ids),
        "vocab_embeds": vocab_embeds,
        "prefix_cache": prefix_cache,
    }


def _gradient(method, components: dict, control_ids: torch.Tensor, *, cached: bool):
    model = method.model
    vocab_embeds = components["vocab_embeds"]
    onehot = torch.zeros(
        (1, control_ids.shape[1], vocab_embeds.shape[0]),
        device=model.device,
        dtype=model.dtype,
    )
    onehot.scatter_(2, control_ids.unsqueeze(2), 1.0).requires_grad_()
    control_embeds = torch.matmul(onehot.squeeze(0), vocab_embeds).unsqueeze(0)
    pieces = [control_embeds, components["after_embeds"], components["target_embeds"]]
    kwargs = {"use_cache": True, "past_key_values": components["prefix_cache"]}
    if not cached:
        pieces.insert(0, components["prefix_embeds"])
        kwargs = {"use_cache": False}
    inputs = torch.cat(pieces, dim=1)
    outputs = model(inputs_embeds=inputs, **kwargs)
    target_ids = components["target_ids"]
    target_length = target_ids.shape[1]
    logits = outputs.logits[..., -target_length - 1 : -1, :].contiguous()
    loss = CrossEntropyLoss()(
        logits.view(-1, logits.size(-1)), target_ids.view(-1)
    )
    gradient = torch.autograd.grad(loss, onehot)[0].squeeze(0)
    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError("non-finite target logits")
    if not math.isfinite(float(loss.item())):
        raise FloatingPointError("non-finite target loss")
    if not bool(torch.isfinite(gradient).all().item()):
        raise FloatingPointError("non-finite control gradient")
    return float(loss.item()), gradient.detach()


def _candidate_losses(
    method,
    components: dict,
    candidates: torch.Tensor,
    *,
    cached: bool,
) -> torch.Tensor:
    count = candidates.shape[0]
    candidate_embeds = components["embed"](candidates)
    pieces = [
        candidate_embeds,
        components["after_embeds"].repeat(count, 1, 1),
        components["target_embeds"].repeat(count, 1, 1),
    ]
    if not cached:
        pieces.insert(0, components["prefix_embeds"].repeat(count, 1, 1))
    inputs = torch.cat(pieces, dim=1)
    method.use_prefix_cache = cached
    method.prefix_cache = components["prefix_cache"]
    losses = find_executable_batch_size(
        method.compute_candidates_loss, method.search_batch_size
    )(inputs, components["target_ids"])
    if losses.shape != (count,):
        raise ValueError("candidate loss shape mismatch")
    if not bool(torch.isfinite(losses).all().item()):
        raise FloatingPointError("non-finite candidate loss")
    return losses


def _memory_and_time(start: float) -> dict:
    torch.cuda.synchronize()
    return {
        "elapsed_seconds": time.perf_counter() - start,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_vram_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_vram_mib": float(torch.cuda.max_memory_reserved() / 2**20),
    }


def run_optimization(
    method,
    nonascii_fn,
    checker,
    record: dict,
    config: dict,
    *,
    updates: int,
) -> dict:
    optimization = config["optimization"]
    stopping = config["early_stopping"]
    method.search_batch_size = 512
    components = _prepare(method, record, config)
    base = components["base"]
    current_layout = base
    optim_ids = torch.tensor(
        [base.control_ids], dtype=torch.long, device=method.model.device
    )
    not_allowed = torch.unique(nonascii_fn(method.tokenizer)).to(method.model.device)
    generator = torch.Generator(device=method.model.device).manual_seed(
        record["example_seed"]
    )
    all_losses = []
    valid_counts = []
    rejected_counts = []
    checks = []
    traces = []
    termination_reason = "max_updates_reached"
    termination_step = updates - 1
    for step in range(updates):
        step_start = time.perf_counter()
        gradient_loss, gradient = _gradient(
            method, components, optim_ids, cached=True
        )
        sampled = sample_control_isolated(
            optim_ids.squeeze(0),
            gradient,
            512,
            topk=256,
            not_allowed_tokens=not_allowed,
            generator=generator,
        )
        sampled_hash = _tensor_sha256(sampled)
        candidates, layouts, diagnostics = filter_aligned_candidates(
            method.tokenizer, base, sampled
        )
        candidate_hash = _tensor_sha256(candidates)
        losses = _candidate_losses(method, components, candidates, cached=True)
        best = int(losses.argmin().item())
        optim_ids = candidates[best].unsqueeze(0)
        current_layout = layouts[best]
        current_loss = float(losses[best].item())
        all_losses.append(current_loss)
        valid_counts.append(diagnostics["valid_candidate_count"])
        rejected_counts.append(diagnostics["rejected_candidate_count"])
        traces.append(
            {
                "step_zero_based": step,
                "gradient_loss": gradient_loss,
                "gradient_sha256": _tensor_sha256(gradient),
                "gradient_abs_max": float(gradient.abs().max().item()),
                "sampled_candidates_sha256": sampled_hash,
                "valid_candidates_sha256": candidate_hash,
                "candidate_losses_sha256": _tensor_sha256(losses),
                "valid_candidate_count": diagnostics["valid_candidate_count"],
                "rejected_candidate_count": diagnostics["rejected_candidate_count"],
                "selected_candidate_position": best,
                "selected_suffix_token_ids": list(current_layout.control_ids),
                "selected_suffix_text": current_layout.suffix_text,
                "candidate_loss_min": current_loss,
                "candidate_loss_max": float(losses.max().item()),
                "ranges": current_layout.ranges_payload(),
                "non_target_input_ids_sha256": canonical_json_sha256(
                    list(current_layout.non_target_input_ids)
                ),
                "elapsed_seconds": time.perf_counter() - step_start,
            }
        )
        should_eval = step % stopping["eval_interval_updates"] == 0 or step == 499
        if (
            updates == 500
            and should_eval
            and current_loss < stopping["check_refusal_loss_strictly_less_than"]
        ):
            result = _early_stop_completion(
                method.model,
                method.tokenizer,
                current_layout.non_target_input_ids,
                checker,
            )
            checks.append({"step_zero_based": step, "loss": current_loss, **result})
            if not result["is_refusal"]:
                termination_reason = "refusal_check_non_refusal"
                termination_step = step
                break
        print(
            f"[update {step + 1}/{updates}] loss={current_loss:.8f} "
            f"valid={diagnostics['valid_candidate_count']}/512",
            flush=True,
        )
        del gradient, sampled, candidates, losses
        torch.cuda.empty_cache()
        gc.collect()
    formatted_ids = _token_ids(method.tokenizer, current_layout.formatted_prompt)
    if formatted_ids != list(current_layout.non_target_input_ids):
        raise ValueError("final optimization/evaluation input IDs differ")
    return {
        "base_layout": base,
        "final_layout": current_layout,
        "all_losses": all_losses,
        "valid_candidate_counts": valid_counts,
        "rejected_candidate_counts": rejected_counts,
        "early_stop_check_history": checks,
        "termination_reason": termination_reason,
        "termination_step_zero_based": termination_step,
        "updates_completed": len(all_losses),
        "effective_candidate_batch_size": method.search_batch_size,
        "traces": traces,
    }


def short_stage(args, config, record, fingerprint):
    if args.label not in {"short_a", "short_b"}:
        raise ValueError("--label short_a or short_b is required for short stage")
    output_path = Path(args.run_dir).resolve() / args.label / "result.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    method, nonascii_fn, checker = _load_attack_method(config)
    result = run_optimization(
        method, nonascii_fn, checker, record, config, updates=5
    )
    layout = result["final_layout"]
    payload = {
        "schema_version": 1,
        "stage": args.label,
        "evaluation_role": "development_smoke",
        "dataset_index": 68,
        "behavior_id": "advbench_69",
        "example_seed": 110,
        **fingerprint,
        "updates_completed": result["updates_completed"],
        "termination_reason": "development_short_limit",
        "all_losses": result["all_losses"],
        "valid_candidate_counts": result["valid_candidate_counts"],
        "rejected_candidate_counts": result["rejected_candidate_counts"],
        "final_suffix_text": layout.suffix_text,
        "final_suffix_token_ids": list(layout.control_ids),
        "formatted_prompt": layout.formatted_prompt,
        "non_target_input_ids": list(layout.non_target_input_ids),
        "target_token_ids": list(layout.target_ids),
        "ranges": layout.ranges_payload(),
        "effective_candidate_batch_size": result["effective_candidate_batch_size"],
        "traces": result["traces"],
        "finite_forward_gradient_and_losses": True,
        **_memory_and_time(start),
        "completed_at": utc_now(),
    }
    atomic_write_json_exclusive(output_path, payload)
    return payload


def cache_compare_stage(args, config, record, fingerprint):
    output_path = Path(args.run_dir).resolve() / "cache_compare" / "result.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    method, nonascii_fn, _ = _load_attack_method(config)
    method.search_batch_size = 512
    components = _prepare(method, record, config)
    base = components["base"]
    control = torch.tensor(
        [base.control_ids], dtype=torch.long, device=method.model.device
    )
    cached_loss, cached_gradient = _gradient(
        method, components, control, cached=True
    )
    uncached_loss, uncached_gradient = _gradient(
        method, components, control, cached=False
    )
    generator = torch.Generator(device=method.model.device).manual_seed(110)
    not_allowed = torch.unique(nonascii_fn(method.tokenizer)).to(method.model.device)
    sampled = sample_control_isolated(
        control.squeeze(0),
        cached_gradient,
        512,
        topk=256,
        not_allowed_tokens=not_allowed,
        generator=generator,
    )
    candidates, layouts, diagnostics = filter_aligned_candidates(
        method.tokenizer, base, sampled
    )
    method.search_batch_size = 512
    cached_candidate_losses = _candidate_losses(
        method, components, candidates, cached=True
    )
    cached_batch_size = method.search_batch_size
    method.search_batch_size = 512
    uncached_candidate_losses = _candidate_losses(
        method, components, candidates, cached=False
    )
    uncached_batch_size = method.search_batch_size
    gradient_delta = cached_gradient.float() - uncached_gradient.float()
    loss_delta = cached_candidate_losses.float() - uncached_candidate_losses.float()
    cached_best = int(cached_candidate_losses.argmin().item())
    uncached_best = int(uncached_candidate_losses.argmin().item())
    cached_topk = (-cached_gradient).topk(256, dim=1).indices
    uncached_topk = (-uncached_gradient).topk(256, dim=1).indices
    topk_overlap = []
    for left, right in zip(cached_topk, uncached_topk):
        topk_overlap.append(
            len(set(left.tolist()) & set(right.tolist())) / 256
        )
    payload = {
        "schema_version": 1,
        "stage": "cache_compare",
        "evaluation_role": "development_smoke",
        "dataset_index": 68,
        "behavior_id": "advbench_69",
        "example_seed": 110,
        **fingerprint,
        "same_candidate_tensor_sha256": _tensor_sha256(candidates),
        "valid_candidate_count": diagnostics["valid_candidate_count"],
        "rejected_candidate_count": diagnostics["rejected_candidate_count"],
        "cached_gradient_loss": cached_loss,
        "uncached_gradient_loss": uncached_loss,
        "gradient_loss_abs_diff": abs(cached_loss - uncached_loss),
        "gradient_abs_diff_max": float(gradient_delta.abs().max().item()),
        "gradient_abs_diff_mean": float(gradient_delta.abs().mean().item()),
        "gradient_l2_diff": float(torch.linalg.vector_norm(gradient_delta).item()),
        "gradient_cached_abs_max": float(cached_gradient.abs().max().item()),
        "gradient_uncached_abs_max": float(uncached_gradient.abs().max().item()),
        "coordinate_topk_exact_match_fraction": float(
            (cached_topk == uncached_topk).float().mean().item()
        ),
        "coordinate_topk_overlap_mean": sum(topk_overlap) / len(topk_overlap),
        "candidate_loss_abs_diff_max": float(loss_delta.abs().max().item()),
        "candidate_loss_abs_diff_mean": float(loss_delta.abs().mean().item()),
        "cached_selected_position": cached_best,
        "uncached_selected_position": uncached_best,
        "candidate_selection_equal": cached_best == uncached_best,
        "cached_selected_suffix_token_ids": candidates[cached_best].tolist(),
        "uncached_selected_suffix_token_ids": candidates[uncached_best].tolist(),
        "cached_selected_suffix_text": layouts[cached_best].suffix_text,
        "uncached_selected_suffix_text": layouts[uncached_best].suffix_text,
        "cached_selected_loss": float(cached_candidate_losses[cached_best].item()),
        "uncached_selected_loss": float(uncached_candidate_losses[uncached_best].item()),
        "cached_effective_candidate_batch_size": cached_batch_size,
        "uncached_effective_candidate_batch_size": uncached_batch_size,
        "all_values_finite": True,
        **_memory_and_time(start),
        "completed_at": utc_now(),
    }
    atomic_write_json_exclusive(output_path, payload)
    if cached_best != uncached_best:
        raise RuntimeError("prefix cache changed candidate selection; stop before full run")
    return payload


def replay_compare_stage(args, fingerprint):
    run_dir = Path(args.run_dir).resolve()
    left = read_json(run_dir / "short_a/result.json")
    right = read_json(run_dir / "short_b/result.json")
    for payload in (left, right):
        for key, value in fingerprint.items():
            if payload.get(key) != value:
                raise ValueError(f"short-run fingerprint mismatch: {key}")
    trace_keys = (
        "gradient_sha256",
        "sampled_candidates_sha256",
        "valid_candidates_sha256",
        "candidate_losses_sha256",
        "selected_candidate_position",
        "selected_suffix_token_ids",
    )
    per_step = []
    for left_trace, right_trace in zip(left["traces"], right["traces"], strict=True):
        equality = {key: left_trace[key] == right_trace[key] for key in trace_keys}
        per_step.append(
            {
                "step_zero_based": left_trace["step_zero_based"],
                **equality,
                "loss_exact": left_trace["candidate_loss_min"]
                == right_trace["candidate_loss_min"],
                "loss_abs_diff": abs(
                    left_trace["candidate_loss_min"]
                    - right_trace["candidate_loss_min"]
                ),
            }
        )
    exact = (
        len(left["traces"]) == len(right["traces"]) == 5
        and all(
            all(
                value is True
                for key, value in row.items()
                if key not in {"step_zero_based", "loss_abs_diff"}
            )
            and row["loss_abs_diff"] == 0.0
            for row in per_step
        )
        and left["all_losses"] == right["all_losses"]
        and left["final_suffix_token_ids"] == right["final_suffix_token_ids"]
        and left["non_target_input_ids"] == right["non_target_input_ids"]
    )
    payload = {
        "schema_version": 1,
        "stage": "replay_compare",
        **fingerprint,
        "exact_reproduction": exact,
        "per_step": per_step,
        "all_losses_equal": left["all_losses"] == right["all_losses"],
        "final_suffix_equal": left["final_suffix_token_ids"]
        == right["final_suffix_token_ids"],
        "input_ids_equal": left["non_target_input_ids"]
        == right["non_target_input_ids"],
        "left_elapsed_seconds": left["elapsed_seconds"],
        "right_elapsed_seconds": right["elapsed_seconds"],
        "completed_at": utc_now(),
    }
    output = run_dir / "replay_compare/result.json"
    atomic_write_json_exclusive(output, payload)
    if not exact:
        raise RuntimeError("short-run reproduction mismatch; stop before full run")
    return payload


def full_stage(args, config, record, fingerprint):
    full_dir = Path(args.run_dir).resolve() / "full"
    tokenizer = load_fast_tokenizer(config)
    existing = load_or_skip_completed_example(
        full_dir, record, fingerprint, tokenizer=tokenizer
    )
    if existing is not None:
        return {
            "stage": "full",
            "status": "validated_skip",
            "output": str(full_dir / "examples/advbench_69.json"),
            "updates_completed": existing["updates_completed"],
            "termination_reason": existing["termination_reason"],
        }
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    method, nonascii_fn, checker = _load_attack_method(config)
    try:
        result = run_optimization(
            method, nonascii_fn, checker, record, config, updates=500
        )
        layout = result["final_layout"]
        payload = {
            "schema_version": 1,
            "status": "complete",
            "dataset_index": record["dataset_index"],
            "behavior_id": record["behavior_id"],
            "record_sha256": record["record_sha256"],
            "behavior": record["behavior"],
            "target": record["target"],
            "example_seed": record["example_seed"],
            **fingerprint,
            "control_token_count": 20,
            "final_suffix_text": layout.suffix_text,
            "final_suffix_token_ids": list(layout.control_ids),
            "test_case": layout.test_case,
            "formatted_prompt": layout.formatted_prompt,
            "non_target_input_ids": list(layout.non_target_input_ids),
            "target_token_ids": list(layout.target_ids),
            "ranges": layout.ranges_payload(),
            "termination_reason": result["termination_reason"],
            "termination_step_zero_based": result["termination_step_zero_based"],
            "updates_completed": result["updates_completed"],
            "final_loss": result["all_losses"][-1],
            "all_losses": result["all_losses"],
            "valid_candidate_counts": result["valid_candidate_counts"],
            "rejected_candidate_counts": result["rejected_candidate_counts"],
            "early_stop_check_history": result["early_stop_check_history"],
            "effective_candidate_batch_size": result[
                "effective_candidate_batch_size"
            ],
            "temperature_effective": False,
            "finite_forward_gradient_and_losses": True,
            "trace_sha256": canonical_json_sha256(result["traces"]),
            "traces": result["traces"],
            **_memory_and_time(start),
            "generated_at": utc_now(),
        }
        path = save_completed_example(
            full_dir,
            payload,
            record,
            fingerprint,
            tokenizer=method.tokenizer,
        )
        return {**payload, "output": str(path)}
    except Exception as exc:
        save_failure_attempt(full_dir, record, fingerprint, exc)
        raise


def _json_safe_record(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key not in {"input_ids", "first_logits"}
    }


def evaluate_stage(args, config, record, fingerprint):
    run_dir = Path(args.run_dir).resolve()
    output_path = run_dir / "evaluation/result.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    tokenizer_only = load_fast_tokenizer(config)
    completed = load_or_skip_completed_example(
        run_dir / "full", record, fingerprint, tokenizer=tokenizer_only
    )
    if completed is None:
        raise FileNotFoundError("validated full development output is missing")
    evaluation_records = [
        {
            "prompt_id": "llama:gcg-development:68",
            "row_position": 0,
            "dataset_index": 68,
            "behavior_id": "advbench_69",
            "behavior": record["behavior"],
            "target": record["target"],
            "prompt": completed["test_case"],
            "saved_formatted_prompt": completed["formatted_prompt"],
            "saved_non_target_input_ids": completed["non_target_input_ids"],
            "example_seed": 110,
        }
    ]
    artifact_dir = Path(args.defense_artifact).resolve()
    model_path = Path(config["model"]["local_path"]).resolve()
    selection, verification, metadata, _ = load_selection(
        artifact_dir, k_values=[500], model_path=model_path
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(model_path)
    formatted = format_records(evaluation_records, tokenizer)
    validate_formatted_evaluation_identity(formatted)
    baseline = run_baseline(
        formatted,
        model=model,
        tokenizer=tokenizer,
        label="development68 baseline",
        require_saved_reproduction=False,
    )
    rows, summary = run_condition(
        formatted,
        {68: baseline[0]},
        model=model,
        tokenizer=tokenizer,
        selection=selection,
        k=500,
        alpha=1.0,
        label="development68",
    )
    payload = {
        "schema_version": 1,
        "stage": "evaluate",
        "evaluation_role": "development_smoke",
        "dataset_index": 68,
        **fingerprint,
        "optimization_evaluation_input_ids_equal": True,
        "defense_selection_manifest_sha256": verification[
            "selection_manifest_sha256"
        ],
        "defense_score_artifact_sha256": verification["score_artifact_sha256"],
        "score_capture_position": metadata["score_capture_position"],
        "baseline": _json_safe_record(baseline[0]),
        "top500": _json_safe_record(rows[0]),
        "top500_summary": summary,
        **_memory_and_time(start),
        "completed_at": utc_now(),
    }
    atomic_write_json_exclusive(output_path, payload)
    return payload


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    config, _, record, provenance = validate_inputs(args)
    fingerprint = _run_fingerprint(args)
    _ensure_run_manifest(args, record, provenance)
    if args.stage == "short":
        result = short_stage(args, config, record, fingerprint)
    elif args.stage == "cache-compare":
        result = cache_compare_stage(args, config, record, fingerprint)
    elif args.stage == "replay-compare":
        result = replay_compare_stage(args, fingerprint)
    elif args.stage == "full":
        result = full_stage(args, config, record, fingerprint)
    else:
        result = evaluate_stage(args, config, record, fingerprint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
