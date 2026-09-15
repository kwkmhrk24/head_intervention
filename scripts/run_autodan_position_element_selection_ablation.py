#!/usr/bin/env python3
"""Build AutoDAN Top-500 artifacts and run the fixed four-way ablation."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_gcg_position_element_topk_intervention import (  # noqa: E402
    AUTODAN_OFFICIAL_REPOSITORY_COMMIT,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_AUTODAN_INPUT,
    DEFAULT_MODEL,
    EXPECTED_AUTODAN_SHA256,
    finish_cuda_measurement,
    format_autodan_records,
    generate_with_first_logits,
    load_model_and_tokenizer,
    load_selection,
    serialized_input_ids,
    start_cuda_measurement,
    validate_first_token_outcome_invariant,
    validate_prompt_runtime_invariants,
)
from utils.asym_score_intervention import (  # noqa: E402
    temporarily_disable_default_dynamic_cache,
)
from utils.experiment_runtime import git_value, seed_everything  # noqa: E402
from utils.gcg_position_element_topk_intervention import (  # noqa: E402
    RANKING_METRIC,
    PositionElementSelection,
    build_position_element_selection,
    patch_llama_position_element_attention,
    selection_diagnostics,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    TAIL_TOKENS,
    extract_prompt_scores_from_input_ids,
    llama_model_config_summary,
    read_json,
    read_jsonl,
    save_torch_artifact,
    sha256_file,
    sha256_text,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.gcg_wasserstein_reference_evaluation import (  # noqa: E402
    classify_transition,
    ensure_output_absent,
)


OFFICIAL_FORMAT = "official_fastchat_34062e9"
OFFICIAL_BASELINE_DIR = (
    "data/outputs/llama2/autodan/formatting_ablation/"
    "official_fastchat_34062e9_full130_20260914T151610Z"
)
OFFICIAL_BASELINE_HASHES = {
    "artifact_manifest.json": (
        "11c7a8a90a1179245740d482974bc5af4ef6d1f2a87c8eb5b863b4bf7dba9eb3"
    ),
    "run_manifest.json": (
        "52e717bf25f9a0f479a087aad7d0f16d950f191fc71081bf6c6ccee5dedcf76a"
    ),
    "paired_results.jsonl": (
        "409bc0f06542824920cac8b5cc5c0c2c87ce578aa0fea3d328b1a3523a8432a9"
    ),
    "summary.json": (
        "882454fa87387d2f66f81c8bc425b07b0502f4ead844a32a42d5a0e3acccd959"
    ),
}
OFFICIAL_CONDITION_A_RUN = (
    "data/outputs/gcg_position_element_topk/"
    "autodan130_official_format_transfer/full130_20260914T152933Z"
)
OFFICIAL_CONDITION_A_HASHES = {
    "artifact_manifest.json": (
        "fbfb36b39be53623366bb3095e8acceb9ba914e1cfa1839ff0a7564d773e5047"
    ),
    "run_manifest.json": (
        "64c1c6ea3bbb9bfe08733fa13db150714528a829085d4da8fae5b27f38fa2c2f"
    ),
    "defense_results.jsonl": (
        "9575fb47dbf8ba1fc0d1925943e54c20ba82c24fee3b525ec8bc5d9996dbed77"
    ),
}
DEFAULT_ADVBENCH = "data/raw/advbench.jsonl"
DEFAULT_DERIVED_ORIGINAL = "data/derived/llama2/autodan/original_prompts.jsonl"
DEFAULT_DERIVED_JAILBREAK = "data/derived/llama2/autodan/jailbreak_prompts.jsonl"
MODEL_REPOSITORY_HEAD = "f5db02db724555f92da89c216ac04704f23d4590"
TOTAL_COUNT = 130
SELECTION_COUNT = 20
EVALUATION_COUNT = 110
MALFORMED_INDEX = 10
SEED = 42
K = 500
ALPHA = 1.0
INTERVENTION_TIMING = "first_token_only"
STRUCTURAL_K_VALUES = (100, 500, 1000)
EXPECTED_SELECTION_INDICES = [
    1,
    2,
    4,
    8,
    11,
    12,
    14,
    30,
    31,
    33,
    36,
    39,
    41,
    58,
    82,
    89,
    91,
    106,
    108,
    114,
]
CONDITIONS = {
    "A": {"elements": "GCG", "reference": "GCG", "reuse_response": True},
    "B": {"elements": "AutoDAN", "reference": "GCG", "reuse_response": False},
    "C": {"elements": "GCG", "reference": "AutoDAN", "reuse_response": False},
    "D": {
        "elements": "AutoDAN",
        "reference": "AutoDAN",
        "reuse_response": False,
    },
}
SELECTION_ARTIFACT_FILES = (
    "selection_manifest.json",
    "selection_original_scores.pt",
    "selection_autodan_scores.pt",
    "full_ranking.jsonl",
    "top500.json",
    "original_mean_reference.pt",
    "structural_comparison.json",
    "run_manifest.json",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("cpu-validate", "build-selection", "evaluate"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--autodan-input", default=DEFAULT_AUTODAN_INPUT)
    parser.add_argument("--advbench", default=DEFAULT_ADVBENCH)
    parser.add_argument("--derived-original", default=DEFAULT_DERIVED_ORIGINAL)
    parser.add_argument("--derived-jailbreak", default=DEFAULT_DERIVED_JAILBREAK)
    parser.add_argument("--official-baseline-dir", default=OFFICIAL_BASELINE_DIR)
    parser.add_argument("--gcg-artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--condition-a-run", default=OFFICIAL_CONDITION_A_RUN
    )
    parser.add_argument("--selection-artifact-dir")
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def verify_fixed_hashes(directory: Path, expected: Mapping[str, str]) -> None:
    for name, digest in expected.items():
        path = directory / name
        if sha256_file(path) != digest:
            raise ValueError(f"fixed source hash mismatch: {path}")


def model_repository_head(model_path: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={model_path}",
            "-C",
            str(model_path),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_selection_manifest(
    *,
    raw_rows: Sequence[Mapping[str, object]],
    advbench_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
    source_provenance: Mapping[str, object],
) -> dict:
    if not (
        len(raw_rows) == len(baseline_rows) == TOTAL_COUNT
        and len(advbench_rows) >= TOTAL_COUNT
    ):
        raise ValueError("AutoDAN/AdvBench/baseline record count mismatch")
    eligible_success_indices = []
    records = []
    for index, (raw, original, baseline) in enumerate(
        zip(raw_rows, advbench_rows, baseline_rows)
    ):
        if raw.get("index") != index or original.get("index") != index:
            raise ValueError(f"dataset index mismatch at index={index}")
        original_prompt = original.get("prompt")
        jailbreak_prompt = raw.get("user prompt")
        if not isinstance(original_prompt, str) or not original_prompt:
            raise ValueError(f"invalid AdvBench behavior at index={index}")
        if not isinstance(jailbreak_prompt, str) or not jailbreak_prompt:
            raise ValueError(f"invalid AutoDAN prompt at index={index}")
        if baseline.get("dataset_index") != index:
            raise ValueError(f"official baseline index mismatch at index={index}")
        if baseline.get("prompt") != jailbreak_prompt:
            raise ValueError(f"official baseline prompt mismatch at index={index}")
        success = baseline.get("official_format_success")
        if type(success) is not bool or baseline.get("generation_failed") is not False:
            raise ValueError(f"invalid official baseline outcome at index={index}")
        malformed = index == MALFORMED_INDEX and "[REPLACE]" in jailbreak_prompt
        if success and not malformed:
            eligible_success_indices.append(index)
        records.append(
            {
                "dataset_index": index,
                "row_position": index,
                "prompt_id": f"autodan_{index}",
                "split": None,
                "official_baseline_success": success,
                "advbench_pair_valid": True,
                "known_malformed": malformed,
                "original_prompt_sha256": sha256_text(original_prompt),
                "autodan_prompt_sha256": sha256_text(jailbreak_prompt),
            }
        )
    if len(eligible_success_indices) != 48:
        raise ValueError("expected 48 eligible official-format baseline successes")
    selected = sorted(
        random.Random(SEED).sample(eligible_success_indices, SELECTION_COUNT)
    )
    if selected != EXPECTED_SELECTION_INDICES:
        raise ValueError("AutoDAN Selection20 differs from the frozen indices")
    selection_set = set(selected)
    evaluation = [index for index in range(TOTAL_COUNT) if index not in selection_set]
    for record in records:
        record["split"] = (
            "selection" if record["dataset_index"] in selection_set else "evaluation"
        )
    if len(evaluation) != EVALUATION_COUNT or MALFORMED_INDEX not in evaluation:
        raise AssertionError("invalid AutoDAN evaluation110 partition")
    baseline_success_all = sum(
        bool(record["official_baseline_success"]) for record in records
    )
    baseline_success_evaluation = sum(
        bool(records[index]["official_baseline_success"]) for index in evaluation
    )
    if baseline_success_all != 49 or baseline_success_evaluation != 29:
        raise ValueError("official baseline success counts changed")
    return {
        "schema_version": 1,
        "dataset": "Llama-2 AutoDAN 130 paired with AdvBench by dataset index",
        "selection_role": "development_selection_not_evaluation",
        "selection_rule_source": (
            "GCG rule recovered from utils/gcg_wasserstein_reference.py and "
            "the audited GCG split manifest"
        ),
        "selection_rule": (
            "validate index-aligned pairs; take official-format baseline successes; "
            "exclude known malformed index 10 before sampling; "
            "sorted(random.Random(42).sample(eligible_success_indices, 20))"
        ),
        "selection_does_not_use": [
            "GCG defense outcomes",
            "AutoDAN defense outcomes",
            "stored historical AutoDAN success",
            "AutoDAN loss",
            "response content beyond the fixed refusal-substring success label",
        ],
        "seed": SEED,
        "selection_count": SELECTION_COUNT,
        "evaluation_count": EVALUATION_COUNT,
        "official_baseline_success_count": baseline_success_all,
        "eligible_success_count": len(eligible_success_indices),
        "eligible_success_indices": eligible_success_indices,
        "excluded_malformed_indices": [MALFORMED_INDEX],
        "selection_indices": selected,
        "evaluation_indices": evaluation,
        "evaluation_baseline_success_count": baseline_success_evaluation,
        "format": OFFICIAL_FORMAT,
        "autodan_repository_commit": AUTODAN_OFFICIAL_REPOSITORY_COMMIT,
        "source_provenance": dict(source_provenance),
        "records": records,
    }


def validate_sources(args, *, tokenizer=None) -> tuple[list[dict], list[dict], list[dict], dict]:
    raw_path = Path(args.autodan_input).resolve()
    advbench_path = Path(args.advbench).resolve()
    derived_original_path = Path(args.derived_original).resolve()
    derived_jailbreak_path = Path(args.derived_jailbreak).resolve()
    baseline_dir = Path(args.official_baseline_dir).resolve()
    if sha256_file(raw_path) != EXPECTED_AUTODAN_SHA256:
        raise ValueError("AutoDAN raw source hash mismatch")
    verify_fixed_hashes(baseline_dir, OFFICIAL_BASELINE_HASHES)
    raw_rows = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise ValueError("AutoDAN raw source must be a JSON list")
    advbench_rows = read_jsonl(advbench_path)
    derived_original = read_jsonl(derived_original_path)
    derived_jailbreak = read_jsonl(derived_jailbreak_path)
    baseline_rows = read_jsonl(baseline_dir / "paired_results.jsonl")
    if len(derived_original) != TOTAL_COUNT or len(derived_jailbreak) != TOTAL_COUNT:
        raise ValueError("derived AutoDAN paired sources must contain 130 rows")
    for index in range(TOTAL_COUNT):
        if derived_original[index].get("prompt") != advbench_rows[index].get("prompt"):
            raise ValueError(f"derived original/AdvBench mismatch at index={index}")
        if derived_jailbreak[index].get("prompt") != raw_rows[index].get("user prompt"):
            raise ValueError(f"derived jailbreak/raw mismatch at index={index}")
    baseline_manifest = read_json(baseline_dir / "run_manifest.json")
    if baseline_manifest.get("autodan_repository_commit") != (
        AUTODAN_OFFICIAL_REPOSITORY_COMMIT
    ):
        raise ValueError("official baseline AutoDAN revision mismatch")
    if baseline_manifest.get("model_repository_head") != MODEL_REPOSITORY_HEAD:
        raise ValueError("official baseline model revision mismatch")
    source_provenance = {
        "autodan_path": str(raw_path),
        "autodan_sha256": sha256_file(raw_path),
        "advbench_path": str(advbench_path),
        "advbench_sha256": sha256_file(advbench_path),
        "derived_original_path": str(derived_original_path),
        "derived_original_sha256": sha256_file(derived_original_path),
        "derived_jailbreak_path": str(derived_jailbreak_path),
        "derived_jailbreak_sha256": sha256_file(derived_jailbreak_path),
        "official_baseline_dir": str(baseline_dir),
        "official_baseline_hashes": dict(OFFICIAL_BASELINE_HASHES),
    }
    split = build_selection_manifest(
        raw_rows=raw_rows,
        advbench_rows=advbench_rows,
        baseline_rows=baseline_rows,
        source_provenance=source_provenance,
    )
    if tokenizer is not None:
        records = [
            {"prompt": row["user prompt"], "dataset_index": index}
            for index, row in enumerate(raw_rows)
        ]
        formatted = format_autodan_records(
            records, tokenizer, prompt_format=OFFICIAL_FORMAT
        )
        for index, (record, baseline) in enumerate(zip(formatted, baseline_rows)):
            if record["formatted_prompt"] != baseline["official_formatted_prompt"]:
                raise ValueError(f"official formatted text mismatch at index={index}")
            if serialized_input_ids(record["input_ids"]) != baseline["official_input_ids"]:
                raise ValueError(f"official input IDs mismatch at index={index}")
        split["official_format_text_and_input_ids_reproduced_count"] = TOTAL_COUNT
    return raw_rows, advbench_rows, baseline_rows, split


def pearson(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("correlation inputs must be matching nontrivial vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("correlation inputs contain NaN or infinity")
    value = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(value):
        raise ValueError("correlation is not finite")
    return value


def count_by(records: Sequence[Mapping[str, object]], keys: Sequence[str]) -> list[dict]:
    counts: dict[tuple[int, ...], int] = {}
    for record in records:
        identity = tuple(int(record[key]) for key in keys)
        counts[identity] = counts.get(identity, 0) + 1
    domains = (
        [(layer,) for layer in range(NUM_LAYERS)]
        if tuple(keys) == ("layer",)
        else [
            (layer, head)
            for layer in range(NUM_LAYERS)
            for head in range(NUM_HEADS)
        ]
    )
    return [
        {**dict(zip(keys, identity)), "count": counts.get(identity, 0)}
        for identity in domains
    ]


def structural_comparison(
    gcg: PositionElementSelection, autodan: PositionElementSelection
) -> dict:
    if gcg.k_values != STRUCTURAL_K_VALUES or autodan.k_values != STRUCTURAL_K_VALUES:
        raise ValueError("structural comparison requires Top-100/500/1000")
    gcg_by_flat = {int(row["flat_index"]): row for row in gcg.ranking}
    autodan_by_flat = {int(row["flat_index"]): row for row in autodan.ranking}
    if set(gcg_by_flat) != set(autodan_by_flat) or len(gcg_by_flat) != 122880:
        raise ValueError("rankings do not cover the same 122,880 elements")
    overlaps = {}
    for k in STRUCTURAL_K_VALUES:
        left = {int(row["flat_index"]) for row in gcg.topk_records(k)}
        right = {int(row["flat_index"]) for row in autodan.topk_records(k)}
        intersection = left & right
        union = left | right
        overlaps[str(k)] = {
            "intersection_count": len(intersection),
            "jaccard": len(intersection) / len(union),
            "overlap_rate_each_set": len(intersection) / k,
        }
    flat_order = sorted(gcg_by_flat)
    score_correlation = pearson(
        [float(gcg_by_flat[index]["ranking_score"]) for index in flat_order],
        [float(autodan_by_flat[index]["ranking_score"]) for index in flat_order],
    )
    rank_correlation = pearson(
        [int(gcg_by_flat[index]["rank"]) for index in flat_order],
        [int(autodan_by_flat[index]["rank"]) for index in flat_order],
    )
    gcg500 = list(gcg.topk_records(500))
    autodan500 = list(autodan.topk_records(500))
    common = sorted(
        {int(row["flat_index"]) for row in gcg500}
        & {int(row["flat_index"]) for row in autodan500}
    )
    sign_matches = sum(
        int(gcg_by_flat[index]["mean_delta_sign"])
        == int(autodan_by_flat[index]["mean_delta_sign"])
        for index in common
    )
    return {
        "schema_version": 1,
        "candidate_element_count": 122880,
        "ranking_metric": RANKING_METRIC,
        "overlap_by_k": overlaps,
        "top500_intersection_count": overlaps["500"]["intersection_count"],
        "top500_jaccard": overlaps["500"]["jaccard"],
        "top500_overlap_rate_each_set": overlaps["500"]["overlap_rate_each_set"],
        "all_element_ranking_score_pearson": score_correlation,
        "all_element_rank_spearman": rank_correlation,
        "top500_layer_counts": {
            "GCG": count_by(gcg500, ("layer",)),
            "AutoDAN": count_by(autodan500, ("layer",)),
        },
        "top500_head_counts": {
            "GCG": count_by(gcg500, ("layer", "head")),
            "AutoDAN": count_by(autodan500, ("layer", "head")),
        },
        "top500_relative_position_counts": {
            "GCG": {str(i): sum(int(row["position"]) == i for row in gcg500) for i in range(ELEMENTS_PER_HEAD)},
            "AutoDAN": {str(i): sum(int(row["position"]) == i for row in autodan500) for i in range(ELEMENTS_PER_HEAD)},
        },
        "top500_lag_counts": {
            "GCG": {str(i): sum(int(row["lag"]) == i for row in gcg500) for i in range(1, TAIL_TOKENS)},
            "AutoDAN": {str(i): sum(int(row["lag"]) == i for row in autodan500) for i in range(1, TAIL_TOKENS)},
        },
        "common_top500_mean_delta_sign": {
            "common_count": len(common),
            "matching_count": sign_matches,
            "agreement_rate": sign_matches / len(common) if common else None,
        },
    }


def write_artifact_manifest(output_dir: Path, file_names: Sequence[str]) -> dict:
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "files": [
            {
                "relative_path": name,
                "sha256": sha256_file(output_dir / name),
                "size_bytes": (output_dir / name).stat().st_size,
            }
            for name in file_names
        ],
    }
    write_json(output_dir / "artifact_manifest.json", payload)
    return payload


def verify_artifact_manifest(output_dir: Path) -> dict:
    manifest = read_json(output_dir / "artifact_manifest.json")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("artifact manifest is empty")
    for record in files:
        path = output_dir / record["relative_path"]
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"artifact hash mismatch: {path}")
    return manifest


def load_autodan_selection_artifacts(
    artifact_dir: Path,
) -> tuple[PositionElementSelection, dict, dict]:
    verify_artifact_manifest(artifact_dir)
    split = read_json(artifact_dir / "selection_manifest.json")
    if split.get("selection_indices") != EXPECTED_SELECTION_INDICES:
        raise ValueError("AutoDAN selection indices mismatch")
    original_payload = torch.load(
        artifact_dir / "selection_original_scores.pt",
        map_location="cpu",
        weights_only=True,
    )
    autodan_payload = torch.load(
        artifact_dir / "selection_autodan_scores.pt",
        map_location="cpu",
        weights_only=True,
    )
    if original_payload.get("metadata") != autodan_payload.get("metadata"):
        raise ValueError("AutoDAN score artifact metadata mismatch")
    selection = build_position_element_selection(
        original_payload["values"],
        autodan_payload["values"],
        k_values=STRUCTURAL_K_VALUES,
    )
    reference_payload = torch.load(
        artifact_dir / "original_mean_reference.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not torch.equal(reference_payload.get("mean"), selection.original_mean_reference):
        raise ValueError("AutoDAN reference does not match original score mean")
    top500 = read_json(artifact_dir / "top500.json")
    expected = [
        {key: row[key] for key in ("rank", "flat_index", "layer", "head", "position")}
        for row in selection.topk_records(K)
    ]
    observed = [
        {key: row[key] for key in ("rank", "flat_index", "layer", "head", "position")}
        for row in top500["elements"]
    ]
    if observed != expected:
        raise ValueError("saved AutoDAN Top-500 identities mismatch")
    return selection, split, original_payload["metadata"]


def cpu_validate(args) -> dict:
    model_path = Path(args.model).resolve()
    if model_repository_head(model_path) != MODEL_REPOSITORY_HEAD:
        raise ValueError("model repository revision mismatch")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    _, _, _, split = validate_sources(args, tokenizer=tokenizer)
    gcg, verification, _, _ = load_selection(
        Path(args.gcg_artifact_dir).resolve(),
        k_values=STRUCTURAL_K_VALUES,
        model_path=model_path,
    )
    result = {
        "status": "cpu_validation_passed",
        "gpu_used": False,
        "artifacts_written": False,
        "model_revision": MODEL_REPOSITORY_HEAD,
        "selection_indices": split["selection_indices"],
        "evaluation_count": split["evaluation_count"],
        "evaluation_baseline_success_count": split[
            "evaluation_baseline_success_count"
        ],
        "official_format_text_and_input_ids_reproduced_count": split[
            "official_format_text_and_input_ids_reproduced_count"
        ],
        "gcg_artifact_verification": verification,
        "gcg_candidate_element_count": len(gcg.ranking),
    }
    if args.selection_artifact_dir:
        auto, saved_split, _ = load_autodan_selection_artifacts(
            Path(args.selection_artifact_dir).resolve()
        )
        result["saved_autodan_artifacts_verified"] = True
        result["saved_selection_indices"] = saved_split["selection_indices"]
        result["saved_candidate_element_count"] = len(auto.ranking)
    return result


def build_selection(args) -> dict:
    if not args.selection_artifact_dir:
        raise ValueError("--selection-artifact-dir is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for score extraction")
    output_dir = ensure_output_absent(args.selection_artifact_dir).resolve()
    model_path = Path(args.model).resolve()
    if model_repository_head(model_path) != MODEL_REPOSITORY_HEAD:
        raise ValueError("model repository revision mismatch")
    raw_rows, advbench_rows, baseline_rows, split = validate_sources(args)
    seed_everything(SEED)
    started = start_cuda_measurement()
    model, tokenizer = load_model_and_tokenizer(model_path)
    raw_rows, advbench_rows, baseline_rows, split = validate_sources(
        args, tokenizer=tokenizer
    )
    selected = split["selection_indices"]
    original_values = torch.empty(
        (SELECTION_COUNT, NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD),
        dtype=torch.float32,
    )
    autodan_values = torch.empty_like(original_values)
    sequence_lengths = []
    for selection_position, index in enumerate(selected):
        original_formatted = format_autodan_records(
            [{"prompt": advbench_rows[index]["prompt"], "dataset_index": index}],
            tokenizer,
            prompt_format=OFFICIAL_FORMAT,
        )[0]
        autodan_formatted = format_autodan_records(
            [{"prompt": raw_rows[index]["user prompt"], "dataset_index": index}],
            tokenizer,
            prompt_format=OFFICIAL_FORMAT,
        )[0]
        if serialized_input_ids(autodan_formatted["input_ids"]) != baseline_rows[index][
            "official_input_ids"
        ]:
            raise ValueError(f"selection AutoDAN input mismatch at index={index}")
        original_score, original_length = extract_prompt_scores_from_input_ids(
            model=model,
            input_ids=original_formatted["input_ids"],
            row_position=index,
            dataset_index=index,
        )
        autodan_score, autodan_length = extract_prompt_scores_from_input_ids(
            model=model,
            input_ids=autodan_formatted["input_ids"],
            row_position=index,
            dataset_index=index,
        )
        original_values[selection_position].copy_(original_score)
        autodan_values[selection_position].copy_(autodan_score)
        sequence_lengths.append(
            {
                "selection_position": selection_position,
                "dataset_index": index,
                "original_length": original_length,
                "autodan_length": autodan_length,
            }
        )
        print(f"[score {selection_position + 1}/20] index={index}", flush=True)
    selection = build_position_element_selection(
        original_values, autodan_values, k_values=STRUCTURAL_K_VALUES
    )
    gcg, gcg_verification, _, gcg_score_paths = load_selection(
        Path(args.gcg_artifact_dir).resolve(),
        k_values=STRUCTURAL_K_VALUES,
        model_path=model_path,
    )
    comparison = structural_comparison(gcg, selection)
    runtime = finish_cuda_measurement(started)
    output_dir.mkdir(parents=True)
    split["created_at"] = utc_now()
    split["execution_git_head"] = git_value(REPO_ROOT, "rev-parse", "HEAD")
    split["execution_git_branch"] = git_value(REPO_ROOT, "branch", "--show-current")
    write_json(output_dir / "selection_manifest.json", split)
    split_hash = sha256_file(output_dir / "selection_manifest.json")
    metadata = {
        "selection_manifest_sha256": split_hash,
        "selection_indices": selected,
        "model_path": str(model_path),
        "model_repository_head": MODEL_REPOSITORY_HEAD,
        "model_config": llama_model_config_summary(model),
        "format": OFFICIAL_FORMAT,
        "autodan_repository_commit": AUTODAN_OFFICIAL_REPOSITORY_COMMIT,
        "torch_dtype": "torch.float16",
        "attention_implementation": "eager",
        "use_cache": False,
        "tail_tokens": TAIL_TOKENS,
        "score_capture_position": (
            "post-RoPE, after repeat_kv, immediately after QK^T/sqrt(head_dim), "
            "pre-mask, pre-softmax"
        ),
        "score_calculation_dtype": (
            "torch.float16 Q/K matmul and scaling; S_asym computed in torch.float16"
        ),
        "stored_value_dtype": "torch.float32",
        "sequence_lengths": sequence_lengths,
    }
    save_torch_artifact(
        output_dir / "selection_original_scores.pt",
        {"condition": "original", "values": original_values, "metadata": metadata},
    )
    save_torch_artifact(
        output_dir / "selection_autodan_scores.pt",
        {"condition": "AutoDAN", "values": autodan_values, "metadata": metadata},
    )
    write_jsonl(output_dir / "full_ranking.jsonl", list(selection.ranking))
    write_json(
        output_dir / "top500.json",
        {
            "ranking_metric": RANKING_METRIC,
            "k": K,
            "elements": [dict(row) for row in selection.topk_records(K)],
            "diagnostics": selection_diagnostics(selection),
        },
    )
    save_torch_artifact(
        output_dir / "original_mean_reference.pt",
        {"mean": selection.original_mean_reference, "metadata": metadata},
    )
    write_json(output_dir / "structural_comparison.json", comparison)
    run_manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": utc_now(),
        "stage": "build-selection",
        "execution_git_head": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "execution_git_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "execution_git_status_porcelain": git_value(
            REPO_ROOT, "status", "--porcelain"
        ).splitlines(),
        "model": str(model_path),
        "model_repository_head": MODEL_REPOSITORY_HEAD,
        "selection": selection_diagnostics(selection),
        "gcg_artifact_verification": gcg_verification,
        "gcg_score_hashes": {
            name: sha256_file(path) for name, path in gcg_score_paths.items()
        },
        "runtime": runtime,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    artifact_manifest = write_artifact_manifest(output_dir, SELECTION_ARTIFACT_FILES)
    return {
        "status": "complete",
        "selection_artifact_dir": str(output_dir),
        "selection_indices": selected,
        "top500_intersection_count": comparison["top500_intersection_count"],
        "artifact_manifest_sha256": sha256_file(
            output_dir / "artifact_manifest.json"
        ),
        "runtime": runtime,
        "artifact_count": len(artifact_manifest["files"]),
    }


def first_forward_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(), temporarily_disable_default_dynamic_cache(model):
        output = model(
            input_ids=input_ids.to(model.device),
            use_cache=False,
            return_dict=True,
        )
    logits = output.logits[0, -1].float().detach().cpu()
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("baseline logits contain NaN or infinity")
    return logits


def logit_diagnostics(baseline: torch.Tensor, condition: torch.Tensor) -> dict:
    baseline = baseline.double().cpu()
    condition = condition.double().cpu()
    if baseline.shape != condition.shape or baseline.ndim != 1:
        raise ValueError("logit vectors must have matching rank-1 shapes")
    delta = condition - baseline
    top2 = torch.topk(baseline, 2)
    log_p = torch.log_softmax(baseline, dim=0)
    log_q = torch.log_softmax(condition, dim=0)
    kl = float(torch.sum(torch.exp(log_p) * (log_p - log_q)).item())
    if kl < 0 and kl > -1e-12:
        kl = 0.0
    result = {
        "max_absolute_logit_difference": float(delta.abs().max().item()),
        "logit_l2_distance": float(torch.linalg.vector_norm(delta).item()),
        "kl_baseline_to_condition": kl,
        "baseline_top1_top2_margin": float(
            (top2.values[0] - top2.values[1]).item()
        ),
        "baseline_top1_token_id": int(top2.indices[0].item()),
        "condition_top1_token_id": int(torch.argmax(condition).item()),
        "argmax_equal": bool(torch.argmax(baseline) == torch.argmax(condition)),
    }
    if not all(
        math.isfinite(float(value))
        for value in result.values()
        if type(value) in (int, float)
    ):
        raise ValueError("logit diagnostics contain NaN or infinity")
    return result


def aggregate_scalar(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("metric values must be a non-empty finite vector")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_condition(
    rows: Sequence[Mapping[str, object]], baseline_rows: Sequence[Mapping[str, object]]
) -> dict:
    if len(rows) != len(baseline_rows) or not rows:
        raise ValueError("condition/baseline row count mismatch")
    transitions = {
        "newly_defended": 0,
        "still_attack_success": 0,
        "newly_worsened": 0,
        "still_defended": 0,
    }
    for row in rows:
        transitions[row["transition"]] += 1
    baseline_success = sum(bool(row["success"]) for row in baseline_rows)
    condition_success = sum(bool(row["success"]) for row in rows)
    metrics = (
        "max_absolute_logit_difference",
        "logit_l2_distance",
        "kl_baseline_to_condition",
        "baseline_top1_top2_margin",
    )
    return {
        "prompt_count": len(rows),
        "baseline_attack_success_count": baseline_success,
        "attack_success_count": condition_success,
        "asr": condition_success / len(rows),
        "asr_delta_from_baseline": (condition_success - baseline_success) / len(rows),
        "relative_attack_success_reduction": (
            (baseline_success - condition_success) / baseline_success
            if baseline_success
            else None
        ),
        **{f"{name}_count": count for name, count in transitions.items()},
        "response_changed_count": sum(bool(row["response_changed"]) for row in rows),
        "first_token_argmax_changed_count": sum(
            not bool(row["logit_diagnostics"]["argmax_equal"]) for row in rows
        ),
        "generation_failure_count": sum(
            bool(row["generation_failed"]) for row in rows
        ),
        "logit_metrics": {
            metric: aggregate_scalar(
                [float(row["logit_diagnostics"][metric]) for row in rows]
            )
            for metric in metrics
        },
    }


def verify_condition_a_run(run_dir: Path, baseline_rows: Sequence[dict]) -> list[dict]:
    verify_fixed_hashes(run_dir, OFFICIAL_CONDITION_A_HASHES)
    artifact_manifest = read_json(run_dir / "artifact_manifest.json")
    for record in artifact_manifest.get("artifacts", []):
        path = run_dir / record["relative_path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Condition A artifact hash mismatch: {path}")
    manifest = read_json(run_dir / "run_manifest.json")
    if (
        manifest.get("format_formula")
        != "[INST] <<SYS>>\\n\\n<</SYS>>\\n\\n{prompt}[/INST]"
        or manifest.get("k") != K
        or manifest.get("alpha") != ALPHA
        or manifest.get("intervention_timing") != INTERVENTION_TIMING
    ):
        raise ValueError("Condition A scientific conditions mismatch")
    rows = read_jsonl(run_dir / "defense_results.jsonl")
    if len(rows) != TOTAL_COUNT:
        raise ValueError("Condition A must contain 130 rows")
    for index, (row, baseline) in enumerate(zip(rows, baseline_rows)):
        if row.get("dataset_index") != index:
            raise ValueError(f"Condition A index mismatch at index={index}")
        if row.get("input_ids") != baseline.get("official_input_ids"):
            raise ValueError(f"Condition A input IDs mismatch at index={index}")
        validate_prompt_runtime_invariants(row["diagnostics"], k=K)
    return rows


def condition_selection(
    name: str,
    gcg: PositionElementSelection,
    autodan: PositionElementSelection,
) -> PositionElementSelection:
    definition = CONDITIONS[name]
    elements = gcg if definition["elements"] == "GCG" else autodan
    reference = (
        gcg.original_mean_reference
        if definition["reference"] == "GCG"
        else autodan.original_mean_reference
    )
    return replace(elements, original_mean_reference=reference)


def evaluate(args) -> dict:
    if not args.selection_artifact_dir or not args.output_dir:
        raise ValueError("--selection-artifact-dir and --output-dir are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    output_dir = ensure_output_absent(args.output_dir).resolve()
    model_path = Path(args.model).resolve()
    if model_repository_head(model_path) != MODEL_REPOSITORY_HEAD:
        raise ValueError("model repository revision mismatch")
    autodan, split, autodan_metadata = load_autodan_selection_artifacts(
        Path(args.selection_artifact_dir).resolve()
    )
    gcg, gcg_verification, _, gcg_score_paths = load_selection(
        Path(args.gcg_artifact_dir).resolve(),
        k_values=STRUCTURAL_K_VALUES,
        model_path=model_path,
    )
    raw_rows, _, baseline_all, current_split = validate_sources(args)
    if current_split["selection_indices"] != split["selection_indices"]:
        raise ValueError("current/saved AutoDAN split mismatch")
    condition_a_all = verify_condition_a_run(
        Path(args.condition_a_run).resolve(), baseline_all
    )
    seed_everything(SEED)
    model, tokenizer = load_model_and_tokenizer(model_path)
    raw_rows, _, baseline_all, current_split = validate_sources(
        args, tokenizer=tokenizer
    )
    evaluation_indices = split["evaluation_indices"]
    formatted_all = format_autodan_records(
        [
            {"prompt": row["user prompt"], "dataset_index": index}
            for index, row in enumerate(raw_rows)
        ],
        tokenizer,
        prompt_format=OFFICIAL_FORMAT,
    )
    records = []
    for index in evaluation_indices:
        baseline = baseline_all[index]
        formatted = formatted_all[index]
        ids = serialized_input_ids(formatted["input_ids"])
        if ids != baseline["official_input_ids"]:
            raise ValueError(f"evaluation baseline input mismatch at index={index}")
        records.append(
            {
                "dataset_index": index,
                "prompt_id": f"autodan_{index}",
                "prompt": raw_rows[index]["user prompt"],
                "formatted_prompt": formatted["formatted_prompt"],
                "prompt_length": formatted["prompt_length"],
                "input_ids": formatted["input_ids"],
                "baseline_response": baseline["official_format_response"],
                "baseline_response_token_ids": baseline[
                    "official_response_token_ids"
                ],
                "baseline_success": baseline["official_format_success"],
            }
        )
    baseline_logits = {}
    baseline_started = start_cuda_measurement()
    for position, record in enumerate(records, 1):
        logits = first_forward_logits(model, record["input_ids"])
        if int(torch.argmax(logits).item()) != record["baseline_response_token_ids"][0]:
            raise RuntimeError(
                f"baseline first token mismatch at index={record['dataset_index']}"
            )
        baseline_logits[record["dataset_index"]] = logits
        print(
            f"[baseline-logit {position}/{len(records)}] "
            f"index={record['dataset_index']}",
            flush=True,
        )
    baseline_runtime = finish_cuda_measurement(baseline_started)
    condition_rows = {}
    condition_runtimes = {}
    condition_a_by_index = {
        row["dataset_index"]: row for row in condition_a_all
    }
    for name in CONDITIONS:
        selected = condition_selection(name, gcg, autodan)
        rows = []
        started = start_cuda_measurement()
        with patch_llama_position_element_attention(
            model,
            selection=selected,
            k=K,
            alpha=ALPHA,
            intervention_timing=INTERVENTION_TIMING,
        ) as controller:
            for position, record in enumerate(records, 1):
                controller.clear()
                controller.set_prompt_context(prompt_length=record["prompt_length"])
                if CONDITIONS[name]["reuse_response"]:
                    condition_logits = first_forward_logits(model, record["input_ids"])
                    saved = condition_a_by_index[record["dataset_index"]]
                    response = saved["response"]
                    response_ids = saved["response_token_ids"]
                    success = saved["success"]
                    generation_failed = saved["generation_failed"]
                    matched = saved["matched_refusal_substrings"]
                    decision = saved["success_decision_reason"]
                else:
                    generation = generate_with_first_logits(
                        model, tokenizer, record["input_ids"]
                    )
                    condition_logits = generation["first_logits"]
                    response = generation["response"]
                    response_ids = generation["response_token_ids"]
                    success = generation["success"]
                    generation_failed = generation["generation_failed"]
                    matched = generation["matched_refusal_substrings"]
                    decision = generation["success_decision_reason"]
                if generation_failed:
                    raise RuntimeError(
                        f"generation failed at condition={name} "
                        f"index={record['dataset_index']}"
                    )
                diagnostics = controller.summary()
                validate_prompt_runtime_invariants(diagnostics, k=K)
                logit = logit_diagnostics(
                    baseline_logits[record["dataset_index"]], condition_logits
                )
                if int(response_ids[0]) != logit["condition_top1_token_id"]:
                    raise RuntimeError("condition response/logit first token mismatch")
                response_changed = response != record["baseline_response"]
                validate_first_token_outcome_invariant(
                    response_equal=not response_changed,
                    first_token_argmax_equal=logit["argmax_equal"],
                )
                transition = classify_transition(record["baseline_success"], success)
                rows.append(
                    {
                        "condition": name,
                        "elements_source": CONDITIONS[name]["elements"],
                        "reference_source": CONDITIONS[name]["reference"],
                        "dataset_index": record["dataset_index"],
                        "prompt_id": record["prompt_id"],
                        "prompt": record["prompt"],
                        "formatted_prompt": record["formatted_prompt"],
                        "prompt_length": record["prompt_length"],
                        "input_ids": serialized_input_ids(record["input_ids"]),
                        "input_ids_match_baseline": True,
                        "k": K,
                        "alpha": ALPHA,
                        "intervention_timing": INTERVENTION_TIMING,
                        "baseline_response": record["baseline_response"],
                        "baseline_success": record["baseline_success"],
                        "response": response,
                        "response_token_ids": response_ids,
                        "success": success,
                        "matched_refusal_substrings": matched,
                        "success_decision_reason": decision,
                        "generation_failed": generation_failed,
                        "response_changed": response_changed,
                        "transition": transition,
                        "logit_diagnostics": logit,
                        "runtime_diagnostics": diagnostics,
                        "response_reused_from_fixed_condition_a_run": (
                            CONDITIONS[name]["reuse_response"]
                        ),
                    }
                )
                print(
                    f"[condition {name} {position}/{len(records)}] "
                    f"index={record['dataset_index']}",
                    flush=True,
                )
        condition_runtimes[name] = finish_cuda_measurement(started)
        condition_rows[name] = rows

    output_dir.mkdir(parents=True)
    baseline_output = [
        {
            **{key: value for key, value in record.items() if key != "input_ids"},
            "input_ids": serialized_input_ids(record["input_ids"]),
            "success": record["baseline_success"],
            "response": record["baseline_response"],
            "response_token_ids": record["baseline_response_token_ids"],
            "generation_failed": False,
            "source_reused": True,
        }
        for record in records
    ]
    write_jsonl(output_dir / "evaluation110_baseline.jsonl", baseline_output)
    for name, rows in condition_rows.items():
        write_jsonl(output_dir / f"condition_{name.lower()}_results.jsonl", rows)
    baseline_summary = {
        "prompt_count": len(baseline_output),
        "attack_success_count": sum(row["success"] for row in baseline_output),
        "asr": sum(row["success"] for row in baseline_output) / len(baseline_output),
        "generation_failure_count": 0,
        "source_reused": True,
        "runtime_for_fresh_first_logits": baseline_runtime,
    }
    primary = {
        name: summarize_condition(rows, baseline_output)
        for name, rows in condition_rows.items()
    }
    sensitivity_baseline = [
        row for row in baseline_output if row["dataset_index"] != MALFORMED_INDEX
    ]
    sensitivity = {
        name: summarize_condition(
            [row for row in rows if row["dataset_index"] != MALFORMED_INDEX],
            sensitivity_baseline,
        )
        for name, rows in condition_rows.items()
    }
    comparison_summary = {
        "schema_version": 1,
        "baseline": baseline_summary,
        "conditions": CONDITIONS,
        "primary_evaluation110": primary,
        "sensitivity_evaluation109_excluding_index10": {
            "baseline": {
                "prompt_count": len(sensitivity_baseline),
                "attack_success_count": sum(
                    row["success"] for row in sensitivity_baseline
                ),
                "asr": sum(row["success"] for row in sensitivity_baseline)
                / len(sensitivity_baseline),
            },
            "conditions": sensitivity,
        },
    }
    write_json(output_dir / "comparison_summary.json", comparison_summary)
    artifact_names = [
        "evaluation110_baseline.jsonl",
        "condition_a_results.jsonl",
        "condition_b_results.jsonl",
        "condition_c_results.jsonl",
        "condition_d_results.jsonl",
        "comparison_summary.json",
    ]
    run_manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": utc_now(),
        "execution_git_head": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "execution_git_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "execution_git_status_porcelain": git_value(
            REPO_ROOT, "status", "--porcelain"
        ).splitlines(),
        "model": str(model_path),
        "model_repository_head": MODEL_REPOSITORY_HEAD,
        "model_config": llama_model_config_summary(model),
        "format": OFFICIAL_FORMAT,
        "autodan_repository_commit": AUTODAN_OFFICIAL_REPOSITORY_COMMIT,
        "generation_settings": {
            "torch_dtype": "torch.float16",
            "attention_implementation": "eager",
            "model_eval": True,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": 32,
            "use_cache": False,
            "batch_size": 1,
            "seed": SEED,
        },
        "success_checker": "current 43 case-sensitive refusal substrings",
        "evaluation_indices": evaluation_indices,
        "malformed_index10_retained_in_primary": True,
        "condition_a_response_source": {
            "path": str(Path(args.condition_a_run).resolve()),
            "hashes": dict(OFFICIAL_CONDITION_A_HASHES),
        },
        "official_baseline_source": {
            "path": str(Path(args.official_baseline_dir).resolve()),
            "hashes": dict(OFFICIAL_BASELINE_HASHES),
        },
        "autodan_selection_artifact_dir": str(
            Path(args.selection_artifact_dir).resolve()
        ),
        "autodan_selection_artifact_manifest_sha256": sha256_file(
            Path(args.selection_artifact_dir).resolve() / "artifact_manifest.json"
        ),
        "autodan_score_metadata": autodan_metadata,
        "gcg_artifact_verification": gcg_verification,
        "gcg_score_hashes": {
            name: sha256_file(path) for name, path in gcg_score_paths.items()
        },
        "conditions": CONDITIONS,
        "runtime": {
            "baseline_first_logits": baseline_runtime,
            "conditions": condition_runtimes,
        },
        "validation": {
            "evaluation_count": len(records),
            "condition_input_ids_match_count": {
                name: sum(row["input_ids_match_baseline"] for row in rows)
                for name, rows in condition_rows.items()
            },
            "runtime_invariant_pass_count": {
                name: len(rows) for name, rows in condition_rows.items()
            },
            "generation_failure_count": {
                name: sum(row["generation_failed"] for row in rows)
                for name, rows in condition_rows.items()
            },
        },
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    artifact_names.append("run_manifest.json")
    artifact_manifest = write_artifact_manifest(output_dir, artifact_names)
    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "comparison_summary": comparison_summary,
        "artifact_manifest_sha256": sha256_file(
            output_dir / "artifact_manifest.json"
        ),
        "artifact_count": len(artifact_manifest["files"]),
    }


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "cpu-validate":
        result = cpu_validate(args)
    elif args.stage == "build-selection":
        result = build_selection(args)
    else:
        result = evaluate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
