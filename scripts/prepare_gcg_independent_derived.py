#!/usr/bin/env python3
"""Materialize and validate the frozen AdvBench 220..319 GCG test set."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.artifact_io import (  # noqa: E402
    ensure_output_directory_absent,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.harmbench_gcg_independent_test import (  # noqa: E402
    example_output_path,
    git_capture,
    load_fast_tokenizer,
    load_or_skip_completed_example,
    scientific_fingerprint,
    validate_fixed_config,
    validate_harmbench_revision,
    validate_input_manifest,
    validate_model_revision,
    verify_all_configured_sources,
)


DEFAULT_SOURCE_RUN = (
    REPO_ROOT
    / "data/outputs/gcg_independent_test/llama2/advbench_220_319/"
    "gcg_v1_55c5098_20260910T011413Z"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data/derived/llama2/gcg/independent_advbench_220_319"
)
DEFAULT_CONFIG = REPO_ROOT / "configs/gcg_independent_test_v1/attack_config.json"
DEFAULT_INPUT_MANIFEST = (
    REPO_ROOT / "configs/gcg_independent_test_v1/input_manifest.json"
)
DEFAULT_PROVENANCE = (
    REPO_ROOT / "configs/gcg_independent_test_v1/provenance_manifest.json"
)
DEFAULT_ADAPTER = REPO_ROOT / "scripts/generate_harmbench_gcg_independent_test.py"
EXPECTED_INDICES = tuple(range(220, 320))
EXPECTED_FILES = {
    "original_prompts.jsonl",
    "jailbreak_prompts.jsonl",
    "metadata.jsonl",
    "manifest.json",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "validate"), default="validate")
    parser.add_argument("--source-run", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--provenance", default=str(DEFAULT_PROVENANCE))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    return parser.parse_args(argv)


def repo_relative(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def require_exact_indices(rows: Sequence[Mapping[str, object]], key: str) -> None:
    observed = [row.get(key) for row in rows]
    if tuple(observed) != EXPECTED_INDICES:
        raise ValueError(
            f"{key} must be the unique ascending AdvBench indices 220..319"
        )


def load_validated_source(args) -> tuple[dict, list[dict], dict]:
    config_path = Path(args.config).resolve()
    input_manifest_path = Path(args.input_manifest).resolve()
    provenance_path = Path(args.provenance).resolve()
    adapter_path = Path(args.adapter).resolve()
    source_run = Path(args.source_run).resolve()
    config = read_json(config_path)
    validate_fixed_config(config)
    verify_all_configured_sources(config)
    validate_harmbench_revision(config)
    validate_model_revision(config)
    source_records = validate_input_manifest(
        read_json(input_manifest_path), config
    )
    tokenizer = load_fast_tokenizer(config)
    fingerprint = scientific_fingerprint(
        config_path, input_manifest_path, adapter_path
    )
    final_manifest_path = source_run / "final_manifest.json"
    run_manifest_path = source_run / "run_manifest.json"
    final_manifest = read_json(final_manifest_path)
    for key, expected in (
        ("status", "complete_validated"),
        ("complete_count", 100),
        ("missing_count", 0),
        ("duplicate_count", 0),
        ("processing_failure_history_count", 0),
    ):
        if final_manifest.get(key) != expected:
            raise ValueError(
                f"source final manifest {key} mismatch: "
                f"{final_manifest.get(key)!r} != {expected!r}"
            )
    for key, expected in fingerprint.items():
        if final_manifest.get(key) != expected:
            raise ValueError(f"source final manifest fingerprint mismatch: {key}")
    manifest_examples = final_manifest.get("examples")
    if not isinstance(manifest_examples, list) or len(manifest_examples) != 100:
        raise ValueError("source final manifest must contain 100 example records")
    manifest_by_index = {row.get("dataset_index"): row for row in manifest_examples}
    if set(manifest_by_index) != set(EXPECTED_INDICES):
        raise ValueError("source final manifest example index set mismatch")

    payloads = []
    for record in source_records:
        payload = load_or_skip_completed_example(
            source_run, record, fingerprint, tokenizer=tokenizer
        )
        if payload is None:
            raise ValueError(
                f"missing source output: dataset_index={record['dataset_index']}"
            )
        source_path = example_output_path(source_run, record["behavior_id"])
        if sha256_file(source_path) != manifest_by_index[record["dataset_index"]].get(
            "output_sha256"
        ):
            raise ValueError(
                f"source output hash mismatch: dataset_index={record['dataset_index']}"
            )
        payloads.append(payload)
    require_exact_indices(payloads, "dataset_index")
    if len({row["behavior_id"] for row in payloads}) != 100:
        raise ValueError("source behavior IDs are not unique")

    provenance = {
        "source_run_directory": repo_relative(source_run),
        "source_final_manifest_sha256": sha256_file(final_manifest_path),
        "source_run_manifest_sha256": sha256_file(run_manifest_path),
        "attack_config_path": repo_relative(config_path),
        "attack_config_sha256": sha256_file(config_path),
        "attack_input_manifest_path": repo_relative(input_manifest_path),
        "attack_input_manifest_sha256": sha256_file(input_manifest_path),
        "attack_provenance_manifest_path": repo_relative(provenance_path),
        "attack_provenance_manifest_sha256": sha256_file(provenance_path),
        "attack_adapter_path": repo_relative(adapter_path),
        "attack_adapter_sha256": sha256_file(adapter_path),
        "attack_adapter_utility_sha256": fingerprint["adapter_utility_sha256"],
    }
    return config, payloads, provenance


def build_rows(payloads: Sequence[Mapping[str, object]], source_run: Path):
    original_rows = []
    jailbreak_rows = []
    metadata_rows = []
    for payload in payloads:
        index = payload["dataset_index"]
        behavior = payload["behavior"]
        suffix = payload["final_suffix_text"]
        jailbreak = payload["test_case"]
        if jailbreak != f"{behavior} {suffix}":
            raise ValueError(f"prompt reconstruction mismatch: dataset_index={index}")
        source_path = example_output_path(source_run, payload["behavior_id"])
        original_rows.append({"index": index, "prompt": behavior})
        jailbreak_rows.append({"index": index, "prompt": jailbreak})
        metadata_rows.append(
            {
                "index": index,
                "behavior_id": payload["behavior_id"],
                "seed": payload["example_seed"],
                "final_suffix": suffix,
                "final_suffix_token_ids": payload["final_suffix_token_ids"],
                "updates": payload["updates_completed"],
                "termination_reason": payload["termination_reason"],
                "source_generation_file": repo_relative(source_path),
                "source_generation_file_sha256": sha256_file(source_path),
                "source_record_sha256": payload["record_sha256"],
            }
        )
    require_exact_indices(original_rows, "index")
    require_exact_indices(jailbreak_rows, "index")
    require_exact_indices(metadata_rows, "index")
    return original_rows, jailbreak_rows, metadata_rows


def expected_manifest_core(config: Mapping[str, object], provenance: dict) -> dict:
    return {
        "schema_version": 1,
        "dataset_id": "gcg_independent_advbench_220_319_v1",
        "role": "final_independent_test",
        "attack": "GCG",
        "model": {
            "name": config["model"]["model_name"],
            "local_path": config["model"]["local_path"],
            "revision": config["model"]["revision"],
        },
        "advbench_index_start": 220,
        "advbench_index_end_inclusive": 319,
        "count": 100,
        "ordering": "ascending_advbench_index",
        "prompt_construction": "jailbreak prompt is exact source behavior + one ASCII space + final GCG suffix",
        "filtering": "none; retain all 100 regardless of attack outcome",
        **provenance,
    }


def validate_derived(output_dir: Path, args) -> dict:
    config, payloads, provenance = load_validated_source(args)
    observed_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    if observed_files != EXPECTED_FILES:
        raise ValueError(
            f"derived file set mismatch: {sorted(observed_files)} != "
            f"{sorted(EXPECTED_FILES)}"
        )
    expected_original, expected_jailbreak, expected_metadata = build_rows(
        payloads, Path(args.source_run).resolve()
    )
    actual_original = read_jsonl(output_dir / "original_prompts.jsonl")
    actual_jailbreak = read_jsonl(output_dir / "jailbreak_prompts.jsonl")
    actual_metadata = read_jsonl(output_dir / "metadata.jsonl")
    if actual_original != expected_original:
        raise ValueError("derived original prompts differ from source behaviors")
    if actual_jailbreak != expected_jailbreak:
        raise ValueError("derived jailbreak prompts differ from source test cases")
    if actual_metadata != expected_metadata:
        raise ValueError("derived metadata differs from source generation records")
    manifest = read_json(output_dir / "manifest.json")
    for key, expected in expected_manifest_core(config, provenance).items():
        if manifest.get(key) != expected:
            raise ValueError(f"derived manifest mismatch: {key}")
    expected_hashes = {
        name: sha256_file(output_dir / name)
        for name in (
            "original_prompts.jsonl",
            "jailbreak_prompts.jsonl",
            "metadata.jsonl",
        )
    }
    if manifest.get("derived_files_sha256") != expected_hashes:
        raise ValueError("derived file hash manifest mismatch")
    validation = manifest.get("validation")
    expected_validation = {
        "status": "passed",
        "count": 100,
        "missing_index_count": 0,
        "duplicate_index_count": 0,
        "seed_formula": "42 + AdvBench index",
        "seed_mismatch_count": 0,
        "prompt_reproduction_mismatch_count": 0,
        "source_output_hash_mismatch_count": 0,
    }
    if validation != expected_validation:
        raise ValueError("derived validation declaration mismatch")
    return {
        "status": "derived_validation_passed",
        "output_dir": str(output_dir),
        "count": 100,
        "index_bounds": [220, 319],
        "missing_index_count": 0,
        "duplicate_index_count": 0,
        "prompt_reproduction_mismatch_count": 0,
        "derived_files_sha256": expected_hashes,
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
    }


def build_derived(output_dir: Path, args) -> dict:
    ensure_output_directory_absent(output_dir)
    config, payloads, provenance = load_validated_source(args)
    original_rows, jailbreak_rows, metadata_rows = build_rows(
        payloads, Path(args.source_run).resolve()
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        write_jsonl(temporary / "original_prompts.jsonl", original_rows)
        write_jsonl(temporary / "jailbreak_prompts.jsonl", jailbreak_rows)
        write_jsonl(temporary / "metadata.jsonl", metadata_rows)
        derived_hashes = {
            name: sha256_file(temporary / name)
            for name in (
                "original_prompts.jsonl",
                "jailbreak_prompts.jsonl",
                "metadata.jsonl",
            )
        }
        manifest = {
            **expected_manifest_core(config, provenance),
            "created_at": utc_now(),
            "builder": {
                "path": repo_relative(__file__),
                "sha256": sha256_file(__file__),
                "git": git_capture(REPO_ROOT),
            },
            "schemas": {
                "original_prompts.jsonl": {"fields": ["index", "prompt"]},
                "jailbreak_prompts.jsonl": {"fields": ["index", "prompt"]},
                "metadata.jsonl": {
                    "fields": [
                        "index",
                        "behavior_id",
                        "seed",
                        "final_suffix",
                        "final_suffix_token_ids",
                        "updates",
                        "termination_reason",
                        "source_generation_file",
                        "source_generation_file_sha256",
                        "source_record_sha256",
                    ]
                },
            },
            "derived_files_sha256": derived_hashes,
            "validation": {
                "status": "passed",
                "count": 100,
                "missing_index_count": 0,
                "duplicate_index_count": 0,
                "seed_formula": "42 + AdvBench index",
                "seed_mismatch_count": sum(
                    row["seed"] != 42 + row["index"] for row in metadata_rows
                ),
                "prompt_reproduction_mismatch_count": 0,
                "source_output_hash_mismatch_count": 0,
            },
        }
        write_json(temporary / "manifest.json", manifest)
        validate_derived(temporary, args)
        os.rename(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_derived(output_dir, args)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    result = (
        build_derived(output_dir, args)
        if args.stage == "build"
        else validate_derived(output_dir, args)
    )
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
