#!/usr/bin/env python3
"""Create the CPU-only GCG Rank Element Top-500 v1 selection artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_gcg_reference_factorization import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MODEL,
    EXPECTED_SCORE_ARTIFACT_SHA256,
    EXPECTED_SELECTION_MANIFEST_SHA256,
)
from utils.experiment_runtime import git_value  # noqa: E402
from utils.gcg_rank_element_topk_intervention import (  # noqa: E402
    RANK_ARTIFACT_FILES,
    RANK_SORT,
    RANK_TIE_BREAK,
    RANKING_METRIC,
    TOP_K,
    TOP_K_VALUES,
    build_rank_element_selection,
    load_rank_selection_artifact,
    offline_rank_crossing_diagnostics,
    selection_diagnostics,
    validate_rank_selection_payload,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    TAIL_TOKENS,
    ensure_output_directory_absent,
    save_torch_artifact,
    sha256_file,
    utc_now,
    verify_artifact_directory,
    write_declared_artifact_manifest,
    write_json,
    write_jsonl,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--source-artifact-dir",
        default=DEFAULT_ARTIFACT_DIR,
        help="Audited Selection20 raw-score artifact directory.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def load_source_scores(source_dir: Path, *, model_path: Path):
    """Load only the audited Selection20 score tensors used for selection."""

    verification = verify_artifact_directory(source_dir)
    if verification.get("verified") is not True:
        raise ValueError("source artifact directory verification failed")
    if (
        verification.get("selection_manifest_sha256")
        != EXPECTED_SELECTION_MANIFEST_SHA256
    ):
        raise ValueError("selection manifest hash mismatch")
    if (
        verification.get("score_artifact_sha256")
        != EXPECTED_SCORE_ARTIFACT_SHA256
    ):
        raise ValueError("score artifact hash mismatch")
    paths = {
        "original": source_dir / "selection_original_scores.pt",
        "jailbreak": source_dir / "selection_jailbreak_scores.pt",
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
    if (
        not isinstance(original_metadata, dict)
        or original_metadata != jailbreak_metadata
    ):
        raise ValueError("original/jailbreak score metadata mismatch")
    if Path(original_metadata.get("model_path", "")).resolve() != model_path:
        raise ValueError("selection score model path mismatch")
    if original_metadata.get("selection_manifest_sha256") != (
        EXPECTED_SELECTION_MANIFEST_SHA256
    ):
        raise ValueError("selection score split hash mismatch")
    return (
        payloads["original"].get("values"),
        payloads["jailbreak"].get("values"),
        original_metadata,
        verification,
        paths,
    )


def build_config(
    *,
    model_path: Path,
    source_dir: Path,
    metadata: dict,
    verification: dict,
    paths: dict[str, Path],
) -> dict:
    return {
        "schema_version": 1,
        "method": "rank_element_topk_v1",
        "created_at": utc_now(),
        "execution_git_head": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "execution_git_branch": git_value(
            REPO_ROOT, "branch", "--show-current"
        ),
        "execution_git_status_porcelain": git_value(
            REPO_ROOT, "status", "--porcelain"
        ).splitlines(),
        "preparation_source_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_source_sha256": sha256_file(
            REPO_ROOT / "utils/gcg_rank_element_topk_intervention.py"
        ),
        "model": str(model_path),
        "model_config": metadata["model_config"],
        "source_artifact_dir": str(source_dir),
        "source_artifact_verified": verification["verified"],
        "source_artifact_manifest_sha256": sha256_file(
            source_dir / "artifact_manifest.json"
        ),
        "source_selection_manifest_sha256": verification[
            "selection_manifest_sha256"
        ],
        "source_score_artifact_sha256": verification[
            "score_artifact_sha256"
        ],
        "original_score_sha256": sha256_file(paths["original"]),
        "jailbreak_score_sha256": sha256_file(paths["jailbreak"]),
        "selection_row_positions": metadata["selection_row_positions"],
        "selection_dataset_indices": metadata["selection_dataset_indices"],
        "selection_prompt_count": SELECTION_COUNT,
        "candidate_rank_element_count": (
            NUM_LAYERS * NUM_HEADS * ELEMENTS_PER_HEAD
        ),
        "rank_sort": RANK_SORT,
        "rank_tie_break": RANK_TIE_BREAK,
        "rank_numbering": "one_based_1_to_120",
        "ranking_metric": RANKING_METRIC,
        "k": TOP_K,
        "reference": (
            "mean over Selection20 original independently signed-sorted "
            "S_asym strict-lower values at each rank"
        ),
        "scientific_conditions": {
            "model": "Llama-2-7B-Chat",
            "selection_data": "Selection20",
            "window": "tail16",
            "component": "S_asym=(S-S^T)/2",
            "triangle": "strict_lower",
            "rank_sort": "signed_value_ascending",
            "ranking": "abs(mean(GCG-original))",
            "k": TOP_K,
            "alpha": 1.0,
            "intervention_timing": "first_token_only",
            "score_capture_position": metadata["score_capture_position"],
            "score_calculation_dtype": metadata.get(
                "score_calculation_dtype", "unknown"
            ),
            "stored_value_dtype": metadata["stored_value_dtype"],
            "generation_and_asr": "unchanged from Position Element Top-k",
        },
        "runtime_mapping": (
            "sort current signed values once, retain rank-to-position mapping, "
            "replace selected ranks, write to pre-intervention positions, do "
            "not re-sort"
        ),
        "symmetric_component_preservation": (
            "mirror every lower raw-score delta to upper with opposite sign"
        ),
        "crossing_policy": "diagnose_only; no correction and no re-sorting",
        "data_usage": {
            "selection20_used": True,
            "development_or_validation_only": True,
            "advbench_220_319_independent100_used": False,
        },
    }


def main(argv=None):
    args = parse_args(argv)
    model_path = Path(args.model).resolve()
    source_dir = Path(args.source_artifact_dir).resolve()
    output_dir = ensure_output_directory_absent(args.output_dir).resolve()
    original, jailbreak, metadata, verification, paths = load_source_scores(
        source_dir, model_path=model_path
    )
    selection = build_rank_element_selection(
        original,
        jailbreak,
        k_values=TOP_K_VALUES,
    )
    diagnostics = selection_diagnostics(selection)
    diagnostics["offline_selection20_crossing_diagnostics"] = {
        "original": offline_rank_crossing_diagnostics(
            original, selection, k=TOP_K, alpha=1.0
        ),
        "gcg": offline_rank_crossing_diagnostics(
            jailbreak, selection, k=TOP_K, alpha=1.0
        ),
        "selection_use": "diagnostic_only_not_used_for_ranking_or_selection",
    }
    selected = [dict(row) for row in selection.topk_records(TOP_K)]
    config = build_config(
        model_path=model_path,
        source_dir=source_dir,
        metadata=metadata,
        verification=verification,
        paths=paths,
    )
    selected_features = torch.tensor(
        [
            [
                row["selection_rank"],
                row["layer"],
                row["head"],
                row["rank"],
            ]
            for row in selected
        ],
        dtype=torch.int64,
    )
    tensor_payload = {
        "schema_version": 1,
        "method": "rank_element_topk_v1",
        "k": TOP_K,
        "rank_mask": selection.mask_for_k(TOP_K),
        "original_rank_reference": selection.original_rank_reference,
        "selected_features": selected_features,
        "metadata": {
            key: config[key]
            for key in (
                "source_selection_manifest_sha256",
                "source_score_artifact_sha256",
                "original_score_sha256",
                "jailbreak_score_sha256",
                "selection_row_positions",
                "selection_dataset_indices",
                "rank_sort",
                "rank_tie_break",
                "ranking_metric",
            )
        },
    }
    validate_rank_selection_payload(tensor_payload)

    ensure_output_directory_absent(output_dir)
    output_dir.mkdir(parents=True)
    save_torch_artifact(output_dir / "rank_selection.pt", tensor_payload)
    write_jsonl(output_dir / "selected_rank_elements.jsonl", selected)
    write_json(output_dir / "selection_diagnostics.json", diagnostics)
    write_json(output_dir / "selection_config.json", config)
    write_declared_artifact_manifest(
        output_dir=output_dir,
        file_names=RANK_ARTIFACT_FILES,
        metadata={
            "experiment": "llama2_gcg_rank_element_topk_v1",
            "git_head": config["execution_git_head"],
            "git_branch": config["execution_git_branch"],
            "source_selection_manifest_sha256": config[
                "source_selection_manifest_sha256"
            ],
            "source_score_artifact_sha256": config[
                "source_score_artifact_sha256"
            ],
            "k": TOP_K,
        },
    )
    _, _, _, artifact_verification = load_rank_selection_artifact(output_dir)
    result = {
        "status": "complete_validated",
        "model_loaded": False,
        "gpu_used": False,
        "output_dir": str(output_dir),
        "artifact_verification": artifact_verification,
        "selection": diagnostics,
        "top_rank_elements": selected[:10],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
