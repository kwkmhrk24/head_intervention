#!/usr/bin/env python3
"""Create the CPU-only Selection20 Rank Distribution Shift Cancellation v1 artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_gcg_reference_factorization import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MODEL,
)
from scripts.prepare_gcg_rank_element_topk import load_source_scores  # noqa: E402
from utils.experiment_runtime import git_value  # noqa: E402
from utils.gcg_rank_distribution_shift_cancellation import (  # noqa: E402
    ALL_RANK_COUNT,
    ALPHA_DEFAULT,
    METHOD,
    SHIFT_ARTIFACT_FILES,
    compute_mean_shift,
    load_shift_artifact,
    offline_shift_crossing_diagnostics,
    validate_shift_payload,
)
from utils.gcg_rank_element_topk_intervention import RANK_SORT, RANK_TIE_BREAK  # noqa: E402
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
    write_declared_artifact_manifest,
    write_json,
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


def build_config(*, model_path: Path, source_dir: Path, metadata: dict, verification: dict, paths: dict) -> dict:
    conditions = {
        "model": "Llama-2-7B-Chat",
        "selection_data": "Selection20 paired Original/GCG only",
        "window": "tail16",
        "component": "S_asym=(S-S^T)/2",
        "triangle": "strict_lower",
        "elements_per_head": ELEMENTS_PER_HEAD,
        "layers": NUM_LAYERS,
        "heads_per_layer": NUM_HEADS,
        "rank_sort": RANK_SORT,
        "rank_tie_break": RANK_TIE_BREAK,
        "selection_rule": "all_rank_elements; no Top-k ranking",
        "alpha_default": ALPHA_DEFAULT,
        "intervention_timing": "first_token_only",
        "score_capture_position": metadata["score_capture_position"],
        "score_calculation_dtype": metadata.get("score_calculation_dtype", "unknown"),
        "stored_value_dtype": metadata["stored_value_dtype"],
        "raw_score_intervention": True,
        "generation_and_asr": "unchanged from Rank/Position Element Top-k",
        "crossing_policy": "diagnose_only; no clipping, isotonic regression, correction or re-sort",
    }
    return {
        "schema_version": 1,
        "method": METHOD,
        "created_at": utc_now(),
        "execution_git_head": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "execution_git_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "execution_git_status_porcelain": git_value(REPO_ROOT, "status", "--porcelain").splitlines(),
        "preparation_source_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_source_sha256": sha256_file(
            REPO_ROOT / "utils/gcg_rank_distribution_shift_cancellation.py"
        ),
        "rank_runtime_source_sha256": sha256_file(
            REPO_ROOT / "utils/gcg_rank_element_topk_intervention.py"
        ),
        "model": str(model_path),
        "model_config": metadata["model_config"],
        "source_artifact_dir": str(source_dir),
        "source_artifact_verified": verification["verified"],
        "source_artifact_manifest_sha256": sha256_file(source_dir / "artifact_manifest.json"),
        "source_selection_manifest_sha256": verification["selection_manifest_sha256"],
        "source_score_artifact_sha256": verification["score_artifact_sha256"],
        "original_score_sha256": sha256_file(paths["original"]),
        "jailbreak_score_sha256": sha256_file(paths["jailbreak"]),
        "selection_row_positions": metadata["selection_row_positions"],
        "selection_dataset_indices": metadata["selection_dataset_indices"],
        "selection_prompt_count": SELECTION_COUNT,
        "tail_tokens": TAIL_TOKENS,
        "candidate_rank_element_count": ALL_RANK_COUNT,
        "rank_sort": RANK_SORT,
        "rank_tie_break": RANK_TIE_BREAK,
        "alpha_default": ALPHA_DEFAULT,
        "shift_formula": "D=mean_p(sorted_GCG-sorted_original)",
        "runtime_formula": "Q_prime=Q-alpha*D",
        "scientific_conditions": conditions,
        "runtime_mapping": "stable sort once; retain rank-to-position mapping; scatter without re-sort",
        "symmetric_component_preservation": "mirror strict-lower raw-S delta to upper with opposite sign",
        "data_usage": {
            "selection20_used_for_shift": True,
            "holdout48_used_for_shift": False,
            "independent100_used": False,
            "parameter_tuning": False,
        },
    }


def main(argv=None):
    args = parse_args(argv)
    model_path = Path(args.model).resolve()
    source_dir = Path(args.source_artifact_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_output_directory_absent(output_dir)
    original, gcg, metadata, verification, paths = load_source_scores(
        source_dir, model_path=model_path
    )
    mean_shift = compute_mean_shift(original, gcg)
    config = build_config(
        model_path=model_path,
        source_dir=source_dir,
        metadata=metadata,
        verification=verification,
        paths=paths,
    )
    metadata_fields = (
        "source_selection_manifest_sha256",
        "source_score_artifact_sha256",
        "original_score_sha256",
        "jailbreak_score_sha256",
        "selection_prompt_count",
        "rank_sort",
        "rank_tie_break",
        "alpha_default",
        "scientific_conditions",
    )
    payload = {
        "schema_version": 1,
        "method": METHOD,
        "mean_shift": mean_shift,
        "metadata": {field: config[field] for field in metadata_fields},
    }
    validate_shift_payload(payload)
    config["offline_selection20_crossing_diagnostics"] = {
        "original": offline_shift_crossing_diagnostics(original, mean_shift),
        "gcg": offline_shift_crossing_diagnostics(gcg, mean_shift),
        "purpose": "development diagnostic only; no tuning",
    }
    output_dir.mkdir(parents=True)
    save_torch_artifact(output_dir / "mean_shift.pt", payload)
    write_json(output_dir / "shift_config.json", config)
    write_declared_artifact_manifest(
        output_dir=output_dir,
        file_names=SHIFT_ARTIFACT_FILES,
        metadata={
            "experiment": METHOD,
            "git_head": config["execution_git_head"],
            "git_branch": config["execution_git_branch"],
            "source_selection_manifest_sha256": config["source_selection_manifest_sha256"],
            "source_score_artifact_sha256": config["source_score_artifact_sha256"],
        },
    )
    loaded, _, artifact_verification = load_shift_artifact(output_dir)
    if not loaded.equal(mean_shift):
        raise AssertionError("saved shift differs from computed Selection20 shift")
    print(json.dumps({
        "status": "complete_validated",
        "model_loaded": False,
        "gpu_used": False,
        "output_dir": str(output_dir),
        "mean_shift_shape": list(mean_shift.shape),
        "artifact_verification": artifact_verification,
        "offline_selection20_crossing_diagnostics": config["offline_selection20_crossing_diagnostics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
