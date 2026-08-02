#!/usr/bin/env python3
"""Print the ranking, reference checks, split counts, and optional refusal summary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.gcg_wasserstein_reference import (  # noqa: E402
    read_json,
    validate_reference_payload,
    verify_artifact_directory,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    run_dir = Path(args.run_dir)
    verification = verify_artifact_directory(run_dir)
    with (run_dir / "head_wasserstein_ranking.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        ranking = list(csv.DictReader(handle))
    reference = torch.load(
        run_dir / "original_reference.pt",
        map_location="cpu",
        weights_only=True,
    )
    validate_reference_payload(reference)
    mean = reference["mean"]
    std = reference["std"]
    sem = reference["sem"]
    split = read_json(run_dir / "split_manifest.json")
    refusal_path = run_dir / "selection_original_refusal_summary.json"
    refusal = read_json(refusal_path) if refusal_path.is_file() else None

    payload = {
        "verification": verification,
        "top_ranking": [
            {
                "rank": int(row["rank"]),
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "mean": float(row["mean"]),
                "median": float(row["median"]),
                "std": float(row["std"]),
                "sem": float(row["sem"]),
                "min": float(row["min"]),
                "max": float(row["max"]),
            }
            for row in ranking[: args.top_k]
        ],
        "top5_heads": [
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "mean": float(row["mean"]),
                "median": float(row["median"]),
                "std": float(row["std"]),
            }
            for row in ranking[:5]
        ],
        "reference": {
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
            "sem_shape": list(sem.shape),
            "dtype": str(mean.dtype),
            "finite": all(
                bool(torch.isfinite(value).all().item())
                for value in (mean, std, sem)
            ),
            "mean_tensor_mean": float(mean.mean().item()),
            "mean_tensor_std": float(mean.std(correction=1).item()),
            "mean_tensor_min": float(mean.min().item()),
            "mean_tensor_max": float(mean.max().item()),
            "std_tensor_mean": float(std.mean().item()),
            "std_tensor_min": float(std.min().item()),
            "std_tensor_max": float(std.max().item()),
        },
        "split": {
            "selection_count": split["selection_count"],
            "evaluation_count": split["evaluation_count"],
            "baseline_success_count_selection": split[
                "baseline_success_count_selection"
            ],
            "baseline_success_count_evaluation": split[
                "baseline_success_count_evaluation"
            ],
            "baseline_asr_evaluation": split["baseline_asr_evaluation"],
        },
        "original_refusal_diagnostic": (
            {
                "selection_count": refusal["selection_count"],
                "refusal_count": refusal["refusal_count"],
                "non_refusal_count": refusal["non_refusal_count"],
                "generation_failure_count": refusal[
                    "generation_failure_count"
                ],
                "refusal_dataset_indices": refusal[
                    "refusal_dataset_indices"
                ],
                "non_refusal_dataset_indices": refusal[
                    "non_refusal_dataset_indices"
                ],
            }
            if refusal is not None
            else "not run"
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
