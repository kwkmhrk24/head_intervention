#!/usr/bin/env python3
"""Analyze prompt-level transitions across saved multidataset evaluation runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.intervention_transitions import analyze_runs, write_outputs  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    analysis = analyze_runs(
        args.run_dir,
        seed=args.seed,
        resamples=args.bootstrap_resamples,
    )
    write_outputs(args.output_dir, analysis)
    print(f"Wrote cross-run transition analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
