#!/usr/bin/env python3
"""Audit completed GCG Wasserstein-reference artifacts and evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.gcg_reference_audit import (  # noqa: E402
    audit_evaluation,
    audit_first_token,
    summarize_artifacts,
    verify_artifacts,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("artifacts", "evaluation", "first-token", "summary"):
        subparser = subparsers.add_parser(mode)
        subparser.add_argument("--run-dir", required=True)
        if mode == "summary":
            subparser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "artifacts":
        result = verify_artifacts(args.run_dir)
    elif args.mode == "evaluation":
        result = audit_evaluation(args.run_dir)
    elif args.mode == "first-token":
        result = audit_first_token(args.run_dir)
    else:
        result = summarize_artifacts(args.run_dir, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
