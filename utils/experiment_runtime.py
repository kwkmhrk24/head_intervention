"""Reproducible runtime helpers shared by model-backed experiment entrypoints."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, Torch, and every visible CUDA device."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_value(repo_root: str | Path, *arguments: str) -> str:
    """Return stripped output from a required Git query."""

    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_head(repo_root: str | Path) -> str:
    """Return the commit recorded in experiment manifests."""

    return git_value(repo_root, "rev-parse", "HEAD")


def git_state(repo_root: str | Path) -> tuple[str, str]:
    """Return the commit and current branch recorded in manifests."""

    return (
        git_head(repo_root),
        git_value(repo_root, "branch", "--show-current"),
    )
