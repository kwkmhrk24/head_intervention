"""CPU-only correlations, numerically identical to the legacy implementation.

Copied without changing operation order from
scripts/analyze_attention_head_metric_relationships.py at 5a5fe74.
The legacy tensor-loading module remains unchanged.
"""
from typing import Sequence
import numpy as np
import pandas as pd


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.size != y_values.size or x_values.size < 2:
        raise ValueError("Correlation inputs must have equal length >= 2")
    if np.std(x_values) == 0 or np.std(y_values) == 0:
        raise ValueError("Correlation is undefined for a constant metric")
    return float(np.corrcoef(x_values, y_values)[0, 1])


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy(np.float64)
    y_rank = pd.Series(y).rank(method="average").to_numpy(np.float64)
    return pearson_correlation(x_rank, y_rank)
