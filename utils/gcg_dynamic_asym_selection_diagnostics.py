"""Leave-one-out selection diagnostics for full-prompt dynamic Asym scores."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager

import torch

from utils.gcg_dynamic_asym_intervention import (
    NUMERICAL_SIGMA_FLOOR,
    _patch_llama_attention_modules,
    _restore_llama_attention_modules,
    compute_equal_prompt_reference_statistics,
)
from utils.gcg_wasserstein_reference import strict_lower_indices
from utils.reference_asym_score_intervention import ReferenceInterventionRuntimeError


FIXED_TOP_N = 500
TOTAL_SELECTION_BUDGET = 20 * FIXED_TOP_N
QUANTILES = (0.90, 0.95, 0.99, 0.995, 0.999)
QUANTILE_NAMES = ("q90", "q95", "q99", "q995", "q999")


def candidate_count(
    prompt_length: int,
    *,
    layer_count: int = 32,
    head_count: int = 32,
) -> int:
    if type(prompt_length) is not int or prompt_length < 2:
        raise ValueError("prompt_length must be an integer >= 2")
    if type(layer_count) is not int or layer_count <= 0:
        raise ValueError("layer_count must be a positive integer")
    if type(head_count) is not int or head_count <= 0:
        raise ValueError("head_count must be a positive integer")
    return layer_count * head_count * prompt_length * (prompt_length - 1) // 2


def proportional_top_n_allocation(
    candidate_counts: Sequence[int],
    *,
    total_budget: int = TOTAL_SELECTION_BUDGET,
) -> tuple[int, ...]:
    """Allocate an integer budget by exact largest-remainder apportionment."""

    if isinstance(candidate_counts, (str, bytes)) or not candidate_counts:
        raise ValueError("candidate_counts must be a non-empty sequence")
    counts = tuple(candidate_counts)
    if any(type(value) is not int or value <= 0 for value in counts):
        raise ValueError("candidate counts must be positive integers")
    if type(total_budget) is not int or total_budget <= 0:
        raise ValueError("total_budget must be a positive integer")
    total_candidates = sum(counts)
    if total_budget > total_candidates:
        raise ValueError("selection budget exceeds total candidate count")

    numerators = tuple(total_budget * value for value in counts)
    allocated = [numerator // total_candidates for numerator in numerators]
    remainders = [numerator % total_candidates for numerator in numerators]
    remaining = total_budget - sum(allocated)
    order = sorted(range(len(counts)), key=lambda index: (-remainders[index], index))
    for index in order[:remaining]:
        allocated[index] += 1
    if sum(allocated) != total_budget:
        raise AssertionError("proportional allocation budget mismatch")
    if any(value < 0 or value > count for value, count in zip(allocated, counts)):
        raise AssertionError("proportional allocation is outside candidate bounds")
    return tuple(allocated)


def compute_leave_one_out_statistics(
    prompt_means: torch.Tensor,
    prompt_mean_squares: torch.Tensor,
) -> tuple[dict, ...]:
    """Compute one 19-prompt Layer/Head reference for every excluded prompt."""

    if not isinstance(prompt_means, torch.Tensor) or prompt_means.ndim != 3:
        raise TypeError("prompt_means must have shape prompt,layer,head")
    if (
        not isinstance(prompt_mean_squares, torch.Tensor)
        or prompt_mean_squares.shape != prompt_means.shape
    ):
        raise ValueError("prompt_mean_squares shape mismatch")
    prompt_count = int(prompt_means.shape[0])
    if prompt_count < 2:
        raise ValueError("leave-one-out requires at least two prompts")
    if prompt_means.device.type != "cpu" or prompt_mean_squares.device.type != "cpu":
        raise ValueError("leave-one-out prompt moments must be on CPU")
    if not bool(torch.isfinite(prompt_means).all().item()) or not bool(
        torch.isfinite(prompt_mean_squares).all().item()
    ):
        raise ValueError("leave-one-out prompt moments contain NaN or infinity")

    all_positions = tuple(range(prompt_count))
    total_means = prompt_means.to(torch.float64).sum(dim=0)
    total_squares = prompt_mean_squares.to(torch.float64).sum(dim=0)
    results = []
    for excluded_position in all_positions:
        included_positions = tuple(
            position for position in all_positions if position != excluded_position
        )
        included_means = prompt_means[list(included_positions)]
        included_squares = prompt_mean_squares[list(included_positions)]
        statistics = compute_equal_prompt_reference_statistics(
            included_means,
            included_squares,
            expected_prompt_count=prompt_count - 1,
        )
        direct_mu = (
            total_means - prompt_means[excluded_position].to(torch.float64)
        ) / (prompt_count - 1)
        direct_mean_q = (
            total_squares
            - prompt_mean_squares[excluded_position].to(torch.float64)
        ) / (prompt_count - 1)
        torch.testing.assert_close(
            statistics["mu"], direct_mu.to(torch.float32), rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            statistics["mean_q"],
            direct_mean_q.to(torch.float32),
            rtol=1e-6,
            atol=1e-7,
        )
        if excluded_position in included_positions:
            raise AssertionError("LOO reference includes its excluded prompt")
        results.append(
            {
                "excluded_selection_position": excluded_position,
                "included_selection_positions": included_positions,
                "reference_prompt_count": prompt_count - 1,
                "self_excluded": True,
                "mu": statistics["mu"],
                "sigma": statistics["sigma"],
                "variance": statistics["variance"],
                "mean_q": statistics["mean_q"],
                "clamped_negative_variance_count": statistics[
                    "clamped_negative_variance_count"
                ],
                "sigma_floor_head_count": statistics["sigma_floor_head_count"],
            }
        )
    return tuple(results)


def distribution_summary(values: torch.Tensor) -> dict:
    if not isinstance(values, torch.Tensor) or values.ndim != 1 or values.numel() == 0:
        raise ValueError("distribution values must be a non-empty rank-1 tensor")
    if not values.is_floating_point():
        raise TypeError("distribution values must be floating point")
    nonfinite_count = int((~torch.isfinite(values)).sum().item())
    if nonfinite_count:
        raise ValueError(f"distribution contains {nonfinite_count} nonfinite values")
    values = values.detach().to(device="cpu", dtype=torch.float32).contiguous()
    quantile_levels = torch.tensor(
        (0.5, *QUANTILES), dtype=torch.float32, device="cpu"
    )
    quantile_values = torch.quantile(values, quantile_levels)
    summary = {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "median": float(quantile_values[0].item()),
        "mean": float(values.to(torch.float64).mean().item()),
        "max": float(values.max().item()),
        "nonfinite_count": 0,
    }
    for name, value in zip(QUANTILE_NAMES, quantile_values[1:]):
        summary[name] = float(value.item())
    return summary


def stable_top_indices(values: torch.Tensor, count: int) -> torch.Tensor:
    """Rank descending with lower flat candidate identity breaking ties."""

    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise TypeError("ranking values must be a rank-1 tensor")
    if type(count) is not int or count <= 0 or count > values.numel():
        raise ValueError("selection count is outside candidate bounds")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("ranking values contain NaN or infinity")
    cpu_values = values.detach().to(device="cpu", dtype=torch.float32)
    ranking = torch.argsort(cpu_values, descending=True, stable=True)
    selected = ranking[:count].to(torch.int64).contiguous()
    if selected.unique().numel() != count:
        raise AssertionError("selected candidate identities are not unique")
    return selected


def selection_count_summary(
    selected_indices: torch.Tensor,
    *,
    layer_count: int,
    head_count: int,
    elements_per_head: int,
) -> dict:
    if not isinstance(selected_indices, torch.Tensor) or selected_indices.ndim != 1:
        raise TypeError("selected_indices must be a rank-1 tensor")
    if selected_indices.dtype != torch.int64 or selected_indices.device.type != "cpu":
        raise ValueError("selected_indices must be CPU int64")
    total_candidates = layer_count * head_count * elements_per_head
    if selected_indices.numel() == 0:
        raise ValueError("selected_indices must not be empty")
    if int(selected_indices.min()) < 0 or int(selected_indices.max()) >= total_candidates:
        raise ValueError("selected index is outside candidate bounds")
    layer_head_flat = torch.div(
        selected_indices, elements_per_head, rounding_mode="floor"
    )
    layers = torch.div(layer_head_flat, head_count, rounding_mode="floor")
    heads = layer_head_flat.remainder(head_count)
    layer_counts = torch.bincount(layers, minlength=layer_count)
    head_counts = torch.bincount(heads, minlength=head_count)
    layer_head_counts = torch.bincount(
        layer_head_flat, minlength=layer_count * head_count
    ).reshape(layer_count, head_count)
    selected_count = int(selected_indices.numel())
    if (
        int(layer_counts.sum()) != selected_count
        or int(head_counts.sum()) != selected_count
        or int(layer_head_counts.sum()) != selected_count
    ):
        raise AssertionError("selection count aggregation mismatch")
    return {
        "layer_counts": [int(value) for value in layer_counts.tolist()],
        "head_counts": [int(value) for value in head_counts.tolist()],
        "layer_head_counts": [
            [int(value) for value in row] for row in layer_head_counts.tolist()
        ],
    }


def concentration_summary(counts: Sequence[int], *, labels: Sequence[object]) -> dict:
    if len(counts) != len(labels) or not counts:
        raise ValueError("concentration counts/labels mismatch")
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("concentration counts must be non-negative integers")
    total = sum(counts)
    if total <= 0:
        raise ValueError("concentration total must be positive")
    order = sorted(range(len(counts)), key=lambda index: (-counts[index], index))
    shares = [value / total for value in counts]
    hhi = sum(share * share for share in shares)

    def top_share(k: int) -> float:
        return sum(counts[index] for index in order[:k]) / total

    top = []
    for index in order[:10]:
        if counts[index] == 0:
            break
        label = labels[index]
        record = {
            "count": counts[index],
            "fraction": counts[index] / total,
        }
        if isinstance(label, tuple):
            record.update({"layer": int(label[0]), "head": int(label[1])})
        else:
            record["head"] = int(label)
        top.append(record)
    return {
        "nonzero_group_count": sum(value > 0 for value in counts),
        "max_group_count": counts[order[0]],
        "max_group_fraction": counts[order[0]] / total,
        "top1_fraction": top_share(1),
        "top5_fraction": top_share(min(5, len(counts))),
        "top10_fraction": top_share(min(10, len(counts))),
        "herfindahl_hirschman_index": hhi,
        "effective_group_count": 1.0 / hhi,
        "top_groups": top,
    }


def add_count_summaries(first: dict, second: dict) -> dict:
    if len(first["layer_counts"]) != len(second["layer_counts"]):
        raise ValueError("layer count summary shapes differ")
    if len(first["head_counts"]) != len(second["head_counts"]):
        raise ValueError("head count summary shapes differ")
    if len(first["layer_head_counts"]) != len(second["layer_head_counts"]):
        raise ValueError("layer-head count summary shapes differ")
    layer_counts = [
        left + right
        for left, right in zip(first["layer_counts"], second["layer_counts"])
    ]
    head_counts = [
        left + right
        for left, right in zip(first["head_counts"], second["head_counts"])
    ]
    layer_head_counts = [
        [left + right for left, right in zip(first_row, second_row)]
        for first_row, second_row in zip(
            first["layer_head_counts"], second["layer_head_counts"]
        )
    ]
    return {
        "layer_counts": layer_counts,
        "head_counts": head_counts,
        "layer_head_counts": layer_head_counts,
    }


def zero_count_summary(layer_count: int, head_count: int) -> dict:
    return {
        "layer_counts": [0] * layer_count,
        "head_counts": [0] * head_count,
        "layer_head_counts": [[0] * head_count for _ in range(layer_count)],
    }


def prompt_selected_count_summary(counts: Sequence[int]) -> dict:
    """Summarize per-prompt selection counts using population standard deviation."""

    if isinstance(counts, (str, bytes)) or not counts:
        raise ValueError("prompt selection counts must be a non-empty sequence")
    values = tuple(counts)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("prompt selection counts must be non-negative integers")
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "prompt_count": len(values),
        "mean": float(tensor.mean().item()),
        "median": float(torch.quantile(tensor, 0.5).item()),
        "min": int(tensor.min().item()),
        "max": int(tensor.max().item()),
        "std": float(tensor.std(correction=0).item()),
        "std_definition": "population_ddof_0",
    }


def threshold_selection_summary(
    abs_z: torch.Tensor,
    *,
    tau: float,
    layer_count: int,
    head_count: int,
    elements_per_head: int,
) -> tuple[dict, dict, torch.Tensor]:
    """Select strictly above ``tau`` and summarize the selected identities."""

    if not isinstance(tau, (int, float)) or isinstance(tau, bool):
        raise TypeError("tau must be a real number")
    tau = float(tau)
    if not math.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative")
    if not isinstance(abs_z, torch.Tensor) or abs_z.ndim != 1:
        raise TypeError("abs_z must be a rank-1 tensor")
    if not abs_z.is_floating_point():
        raise TypeError("abs_z must be floating point")
    if abs_z.device.type != "cpu":
        raise ValueError("abs_z must be on CPU")
    if not bool(torch.isfinite(abs_z).all().item()):
        raise ValueError("abs_z contains NaN or infinity")
    expected_candidates = layer_count * head_count * elements_per_head
    if abs_z.numel() != expected_candidates:
        raise ValueError("abs_z candidate count does not match its structure")

    selected_indices = torch.nonzero(abs_z > tau, as_tuple=False).flatten().to(torch.int64)
    selected_values = abs_z[selected_indices].contiguous()
    if selected_indices.numel():
        counts = selection_count_summary(
            selected_indices,
            layer_count=layer_count,
            head_count=head_count,
            elements_per_head=elements_per_head,
        )
        layer_head_flat = [
            value for row in counts["layer_head_counts"] for value in row
        ]
        selected_distribution = distribution_summary(selected_values)
        if not selected_distribution["min"] > tau:
            raise AssertionError("threshold selection did not enforce strict > tau")
        head_concentration = concentration_summary(
            counts["head_counts"], labels=tuple(range(head_count))
        )
        layer_head_concentration = concentration_summary(
            layer_head_flat,
            labels=tuple(
                (layer, head)
                for layer in range(layer_count)
                for head in range(head_count)
            ),
        )
    else:
        counts = zero_count_summary(layer_count, head_count)
        selected_distribution = None
        head_concentration = None
        layer_head_concentration = None
    summary = {
        "tau": tau,
        "selection_rule": "abs_z > tau",
        "selected_count": int(selected_indices.numel()),
        "selected_fraction": int(selected_indices.numel()) / expected_candidates,
        "selected_abs_z": selected_distribution,
        **counts,
        "head_concentration": head_concentration,
        "layer_head_concentration": layer_head_concentration,
    }
    return summary, counts, selected_values


class FullPromptAbsZCollector:
    """Collect full strict-lower absolute z-scores without changing scores."""

    def __init__(
        self,
        *,
        layer_count: int,
        head_count: int,
        expected_score_dtype: torch.dtype | None = None,
    ):
        if type(layer_count) is not int or layer_count <= 0:
            raise ValueError("layer_count must be a positive integer")
        if type(head_count) is not int or head_count <= 0:
            raise ValueError("head_count must be a positive integer")
        self.layer_count = layer_count
        self.head_count = head_count
        self.expected_score_dtype = expected_score_dtype
        self._reference_cache: dict[
            tuple[int, str], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.clear()

    def clear(self) -> None:
        self.prompt_length: int | None = None
        self.excluded_selection_position: int | None = None
        self.mu: torch.Tensor | None = None
        self.sigma: torch.Tensor | None = None
        self.sigma_floor_head_count = 0
        self._reference_cache.clear()
        self._abs_z_by_layer: dict[int, torch.Tensor] = {}
        self._element_counts: dict[int, int] = {}
        self.pre_score_nonfinite_count = 0
        self.z_nonfinite_count = 0

    def set_prompt_context(
        self,
        *,
        prompt_length: int,
        excluded_selection_position: int,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> None:
        if self._abs_z_by_layer:
            raise ReferenceInterventionRuntimeError(
                "diagnostic context must be set before attention forwards"
            )
        if type(prompt_length) is not int or prompt_length < 2:
            raise ValueError("prompt_length must be an integer >= 2")
        if type(excluded_selection_position) is not int or excluded_selection_position < 0:
            raise ValueError("excluded_selection_position must be non-negative")
        if not isinstance(mu, torch.Tensor) or not isinstance(sigma, torch.Tensor):
            raise TypeError("mu and sigma must be torch tensors")
        expected_shape = (self.layer_count, self.head_count)
        if tuple(mu.shape) != expected_shape or tuple(sigma.shape) != expected_shape:
            raise ValueError("diagnostic reference shape mismatch")
        for name, value in (("mu", mu), ("sigma", sigma)):
            if value.dtype != torch.float32 or value.device.type != "cpu":
                raise ValueError(f"{name} must be CPU float32")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} contains NaN or infinity")
        if bool((sigma < 0).any().item()):
            raise ValueError("sigma must be non-negative")
        self.prompt_length = prompt_length
        self.excluded_selection_position = excluded_selection_position
        self.mu = mu.contiguous()
        self.sigma = sigma.contiguous()
        self.sigma_floor_head_count = int(
            (self.sigma < NUMERICAL_SIGMA_FLOOR).sum().item()
        )

    def _layer_reference(
        self, layer: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer, str(device))
        cached = self._reference_cache.get(key)
        if cached is None:
            if self.mu is None or self.sigma is None:
                raise ReferenceInterventionRuntimeError(
                    "diagnostic reference context is missing"
                )
            cached = (
                self.mu[layer].to(device=device),
                self.sigma[layer].to(device=device).clamp_min(
                    NUMERICAL_SIGMA_FLOOR
                ),
            )
            self._reference_cache[key] = cached
        return cached

    def apply(self, layer: int, attention_scores: torch.Tensor) -> torch.Tensor:
        if self.prompt_length is None:
            raise ReferenceInterventionRuntimeError("diagnostic prompt context is missing")
        if type(layer) is not int or not 0 <= layer < self.layer_count:
            raise ReferenceInterventionRuntimeError("diagnostic layer index is invalid")
        if layer in self._abs_z_by_layer:
            raise ReferenceInterventionRuntimeError("diagnostic saw a layer more than once")
        if not isinstance(attention_scores, torch.Tensor) or attention_scores.ndim != 4:
            raise ReferenceInterventionRuntimeError(
                "diagnostic scores must have shape batch,head,query,key"
            )
        batch, heads, query_length, key_length = attention_scores.shape
        if (
            batch != 1
            or heads != self.head_count
            or query_length != key_length
            or query_length != self.prompt_length
        ):
            raise ReferenceInterventionRuntimeError("diagnostic score shape mismatch")
        if (
            self.expected_score_dtype is not None
            and attention_scores.dtype != self.expected_score_dtype
        ):
            raise ReferenceInterventionRuntimeError("diagnostic score dtype mismatch")
        nonfinite_scores = int((~torch.isfinite(attention_scores)).sum().item())
        self.pre_score_nonfinite_count += nonfinite_scores
        if nonfinite_scores:
            raise ReferenceInterventionRuntimeError(
                "diagnostic attention scores contain NaN or infinity"
            )

        rows, cols = strict_lower_indices(query_length, device=attention_scores.device)
        elements_per_head = query_length * (query_length - 1) // 2
        scores = attention_scores.float()
        asymmetry = (scores - scores.transpose(-1, -2)) / 2.0
        current_lower = asymmetry[0, :, rows, cols]
        mu, safe_sigma = self._layer_reference(layer, attention_scores.device)
        abs_z = (
            (current_lower - mu.view(-1, 1)) / safe_sigma.view(-1, 1)
        ).abs()
        nonfinite_z = int((~torch.isfinite(abs_z)).sum().item())
        self.z_nonfinite_count += nonfinite_z
        if nonfinite_z:
            raise ReferenceInterventionRuntimeError(
                "diagnostic absolute z-scores contain NaN or infinity"
            )
        if tuple(abs_z.shape) != (self.head_count, elements_per_head):
            raise AssertionError("diagnostic absolute z-score shape mismatch")
        self._abs_z_by_layer[layer] = abs_z.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        self._element_counts[layer] = elements_per_head
        return attention_scores

    def result(self) -> tuple[torch.Tensor, dict]:
        expected_layers = set(range(self.layer_count))
        if set(self._abs_z_by_layer) != expected_layers:
            raise ReferenceInterventionRuntimeError(
                "diagnostic did not observe every attention layer exactly once"
            )
        element_counts = set(self._element_counts.values())
        if len(element_counts) != 1:
            raise ReferenceInterventionRuntimeError(
                "strict-lower element counts differ across diagnostic layers"
            )
        elements_per_head = element_counts.pop()
        stacked = torch.stack(
            [self._abs_z_by_layer[layer] for layer in range(self.layer_count)]
        )
        expected_shape = (self.layer_count, self.head_count, elements_per_head)
        if tuple(stacked.shape) != expected_shape:
            raise AssertionError("diagnostic stacked absolute z-score shape mismatch")
        flattened = stacked.reshape(-1).contiguous()
        expected_candidates = candidate_count(
            self.prompt_length,
            layer_count=self.layer_count,
            head_count=self.head_count,
        )
        if flattened.numel() != expected_candidates:
            raise AssertionError("diagnostic candidate count mismatch")
        return flattened, {
            "prompt_length": self.prompt_length,
            "excluded_selection_position": self.excluded_selection_position,
            "layer_count": self.layer_count,
            "head_count": self.head_count,
            "strict_lower_elements_per_head": elements_per_head,
            "candidate_count": expected_candidates,
            "sigma_floor_head_count": self.sigma_floor_head_count,
            "pre_score_nonfinite_count": self.pre_score_nonfinite_count,
            "z_nonfinite_count": self.z_nonfinite_count,
            "all_finite": True,
        }


@contextmanager
def patch_llama_abs_z_collection(
    model,
    *,
    layer_count: int,
    head_count: int,
):
    """Patch eager Llama attention only to observe full-prompt absolute z-scores."""

    collector = FullPromptAbsZCollector(
        layer_count=layer_count,
        head_count=head_count,
        expected_score_dtype=torch.float16,
    )
    patched = []
    try:
        patched = _patch_llama_attention_modules(model, collector)
        yield collector
    finally:
        _restore_llama_attention_modules(patched)
