"""Analyze prompt-level intervention transitions across saved evaluation runs.

This script is deliberately CPU-only: it reads existing JSON/JSONL outputs and
never imports model or tokenizer code. Prompt and response text are not copied
to any analysis output.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from utils.evaluate_utils import classify_transition


RUN_SPECS = {
    "llama_autodan": ("Llama-2", "AutoDAN"),
    "llama_gcg": ("Llama-2", "GCG"),
    "mistral_autodan": ("Mistral", "AutoDAN"),
    "mistral_gcg": ("Mistral", "GCG"),
}
CONDITIONS = {
    "paper": "paper_beta_0.5",
    "consensus_increase": "consensus_increase_beta_0.5",
    "consensus_decrease": "consensus_decrease_beta_0.5",
}


@dataclass(frozen=True)
class PromptOutcome:
    prompt_id: str
    source_index: int
    success: bool


def _canonical_prompt_id(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        raise ValueError(f"invalid prompt_id: {value!r}")
    result = str(value)
    if not result:
        raise ValueError("prompt_id must be non-empty")
    return result


def _source_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid source_index: {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid source_index: {value!r}") from exc
    if isinstance(value, float) and value != result:
        raise ValueError(f"non-integral source_index: {value!r}")
    if isinstance(value, str) and value.strip() != str(result):
        raise ValueError(f"non-canonical source_index: {value!r}")
    return result


def load_prompt_outcomes(path: Path) -> list[PromptOutcome]:
    """Load only identifiers and binary outcomes from a prompt-results JSONL."""
    if not path.is_file():
        raise FileNotFoundError(f"prompt results not found: {path}")
    outcomes: list[PromptOutcome] = []
    seen_prompt_ids: set[str] = set()
    seen_sources: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            prompt_id = _canonical_prompt_id(row.get("prompt_id"))
            source_index = _source_index(row.get("source_index"))
            success = row.get("success")
            if type(success) is not bool:
                raise ValueError(
                    f"non-binary success at {path}:{line_number}, "
                    f"source_index={source_index}"
                )
            if prompt_id in seen_prompt_ids:
                raise ValueError(f"duplicate prompt_id {prompt_id!r} in {path}")
            if source_index in seen_sources:
                raise ValueError(f"duplicate source_index {source_index} in {path}")
            seen_prompt_ids.add(prompt_id)
            seen_sources.add(source_index)
            outcomes.append(PromptOutcome(prompt_id, source_index, success))
    if not outcomes:
        raise ValueError(f"no prompt outcomes in {path}")
    return outcomes


def transition_label(baseline_success: bool, intervention_success: bool) -> str:
    transition = classify_transition(baseline_success, intervention_success)
    if transition is None:  # Both inputs are validated binary outcomes.
        raise AssertionError("binary transition unexpectedly classified as missing")
    return transition


def pair_outcomes(
    baseline: Sequence[PromptOutcome], intervention: Sequence[PromptOutcome]
) -> list[dict[str, Any]]:
    baseline_by_source = {row.source_index: row for row in baseline}
    intervention_by_source = {row.source_index: row for row in intervention}
    if set(baseline_by_source) != set(intervention_by_source):
        missing = sorted(set(baseline_by_source) - set(intervention_by_source))
        extra = sorted(set(intervention_by_source) - set(baseline_by_source))
        raise ValueError(
            "baseline/intervention source_index mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    paired: list[dict[str, Any]] = []
    for base in baseline:
        current = intervention_by_source[base.source_index]
        if current.prompt_id != base.prompt_id:
            raise ValueError(
                "prompt_id mismatch at source_index "
                f"{base.source_index}: {base.prompt_id!r} != {current.prompt_id!r}"
            )
        paired.append(
            {
                "prompt_id": base.prompt_id,
                "source_index": base.source_index,
                "baseline_success": base.success,
                "intervention_success": current.success,
                "transition": transition_label(base.success, current.success),
            }
        )
    return paired


def summarize_transitions(paired: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {
        name: sum(row["transition"] == name for row in paired)
        for name in ("defended", "regressed", "remained_success", "remained_failure")
    }
    total = len(paired)
    baseline_success = sum(bool(row["baseline_success"]) for row in paired)
    intervention_success = sum(bool(row["intervention_success"]) for row in paired)
    if sum(counts.values()) != total:
        raise AssertionError("transition counts do not sum to total")
    if counts["defended"] + counts["remained_success"] != baseline_success:
        raise AssertionError("baseline success identity failed")
    if counts["regressed"] + counts["remained_failure"] != total - baseline_success:
        raise AssertionError("baseline failure identity failed")
    return {
        "total": total,
        "baseline_success": baseline_success,
        "intervention_success": intervention_success,
        "asr_baseline": baseline_success / total,
        "asr_intervention": intervention_success / total,
        "asr_difference": (intervention_success - baseline_success) / total,
        **counts,
        "net_defended": counts["defended"] - counts["regressed"],
    }


def jaccard_counts(a: set[int], b: set[int]) -> tuple[int, int, float]:
    intersection = len(a & b)
    union = len(a | b)
    return intersection, union, intersection / union if union else 1.0


def mcnemar_exact_p(defended: int, regressed: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant pairs."""
    if defended < 0 or regressed < 0:
        raise ValueError("McNemar counts must be non-negative")
    discordant = defended + regressed
    if discordant == 0:
        return 1.0
    tail = min(defended, regressed)
    numerator = sum(math.comb(discordant, i) for i in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2**discordant))


def paired_bootstrap_ci(
    paired: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    resamples: int = 10_000,
) -> tuple[float, float]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    differences = np.asarray(
        [
            int(bool(row["intervention_success"]))
            - int(bool(row["baseline_success"]))
            for row in paired
        ],
        dtype=np.float64,
    )
    if differences.size == 0:
        raise ValueError("paired bootstrap requires at least one prompt")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    chunk_size = 1_000
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = rng.integers(0, differences.size, size=(stop - start, differences.size))
        means[start:stop] = differences[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def validate_existing_summary(path: Path, calculated: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"condition summary not found: {path}")
    with path.open(encoding="utf-8") as handle:
        existing = json.load(handle)
    key_map = {
        "total": "total_count",
        "baseline_success": "baseline_success_count",
        "intervention_success": "success_count",
        "asr_intervention": "asr",
        "defended": "defended_count",
        "regressed": "regressed_count",
        "remained_success": "remained_success_count",
        "remained_failure": "remained_failure_count",
    }
    for calculated_key, existing_key in key_map.items():
        expected = existing.get(existing_key)
        actual = calculated[calculated_key]
        if isinstance(actual, float):
            matches = isinstance(expected, (int, float)) and math.isclose(
                actual, float(expected), rel_tol=1e-12, abs_tol=1e-12
            )
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"summary mismatch in {path}: {existing_key}={expected!r}, "
                f"recalculated {calculated_key}={actual!r}"
            )


def matched_attack_rows(
    model: str,
    condition: str,
    autodan: Sequence[Mapping[str, Any]],
    gcg: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    auto = {int(row["source_index"]): row for row in autodan}
    gcg_by_source = {int(row["source_index"]): row for row in gcg}
    common = sorted(set(auto) & set(gcg_by_source))
    if not common:
        raise ValueError(f"no matched source_index values for {model} {condition}")
    auto_def = {idx for idx in common if auto[idx]["transition"] == "defended"}
    gcg_def = {idx for idx in common if gcg_by_source[idx]["transition"] == "defended"}
    auto_reg = {idx for idx in common if auto[idx]["transition"] == "regressed"}
    gcg_reg = {idx for idx in common if gcg_by_source[idx]["transition"] == "regressed"}
    same = sum(auto[idx]["transition"] == gcg_by_source[idx]["transition"] for idx in common)
    return {
        "model": model,
        "condition": condition,
        "matched_source_count": len(common),
        "defended_in_both": len(auto_def & gcg_def),
        "defended_autodan_only": len(auto_def - gcg_def),
        "defended_gcg_only": len(gcg_def - auto_def),
        "regressed_in_both": len(auto_reg & gcg_reg),
        "regressed_autodan_only": len(auto_reg - gcg_reg),
        "regressed_gcg_only": len(gcg_reg - auto_reg),
        "same_transition_count": same,
        "same_transition_rate": same / len(common),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze_runs(
    run_dir: Path,
    *,
    seed: int = 42,
    resamples: int = 10_000,
) -> dict[str, list[dict[str, Any]]]:
    transition_summaries: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    statistical_rows: list[dict[str, Any]] = []
    paired_by_run: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for run_name, (model, attack) in RUN_SPECS.items():
        current_dir = run_dir / run_name
        baseline = load_prompt_outcomes(current_dir / "no_intervention/prompt_results.jsonl")
        paired_by_run[run_name] = {}
        for condition, folder in CONDITIONS.items():
            intervention = load_prompt_outcomes(current_dir / folder / "prompt_results.jsonl")
            paired = pair_outcomes(baseline, intervention)
            summary = summarize_transitions(paired)
            validate_existing_summary(current_dir / folder / "summary.json", summary)
            paired_by_run[run_name][condition] = paired
            transition_summaries.append(
                {"model": model, "attack": attack, "condition": condition, **summary}
            )
            prompt_rows.extend(
                {"model": model, "attack": attack, "condition": condition, **row}
                for row in paired
            )
            lower, upper = paired_bootstrap_ci(paired, seed=seed, resamples=resamples)
            statistical_rows.append(
                {
                    "model": model,
                    "attack": attack,
                    "condition": condition,
                    "total": summary["total"],
                    "defended": summary["defended"],
                    "regressed": summary["regressed"],
                    "mcnemar_exact_p": mcnemar_exact_p(
                        summary["defended"], summary["regressed"]
                    ),
                    "asr_difference": summary["asr_difference"],
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "bootstrap_seed": seed,
                    "bootstrap_resamples": resamples,
                }
            )

    overlap_rows: list[dict[str, Any]] = []
    for run_name, (model, attack) in RUN_SPECS.items():
        for condition_a, condition_b in combinations(CONDITIONS, 2):
            rows_a = paired_by_run[run_name][condition_a]
            rows_b = paired_by_run[run_name][condition_b]
            defended_a = {row["source_index"] for row in rows_a if row["transition"] == "defended"}
            defended_b = {row["source_index"] for row in rows_b if row["transition"] == "defended"}
            regressed_a = {row["source_index"] for row in rows_a if row["transition"] == "regressed"}
            regressed_b = {row["source_index"] for row in rows_b if row["transition"] == "regressed"}
            def_inter, def_union, def_jaccard = jaccard_counts(defended_a, defended_b)
            reg_inter, reg_union, reg_jaccard = jaccard_counts(regressed_a, regressed_b)
            overlap_rows.append(
                {
                    "model": model,
                    "attack": attack,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "defended_a": len(defended_a),
                    "defended_b": len(defended_b),
                    "defended_intersection": def_inter,
                    "defended_union": def_union,
                    "defended_jaccard": def_jaccard,
                    "regressed_a": len(regressed_a),
                    "regressed_b": len(regressed_b),
                    "regressed_intersection": reg_inter,
                    "regressed_union": reg_union,
                    "regressed_jaccard": reg_jaccard,
                }
            )

    matched_rows: list[dict[str, Any]] = []
    for model, prefix in (("Llama-2", "llama"), ("Mistral", "mistral")):
        for condition in CONDITIONS:
            matched_rows.append(
                matched_attack_rows(
                    model,
                    condition,
                    paired_by_run[f"{prefix}_autodan"][condition],
                    paired_by_run[f"{prefix}_gcg"][condition],
                )
            )

    return {
        "transition_summary": transition_summaries,
        "prompt_ids": prompt_rows,
        "condition_overlap": overlap_rows,
        "matched_attack": matched_rows,
        "statistical_tests": statistical_rows,
    }


def _validate_finite(rows: Iterable[Mapping[str, Any]]) -> None:
    for row_number, row in enumerate(rows, start=1):
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite output at row {row_number}, column {key}")


def write_outputs(output_dir: Path, analysis: Mapping[str, list[dict[str, Any]]]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    for rows in analysis.values():
        _validate_finite(rows)
    output_dir.mkdir(parents=True)
    write_csv(
        output_dir / "cross_run_transition_summary.csv",
        analysis["transition_summary"],
        [
            "model", "attack", "condition", "total", "baseline_success",
            "intervention_success", "asr_baseline", "asr_intervention",
            "asr_difference", "defended", "regressed", "remained_success",
            "remained_failure", "net_defended",
        ],
    )
    write_csv(
        output_dir / "transition_prompt_ids.csv",
        analysis["prompt_ids"],
        [
            "model", "attack", "condition", "prompt_id", "source_index",
            "baseline_success", "intervention_success", "transition",
        ],
    )
    write_csv(
        output_dir / "condition_overlap_summary.csv",
        analysis["condition_overlap"],
        [
            "model", "attack", "condition_a", "condition_b", "defended_a",
            "defended_b", "defended_intersection", "defended_union",
            "defended_jaccard", "regressed_a", "regressed_b",
            "regressed_intersection", "regressed_union", "regressed_jaccard",
        ],
    )
    write_csv(
        output_dir / "matched_attack_transition_summary.csv",
        analysis["matched_attack"],
        [
            "model", "condition", "matched_source_count", "defended_in_both",
            "defended_autodan_only", "defended_gcg_only", "regressed_in_both",
            "regressed_autodan_only", "regressed_gcg_only",
            "same_transition_count", "same_transition_rate",
        ],
    )
    write_csv(
        output_dir / "statistical_tests.csv",
        analysis["statistical_tests"],
        [
            "model", "attack", "condition", "total", "defended", "regressed",
            "mcnemar_exact_p", "asr_difference", "bootstrap_ci_lower",
            "bootstrap_ci_upper", "bootstrap_seed", "bootstrap_resamples",
        ],
    )

