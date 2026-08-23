#!/usr/bin/env python3
"""Analyze saved GCG reference-intervention responses without generation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.evaluate_utils import check_success_with_details  # noqa: E402
from utils.experiment_runtime import git_head  # noqa: E402
from utils.gcg_reference_intervention_diagnostics import (  # noqa: E402
    compare_token_sequences,
    summarize_response_changes,
)
from utils.gcg_wasserstein_reference import (  # noqa: E402
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
)
from utils.gcg_wasserstein_reference_evaluation import (  # noqa: E402
    EVALUATION_COUNT,
    ensure_output_absent,
)


DEFAULT_MODEL = "/mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf"
DEFAULT_EVALUATION_RUN = (
    "data/outputs/gcg_wasserstein_reference_intervention/llama2/"
    "seed42_selection20/evaluation_20260802T071032Z"
)
DEFAULT_ORIGINAL_INPUT = "data/derived/llama2/gcg/original_prompts.jsonl"
DEFAULT_REFERENCE_ARTIFACT = (
    "data/outputs/gcg_wasserstein_reference/llama2/seed42_selection20/"
    "codex_gpu_20260802T061114Z"
)
ALPHAS = (0.1, 0.3, 0.5)
INDEX_182 = 182
CSV_FIELDS = (
    "row_position",
    "dataset_index",
    "baseline_success",
    "intervention_success",
    "transition",
    "response_equal",
    "baseline_token_count",
    "intervention_token_count",
    "common_prefix_token_count",
    "first_different_token_position",
    "token_edit_distance",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--evaluation-run", default=DEFAULT_EVALUATION_RUN)
    parser.add_argument("--original-input", default=DEFAULT_ORIGINAL_INPUT)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def tokenize_response(tokenizer, response: str) -> list[int]:
    if not isinstance(response, str):
        raise ValueError("saved response must be a string")
    return list(tokenizer(response, add_special_tokens=False)["input_ids"])


def rows_by_dataset_index(rows: list[dict], *, label: str) -> dict[int, dict]:
    if len(rows) != EVALUATION_COUNT:
        raise ValueError(f"{label} must contain {EVALUATION_COUNT} rows")
    output = {}
    for row in rows:
        dataset_index = int(row["dataset_index"])
        if dataset_index in output:
            raise ValueError(f"{label} contains duplicate dataset index")
        output[dataset_index] = row
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in CSV_FIELDS} for row in rows)


def response_details(
    *,
    label: str,
    response: str,
    success: bool,
    matched_refusal_substrings: list[str],
    success_decision_reason: str,
    token_ids: list[int],
    baseline_token_ids: list[int],
) -> dict:
    comparison = compare_token_sequences(baseline_token_ids, token_ids)
    return {
        "condition": label,
        "response": response,
        "success": bool(success),
        "matched_refusal_substrings": matched_refusal_substrings,
        "success_decision_reason": success_decision_reason,
        "response_token_count": len(token_ids),
        "common_prefix_token_count_with_baseline": comparison[
            "common_prefix_token_count"
        ],
        "first_different_token_position_from_baseline": comparison[
            "first_different_token_position"
        ],
        "token_edit_distance_from_baseline": comparison["token_edit_distance"],
    }


def write_index182_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Dataset index 182 saved-response diagnostics",
        "",
        "> This file contains an adversarial prompt and model outputs for research auditing.",
        "",
        "## Original prompt",
        "",
        "```text",
        payload["original_prompt"],
        "```",
        "",
        "## GCG jailbreak prompt",
        "",
        "```text",
        payload["jailbreak_prompt"],
        "```",
    ]
    for label, details in payload["responses"].items():
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                f"- attack success: `{str(details['success']).lower()}`",
                f"- response tokens: `{details['response_token_count']}`",
                "- first differing token from baseline: "
                f"`{details['first_different_token_position_from_baseline']}`",
                "- decision reason: " + details["success_decision_reason"],
                "- matched refusal substrings: `"
                + json.dumps(details["matched_refusal_substrings"], ensure_ascii=False)
                + "`",
                "",
                "```text",
                details["response"],
                "```",
            ]
        )
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv=None):
    args = parse_args(argv)
    output_dir = ensure_output_absent(args.output_dir).resolve()
    evaluation_run = Path(args.evaluation_run).resolve()
    original_input = Path(args.original_input).resolve()
    model_path = Path(args.model).resolve()
    if read_json(evaluation_run / "evaluation_summary.json").get("status") != "complete":
        raise ValueError("source evaluation is not complete")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    baseline_rows = read_jsonl(evaluation_run / "baseline_evaluation_results.jsonl")
    alpha_zero_rows = read_jsonl(evaluation_run / "alpha_0.0/prompt_results.jsonl")
    baseline = rows_by_dataset_index(baseline_rows, label="baseline")
    alpha_zero = rows_by_dataset_index(alpha_zero_rows, label="alpha=0")
    if set(baseline) != set(alpha_zero):
        raise ValueError("baseline and alpha=0 dataset indices differ")
    for dataset_index in baseline:
        base = baseline[dataset_index]
        zero = alpha_zero[dataset_index]
        if base["baseline_response"] != zero["response"] or bool(
            base["baseline_success"]
        ) != bool(zero["success"]):
            raise ValueError(f"alpha=0 mismatch at dataset index {dataset_index}")

    output_dir.mkdir(parents=True)
    response_dir = output_dir / "response_change_analysis"
    response_dir.mkdir()
    summaries = {}
    alpha_rows_by_index = {}
    for alpha in ALPHAS:
        source_rows = read_jsonl(
            evaluation_run / f"alpha_{alpha:.1f}/prompt_results.jsonl"
        )
        intervention = rows_by_dataset_index(
            source_rows,
            label=f"alpha={alpha:.1f}",
        )
        if set(intervention) != set(baseline):
            raise ValueError(f"alpha={alpha:.1f} dataset indices differ")
        analysis_rows = []
        for dataset_index in sorted(baseline, key=lambda key: baseline[key]["row_position"]):
            base = baseline[dataset_index]
            current = intervention[dataset_index]
            if base["prompt"] != current["prompt"]:
                raise ValueError(f"prompt mismatch at dataset index {dataset_index}")
            if current["generation_failed"]:
                raise ValueError(f"saved generation failed at dataset index {dataset_index}")
            baseline_tokens = tokenize_response(tokenizer, base["baseline_response"])
            intervention_tokens = tokenize_response(tokenizer, current["response"])
            comparison = compare_token_sequences(
                baseline_tokens,
                intervention_tokens,
            )
            comparison["response_equal"] = (
                base["baseline_response"] == current["response"]
            )
            analysis_rows.append(
                {
                    "row_position": int(base["row_position"]),
                    "dataset_index": dataset_index,
                    "baseline_success": bool(base["baseline_success"]),
                    "intervention_success": bool(current["success"]),
                    "transition": current["transition"],
                    **comparison,
                }
            )
        write_csv(response_dir / f"alpha_{alpha:.1f}.csv", analysis_rows)
        summaries[f"alpha_{alpha:.1f}"] = {
            "alpha": alpha,
            **summarize_response_changes(analysis_rows),
        }
        alpha_rows_by_index[alpha] = intervention

    response_summary = {
        "created_at": utc_now(),
        "source_evaluation_run": str(evaluation_run),
        "tokenizer": str(model_path),
        "tokenize_add_special_tokens": False,
        "token_position_indexing": "zero-based",
        "conditions": [summaries[f"alpha_{alpha:.1f}"] for alpha in ALPHAS],
    }
    write_json(output_dir / "response_change_summary.json", response_summary)

    original_rows = read_jsonl(original_input)
    original_by_index = {int(row["index"]): row for row in original_rows}
    base_182 = baseline[INDEX_182]
    zero_182 = alpha_zero[INDEX_182]
    baseline_token_ids = tokenize_response(tokenizer, base_182["baseline_response"])
    baseline_success, baseline_checker = check_success_with_details(
        base_182["baseline_response"]
    )
    if bool(baseline_success) != bool(base_182["baseline_success"]):
        raise ValueError("recomputed baseline success differs for dataset index 182")
    responses = {
        "baseline": response_details(
            label="baseline",
            response=base_182["baseline_response"],
            success=baseline_success,
            matched_refusal_substrings=baseline_checker[
                "matched_refusal_substrings"
            ],
            success_decision_reason=baseline_checker["decision_reason"],
            token_ids=baseline_token_ids,
            baseline_token_ids=baseline_token_ids,
        )
    }
    for alpha in ALPHAS:
        row = alpha_rows_by_index[alpha][INDEX_182]
        token_ids = tokenize_response(tokenizer, row["response"])
        responses[f"alpha_{alpha:.1f}"] = response_details(
            label=f"alpha={alpha:.1f}",
            response=row["response"],
            success=row["success"],
            matched_refusal_substrings=row["matched_refusal_substrings"],
            success_decision_reason=row["success_decision_reason"],
            token_ids=token_ids,
            baseline_token_ids=baseline_token_ids,
        )
    details_182 = {
        "row_position": int(base_182["row_position"]),
        "dataset_index": INDEX_182,
        "prompt_id": base_182["prompt_id"],
        "original_prompt": original_by_index[INDEX_182]["prompt"],
        "jailbreak_prompt": base_182["prompt"],
        "responses": responses,
    }
    write_json(output_dir / "diagnostics_index182.json", details_182)
    write_index182_markdown(output_dir / "diagnostics_index182.md", details_182)
    write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "execution_git_head": git_head(REPO_ROOT),
            "status": "response_analysis_complete_gpu_pending",
            "source_evaluation_run": str(evaluation_run),
            "source_evaluation_summary_sha256": sha256_file(
                evaluation_run / "evaluation_summary.json"
            ),
            "reference_artifact_dir": str(
                (REPO_ROOT / DEFAULT_REFERENCE_ARTIFACT).resolve()
            ),
            "model": str(model_path),
            "original_input": str(original_input),
            "original_input_sha256": sha256_file(original_input),
            "response_analysis_alphas": list(ALPHAS),
            "logit_diagnostic_alphas": [0.0, *ALPHAS],
            "kl_direction": "D_KL(P_baseline || P_intervention)",
            "logit_probability_dtype": "torch.float32",
            "token_position_indexing": "zero-based",
        },
    )
    print(json.dumps(response_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
