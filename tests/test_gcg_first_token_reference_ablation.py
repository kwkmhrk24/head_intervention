import csv
import tempfile
import unittest
from pathlib import Path

import torch

from utils.gcg_first_token_reference_ablation import (
    EXPECTED_SELECTION_DATASET_INDICES,
    load_selection_development_records,
    summarize_development_condition,
    summarize_first_token_logits,
    validate_first_token_response_invariant,
    write_comparison_table,
)
from utils.gcg_reference_intervention_diagnostics import (
    compute_logit_step_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = (
    REPO_ROOT
    / "data/outputs/gcg_wasserstein_reference/llama2/seed42_selection20/"
    "codex_gpu_20260802T061114Z"
)
JAILBREAK_INPUT = REPO_ROOT / "data/derived/llama2/gcg/jailbreak_prompts.jsonl"
BASELINE_RESULTS = (
    REPO_ROOT
    / "data/outputs/multidataset_asym_score_intervention/"
    "beta05_full_20260722_041303/llama_gcg/no_intervention/"
    "prompt_results.jsonl"
)


def make_row(
    dataset_index,
    *,
    response_equal=True,
    first_difference=None,
    argmax_equal=True,
    success=True,
):
    return {
        "dataset_index": dataset_index,
        "generation_failed": False,
        "empty_response": False,
        "success": success,
        "response_equal": response_equal,
        "first_different_token_position": first_difference,
        "first_token_argmax_equal": argmax_equal,
        "diagnostics": {
            "all_scores_finite": True,
            "total_intervention_count": 5,
            "later_forward_intervention_count": 0,
        },
    }


class SelectionLoaderTests(unittest.TestCase):
    def test_loads_fixed_selection_twenty(self):
        rows, manifest = load_selection_development_records(
            split_manifest_path=ARTIFACT_DIR / "split_manifest.json",
            jailbreak_input=JAILBREAK_INPUT,
            baseline_results=BASELINE_RESULTS,
        )
        self.assertEqual(len(rows), 20)
        self.assertEqual(
            tuple(row["dataset_index"] for row in rows),
            EXPECTED_SELECTION_DATASET_INDICES,
        )
        self.assertTrue(all(row["split"] == "selection" for row in rows))
        self.assertTrue(all(row["baseline_success"] for row in rows))
        self.assertEqual(manifest["selection_count"], 20)


class ResponseInvariantTests(unittest.TestCase):
    def test_equal_response_requires_no_difference(self):
        validate_first_token_response_invariant(
            response_equal=True,
            first_different_token_position=None,
        )
        with self.assertRaisesRegex(ValueError, "equal responses"):
            validate_first_token_response_invariant(
                response_equal=True,
                first_different_token_position=0,
            )

    def test_changed_response_must_start_at_token_zero(self):
        validate_first_token_response_invariant(
            response_equal=False,
            first_different_token_position=0,
        )
        with self.assertRaisesRegex(ValueError, "after token 0"):
            validate_first_token_response_invariant(
                response_equal=False,
                first_different_token_position=1,
            )

    def test_first_token_summary_requires_response_argmax_set_match(self):
        rows = [
            make_row(68),
            make_row(
                70,
                response_equal=False,
                first_difference=0,
                argmax_equal=False,
                success=False,
            ),
        ]
        summary = summarize_development_condition(
            rows,
            timing="first_token_only",
            alpha=0.5,
            expected_count=2,
        )
        self.assertEqual(summary["newly_defended_count"], 1)
        self.assertEqual(summary["response_changed_count"], 1)
        self.assertEqual(summary["first_token_argmax_changed_count"], 1)
        self.assertTrue(summary["response_change_matches_first_token_argmax_change"])
        rows[1]["first_token_argmax_equal"] = True
        with self.assertRaisesRegex(ValueError, "argmax changes differ"):
            summarize_development_condition(
                rows,
                timing="first_token_only",
                alpha=0.5,
                expected_count=2,
            )


class FirstLogitSummaryTests(unittest.TestCase):
    def test_identical_logits_have_zero_kl_and_margin_change(self):
        logits = torch.tensor([1.0, 3.0, -2.0])
        metric = compute_logit_step_metrics(logits, logits.clone(), step=0)
        self.assertEqual(metric["kl_baseline_to_intervention"], 0.0)
        self.assertTrue(metric["argmax_equal"])
        self.assertEqual(metric["margin_change"], 0.0)

    def test_first_logit_distribution_has_required_statistics(self):
        rows = []
        for offset, dataset_index in enumerate(EXPECTED_SELECTION_DATASET_INDICES):
            baseline = torch.tensor([2.0, 1.0])
            intervention = torch.tensor([2.0 + offset / 100.0, 1.0])
            rows.append(
                {
                    "dataset_index": dataset_index,
                    "alpha": 0.1,
                    "metrics": compute_logit_step_metrics(
                        baseline,
                        intervention,
                        step=0,
                    ),
                }
            )
        summary = summarize_first_token_logits(rows, alphas=(0.1,))
        condition = summary["conditions"][0]
        self.assertEqual(condition["prompt_count"], 20)
        self.assertEqual(condition["argmax_changed_count"], 0)
        self.assertEqual(
            set(condition["max_absolute_logit_difference"]),
            {"min", "median", "mean", "p95", "max"},
        )


class OutputTests(unittest.TestCase):
    def test_comparison_table_refuses_overwrite(self):
        row = {
            "timing": "first_token_only",
            "alpha": 0.1,
            "prompt_count": 20,
            "attack_success_count": 20,
            "newly_defended_count": 0,
            "still_attack_success_count": 20,
            "response_equal_count": 20,
            "response_changed_count": 0,
            "first_token_argmax_changed_count": 0,
            "generation_failure_count": 0,
            "empty_response_count": 0,
            "total_intervention_count": 100,
            "later_forward_intervention_count": 0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "comparison.csv"
            write_comparison_table(path, [row])
            with path.open(encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(len(saved), 1)
            with self.assertRaises(FileExistsError):
                write_comparison_table(path, [row])


if __name__ == "__main__":
    unittest.main()
