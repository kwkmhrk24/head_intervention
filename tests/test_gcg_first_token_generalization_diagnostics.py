import math
import unittest

from utils.gcg_first_token_generalization_diagnostics import (
    ALPHAS,
    EXPECTED_TOP5,
    GROUP_COUNTS,
    SCORE_FIELDS,
    build_all_nonempty_head_subsets,
    build_index88_context,
    head_contribution_conditions,
    minimal_refusal_subsets,
    normalize_logit_metric,
    percentile_rank,
    summarize_group_logits,
    summarize_score_interventions,
)


def metric(value=0.1, margin=1.0, argmax_equal=True):
    return {
        "baseline_top1_token_id": 1,
        "intervention_top1_token_id": 1 if argmax_equal else 2,
        "argmax_equal": argmax_equal,
        "max_absolute_logit_difference": value,
        "mean_absolute_logit_difference": value / 2,
        "kl_baseline_to_intervention": value / 100,
        "baseline_top1_logit": 4.0,
        "intervention_logit_at_baseline_top1": 4.0 - value,
        "top1_logit_change": -value,
        "baseline_top1_top2_margin": margin,
        "intervention_top1_top2_margin": margin - value,
        "margin_change": -value,
    }


def group_logit_rows():
    rows = []
    for group, count in GROUP_COUNTS.items():
        indices = list(range(count))
        if group == "selection20":
            indices[8] = 88
        for alpha in ALPHAS:
            for position, dataset_index in enumerate(indices):
                rows.append(
                    normalize_logit_metric(
                        dataset_group=group,
                        dataset_index=dataset_index,
                        baseline_success=(group == "selection20" or position < 21),
                        alpha=alpha,
                        metric=metric(value=position / 100 + alpha, margin=position + 1),
                    )
                )
    return rows


class GroupLogitTests(unittest.TestCase):
    def test_group_and_baseline_success_summaries(self):
        summary = summarize_group_logits(group_logit_rows())
        self.assertEqual(len(summary["conditions"]), 16)
        holdout_success = next(
            row for row in summary["conditions"]
            if row["dataset_group"] == "holdout48"
            and row["subset"] == "baseline_success"
            and row["alpha"] == 0.5
        )
        self.assertEqual(holdout_success["prompt_count"], 21)
        self.assertEqual(
            set(holdout_success["max_absolute_logit_difference"]),
            {"min", "median", "mean", "p95", "max"},
        )

    def test_percentile_is_inclusive(self):
        self.assertEqual(percentile_rank([1, 2, 3, 4], 2), 50.0)

    def test_index88_context_contains_margin_and_logit_comparisons(self):
        context = build_index88_context(group_logit_rows())
        self.assertEqual(context["alpha"], 0.5)
        self.assertIn("index88_baseline_margin_percentile_within_selection20", context)
        self.assertIn("holdout_logit_difference_at_least_index88_count", context)

    def test_nonfinite_logit_is_rejected(self):
        bad = metric()
        bad["kl_baseline_to_intervention"] = math.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            normalize_logit_metric(
                dataset_group="selection20",
                dataset_index=88,
                baseline_success=True,
                alpha=0.5,
                metric=bad,
            )


class ScoreSummaryTests(unittest.TestCase):
    def test_score_intervention_summary_by_head_and_group(self):
        rows = []
        for group, count in GROUP_COUNTS.items():
            for position in range(count):
                for layer, head in EXPECTED_TOP5:
                    row = {
                        "dataset_group": group,
                        "dataset_index": position,
                        "baseline_success": group == "selection20" or position < 21,
                        "alpha": 0.5,
                        "layer": layer,
                        "head": head,
                    }
                    row.update(
                        {
                            field: 1.0 + position + (0.5 if group == "holdout48" else 0)
                            for field in SCORE_FIELDS
                        }
                    )
                    rows.append(row)
        summary = summarize_score_interventions(rows)
        self.assertEqual(len(summary["heads"]), 5)
        first = summary["heads"][0]
        self.assertEqual(
            first["holdout_minus_selection"]["current_asym_frobenius_norm"]["mean"],
            14.5,
        )

    def test_nonfinite_score_is_rejected(self):
        row = {
            "dataset_group": "selection20",
            "dataset_index": 88,
            "baseline_success": True,
            "alpha": 0.5,
            "layer": 31,
            "head": 22,
            **{field: 1.0 for field in SCORE_FIELDS},
        }
        row["mean_absolute_delta"] = math.inf
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            summarize_score_interventions([row])


class HeadSubsetTests(unittest.TestCase):
    def test_single_and_leave_one_out_conditions(self):
        conditions = head_contribution_conditions()
        self.assertEqual(len(conditions), 11)
        self.assertEqual(sum(item["name"].startswith("single_") for item in conditions), 5)
        self.assertEqual(sum(item["name"].startswith("without_") for item in conditions), 5)

    def test_all_nonempty_subsets_are_unique_and_ordered_by_size(self):
        subsets = build_all_nonempty_head_subsets()
        self.assertEqual(len(subsets), 31)
        self.assertEqual(len({item["selected_heads"] for item in subsets}), 31)
        self.assertEqual([item["subset_size"] for item in subsets], sorted(item["subset_size"] for item in subsets))

    def test_minimal_refusal_subsets(self):
        rows = [
            {
                "subset_size": 1,
                "selected_heads": [{"layer": 31, "head": 22}],
                "first_token_id": 1,
                "success": True,
                "generation_failed": False,
                "response_equal_to_full_top5": False,
            },
            {
                "subset_size": 2,
                "selected_heads": [
                    {"layer": 31, "head": 22},
                    {"layer": 13, "head": 4},
                ],
                "first_token_id": 2,
                "success": False,
                "generation_failed": False,
                "response_equal_to_full_top5": True,
            },
        ]
        result = minimal_refusal_subsets(rows)
        self.assertEqual(result["minimum_subset_size"], 2)
        self.assertEqual(len(result["minimal_subsets"]), 1)


if __name__ == "__main__":
    unittest.main()
