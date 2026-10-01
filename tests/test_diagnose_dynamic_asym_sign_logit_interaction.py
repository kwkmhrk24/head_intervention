import unittest

import torch

from scripts.diagnose_dynamic_asym_sign_logit_interaction import (
    aggregate_summary,
    cliffs_delta,
    interaction_metrics,
    logit_condition_metrics,
    numeric_distribution,
    parse_args,
    validate_request,
)


class DynamicAsymSignLogitInteractionTest(unittest.TestCase):
    def test_exact_additive_interaction_is_zero(self):
        baseline = torch.tensor([3.0, 2.0, 0.0])
        positive = torch.tensor([4.0, 2.0, 0.0])
        negative = torch.tensor([3.0, 2.5, 0.0])
        absolute_all = torch.tensor([4.0, 2.5, 0.0])
        metrics, additive = interaction_metrics(
            baseline, positive, negative, absolute_all
        )
        self.assertEqual(metrics["interaction_l2"], 0.0)
        self.assertEqual(metrics["interaction_max_absolute"], 0.0)
        self.assertEqual(metrics["interaction_over_delta_all"], 0.0)
        self.assertTrue(torch.equal(additive, absolute_all.double()))

    def test_zero_all_delta_has_defined_null_ratio(self):
        baseline = torch.tensor([3.0, 2.0])
        metrics, _ = interaction_metrics(
            baseline,
            torch.tensor([4.0, 2.0]),
            baseline,
            baseline,
        )
        self.assertIsNone(metrics["interaction_over_delta_all"])
        self.assertGreater(metrics["interaction_l2"], 0.0)

    def test_condition_metrics_use_baseline_argmax(self):
        metrics = logit_condition_metrics(torch.tensor([1.0, 3.0, 2.0]), 0)
        self.assertEqual(metrics["top1_token_id"], 1)
        self.assertEqual(metrics["top2_token_id"], 2)
        self.assertEqual(metrics["top1_top2_margin"], 1.0)
        self.assertTrue(metrics["argmax_changed_from_baseline"])

    def test_distribution_and_effect_size(self):
        summary = numeric_distribution([1.0, 2.0, 3.0])
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["median"], 2.0)
        self.assertAlmostEqual(summary["population_std"], (2 / 3) ** 0.5)
        self.assertEqual(cliffs_delta([3.0, 4.0], [1.0, 2.0]), 1.0)

    def test_aggregate_uses_frozen_group_counts(self):
        rows = []
        for index in range(53):
            defended = index < 21
            add_changed = index % 2 == 0
            actual_changed = index % 3 != 0
            rows.append(
                {
                    "dataset_index": 220 + index,
                    "absolute_all_defended": defended,
                    "all_only_defended": index < 3,
                    "logit_change": {
                        "interaction_over_delta_all": 2.0 if defended else 1.0,
                        "interaction_l2": 20.0 if defended else 10.0,
                    },
                    "conditions": {
                        "additive_prediction": {
                            "top1_token_id": 1 if add_changed else 0,
                            "argmax_changed_from_baseline": add_changed,
                        },
                        "all": {
                            "top1_token_id": 1 if actual_changed else 0,
                            "argmax_changed_from_baseline": actual_changed,
                        },
                    },
                }
            )
        summary = aggregate_summary(rows)
        self.assertEqual(summary["defended"]["prompt_count"], 21)
        self.assertEqual(summary["not_defended"]["prompt_count"], 32)
        self.assertEqual(summary["all_only_defended"]["prompt_count"], 3)
        self.assertEqual(
            summary["all_baseline_success"]["interaction_ratio_undefined_count"],
            0,
        )
        self.assertEqual(
            summary["defended_vs_not_defended_effect_size"]["interaction_ratio"],
            1.0,
        )
        self.assertEqual(
            sum(summary["additive_vs_actual_all_decision_classes"].values()),
            53,
        )

    def test_run_requires_output_and_preflight_is_read_only(self):
        self.assertIsNone(validate_request(parse_args([])))
        self.assertEqual(parse_args(["--scope", "baseline-success53"]).scope, "baseline-success53")
        with self.assertRaisesRegex(ValueError, "output-dir"):
            validate_request(parse_args(["--stage", "run"]))


if __name__ == "__main__":
    unittest.main()
