import unittest

import torch

from scripts.diagnose_gcg_autodan_logit_propagation import (
    classify_direction,
    decision_metrics,
    distribution,
    selected_score_metrics,
)


class LogitPropagationDiagnosticsTests(unittest.TestCase):
    def test_boundary_progress_uses_strongest_post_intervention_competitor(self):
        baseline = torch.tensor([3.0, 2.0, 0.0])
        intervention = torch.tensor([2.5, 2.6, 0.0])
        metrics = decision_metrics(baseline, intervention)
        self.assertEqual(metrics["baseline_top1_token_id"], 0)
        self.assertEqual(metrics["baseline_top2_token_id"], 1)
        self.assertEqual(metrics["intervention_top1_token_id"], 1)
        self.assertTrue(metrics["argmax_changed"])
        self.assertAlmostEqual(metrics["baseline_margin"], 1.0)
        self.assertAlmostEqual(metrics["intervention_residual_margin"], -0.1, places=6)
        self.assertAlmostEqual(metrics["boundary_progress"], 1.1, places=6)
        self.assertAlmostEqual(metrics["baseline_top1_delta_logit"], -0.5)

    def test_selected_score_metrics_use_realized_after_and_reject_zero_denominator(self):
        before = [0.0] * 500
        reference = [2.0] * 500
        after = [1.0] * 500
        metrics = selected_score_metrics(before, reference, after, 10.0, 2.0)
        self.assertEqual(metrics["selected_score_change_mean_abs"], 1.0)
        self.assertEqual(metrics["selected_score_change_rms"], 1.0)
        self.assertAlmostEqual(metrics["selected_score_change_l2"], 500**0.5)
        self.assertEqual(metrics["selected_reference_before_mean_abs"], 2.0)
        self.assertAlmostEqual(
            metrics["logit_l2_per_selected_score_change_l2"], 10 / 500**0.5
        )
        with self.assertRaisesRegex(ValueError, "zero"):
            selected_score_metrics(before, reference, before, 10.0, 2.0)

    def test_summary_and_hypothesis_direction_are_deterministic(self):
        first = distribution([1.0, 2.0, 3.0, 4.0])
        second = distribution([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(first, second)
        self.assertEqual(first["median"], 2.5)
        self.assertEqual(first["q25"], 1.75)
        self.assertEqual(first["q75"], 3.25)
        self.assertEqual(
            classify_direction({"bootstrap_95_ci": [1.0, 2.0]}, "higher"),
            "supported",
        )
        self.assertEqual(
            classify_direction({"bootstrap_95_ci": [-2.0, -1.0]}, "higher"),
            "not supported",
        )
        self.assertEqual(
            classify_direction({"bootstrap_95_ci": [-1.0, 1.0]}, "higher"),
            "inconclusive",
        )


if __name__ == "__main__":
    unittest.main()
