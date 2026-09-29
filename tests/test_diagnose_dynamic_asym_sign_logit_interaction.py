import unittest

import torch

from scripts.diagnose_dynamic_asym_sign_logit_interaction import (
    interaction_metrics,
    logit_condition_metrics,
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

    def test_run_requires_output_and_preflight_is_read_only(self):
        self.assertIsNone(validate_request(parse_args([])))
        with self.assertRaisesRegex(ValueError, "output-dir"):
            validate_request(parse_args(["--stage", "run"]))


if __name__ == "__main__":
    unittest.main()
