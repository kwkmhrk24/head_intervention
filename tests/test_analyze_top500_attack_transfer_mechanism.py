import math
import unittest

import torch

from scripts.analyze_top500_attack_transfer_mechanism import (
    K,
    TIE_ABSOLUTE_TOLERANCE,
    TIE_RELATIVE_TOLERANCE,
    compute_logit_diagnostics,
    distribution,
    numerical_tie_tolerance,
    parse_args,
    validate_defense_argmax_reproduction,
    vector_magnitudes,
)


class AttackTransferMechanismAnalysisTests(unittest.TestCase):
    def test_cli_defaults_to_cpu_validation(self):
        args = parse_args([])
        self.assertEqual(args.stage, "cpu-validate")
        self.assertIsNone(args.output_dir)
        self.assertIsNone(args.limit_per_dataset)

    def test_distribution_reports_requested_quantiles_and_sample_std(self):
        summary = distribution([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["median"], 2.5)
        self.assertAlmostEqual(summary["std"], math.sqrt(5 / 3))
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["q25"], 1.75)
        self.assertEqual(summary["q75"], 3.25)
        self.assertEqual(summary["q95"], 3.8499999999999996)
        self.assertEqual(summary["max"], 4.0)

    def test_vector_magnitudes_require_exact_top500(self):
        values = torch.ones(K)
        result = vector_magnitudes(values, prefix="distance")
        self.assertEqual(result["distance_mean"], 1.0)
        self.assertEqual(result["distance_median"], 1.0)
        self.assertEqual(result["distance_max"], 1.0)
        self.assertEqual(result["distance_l1"], 500.0)
        self.assertAlmostEqual(result["distance_l2"], math.sqrt(500))
        with self.assertRaisesRegex(ValueError, "exactly 500"):
            vector_magnitudes(values[:-1], prefix="distance")

    def test_logit_diagnostics_capture_margin_crossing_and_are_finite(self):
        baseline = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
        defense = torch.tensor([3.0, 3.5, 2.0, 1.0, 0.0, -1.0])
        result = compute_logit_diagnostics(baseline, defense)
        self.assertEqual(result["baseline_top1_token_id"], 0)
        self.assertEqual(result["defense_top1_token_id"], 1)
        self.assertEqual(result["baseline_top1_top2_margin"], 1.0)
        self.assertEqual(result["defense_rank_of_baseline_top1"], 2)
        self.assertEqual(result["top5_token_overlap_count"], 5)
        self.assertTrue(result["top1_changed"])
        self.assertLess(result["baseline_top1_decision_headroom_after_defense"], 0)
        self.assertTrue(
            all(
                math.isfinite(float(value))
                for value in result.values()
                if type(value) in (int, float)
            )
        )

    def test_identical_logits_have_zero_distances_and_kl(self):
        logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])
        result = compute_logit_diagnostics(logits, logits.clone())
        self.assertEqual(result["max_absolute_logit_difference"], 0.0)
        self.assertEqual(result["logit_l1_distance"], 0.0)
        self.assertEqual(result["logit_l2_distance"], 0.0)
        self.assertEqual(result["relative_logit_l2_distance"], 0.0)
        self.assertAlmostEqual(result["kl_baseline_to_defense"], 0.0)
        self.assertFalse(result["top1_changed"])

    def test_tie_tolerance_is_fixed_from_float16_precision(self):
        self.assertEqual(TIE_ABSOLUTE_TOLERANCE, 1e-6)
        self.assertEqual(TIE_RELATIVE_TOLERANCE, 2**-10)
        self.assertAlmostEqual(
            numerical_tie_tolerance(10.0, 9.0),
            1e-6 + (2**-10) * 10.0,
        )

    def test_defense_argmax_accepts_only_predeclared_two_sided_tie(self):
        row = {
            "dataset": "GCG",
            "dataset_index": 263,
            "prompt_id": "example",
            "existing_defense_first_token": 1,
            "existing_first_token_logit_metrics": {
                "intervention_top1_top2_margin": 0.0,
                "intervention_logit_at_baseline_top1": 10.0,
            },
        }
        allowed = validate_defense_argmax_reproduction(
            row=row, defense_logits=torch.tensor([10.005, 10.0, 0.0])
        )
        self.assertEqual(allowed["match_type"], "tie_allowed")
        self.assertEqual(allowed["expected_token_id"], 1)
        self.assertEqual(allowed["observed_token_id"], 0)
        rejected = dict(row)
        rejected["existing_first_token_logit_metrics"] = {
            "intervention_top1_top2_margin": 0.1,
            "intervention_logit_at_baseline_top1": 10.0,
        }
        with self.assertRaisesRegex(RuntimeError, "outside tie rule"):
            validate_defense_argmax_reproduction(
                row=rejected,
                defense_logits=torch.tensor([10.005, 10.0, 0.0]),
            )


if __name__ == "__main__":
    unittest.main()
