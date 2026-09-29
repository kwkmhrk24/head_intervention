import unittest

from scripts.diagnose_gcg_dynamic_asym_layer_sign_reproducibility import (
    distribution_comparison,
    distribution_summary,
    mean_prompt_normalized_distribution,
    parse_args,
    validate_request,
)


class LayerSignReproducibilityTest(unittest.TestCase):
    def test_prompt_equal_layer_distribution(self):
        rows = [
            {"counts": [0, 1] + [0] * 30},
            {"counts": [0, 1, 3] + [0] * 29},
        ]
        distribution = mean_prompt_normalized_distribution(rows, "counts")
        self.assertEqual(distribution[0], 0)
        self.assertEqual(distribution[1], 0.625)
        self.assertEqual(distribution[2], 0.375)
        self.assertAlmostEqual(sum(distribution), 1.0)

    def test_distribution_metrics_and_bands(self):
        denominator = sum(range(1, 32))
        values = [0.0] + [layer / denominator for layer in range(1, 32)]
        comparison = distribution_comparison(values, values)
        self.assertAlmostEqual(comparison["pearson_layers_1_31"], 1.0)
        self.assertEqual(comparison["total_variation"], 0.0)
        self.assertEqual(comparison["jensen_shannon_bits"], 0.0)
        summary = distribution_summary(values)
        self.assertAlmostEqual(sum(summary["layer_bands"].values()), 1.0)
        self.assertEqual(len(summary["top5_layers"]), 5)

    def test_run_requires_output_and_preflight_is_read_only(self):
        self.assertIsNone(validate_request(parse_args([])))
        with self.assertRaisesRegex(ValueError, "output-dir"):
            validate_request(parse_args(["--stage", "run"]))


if __name__ == "__main__":
    unittest.main()
