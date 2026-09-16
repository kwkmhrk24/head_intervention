import math
import unittest

import torch

from scripts.diagnose_gcg_dynamic_asym_thresholds import (
    TAUS,
    parse_args,
    validate_request,
)
from utils.gcg_dynamic_asym_selection_diagnostics import (
    prompt_selected_count_summary,
    threshold_selection_summary,
)


class ThresholdSelectionTests(unittest.TestCase):
    def test_threshold_is_strict_and_counts_layer_head_identities(self):
        abs_z = torch.tensor([2.5, 2.6, 3.0, 2.0, 2.7, 4.0, 2.6, 2.59])
        summary, counts, selected = threshold_selection_summary(
            abs_z,
            tau=2.6,
            layer_count=2,
            head_count=2,
            elements_per_head=2,
        )
        self.assertEqual(summary["selected_count"], 3)
        torch.testing.assert_close(selected, torch.tensor([3.0, 2.7, 4.0]))
        self.assertEqual(counts["layer_counts"], [1, 2])
        self.assertEqual(counts["head_counts"], [2, 1])
        self.assertEqual(counts["layer_head_counts"], [[0, 1], [2, 0]])
        self.assertGreater(summary["selected_abs_z"]["min"], 2.6)

    def test_empty_threshold_selection_is_recorded_without_failure(self):
        summary, counts, selected = threshold_selection_summary(
            torch.tensor([0.0, 1.0, 2.0, 3.0]),
            tau=3.0,
            layer_count=1,
            head_count=2,
            elements_per_head=2,
        )
        self.assertEqual(summary["selected_count"], 0)
        self.assertEqual(summary["selected_fraction"], 0.0)
        self.assertIsNone(summary["selected_abs_z"])
        self.assertIsNone(summary["head_concentration"])
        self.assertEqual(counts["layer_head_counts"], [[0, 0]])
        self.assertEqual(selected.numel(), 0)

    def test_invalid_tau_and_nonfinite_scores_are_rejected(self):
        values = torch.ones(4)
        with self.assertRaisesRegex(ValueError, "finite"):
            threshold_selection_summary(
                values,
                tau=math.inf,
                layer_count=1,
                head_count=1,
                elements_per_head=4,
            )
        values[0] = torch.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            threshold_selection_summary(
                values,
                tau=2.5,
                layer_count=1,
                head_count=1,
                elements_per_head=4,
            )


class PromptCountSummaryTests(unittest.TestCase):
    def test_population_statistics_include_even_sample_median(self):
        summary = prompt_selected_count_summary((0, 2, 4, 6))
        self.assertEqual(summary["mean"], 3.0)
        self.assertEqual(summary["median"], 3.0)
        self.assertEqual(summary["min"], 0)
        self.assertEqual(summary["max"], 6)
        self.assertAlmostEqual(summary["std"], math.sqrt(5.0))
        self.assertEqual(summary["std_definition"], "population_ddof_0")


class ThresholdEntrypointTests(unittest.TestCase):
    def test_tau_grid_is_frozen_and_not_cli_tunable(self):
        self.assertEqual(
            TAUS,
            (2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.5),
        )
        args = parse_args([])
        self.assertEqual(args.stage, "cpu-preflight")
        self.assertFalse(hasattr(args, "tau"))
        self.assertIsNone(validate_request(args))

    def test_run_requires_new_output_directory(self):
        args = parse_args(["--stage", "run"])
        with self.assertRaisesRegex(ValueError, "output-dir"):
            validate_request(args)


if __name__ == "__main__":
    unittest.main()
