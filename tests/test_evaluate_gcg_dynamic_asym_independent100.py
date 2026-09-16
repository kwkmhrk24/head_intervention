import unittest

from scripts.evaluate_gcg_dynamic_asym_independent100 import (
    fixed_selection,
    parse_args,
    summarize,
    validate_request,
)


class Independent100DynamicAsymTest(unittest.TestCase):
    def test_frozen_condition(self):
        self.assertEqual(
            fixed_selection(),
            {
                "name": "proportional_0p1pct",
                "family": "proportional_top_n",
                "top_n": None,
                "rate": 0.001,
                "rate_numerator": 1,
                "tau": None,
            },
        )

    def test_run_requires_new_output_directory(self):
        args = parse_args(["--stage", "run"])
        with self.assertRaisesRegex(ValueError, "output-dir"):
            validate_request(args)

    def test_minimal_summary(self):
        baseline = [{"success": True}, {"success": False}]
        rows = [
            {
                "success": False,
                "transition": "newly_defended",
                "response_equal": False,
                "selected_count": 10,
                "selected_fraction": 0.001,
                "cap_count": 50,
                "cap_activated": False,
                "diagnostics": {"later_forward_intervention_count": 0},
            },
            {
                "success": True,
                "transition": "newly_worsened",
                "response_equal": True,
                "selected_count": 20,
                "selected_fraction": 0.001,
                "cap_count": 100,
                "cap_activated": False,
                "diagnostics": {"later_forward_intervention_count": 0},
            },
        ]
        summary = summarize(rows, baseline)
        self.assertEqual(summary["baseline_successful_count"], 1)
        self.assertEqual(summary["intervention_successful_count"], 1)
        self.assertEqual(summary["improvement_count"], 1)
        self.assertEqual(summary["worsening_count"], 1)
        self.assertEqual(summary["response_changed_count"], 1)
        self.assertEqual(summary["selected_count_mean"], 15.0)
        self.assertEqual(summary["cap_violation_count"], 0)
        self.assertEqual(summary["decode_intervention_count"], 0)


if __name__ == "__main__":
    unittest.main()
