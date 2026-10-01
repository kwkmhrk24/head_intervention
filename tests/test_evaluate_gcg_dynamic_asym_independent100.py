import unittest

import torch

from scripts.diagnose_dynamic_asym_score_logit_propagation import (
    evenly_spaced_by_length,
)

from scripts.evaluate_gcg_dynamic_asym_independent100 import (
    autodan_evaluation_indices,
    fixed_selection,
    parse_args,
    summarize,
    validate_request,
)
from utils.gcg_dynamic_asym_stage2 import build_stage2_selection_plan
from utils.gcg_dynamic_asym_intervention import REPLACEMENT_MODES


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

    def test_proportional_half_percent_condition(self):
        self.assertEqual(
            fixed_selection(0.005),
            {
                "name": "proportional_0p5pct",
                "family": "proportional_top_n",
                "top_n": None,
                "rate": 0.005,
                "rate_numerator": 5,
                "tau": None,
            },
        )

    def test_run_requires_new_output_directory(self):
        args = parse_args(["--stage", "run"])
        with self.assertRaisesRegex(ValueError, "output-dir"):
            validate_request(args)

    def test_autodan_dataset_cli(self):
        args = parse_args(["--dataset", "autodan130"])
        self.assertEqual(args.dataset, "autodan130")

    def test_symmetric_component_cli(self):
        args = parse_args(["--component", "symmetric"])
        self.assertEqual(args.component, "symmetric")

    def test_replacement_mode_cli_defaults_to_reference_mean(self):
        self.assertEqual(parse_args([]).replacement_mode, "reference_mean")
        for replacement_mode in REPLACEMENT_MODES[1:]:
            args = parse_args(["--replacement-mode", replacement_mode])
            self.assertEqual(args.replacement_mode, replacement_mode)
            self.assertIsNone(validate_request(args))

    def test_zero_replacement_modes_require_asymmetric_component(self):
        for mode in REPLACEMENT_MODES[1:]:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "component antisymmetric"):
                    validate_request(
                        parse_args(
                            ["--component", "symmetric", "--replacement-mode", mode]
                        )
                    )

    def test_half_percent_rate_cli(self):
        args = parse_args(["--proportional-rate", "0.005"])
        self.assertEqual(args.proportional_rate, 0.005)

    def test_selection_direction_cli(self):
        self.assertEqual(
            parse_args(["--selection-direction", "positive"]).selection_direction,
            "positive",
        )

    def test_absolute_intervention_target_cli_and_validation(self):
        args = parse_args(["--absolute-intervention-target", "positive"])
        self.assertEqual(args.absolute_intervention_target, "positive")
        self.assertIsNone(validate_request(args))
        with self.assertRaisesRegex(ValueError, "selection-direction absolute"):
            validate_request(
                parse_args(
                    [
                        "--selection-direction",
                        "positive",
                        "--absolute-intervention-target",
                        "negative",
                    ]
                )
            )

        matched = parse_args(
            ["--absolute-intervention-target", "positive-count-matched"]
        )
        self.assertEqual(
            matched.absolute_intervention_target, "positive-count-matched"
        )

    def test_half_percent_uses_layer0_excluded_candidates_and_full_cap(self):
        plan = build_stage2_selection_plan(
            torch.arange(4000, dtype=torch.float32),
            selection=fixed_selection(0.005),
            layer_count=4,
            head_count=2,
            elements_per_head=500,
            layer0_included=False,
        )
        self.assertEqual(plan["eligible_candidate_count"], 3000)
        self.assertEqual(plan["requested_or_threshold_eligible_count"], 15)
        self.assertEqual(plan["cap_count"], 20)
        self.assertEqual(plan["selected_count"], 15)
        self.assertFalse(plan["cap_activated"])

    def test_frozen_autodan_evaluation110_partition(self):
        indices = autodan_evaluation_indices()
        self.assertEqual(len(indices), 110)
        self.assertEqual(len(set(indices)), 110)
        self.assertNotIn(1, indices)
        self.assertIn(0, indices)

    def test_minimal_summary(self):
        baseline = [{"success": True}, {"success": False}]
        rows = [
            {
                "success": False,
                "transition": "newly_defended",
                "response_equal": False,
                "nominal_selected_count": 10,
                "sign_eligible_candidate_count": 100,
                "selection_shortfall_count": 0,
                "selected_count": 10,
                "selected_fraction": 0.001,
                "selected_z": {"count": 10, "min": 1.0, "mean": 2.0, "max": 3.0},
                "positive_selected_count": 10,
                "negative_selected_count": 0,
                "cap_count": 50,
                "cap_activated": False,
                "diagnostics": {"later_forward_intervention_count": 0},
            },
            {
                "success": True,
                "transition": "newly_worsened",
                "response_equal": True,
                "nominal_selected_count": 20,
                "sign_eligible_candidate_count": 200,
                "selection_shortfall_count": 0,
                "selected_count": 20,
                "selected_fraction": 0.001,
                "selected_z": {"count": 20, "min": 2.0, "mean": 3.0, "max": 4.0},
                "positive_selected_count": 20,
                "negative_selected_count": 0,
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
        self.assertEqual(summary["nominal_selected_count_mean"], 15.0)
        self.assertEqual(summary["sign_eligible_candidate_count_mean"], 150.0)
        self.assertEqual(summary["selection_shortfall_count_total"], 0)
        self.assertAlmostEqual(summary["selected_z"]["mean"], 8 / 3)
        self.assertEqual(summary["selected_count_mean"], 15.0)
        self.assertEqual(summary["cap_violation_count"], 0)
        self.assertEqual(summary["decode_intervention_count"], 0)

    def test_diagnostic_sample_spans_length_order(self):
        rows = [
            {"dataset_index": index, "prompt_length": length}
            for index, length in enumerate((40, 10, 30, 20, 50))
        ]
        selected = evenly_spaced_by_length(rows, 3)
        self.assertEqual([row["prompt_length"] for row in selected], [10, 30, 50])


if __name__ == "__main__":
    unittest.main()
