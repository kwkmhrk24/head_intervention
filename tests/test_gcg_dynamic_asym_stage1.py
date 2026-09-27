import unittest

import torch

from scripts.evaluate_gcg_dynamic_asym_stage1 import parse_args, validate_request
from utils.gcg_dynamic_asym_intervention import DynamicAsymmetryReference
from utils.gcg_dynamic_asym_stage1 import (
    CAP_RATE,
    PlannedDynamicAsymmetryController,
    build_stage1_selection_plan,
    cap_count,
    stage1_conditions,
    validate_stage1_conditions,
)


def antisymmetric_scores(length: int, value: float, *, heads: int):
    scores = torch.zeros((1, heads, length, length), dtype=torch.float32)
    rows, cols = torch.tril_indices(length, length, offset=-1)
    scores[:, :, rows, cols] = value
    scores[:, :, cols, rows] = -value
    return scores


class Stage1ConditionTests(unittest.TestCase):
    def test_nine_conditions_are_frozen_in_requested_order(self):
        conditions = stage1_conditions()
        validate_stage1_conditions(conditions)
        self.assertEqual(len(conditions), 9)
        self.assertEqual(
            tuple(row["name"] for row in conditions),
            (
                "fixed_top_250",
                "fixed_top_500",
                "fixed_top_750",
                "proportional_0p1pct",
                "proportional_0p2pct",
                "proportional_0p4pct",
                "threshold_tau_3p0",
                "threshold_tau_2p8",
                "threshold_tau_2p7",
            ),
        )
        self.assertEqual(CAP_RATE, 0.005)

    def test_cli_has_no_stage2_alpha_or_layer_zero_controls(self):
        args = parse_args([])
        self.assertEqual(args.stage, "cpu-preflight")
        self.assertFalse(hasattr(args, "alpha"))
        self.assertFalse(hasattr(args, "exclude_layer0"))
        self.assertIsNone(validate_request(args))

    def test_run_requires_new_output_directory(self):
        args = parse_args(["--stage", "run"])
        with self.assertRaisesRegex(ValueError, "output-dir"):
            validate_request(args)


class Stage1SelectionPlanTests(unittest.TestCase):
    def setUp(self):
        self.conditions = stage1_conditions()
        self.abs_z = torch.arange(400, dtype=torch.float32)
        self.structure = {
            "layer_count": 2,
            "head_count": 2,
            "elements_per_head": 100,
        }

    def test_cap_uses_exact_integer_floor(self):
        self.assertEqual(cap_count(400), 2)
        self.assertEqual(cap_count(194560), 972)

    def test_fixed_top_n_is_capped_after_global_ranking(self):
        plan = build_stage1_selection_plan(
            self.abs_z,
            condition=self.conditions[0],
            **self.structure,
        )
        self.assertEqual(plan["selected_indices"].tolist(), [399, 398])
        self.assertEqual(plan["selected_count"], 2)
        self.assertTrue(plan["cap_activated"])
        self.assertEqual(plan["cap_excluded_count"], 248)
        self.assertEqual(sum(plan["layer_counts"]), 2)

    def test_proportional_n_uses_only_this_prompt_candidate_count(self):
        plan = build_stage1_selection_plan(
            self.abs_z,
            condition=self.conditions[5],
            **self.structure,
        )
        self.assertEqual(plan["requested_or_threshold_eligible_count"], 1)
        self.assertEqual(plan["selected_indices"].tolist(), [399])
        self.assertFalse(plan["cap_activated"])

    def test_threshold_is_strict_then_capped_by_global_abs_z_rank(self):
        values = torch.zeros(400)
        values[10] = 3.0
        values[11] = 3.1
        values[12] = 4.0
        values[13] = 5.0
        plan = build_stage1_selection_plan(
            values,
            condition=self.conditions[6],
            **self.structure,
        )
        self.assertEqual(plan["threshold_eligible_count"], 3)
        self.assertEqual(plan["selected_indices"].tolist(), [13, 12])
        self.assertTrue(plan["cap_activated"])
        self.assertEqual(plan["cap_excluded_count"], 1)
        self.assertGreater(plan["selected_abs_z"]["min"], 3.0)

    def test_absolute_direction_exactly_matches_legacy_indices_and_order(self):
        signed_z = torch.arange(-200, 200, dtype=torch.float32)
        absolute_z = signed_z.abs()
        legacy = build_stage1_selection_plan(
            absolute_z,
            condition=self.conditions[0],
            **self.structure,
        )
        directional = build_stage1_selection_plan(
            absolute_z,
            signed_z=signed_z,
            selection_direction="absolute",
            condition=self.conditions[0],
            **self.structure,
        )
        self.assertEqual(directional["selected_count"], legacy["selected_count"])
        self.assertTrue(
            torch.equal(directional["selected_indices"], legacy["selected_indices"])
        )

    def test_positive_and_negative_rank_signed_tails_with_same_requested_n(self):
        signed_z = torch.arange(-1000, 1000, dtype=torch.float32)
        structure = {"layer_count": 2, "head_count": 2, "elements_per_head": 500}
        positive = build_stage1_selection_plan(
            signed_z.abs(),
            signed_z=signed_z,
            selection_direction="positive",
            condition=self.conditions[5],
            **structure,
        )
        negative = build_stage1_selection_plan(
            signed_z.abs(),
            signed_z=signed_z,
            selection_direction="negative",
            condition=self.conditions[5],
            **structure,
        )
        self.assertEqual(positive["requested_count"], 8)
        self.assertEqual(negative["requested_count"], 8)
        self.assertEqual(positive["selected_indices"].tolist(), list(range(1999, 1991, -1)))
        self.assertEqual(negative["selected_indices"].tolist(), list(range(8)))
        self.assertEqual(positive["positive_selected_count"], 8)
        self.assertEqual(positive["negative_selected_count"], 0)
        self.assertEqual(negative["positive_selected_count"], 0)
        self.assertEqual(negative["negative_selected_count"], 8)
        self.assertGreater(positive["selected_z"]["min"], 0)
        self.assertLess(negative["selected_z"]["max"], 0)

    def test_directional_shortfall_is_reported_without_top_up(self):
        signed_z = -torch.arange(1, 401, dtype=torch.float32)
        signed_z[17] = 5.0
        plan = build_stage1_selection_plan(
            signed_z.abs(),
            signed_z=signed_z,
            selection_direction="positive",
            condition=self.conditions[0],
            **self.structure,
        )
        self.assertEqual(plan["requested_count"], 250)
        self.assertEqual(plan["cap_count"], 2)
        self.assertEqual(plan["sign_eligible_candidate_count"], 1)
        self.assertEqual(plan["selected_indices"].tolist(), [17])
        self.assertEqual(plan["actual_selected_count"], 1)
        self.assertEqual(plan["selection_shortfall_count"], 1)


class PlannedControllerTests(unittest.TestCase):
    def setUp(self):
        self.reference = DynamicAsymmetryReference(
            mu=torch.zeros((2, 2), dtype=torch.float32),
            sigma=torch.ones((2, 2), dtype=torch.float32),
            metadata={},
        )
        self.condition = stage1_conditions()[0]

    def test_plan_applies_existing_pure_asym_formula_prefill_only(self):
        controller = PlannedDynamicAsymmetryController(
            reference=self.reference,
            condition=self.condition,
            alpha=1.0,
        )
        plan = {
            "condition": self.condition,
            "candidate_count": 24,
            "selected_indices": torch.tensor([0, 12], dtype=torch.int64),
            "selected_count": 2,
            "selected_fraction": 2 / 24,
            "cap_count": 2,
            "cap_rate": 0.005,
            "cap_activated": False,
            "cap_excluded_count": 0,
            "requested_or_threshold_eligible_count": 2,
            "requested_count": 2,
            "selection_direction": "absolute",
            "sign_eligible_candidate_count": 24,
            "selection_shortfall_count": 0,
            "threshold_eligible_count": None,
            "selected_abs_z": {},
            "selected_z": {},
            "positive_selected_count": 2,
            "negative_selected_count": 0,
            "layer_counts": [1, 1],
            "head_counts": [2, 0],
            "layer_head_counts": [[1, 0], [1, 0]],
        }
        controller.set_prompt_plan(prompt_length=4, plan=plan)
        first = antisymmetric_scores(4, 2.0, heads=2)
        second = antisymmetric_scores(4, 3.0, heads=2)
        first_before_sym = (first + first.transpose(-1, -2)) / 2
        second_before_sym = (second + second.transpose(-1, -2)) / 2
        first_after = controller.apply(0, first)
        second_after = controller.apply(1, second)
        self.assertTrue(
            torch.equal(
                (first_after + first_after.transpose(-1, -2)) / 2,
                first_before_sym,
            )
        )
        self.assertTrue(
            torch.equal(
                (second_after + second_after.transpose(-1, -2)) / 2,
                second_before_sym,
            )
        )
        decode = torch.zeros((1, 2, 5, 5), dtype=torch.float32)
        self.assertIs(controller.apply(0, decode), decode)
        controller.assert_complete_prefill()
        summary = controller.summary()
        self.assertEqual(summary["selected_element_intervention_count"], 2)
        self.assertEqual(summary["later_forward_intervention_count"], 0)
        self.assertEqual(summary["later_forward_skip_count"], 1)
        self.assertTrue(summary["layer0_included"])

    def test_stage1_rejects_nonunit_alpha(self):
        with self.assertRaisesRegex(ValueError, "fixed at 1.0"):
            PlannedDynamicAsymmetryController(
                reference=self.reference,
                condition=self.condition,
                alpha=0.5,
            )


if __name__ == "__main__":
    unittest.main()
