import unittest

import torch

from scripts.evaluate_gcg_dynamic_asym_stage2 import parse_args, validate_request
from utils.gcg_dynamic_asym_intervention import DynamicAsymmetryReference
from utils.gcg_dynamic_asym_stage2 import (
    Stage2PlannedDynamicAsymmetryController,
    build_stage2_selection_plan,
    stage2_new_conditions,
    stage2_selection_conditions,
    validate_stage2_conditions,
)


def antisymmetric_scores(length: int, value: float, *, heads: int):
    scores = torch.zeros((1, heads, length, length), dtype=torch.float32)
    rows, cols = torch.tril_indices(length, length, offset=-1)
    scores[:, :, rows, cols] = value
    scores[:, :, cols, rows] = -value
    return scores


class Stage2ConditionTests(unittest.TestCase):
    def test_four_selections_and_twelve_new_conditions_are_frozen(self):
        selections = stage2_selection_conditions()
        self.assertEqual(
            tuple(row["name"] for row in selections),
            (
                "fixed_top_500",
                "proportional_0p1pct",
                "proportional_0p4pct",
                "threshold_tau_3p0",
            ),
        )
        conditions = stage2_new_conditions()
        validate_stage2_conditions(conditions)
        self.assertEqual(len(conditions), 12)
        self.assertFalse(
            any(
                row["alpha"] == 1.0 and row["layer0_included"]
                for row in conditions
            )
        )

    def test_cli_has_no_scientific_condition_controls(self):
        args = parse_args([])
        self.assertEqual(args.stage, "cpu-preflight")
        self.assertFalse(hasattr(args, "alpha"))
        self.assertFalse(hasattr(args, "tau"))
        self.assertFalse(hasattr(args, "rate"))
        self.assertFalse(hasattr(args, "exclude_layer0"))
        self.assertIsNone(validate_request(args))

    def test_run_requires_new_output_directory(self):
        with self.assertRaisesRegex(ValueError, "output-dir"):
            validate_request(parse_args(["--stage", "run"]))


class Stage2SelectionPlanTests(unittest.TestCase):
    def setUp(self):
        self.selections = stage2_selection_conditions()
        self.structure = {
            "layer_count": 4,
            "head_count": 2,
            "elements_per_head": 500,
        }
        self.abs_z = torch.arange(4000, dtype=torch.float32)

    def test_layer0_is_removed_before_fixed_ranking_but_full_cap_is_kept(self):
        values = self.abs_z.clone()
        values[:1000] += 10000
        plan = build_stage2_selection_plan(
            values,
            selection=self.selections[0],
            layer0_included=False,
            **self.structure,
        )
        self.assertEqual(plan["candidate_count"], 4000)
        self.assertEqual(plan["eligible_candidate_count"], 3000)
        self.assertEqual(plan["excluded_candidate_count"], 1000)
        self.assertEqual(plan["cap_candidate_count"], 4000)
        self.assertEqual(plan["cap_count"], 20)
        self.assertEqual(plan["selected_count"], 20)
        self.assertEqual(plan["layer_counts"][0], 0)
        self.assertTrue((plan["selected_indices"] >= 1000).all().item())

    def test_proportional_n_uses_filtered_candidates_and_full_cap(self):
        included = build_stage2_selection_plan(
            self.abs_z,
            selection=self.selections[1],
            layer0_included=True,
            **self.structure,
        )
        excluded = build_stage2_selection_plan(
            self.abs_z,
            selection=self.selections[1],
            layer0_included=False,
            **self.structure,
        )
        self.assertEqual(included["requested_or_threshold_eligible_count"], 4)
        self.assertEqual(excluded["requested_or_threshold_eligible_count"], 3)
        self.assertEqual(included["cap_count"], excluded["cap_count"])
        self.assertEqual(excluded["layer_counts"][0], 0)

    def test_threshold_does_not_top_up_after_layer0_exclusion(self):
        values = torch.zeros(4000)
        values[0:10] = 4.0
        values[1000] = 3.5
        excluded = build_stage2_selection_plan(
            values,
            selection=self.selections[3],
            layer0_included=False,
            **self.structure,
        )
        self.assertEqual(excluded["threshold_eligible_count"], 1)
        self.assertEqual(excluded["selected_count"], 1)
        self.assertEqual(excluded["selected_indices"].tolist(), [1000])


class Stage2ControllerTests(unittest.TestCase):
    def setUp(self):
        self.reference = DynamicAsymmetryReference(
            mu=torch.zeros((2, 2), dtype=torch.float32),
            sigma=torch.ones((2, 2), dtype=torch.float32),
            metadata={},
        )
        self.selection = stage2_selection_conditions()[0]

    def _plan(self):
        return {
            "condition": self.selection,
            "candidate_count": 24,
            "eligible_candidate_count": 12,
            "excluded_candidate_count": 12,
            "cap_candidate_count": 24,
            "selected_indices": torch.tensor([12], dtype=torch.int64),
            "selected_count": 1,
            "selected_fraction": 1 / 24,
            "cap_count": 2,
            "cap_rate": 0.005,
            "cap_activated": False,
            "cap_excluded_count": 0,
            "requested_or_threshold_eligible_count": 1,
            "threshold_eligible_count": None,
            "selected_abs_z": {},
            "layer_counts": [0, 1],
            "head_counts": [1, 0],
            "layer_head_counts": [[0, 0], [1, 0]],
            "layer0_included": False,
        }

    def test_half_alpha_uses_existing_pure_asym_formula_and_prefill_only(self):
        controller = Stage2PlannedDynamicAsymmetryController(
            reference=self.reference,
            selection=self.selection,
            alpha=0.5,
            layer0_included=False,
        )
        controller.set_prompt_plan(prompt_length=4, plan=self._plan())
        layer0 = antisymmetric_scores(4, 2.0, heads=2)
        layer1 = antisymmetric_scores(4, 2.0, heads=2)
        layer0_after = controller.apply(0, layer0)
        layer1_after = controller.apply(1, layer1)
        self.assertTrue(torch.equal(layer0_after, layer0))
        before_sym = (layer1 + layer1.transpose(-1, -2)) / 2
        after_sym = (layer1_after + layer1_after.transpose(-1, -2)) / 2
        self.assertTrue(torch.equal(after_sym, before_sym))
        rows, cols = torch.tril_indices(4, 4, offset=-1)
        self.assertEqual(float(layer1_after[0, 0, rows[0], cols[0]]), 1.0)
        self.assertEqual(float(layer1_after[0, 0, cols[0], rows[0]]), -1.0)
        decode = torch.zeros((1, 2, 5, 5), dtype=torch.float32)
        self.assertIs(controller.apply(0, decode), decode)
        controller.assert_complete_prefill()
        summary = controller.summary()
        self.assertEqual(summary["alpha"], 0.5)
        self.assertFalse(summary["layer0_included"])
        self.assertEqual(summary["later_forward_intervention_count"], 0)

    def test_rejects_unrequested_alpha_and_layer0_selection(self):
        with self.assertRaisesRegex(ValueError, "0.5 or 1.0"):
            Stage2PlannedDynamicAsymmetryController(
                reference=self.reference,
                selection=self.selection,
                alpha=0.25,
                layer0_included=False,
            )
        controller = Stage2PlannedDynamicAsymmetryController(
            reference=self.reference,
            selection=self.selection,
            alpha=1.0,
            layer0_included=False,
        )
        bad_plan = self._plan()
        bad_plan["layer_counts"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "Layer 0"):
            controller.set_prompt_plan(prompt_length=4, plan=bad_plan)


if __name__ == "__main__":
    unittest.main()
