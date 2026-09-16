import unittest

import torch

from scripts.evaluate_gcg_pure_symmetric_position_element_topk_intervention import (
    INTERVENTION_VARIANT,
    validate_prompt_runtime_invariants,
)
from utils.gcg_position_element_topk_intervention import (
    PureSymmetricPositionElementTopKController,
    SymmetricLowerOnly2DeltaPositionElementTopKController,
)
from utils.gcg_wasserstein_reference import ELEMENTS_PER_HEAD, strict_lower_indices


class PureSymmetricInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.scores = torch.randn(1, 4, 20, 20, dtype=torch.float32)
        self.rows, self.cols = strict_lower_indices(16)
        self.reference = torch.zeros((2, 4, ELEMENTS_PER_HEAD), dtype=torch.float32)
        self.mask = torch.zeros_like(self.reference, dtype=torch.bool)
        self.reference[0, 1, 5] = 2.0
        self.reference[0, 3, 119] = -1.0
        self.mask[0, 1, 5] = True
        self.mask[0, 3, 119] = True

    def controller(self, alpha=1.0):
        return PureSymmetricPositionElementTopKController(
            reference_mean=self.reference,
            element_mask=self.mask,
            alpha=alpha,
            intervention_timing="first_token_only",
        )

    def component(self, scores, head, position):
        row = int(self.rows[position]) + 4
        col = int(self.cols[position]) + 4
        sym = (scores[0, head, row, col] + scores[0, head, col, row]) / 2
        asym = (scores[0, head, row, col] - scores[0, head, col, row]) / 2
        return sym, asym, row, col

    def test_pairwise_equal_delta_changes_sym_by_delta(self):
        controller = self.controller(alpha=0.5)
        controller.set_prompt_context(prompt_length=20)
        before_sym, _, row, col = self.component(self.scores, 1, 5)
        expected_delta = 0.5 * (torch.tensor(2.0) - before_sym)
        result = controller.apply(0, self.scores)
        after_sym, _, _, _ = self.component(result, 1, 5)
        lower_delta = result[0, 1, row, col] - self.scores[0, 1, row, col]
        upper_delta = result[0, 1, col, row] - self.scores[0, 1, col, row]
        torch.testing.assert_close(after_sym - before_sym, expected_delta, atol=2e-7, rtol=0)
        torch.testing.assert_close(lower_delta, expected_delta, atol=2e-7, rtol=0)
        torch.testing.assert_close(upper_delta, expected_delta, atol=2e-7, rtol=0)
        self.assertEqual(
            controller.summary()["max_constructed_pair_delta_difference_float32"],
            0.0,
        )

    def test_antisymmetric_component_is_preserved(self):
        controller = self.controller(alpha=1.0)
        controller.set_prompt_context(prompt_length=20)
        _, before_asym, _, _ = self.component(self.scores, 1, 5)
        result = controller.apply(0, self.scores)
        _, after_asym, _, _ = self.component(result, 1, 5)
        torch.testing.assert_close(after_asym, before_asym, atol=2e-7, rtol=0)
        summary = controller.summary()
        self.assertEqual(summary["max_antisymmetric_component_change_float32"], 0.0)
        self.assertLessEqual(
            summary["max_recomputed_antisymmetric_component_change_float32"],
            2e-7,
        )

    def test_alpha_one_reaches_reference(self):
        controller = self.controller(alpha=1.0)
        controller.set_prompt_context(prompt_length=20)
        result = controller.apply(0, self.scores)
        observed, _, _, _ = self.component(result, 1, 5)
        torch.testing.assert_close(observed, torch.tensor(2.0), atol=2e-7, rtol=0)
        self.assertLessEqual(
            controller.summary()["max_symmetric_target_error_float32"], 2e-7
        )

    def test_alpha_half_moves_halfway_to_reference(self):
        controller = self.controller(alpha=0.5)
        controller.set_prompt_context(prompt_length=20)
        before, _, _, _ = self.component(self.scores, 1, 5)
        result = controller.apply(0, self.scores)
        observed, _, _, _ = self.component(result, 1, 5)
        expected = 0.5 * before + 1.0
        torch.testing.assert_close(observed, expected, atol=2e-7, rtol=0)

    def test_only_selected_pairs_change(self):
        controller = self.controller(alpha=1.0)
        controller.set_prompt_context(prompt_length=20)
        result = controller.apply(0, self.scores)
        delta = result - self.scores
        allowed = torch.zeros_like(delta, dtype=torch.bool)
        for head, position in ((1, 5), (3, 119)):
            row = int(self.rows[position]) + 4
            col = int(self.cols[position]) + 4
            allowed[0, head, row, col] = True
            allowed[0, head, col, row] = True
        self.assertEqual(float(delta.masked_select(~allowed).abs().max()), 0.0)
        self.assertTrue(bool(delta.masked_select(allowed).abs().gt(0).all()))
        summary = controller.summary()
        self.assertEqual(summary["max_unselected_element_change"], 0.0)
        self.assertEqual(summary["max_unselected_upper_triangle_change"], 0.0)

    def test_diagonal_and_outside_tail_are_unchanged(self):
        controller = self.controller(alpha=1.0)
        controller.set_prompt_context(prompt_length=20)
        result = controller.apply(0, self.scores)
        delta = result - self.scores
        self.assertEqual(float(delta[:, :, :4, :].abs().max()), 0.0)
        self.assertEqual(float(delta[:, :, 4:, :4].abs().max()), 0.0)
        self.assertEqual(
            float(torch.diagonal(delta, dim1=-2, dim2=-1).abs().max()), 0.0
        )
        summary = controller.summary()
        self.assertEqual(summary["max_outside_tail_change"], 0.0)
        self.assertEqual(summary["max_diagonal_change_float32"], 0.0)

    def test_later_token_forward_is_not_intervened(self):
        controller = self.controller(alpha=1.0)
        controller.set_prompt_context(prompt_length=20)
        controller.apply(0, self.scores)
        later = torch.randn(1, 4, 21, 21)
        self.assertIs(controller.apply(0, later), later)
        self.assertEqual(controller.summary()["later_forward_intervention_count"], 0)

    def test_historical_and_pure_variants_are_distinct(self):
        historical = SymmetricLowerOnly2DeltaPositionElementTopKController(
            reference_mean=self.reference,
            element_mask=self.mask,
            alpha=1.0,
            intervention_timing="first_token_only",
        )
        pure = self.controller(alpha=1.0)
        historical.set_prompt_context(prompt_length=20)
        pure.set_prompt_context(prompt_length=20)
        historical.apply(0, self.scores)
        pure.apply(0, self.scores)
        self.assertEqual(
            historical.summary()["intervention_variant"], "sym_lower_only_2delta"
        )
        self.assertEqual(pure.summary()["intervention_variant"], INTERVENTION_VARIANT)

    def test_runtime_validator_checks_pure_invariants(self):
        diagnostics = {
            "component": "symmetric",
            "intervention_variant": "sym_pure_pairwise_delta",
            "selected_element_intervention_count": 500,
            "selected_lower_changed_count": 500,
            "selected_upper_changed_count": 500,
            "first_forward_detected": True,
            "first_forward_intervened": True,
            "later_forward_intervention_count": 0,
            "short_sequence_skip_count": 0,
            "all_scores_finite": True,
            "max_constructed_lower_delta_error_float32": 0.0,
            "max_constructed_upper_delta_error_float32": 0.0,
            "max_constructed_pair_delta_difference_float32": 0.0,
            "max_symmetric_target_error_float32": 1e-7,
            "max_antisymmetric_component_change_float32": 0.0,
            "max_recomputed_antisymmetric_component_change_float32": 1e-7,
            "max_unselected_upper_triangle_change": 0.0,
            "max_unselected_element_change": 0.0,
            "max_outside_tail_change": 0.0,
            "max_diagonal_change_float32": 0.0,
        }
        validate_prompt_runtime_invariants(diagnostics)
        diagnostics["max_constructed_pair_delta_difference_float32"] = 1e-7
        with self.assertRaisesRegex(
            RuntimeError, "max_constructed_pair_delta_difference_float32"
        ):
            validate_prompt_runtime_invariants(diagnostics)


if __name__ == "__main__":
    unittest.main()
