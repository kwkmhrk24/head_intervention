import unittest

import torch

from scripts.evaluate_gcg_symmetric_position_element_topk_intervention import (
    RAW_SCORE_INTERVENTION,
    structural_comparison,
    validate_prompt_runtime_invariants,
)
from utils.gcg_position_element_topk_intervention import (
    SYMMETRIC_RANKING_METRIC,
    SymmetricLowerOnly2DeltaPositionElementTopKController,
    build_position_element_selection,
)
from utils.gcg_wasserstein_reference import (
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    strict_lower_indices,
)


class SymmetricSelectionTests(unittest.TestCase):
    def test_ranking_and_reference_are_symmetric_labeled(self):
        shape = (SELECTION_COUNT, NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)
        original = torch.zeros(shape, dtype=torch.float32)
        jailbreak = original.clone()
        jailbreak[:, 2, 3, 4] = -3.0
        jailbreak[:, 0, 0, 1] = 2.0
        selection = build_position_element_selection(
            original,
            jailbreak,
            k_values=(100, 500, 1000),
            component="symmetric",
        )
        self.assertEqual(selection.component, "symmetric")
        self.assertEqual(
            (selection.ranking[0]["layer"], selection.ranking[0]["head"], selection.ranking[0]["position"]),
            (2, 3, 4),
        )
        self.assertEqual(SYMMETRIC_RANKING_METRIC, "abs(mean_p(jailbreak_sym-original_sym))")
        self.assertEqual(int(selection.mask_for_k(500).sum()), 500)
        self.assertEqual(len(selection.ranking), 122880)

    def test_structural_comparison_aligns_by_element_identity(self):
        total = NUM_LAYERS * NUM_HEADS * ELEMENTS_PER_HEAD
        base = []
        for flat in range(total):
            position = flat % ELEMENTS_PER_HEAD
            head_flat = flat // ELEMENTS_PER_HEAD
            head = head_flat % NUM_HEADS
            layer = head_flat // NUM_HEADS
            base.append(
                {
                    "flat_index": flat,
                    "layer": layer,
                    "head": head,
                    "position": position,
                    "lag": position % 15 + 1,
                    "ranking_score": float(total - flat),
                }
            )
        result = structural_comparison(base, list(base))
        self.assertEqual(result["top100_overlap"], 100)
        self.assertEqual(result["top500_overlap"], 500)
        self.assertEqual(result["top500_jaccard"], 1.0)
        self.assertEqual(result["top1000_overlap"], 1000)
        self.assertAlmostEqual(result["pearson_ranking_score_all_candidates"], 1.0)


class SymmetricInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.scores = torch.randn(1, 4, 20, 20, dtype=torch.float32)
        self.rows, self.cols = strict_lower_indices(16)
        reference = torch.zeros((2, 4, ELEMENTS_PER_HEAD), dtype=torch.float32)
        mask = torch.zeros_like(reference, dtype=torch.bool)
        reference[0, 1, 5] = 2.0
        reference[0, 3, 119] = -1.0
        mask[0, 1, 5] = True
        mask[0, 3, 119] = True
        self.controller = SymmetricLowerOnly2DeltaPositionElementTopKController(
            reference_mean=reference,
            element_mask=mask,
            alpha=1.0,
            intervention_timing="first_token_only",
        )

    def test_only_selected_lower_raw_scores_change_and_sym_hits_target(self):
        self.controller.set_prompt_context(prompt_length=20)
        before = self.scores.clone()
        result = self.controller.apply(0, self.scores)
        delta = result - before
        allowed = torch.zeros_like(delta, dtype=torch.bool)
        for head, position in ((1, 5), (3, 119)):
            row = int(self.rows[position]) + 4
            col = int(self.cols[position]) + 4
            allowed[0, head, row, col] = True
        self.assertEqual(float(delta.masked_select(~allowed).abs().max()), 0.0)
        self.assertTrue(bool(delta.masked_select(allowed).abs().gt(0).all()))
        targets = {(1, 5): 2.0, (3, 119): -1.0}
        for (head, position), target in targets.items():
            row = int(self.rows[position]) + 4
            col = int(self.cols[position]) + 4
            observed = (result[0, head, row, col] + result[0, head, col, row]) / 2
            torch.testing.assert_close(observed, torch.tensor(target), atol=3e-7, rtol=0)
        summary = self.controller.summary()
        self.assertEqual(summary["intervention_variant"], "sym_lower_only_2delta")
        self.assertEqual(summary["max_upper_triangle_change"], 0.0)
        self.assertEqual(summary["max_diagonal_change_float32"], 0.0)
        self.assertEqual(summary["max_unselected_element_change"], 0.0)
        self.assertLessEqual(summary["max_symmetric_target_error_float32"], 1e-7)
        self.assertEqual(summary["selected_lower_changed_count"], 2)
        self.assertIn("2*alpha", RAW_SCORE_INTERVENTION)

    def test_alpha_half_moves_component_halfway_with_raw_double_delta(self):
        self.controller.alpha = 0.5
        self.controller.set_prompt_context(prompt_length=20)
        before = self.scores.clone()
        position = 5
        row = int(self.rows[position]) + 4
        col = int(self.cols[position]) + 4
        current = (before[0, 1, row, col] + before[0, 1, col, row]) / 2
        result = self.controller.apply(0, self.scores)
        expected_sym = 0.5 * current + 1.0
        observed_sym = (result[0, 1, row, col] + result[0, 1, col, row]) / 2
        observed_raw_delta = result[0, 1, row, col] - before[0, 1, row, col]
        torch.testing.assert_close(observed_sym, expected_sym, atol=3e-7, rtol=0)
        torch.testing.assert_close(observed_raw_delta, 2 * (expected_sym - current), atol=3e-7, rtol=0)

    def test_later_forward_is_skipped(self):
        self.controller.set_prompt_context(prompt_length=20)
        self.controller.apply(0, self.scores)
        later = torch.randn(1, 4, 21, 21)
        self.assertIs(self.controller.apply(0, later), later)
        self.assertEqual(self.controller.summary()["later_forward_intervention_count"], 0)

    def test_runtime_validation(self):
        diagnostics = {
            "component": "symmetric",
            "intervention_variant": "sym_lower_only_2delta",
            "selected_element_intervention_count": 500,
            "selected_lower_changed_count": 500,
            "first_forward_detected": True,
            "first_forward_intervened": True,
            "later_forward_intervention_count": 0,
            "short_sequence_skip_count": 0,
            "all_scores_finite": True,
            "max_unselected_element_change": 0.0,
            "max_outside_tail_change": 0.0,
            "max_upper_triangle_change": 0.0,
            "max_diagonal_change_float32": 0.0,
            "max_symmetric_target_error_float32": 0.0,
        }
        validate_prompt_runtime_invariants(diagnostics)
        diagnostics["max_upper_triangle_change"] = 1e-7
        with self.assertRaisesRegex(RuntimeError, "max_upper_triangle_change"):
            validate_prompt_runtime_invariants(diagnostics)


if __name__ == "__main__":
    unittest.main()
