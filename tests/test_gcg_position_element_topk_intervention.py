import math
import unittest

import torch

from scripts.evaluate_gcg_position_element_topk_intervention import parse_args
from utils.gcg_position_element_topk_intervention import (
    RANKING_METRIC,
    TOP_K_VALUES,
    PositionElementTopKController,
    build_position_element_selection,
    compute_original_mean_reference,
    compute_position_element_ranking,
    selection_diagnostics,
    validate_top_k_values,
)
from utils.gcg_wasserstein_reference import (
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    strict_lower_indices,
)


SCORE_SHAPE = (SELECTION_COUNT, NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)


class PositionElementSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = torch.zeros(SCORE_SHAPE, dtype=torch.float32)
        cls.original[:, 0, 0, 1] = torch.arange(SELECTION_COUNT).float()
        cls.jailbreak = cls.original.clone()
        cls.jailbreak[:, 2, 3, 4] -= 3.0
        cls.jailbreak[:, 0, 0, 1] += 2.0
        cls.jailbreak[:, 0, 0, 2] += 2.0
        cls.jailbreak[:10, 0, 1, 2] += 4.0
        cls.jailbreak[10:, 0, 1, 2] -= 2.0
        cls.ranking = compute_position_element_ranking(
            cls.original, cls.jailbreak
        )

    def test_ranking_uses_absolute_paired_mean_delta_and_stable_ties(self):
        self.assertEqual(RANKING_METRIC, "abs(mean_p(jailbreak_asym-original_asym))")
        identities = [
            (row["layer"], row["head"], row["position"])
            for row in self.ranking[:4]
        ]
        self.assertEqual(
            identities,
            [(2, 3, 4), (0, 0, 1), (0, 0, 2), (0, 1, 2)],
        )
        self.assertEqual([row["rank"] for row in self.ranking[:4]], [1, 2, 3, 4])
        self.assertEqual(self.ranking[0]["mean_delta"], -3.0)
        self.assertEqual(self.ranking[0]["ranking_score"], 3.0)

    def test_sign_consistency_follows_signed_mean_direction(self):
        always_negative = self.ranking[0]
        mixed_positive_mean = next(
            row
            for row in self.ranking
            if (row["layer"], row["head"], row["position"]) == (0, 1, 2)
        )
        zero_delta = next(
            row
            for row in self.ranking
            if (row["layer"], row["head"], row["position"]) == (0, 0, 0)
        )
        self.assertEqual(always_negative["mean_delta_sign"], -1)
        self.assertEqual(always_negative["sign_consistency"], 1.0)
        self.assertEqual(mixed_positive_mean["mean_delta"], 1.0)
        self.assertEqual(mixed_positive_mean["positive_rate"], 0.5)
        self.assertEqual(mixed_positive_mean["negative_rate"], 0.5)
        self.assertEqual(mixed_positive_mean["sign_consistency"], 0.5)
        self.assertEqual(zero_delta["mean_delta_sign"], 0)
        self.assertEqual(zero_delta["zero_rate"], 1.0)
        self.assertEqual(zero_delta["sign_consistency"], 1.0)

    def test_reference_is_selection20_original_mean_only(self):
        reference = compute_original_mean_reference(self.original)
        self.assertEqual(tuple(reference.shape), (32, 32, 120))
        self.assertEqual(reference.dtype, torch.float32)
        self.assertEqual(reference.device.type, "cpu")
        self.assertEqual(reference[0, 0, 1].item(), 9.5)
        modified_jailbreak = self.jailbreak + 1000.0
        second = build_position_element_selection(
            self.original,
            modified_jailbreak,
            k_values=(10,),
        )
        self.assertTrue(torch.equal(second.original_mean_reference, reference))

    def test_all_required_k_masks_are_exact_and_nested(self):
        selection = build_position_element_selection(
            self.original,
            self.jailbreak,
            k_values=TOP_K_VALUES,
        )
        previous = None
        for k in TOP_K_VALUES:
            mask = selection.mask_for_k(k)
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(mask.device.type, "cpu")
            self.assertEqual(int(mask.sum()), k)
            if previous is not None:
                self.assertFalse(bool((previous & ~mask).any()))
            previous = mask
        diagnostics = selection_diagnostics(selection)
        self.assertEqual(diagnostics["k_values"], [10, 50, 100, 500])
        self.assertEqual(
            [row["selected_element_count"] for row in diagnostics["conditions"]],
            [10, 50, 100, 500],
        )

    def test_invalid_inputs_are_rejected(self):
        self.assertEqual(validate_top_k_values(TOP_K_VALUES), TOP_K_VALUES)
        for invalid in ((0,), (10, 10), (50, 10), (200000,)):
            with self.assertRaises((TypeError, ValueError)):
                validate_top_k_values(invalid)
        bad = self.original.clone()
        bad[0, 0, 0, 0] = math.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            compute_position_element_ranking(bad, self.jailbreak)

    def test_entrypoint_defaults_to_read_only_cpu_validation(self):
        args = parse_args([])
        self.assertEqual(args.stage, "cpu-validate")
        self.assertIsNone(args.output_dir)
        self.assertEqual(tuple(args.k_values), TOP_K_VALUES)
        self.assertEqual(tuple(args.dataset_groups), ("selection20", "holdout48"))


def make_controller(alpha=0.5, *, timing="first_token_only"):
    reference = torch.zeros((2, 4, ELEMENTS_PER_HEAD), dtype=torch.float32)
    mask = torch.zeros_like(reference, dtype=torch.bool)
    reference[0, 1, 5] = 2.0
    reference[0, 3, 119] = -1.0
    mask[0, 1, 5] = True
    mask[0, 3, 119] = True
    return PositionElementTopKController(
        reference_mean=reference,
        element_mask=mask,
        alpha=alpha,
        intervention_timing=timing,
    )


class PositionElementInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.scores = torch.randn(1, 4, 20, 20, dtype=torch.float32)
        self.indices = strict_lower_indices(16)

    def test_selected_lower_values_interpolate_to_reference(self):
        controller = make_controller(alpha=0.5)
        controller.set_prompt_context(prompt_length=20)
        before = self.scores.clone()
        current = (
            before[0, 1, -16:, -16:]
            - before[0, 1, -16:, -16:].T
        ) / 2
        row = int(self.indices[0, 5])
        col = int(self.indices[1, 5])
        expected = 0.5 * current[row, col] + 0.5 * 2.0
        result = controller.apply(0, self.scores)
        observed = (
            result[0, 1, -16:, -16:]
            - result[0, 1, -16:, -16:].T
        ) / 2
        torch.testing.assert_close(observed[row, col], expected, atol=2e-7, rtol=0)

    def test_only_selected_mirrored_pairs_change(self):
        controller = make_controller(alpha=1.0)
        controller.set_prompt_context(prompt_length=20)
        before = self.scores.clone()
        result = controller.apply(0, self.scores)
        delta = result - before
        allowed = torch.zeros_like(delta, dtype=torch.bool)
        for head, position in ((1, 5), (3, 119)):
            row = int(self.indices[0, position]) + 4
            col = int(self.indices[1, position]) + 4
            allowed[0, head, row, col] = True
            allowed[0, head, col, row] = True
        self.assertEqual(float(delta.masked_select(~allowed).abs().max()), 0.0)
        self.assertTrue(bool(delta.masked_select(allowed).abs().gt(0).all()))
        summary = controller.summary()
        self.assertEqual(summary["selected_element_count"], 2)
        self.assertEqual(summary["selected_element_intervention_count"], 2)
        self.assertEqual(summary["max_unselected_element_change"], 0.0)
        self.assertEqual(summary["max_outside_tail_change"], 0.0)

    def test_antisymmetry_preserves_symmetric_component_and_diagonal(self):
        controller = make_controller(alpha=0.5)
        controller.set_prompt_context(prompt_length=20)
        before = self.scores.clone()
        result = controller.apply(0, self.scores)
        delta = result[:, :, -16:, -16:] - before[:, :, -16:, -16:]
        torch.testing.assert_close(
            delta + delta.transpose(-1, -2),
            torch.zeros_like(delta),
            atol=2e-7,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.diagonal(delta, dim1=-2, dim2=-1),
            torch.zeros((1, 4, 16)),
            atol=0,
            rtol=0,
        )
        summary = controller.summary()
        self.assertEqual(
            summary["max_constructed_delta_antisymmetry_error_float32"], 0.0
        )
        self.assertLessEqual(
            summary["max_symmetric_component_error_float32"], 2e-7
        )
        self.assertEqual(summary["max_diagonal_change_float32"], 0.0)

    def test_alpha_zero_is_exact_object_identity(self):
        controller = make_controller(alpha=0.0)
        before = self.scores.clone()
        result = controller.apply(0, self.scores)
        self.assertIs(result, self.scores)
        self.assertTrue(torch.equal(result, before))
        self.assertEqual(controller.summary()["alpha_zero_bypass_count"], 1)

    def test_first_token_only_skips_later_forwards(self):
        controller = make_controller(alpha=0.5)
        controller.set_prompt_context(prompt_length=20)
        controller.apply(0, self.scores)
        later = torch.randn(1, 4, 21, 21)
        result = controller.apply(0, later)
        self.assertIs(result, later)
        summary = controller.summary()
        self.assertTrue(summary["first_forward_intervened"])
        self.assertEqual(summary["later_forward_intervention_count"], 0)
        self.assertEqual(summary["later_forward_skip_count"], 1)

    def test_unselected_layer_and_short_sequence_are_identity(self):
        controller = make_controller(alpha=0.5, timing="dynamic_all_steps")
        self.assertIs(controller.apply(1, self.scores), self.scores)
        short = torch.randn(1, 4, 15, 15)
        self.assertIs(controller.apply(0, short), short)
        self.assertEqual(controller.summary()["short_sequence_skip_count"], 1)


if __name__ == "__main__":
    unittest.main()
