import unittest

import torch

from scripts.evaluate_gcg_rank_distribution_shift_cancellation import (
    DATASET_GROUPS,
    INTERVENTION_TIMING,
    parse_args,
)
from scripts.prepare_gcg_rank_distribution_shift_cancellation import (
    parse_args as parse_prepare_args,
)
from utils.gcg_rank_distribution_shift_cancellation import (
    ALL_RANK_COUNT,
    ALPHA_DEFAULT,
    METHOD,
    RankDistributionShiftCancellationController,
    compute_mean_shift,
    offline_shift_crossing_diagnostics,
    validate_shift_payload,
)
from utils.gcg_wasserstein_reference import (
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    strict_lower_indices,
)


SCORE_SHAPE = (SELECTION_COUNT, NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)
SHIFT_SHAPE = (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)


def make_scores(values):
    """Place one unsorted head's strict-lower values in raw S with S_sym=0.25."""

    rows, cols = strict_lower_indices(16)
    scores = torch.zeros((1, NUM_HEADS, 20, 20), dtype=torch.float32)
    scores[0, 0, -16 + rows, -16 + cols] = values + 0.25
    scores[0, 0, -16 + cols, -16 + rows] = -values + 0.25
    return scores


class ShiftConstructionTests(unittest.TestCase):
    def test_independent_signed_sort_and_paired_mean_delta(self):
        original = torch.zeros(SCORE_SHAPE, dtype=torch.float32)
        gcg = torch.zeros_like(original)
        original[0, 0, 0, :3] = torch.tensor([3.0, -2.0, 1.0])
        gcg[0, 0, 0, :3] = torch.tensor([2.0, 4.0, -3.0])
        original[1, 0, 0, :3] = torch.tensor([5.0, -1.0, 2.0])
        gcg[1, 0, 0, :3] = torch.tensor([3.0, 6.0, -4.0])
        # The source positions differ, and GCG must be sorted separately.
        expected = (
            torch.sort(gcg, dim=-1, stable=True).values
            - torch.sort(original, dim=-1, stable=True).values
        ).mean(0)
        shift = compute_mean_shift(original, gcg)
        torch.testing.assert_close(shift, expected, rtol=0, atol=0)
        self.assertEqual(tuple(shift.shape), SHIFT_SHAPE)
        self.assertAlmostEqual(shift[0, 0, 0].item(), -0.2, places=6)
        self.assertAlmostEqual(shift[0, 0, -1].item(), 0.1, places=6)

    def test_artifact_requires_method_hashes_and_conditions(self):
        metadata = {
            "source_selection_manifest_sha256": "a" * 64,
            "source_score_artifact_sha256": "b" * 64,
            "original_score_sha256": "c" * 64,
            "jailbreak_score_sha256": "d" * 64,
            "selection_prompt_count": SELECTION_COUNT,
            "rank_sort": "signed_value_ascending",
            "rank_tie_break": "stable strict-lower row-major position order",
            "alpha_default": ALPHA_DEFAULT,
            "scientific_conditions": {"selection_data": "Selection20"},
        }
        payload = {"schema_version": 1, "method": METHOD,
                   "mean_shift": torch.zeros(SHIFT_SHAPE), "metadata": metadata}
        validate_shift_payload(payload)
        invalid = dict(payload)
        invalid["metadata"] = dict(metadata, selection_prompt_count=48)
        with self.assertRaisesRegex(ValueError, "Selection20"):
            validate_shift_payload(invalid)

    def test_development_entrypoints_reject_independent100(self):
        args = parse_args(["--shift-artifact-dir", "/tmp/shift"])
        self.assertEqual(args.stage, "cpu-validate")
        self.assertEqual(args.alpha, 1.0)
        self.assertEqual(tuple(args.dataset_groups), DATASET_GROUPS)
        self.assertEqual(INTERVENTION_TIMING, "first_token_only")
        self.assertEqual(ALL_RANK_COUNT, 122880)
        with self.assertRaises(SystemExit):
            parse_args(["--shift-artifact-dir", "/tmp/shift",
                        "--dataset-groups", "independent100"])
        with self.assertRaises(SystemExit):
            parse_prepare_args(["--output-dir", "/tmp/out",
                                "--dataset-groups", "independent100"])


class ShiftRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.cols = strict_lower_indices(16)
        self.physical_values = torch.arange(120, dtype=torch.float32).roll(17)
        self.scores = make_scores(self.physical_values)
        self.shift = torch.zeros(SHIFT_SHAPE, dtype=torch.float32)
        self.shift[0, 0, 1] = 2.0  # Q'(1)=-1, crossing Q'(0)=0.

    def controller(self, *, alpha=1.0):
        controller = RankDistributionShiftCancellationController(
            mean_shift=self.shift, alpha=alpha
        )
        controller.set_prompt_context(prompt_length=20)
        return controller

    def test_runtime_formula_scatter_and_no_resort_on_crossing(self):
        controller = self.controller()
        before = self.scores.clone()
        lower_before = ((before[0, 0, -16:, -16:]
                         - before[0, 0, -16:, -16:].T) / 2)[self.rows, self.cols]
        sorted_before, positions = torch.sort(lower_before, stable=True)
        result = controller.apply(0, before)
        lower_after = ((result[0, 0, -16:, -16:]
                        - result[0, 0, -16:, -16:].T) / 2)[self.rows, self.cols]
        expected = sorted_before - self.shift[0, 0]
        torch.testing.assert_close(lower_after[positions], expected, rtol=0, atol=0)
        self.assertEqual(lower_after[positions[0]].item(), 0.0)
        self.assertEqual(lower_after[positions[1]].item(), -1.0)
        self.assertGreater(controller.summary()["selected_value_crossing_count"], 0)
        self.assertGreater(controller.summary()["adjacent_rank_pair_crossing_count"], 0)
        self.assertEqual(controller.summary()["selected_rank_intervention_count"],
                         NUM_HEADS * ELEMENTS_PER_HEAD)

    def test_antisymmetric_raw_delta_symmetric_diagonal_and_outer_region(self):
        controller = self.controller(alpha=0.5)
        before = self.scores.clone()
        result = controller.apply(0, before)
        delta = result - before
        torch.testing.assert_close(delta + delta.transpose(-1, -2),
                                   torch.zeros_like(delta), rtol=0, atol=0)
        block_before = before[0, 0, -16:, -16:]
        block_after = result[0, 0, -16:, -16:]
        torch.testing.assert_close((block_after + block_after.T) / 2,
                                   (block_before + block_before.T) / 2,
                                   rtol=0, atol=0)
        torch.testing.assert_close(torch.diagonal(result, dim1=-2, dim2=-1),
                                   torch.diagonal(before, dim1=-2, dim2=-1),
                                   rtol=0, atol=0)
        self.assertTrue(torch.equal(result[:, :, :4, :], before[:, :, :4, :]))
        self.assertTrue(torch.equal(result[:, :, -16:, :4], before[:, :, -16:, :4]))
        self.assertEqual(controller.summary()["max_constructed_delta_antisymmetry_error_float32"], 0.0)
        self.assertEqual(controller.summary()["max_symmetric_component_error_float32"], 0.0)

    def test_first_token_only_and_alpha_zero_complete_noop(self):
        active = self.controller()
        active.apply(0, self.scores)
        later = torch.randn((1, NUM_HEADS, 21, 21), dtype=torch.float32)
        self.assertIs(active.apply(0, later), later)
        self.assertEqual(active.summary()["later_forward_intervention_count"], 0)
        self.assertEqual(active.summary()["later_forward_skip_count"], 1)
        zero = self.controller(alpha=0.0)
        before = self.scores.clone()
        self.assertIs(zero.apply(0, self.scores), self.scores)
        self.assertTrue(torch.equal(self.scores, before))
        self.assertEqual(zero.summary()["selected_rank_intervention_count"], 0)
        self.assertEqual(zero.summary()["alpha_zero_bypass_count"], 1)

    def test_ties_use_stable_strict_lower_position_order(self):
        tied_scores = make_scores(torch.zeros(120, dtype=torch.float32))
        tied_shift = torch.zeros(SHIFT_SHAPE, dtype=torch.float32)
        tied_shift[0, 0, 0] = 1.0
        controller = RankDistributionShiftCancellationController(
            mean_shift=tied_shift, alpha=1.0
        )
        controller.set_prompt_context(prompt_length=20)
        result = controller.apply(0, tied_scores)
        asym = (result[0, 0, -16:, -16:]
                - result[0, 0, -16:, -16:].T) / 2
        lower = asym[self.rows, self.cols]
        self.assertEqual(lower[0].item(), -1.0)
        self.assertTrue(torch.equal(lower[1:], torch.zeros(119)))

    def test_common_shift_preserves_prompt_difference(self):
        a = torch.arange(120, dtype=torch.float32)
        b = torch.arange(120, dtype=torch.float32) + 7
        shift = torch.arange(120, dtype=torch.float32) * 0.5
        torch.testing.assert_close((a - shift) - (b - shift), a - b,
                                   rtol=0, atol=0)

    def test_offline_crossing_is_detected_without_correction(self):
        values = torch.zeros(SCORE_SHAPE, dtype=torch.float32)
        values[:, 0, 0] = torch.arange(120, dtype=torch.float32)
        result = offline_shift_crossing_diagnostics(values, self.shift)
        self.assertEqual(result["rank_element_count"], SELECTION_COUNT * ALL_RANK_COUNT)
        self.assertEqual(result["adjacent_rank_pair_crossing_count"], SELECTION_COUNT)
        self.assertEqual(result["rank_value_crossing_count"], 2 * SELECTION_COUNT)
        self.assertEqual(result["prompt_with_crossing_count"], SELECTION_COUNT)
        self.assertEqual(result["definition"],
                         "strict adjacent inversion in Q-alpha*D; no correction or re-sort")


if __name__ == "__main__":
    unittest.main()
