import math
import unittest

import torch

from scripts.evaluate_gcg_rank_element_topk_intervention import (
    ALPHA,
    DEFAULT_CONDITION,
    INTERVENTION_TIMING,
    RANK_ALL_ELEMENT_COUNT,
    REQUIRED_RUNTIME_INVARIANTS,
    _condition_for_name,
    _runtime_selection_diagnostics,
    _runtime_selection_for_condition,
    parse_args,
    validate_prompt_runtime_invariants,
)
from utils.gcg_rank_element_topk_intervention import (
    RANK_SORT,
    RANKING_METRIC,
    TOP_K,
    RankElementTopKController,
    build_rank_element_selection,
    compute_original_rank_reference,
    compute_rank_element_ranking,
    offline_rank_crossing_diagnostics,
    selection_diagnostics,
    sort_signed_rank_values,
    validate_rank_selection_payload,
)
from utils.gcg_wasserstein_reference import (
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    strict_lower_indices,
)


SCORE_SHAPE = (SELECTION_COUNT, NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)


class RankElementSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = torch.zeros(SCORE_SHAPE, dtype=torch.float32)
        ordered = torch.arange(ELEMENTS_PER_HEAD, dtype=torch.float32)
        cls.original[:, 0, 0, :] = ordered
        cls.original[:, 2, 3, :] = ordered
        cls.jailbreak = cls.original.clone()
        cls.jailbreak[:, 2, 3, :] += 3.0
        cls.jailbreak[:, 0, 0, :] += 2.0
        cls.ranking = compute_rank_element_ranking(
            cls.original, cls.jailbreak
        )

    def test_signed_sort_is_independent_for_original_and_gcg(self):
        values = torch.zeros(SCORE_SHAPE, dtype=torch.float32)
        first = torch.arange(ELEMENTS_PER_HEAD, dtype=torch.float32)
        second = first.flip(0)
        values[0, 0, 0] = second
        sorted_values = sort_signed_rank_values(values)
        torch.testing.assert_close(sorted_values[0, 0, 0], first)
        self.assertEqual(RANK_SORT, "signed_value_ascending")

        permuted = values.clone()
        permuted[0, 0, 0] = values[0, 0, 0].roll(17)
        original_sorted = sort_signed_rank_values(values)
        permuted_sorted = sort_signed_rank_values(permuted)
        torch.testing.assert_close(
            original_sorted[0, 0, 0], permuted_sorted[0, 0, 0]
        )

    def test_ranking_uses_rank_identity_and_deterministic_ties(self):
        self.assertEqual(
            RANKING_METRIC,
            "abs(mean_p(sorted_jailbreak_asym-sorted_original_asym))",
        )
        identities = [
            (
                row["layer"],
                row["head"],
                row["rank"],
                row["rank_index"],
            )
            for row in self.ranking[:3]
        ]
        self.assertEqual(
            identities,
            [(2, 3, 1, 0), (2, 3, 2, 1), (2, 3, 3, 2)],
        )
        self.assertEqual(
            [row["selection_rank"] for row in self.ranking[:3]],
            [1, 2, 3],
        )
        self.assertEqual(self.ranking[0]["mean_delta"], 3.0)
        self.assertEqual(self.ranking[0]["ranking_score"], 3.0)
        self.assertEqual(self.ranking[0]["original_rank_reference"], 0.0)

    def test_reference_is_mean_of_original_order_statistics_only(self):
        original = torch.zeros(SCORE_SHAPE, dtype=torch.float32)
        for prompt in range(SELECTION_COUNT):
            original[prompt, 0, 0, 0] = float(prompt)
            original[prompt, 0, 0, 1] = -float(prompt)
        reference = compute_original_rank_reference(original)
        self.assertEqual(tuple(reference.shape), (32, 32, 120))
        self.assertEqual(reference.dtype, torch.float32)
        self.assertEqual(reference.device.type, "cpu")
        self.assertEqual(reference[0, 0, 0].item(), -9.5)
        self.assertEqual(reference[0, 0, -1].item(), 9.5)
        self.assertTrue(
            bool((reference[..., 1:] >= reference[..., :-1]).all().item())
        )

    def test_top500_mask_and_diagnostics_are_exact(self):
        selection = build_rank_element_selection(
            self.original,
            self.jailbreak,
            k_values=(TOP_K,),
        )
        mask = selection.mask_for_k(TOP_K)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.device.type, "cpu")
        self.assertEqual(int(mask.sum().item()), TOP_K)
        diagnostics = selection_diagnostics(selection)
        condition = diagnostics["conditions"][0]
        self.assertEqual(diagnostics["candidate_rank_element_count"], 122880)
        self.assertEqual(condition["selected_rank_element_count"], TOP_K)
        self.assertEqual(len(condition["rank_band_counts"]), 10)
        self.assertEqual(
            sum(
                row["selected_rank_element_count"]
                for row in condition["rank_band_counts"]
            ),
            TOP_K,
        )
        crossing = offline_rank_crossing_diagnostics(
            self.jailbreak, selection, k=TOP_K, alpha=1.0
        )
        self.assertEqual(
            crossing["selected_rank_intervention_count"],
            SELECTION_COUNT * TOP_K,
        )
        self.assertGreaterEqual(crossing["selected_value_crossing_count"], 0)
        self.assertLessEqual(
            crossing["selected_value_crossing_count"],
            SELECTION_COUNT * TOP_K,
        )
        self.assertEqual(
            sum(
                row["selected_rank_element_count"]
                for row in condition["head_selection_counts"]
            ),
            TOP_K,
        )

    def test_nonfinite_input_is_rejected(self):
        bad = self.original.clone()
        bad[0, 0, 0, 0] = math.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            compute_rank_element_ranking(bad, self.jailbreak)

    def test_tensor_artifact_payload_identity_is_validated(self):
        selection = build_rank_element_selection(
            self.original,
            self.jailbreak,
            k_values=(TOP_K,),
        )
        records = selection.topk_records(TOP_K)
        payload = {
            "schema_version": 1,
            "method": "rank_element_topk_v1",
            "k": TOP_K,
            "rank_mask": selection.mask_for_k(TOP_K),
            "original_rank_reference": selection.original_rank_reference,
            "selected_features": torch.tensor(
                [
                    [
                        row["selection_rank"],
                        row["layer"],
                        row["head"],
                        row["rank"],
                    ]
                    for row in records
                ],
                dtype=torch.int64,
            ),
            "metadata": {},
        }
        validate_rank_selection_payload(payload)
        invalid = dict(payload)
        invalid["rank_mask"] = payload["rank_mask"].clone()
        invalid["rank_mask"][2, 3, 0] = False
        with self.assertRaisesRegex(ValueError, "cardinality"):
            validate_rank_selection_payload(invalid)

    def test_entrypoint_is_fixed_to_v1_and_excludes_independent100(self):
        args = parse_args(["--rank-artifact-dir", "/tmp/rank-artifact"])
        self.assertEqual(args.stage, "cpu-validate")
        self.assertEqual(args.condition, DEFAULT_CONDITION)
        self.assertEqual(tuple(args.dataset_groups), ("selection20", "holdout48"))
        self.assertEqual(ALPHA, 1.0)
        self.assertEqual(INTERVENTION_TIMING, "first_token_only")
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--rank-artifact-dir",
                    "/tmp/rank-artifact",
                    "--dataset-groups",
                    "independent100",
                ]
            )

    def test_rank_all_alpha_half_selects_every_rank_without_ranking(self):
        source = build_rank_element_selection(
            self.original,
            self.jailbreak,
            k_values=(TOP_K,),
        )
        condition = _condition_for_name("rank_all_alpha0.5")
        selection = _runtime_selection_for_condition(source, condition)
        mask = selection.mask_for_k(RANK_ALL_ELEMENT_COUNT)
        self.assertEqual(condition["alpha"], 0.5)
        self.assertFalse(condition["ranking_used_for_selection"])
        self.assertEqual(int(mask.sum().item()), 122880)
        self.assertTrue(bool(mask.all().item()))

        source_diagnostics = selection_diagnostics(source)
        diagnostics = _runtime_selection_diagnostics(
            source_diagnostics, condition
        )
        all_condition = diagnostics["conditions"][0]
        self.assertEqual(all_condition["selected_head_count"], 1024)
        self.assertEqual(all_condition["selected_rank_min"], 1)
        self.assertEqual(all_condition["selected_rank_max"], 120)
        self.assertEqual(
            sum(
                row["selected_rank_element_count"]
                for row in all_condition["rank_band_counts"]
            ),
            122880,
        )
        self.assertTrue(
            all(
                row["selected_rank_element_count"] == 120
                for row in all_condition["head_selection_counts"]
            )
        )

    def test_cli_accepts_only_frozen_rank_all_condition(self):
        args = parse_args(
            [
                "--rank-artifact-dir",
                "/tmp/rank-artifact",
                "--condition",
                "rank_all_alpha0.5",
            ]
        )
        self.assertEqual(args.condition, "rank_all_alpha0.5")
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--rank-artifact-dir",
                    "/tmp/rank-artifact",
                    "--condition",
                    "rank_all_alpha0.75",
                ]
            )


def make_controller(alpha=1.0, *, timing="first_token_only"):
    reference = torch.arange(ELEMENTS_PER_HEAD, dtype=torch.float32)
    reference = reference.view(1, 1, -1).repeat(2, 4, 1)
    mask = torch.zeros_like(reference, dtype=torch.bool)
    mask[0, 1, 5] = True
    mask[0, 3, 119] = True
    return RankElementTopKController(
        reference_mean=reference,
        rank_mask=mask,
        alpha=alpha,
        intervention_timing=timing,
    )


class RankElementInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.scores = torch.randn(1, 4, 20, 20, dtype=torch.float32)
        self.indices = strict_lower_indices(16)

    def test_selected_rank_uses_preintervention_position_mapping(self):
        controller = make_controller(alpha=0.5)
        controller.set_prompt_context(prompt_length=20)
        before = self.scores.clone()
        block = before[0, 1, -16:, -16:]
        current_lower = ((block - block.T) / 2)[self.indices[0], self.indices[1]]
        sorted_values, positions = torch.sort(current_lower, stable=True)
        source_position = int(positions[5].item())
        row = int(self.indices[0, source_position].item())
        col = int(self.indices[1, source_position].item())
        expected = 0.5 * sorted_values[5] + 0.5 * 5.0
        result = controller.apply(0, self.scores)
        observed_block = result[0, 1, -16:, -16:]
        observed = (observed_block - observed_block.T) / 2
        torch.testing.assert_close(observed[row, col], expected, atol=2e-7, rtol=0)

    def test_only_preintervention_rank_positions_and_mirrors_change(self):
        controller = make_controller(alpha=1.0)
        controller.set_prompt_context(prompt_length=20)
        before = self.scores.clone()
        result = controller.apply(0, self.scores)
        delta = result - before
        allowed = torch.zeros_like(delta, dtype=torch.bool)
        for head, rank_index in ((1, 5), (3, 119)):
            block = before[0, head, -16:, -16:]
            lower = ((block - block.T) / 2)[self.indices[0], self.indices[1]]
            positions = torch.sort(lower, stable=True).indices
            position = int(positions[rank_index].item())
            row = int(self.indices[0, position].item()) + 4
            col = int(self.indices[1, position].item()) + 4
            allowed[0, head, row, col] = True
            allowed[0, head, col, row] = True
        self.assertEqual(float(delta.masked_select(~allowed).abs().max()), 0.0)
        self.assertTrue(bool(delta.masked_select(allowed).abs().gt(0).all()))
        summary = controller.summary()
        self.assertEqual(summary["selected_rank_element_count"], 2)
        self.assertEqual(summary["selected_rank_intervention_count"], 2)
        self.assertEqual(summary["max_unselected_element_change"], 0.0)
        self.assertEqual(summary["max_outside_tail_change"], 0.0)

    def test_antisymmetric_delta_preserves_symmetric_component(self):
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
            summary["max_symmetric_component_error_float32"], 2e-6
        )

    def test_crossing_is_diagnosed_without_resorting_or_correction(self):
        reference = torch.arange(ELEMENTS_PER_HEAD, dtype=torch.float32).view(
            1, 1, -1
        )
        mask = torch.zeros_like(reference, dtype=torch.bool)
        mask[0, 0, 5] = True
        controller = RankElementTopKController(
            reference_mean=reference,
            rank_mask=mask,
            alpha=1.0,
        )
        scores = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
        values = torch.arange(100, 220, dtype=torch.float32)
        scores[0, 0, self.indices[0], self.indices[1]] = values
        scores[0, 0, self.indices[1], self.indices[0]] = -values
        controller.set_prompt_context(prompt_length=16)
        result = controller.apply(0, scores)
        updated = (
            result[0, 0] - result[0, 0].transpose(-1, -2)
        ) / 2
        updated_lower = updated[self.indices[0], self.indices[1]]
        self.assertEqual(updated_lower[5].item(), 5.0)
        self.assertEqual(updated_lower[4].item(), 104.0)
        summary = controller.summary()
        self.assertEqual(summary["selected_value_crossing_count"], 1)
        self.assertEqual(summary["selected_value_crossing_fraction"], 1.0)
        self.assertEqual(summary["adjacent_rank_pair_crossing_count"], 1)

    def test_first_token_only_and_alpha_zero(self):
        controller = make_controller(alpha=0.0)
        before = self.scores.clone()
        self.assertIs(controller.apply(0, self.scores), self.scores)
        self.assertTrue(torch.equal(self.scores, before))
        self.assertEqual(controller.summary()["alpha_zero_bypass_count"], 1)

        controller = make_controller(alpha=1.0)
        controller.set_prompt_context(prompt_length=20)
        controller.apply(0, self.scores)
        later = torch.randn(1, 4, 21, 21)
        self.assertIs(controller.apply(0, later), later)
        self.assertEqual(
            controller.summary()["later_forward_intervention_count"], 0
        )


class RuntimeInvariantTests(unittest.TestCase):
    def valid_diagnostics(self):
        return {
            **REQUIRED_RUNTIME_INVARIANTS,
            "selected_rank_intervention_count": TOP_K,
            "selected_value_crossing_count": 125,
            "selected_value_crossing_fraction": 0.25,
        }

    def test_required_runtime_invariants_pass(self):
        validate_prompt_runtime_invariants(self.valid_diagnostics())

    def test_bad_cardinality_or_crossing_diagnostics_fail(self):
        diagnostics = self.valid_diagnostics()
        diagnostics["selected_rank_intervention_count"] = TOP_K - 1
        with self.assertRaisesRegex(RuntimeError, "selected_rank_intervention_count"):
            validate_prompt_runtime_invariants(diagnostics)
        diagnostics = self.valid_diagnostics()
        diagnostics["selected_value_crossing_fraction"] = 0.2
        with self.assertRaisesRegex(RuntimeError, "crossing diagnostics"):
            validate_prompt_runtime_invariants(diagnostics)


if __name__ == "__main__":
    unittest.main()
