import unittest

import torch

from scripts.diagnose_gcg_dynamic_asym_selection import parse_args, validate_request
from utils.gcg_dynamic_asym_selection_diagnostics import (
    FullPromptAbsZCollector,
    candidate_count,
    compute_leave_one_out_statistics,
    concentration_summary,
    distribution_summary,
    patch_llama_abs_z_collection,
    proportional_top_n_allocation,
    selection_count_summary,
    stable_top_indices,
)
from utils.reference_asym_score_intervention import (
    ReferenceInterventionRuntimeError,
)


def antisymmetric_scores(length: int, value: float, *, heads: int):
    scores = torch.zeros((1, heads, length, length), dtype=torch.float32)
    rows, cols = torch.tril_indices(length, length, offset=-1)
    scores[:, :, rows, cols] = value
    scores[:, :, cols, rows] = -value
    return scores


class ProportionalAllocationTests(unittest.TestCase):
    def test_candidate_count_uses_all_layers_heads_and_full_strict_lower(self):
        self.assertEqual(candidate_count(20), 32 * 32 * 190)
        self.assertEqual(candidate_count(29), 32 * 32 * 406)

    def test_largest_remainder_ties_use_selection_position(self):
        self.assertEqual(
            proportional_top_n_allocation((1, 1, 2), total_budget=2),
            (1, 0, 1),
        )
        self.assertEqual(
            proportional_top_n_allocation((1, 1, 1), total_budget=2),
            (1, 1, 0),
        )

    def test_realistic_allocation_is_exact_and_deterministic(self):
        lengths = (20, 24, 21, 26, 21, 21, 21, 22, 29, 22)
        counts = tuple(candidate_count(length) for length in lengths)
        first = proportional_top_n_allocation(counts, total_budget=5000)
        second = proportional_top_n_allocation(counts, total_budget=5000)
        self.assertEqual(first, second)
        self.assertEqual(sum(first), 5000)
        self.assertTrue(all(0 < selected < count for selected, count in zip(first, counts)))


class LeaveOneOutTests(unittest.TestCase):
    def setUp(self):
        means = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64).view(3, 1, 1)
        self.means = means
        self.mean_squares = means.square() + 1.0

    def test_every_prompt_is_excluded_from_its_own_reference(self):
        results = compute_leave_one_out_statistics(self.means, self.mean_squares)
        self.assertEqual(len(results), 3)
        expected_mu = (3.0, 2.0, 1.0)
        for position, (result, expected) in enumerate(zip(results, expected_mu)):
            self.assertEqual(result["excluded_selection_position"], position)
            self.assertNotIn(position, result["included_selection_positions"])
            self.assertEqual(result["reference_prompt_count"], 2)
            self.assertTrue(result["self_excluded"])
            self.assertEqual(result["mu"].item(), expected)

    def test_excluded_prompt_cannot_change_its_loo_statistics(self):
        baseline = compute_leave_one_out_statistics(
            self.means, self.mean_squares
        )[1]
        modified_means = self.means.clone()
        modified_squares = self.mean_squares.clone()
        modified_means[1] = 1000.0
        modified_squares[1] = 1000001.0
        modified = compute_leave_one_out_statistics(
            modified_means, modified_squares
        )[1]
        torch.testing.assert_close(modified["mu"], baseline["mu"])
        torch.testing.assert_close(modified["sigma"], baseline["sigma"])


class RankingAndSummaryTests(unittest.TestCase):
    def test_top_selection_is_descending_and_ties_use_flat_identity(self):
        values = torch.tensor([1.0, 4.0, 4.0, 2.0, 4.0])
        selected = stable_top_indices(values, 4)
        self.assertEqual(selected.tolist(), [1, 2, 4, 3])

    def test_selection_counts_map_flat_identity_to_layer_and_head(self):
        # layer=0/head=0/position=0, layer=0/head=1/position=1,
        # layer=1/head=0/position=0 for E=3, H=2.
        selected = torch.tensor([0, 4, 6], dtype=torch.int64)
        summary = selection_count_summary(
            selected,
            layer_count=2,
            head_count=2,
            elements_per_head=3,
        )
        self.assertEqual(summary["layer_counts"], [2, 1])
        self.assertEqual(summary["head_counts"], [2, 1])
        self.assertEqual(summary["layer_head_counts"], [[1, 1], [1, 0]])

    def test_distribution_and_concentration_summaries_are_finite(self):
        distribution = distribution_summary(torch.arange(1, 101).float())
        self.assertEqual(distribution["count"], 100)
        self.assertEqual(distribution["min"], 1.0)
        self.assertEqual(distribution["max"], 100.0)
        self.assertEqual(distribution["nonfinite_count"], 0)
        concentration = concentration_summary((50, 30, 20), labels=(0, 1, 2))
        self.assertEqual(concentration["max_group_count"], 50)
        self.assertEqual(concentration["top1_fraction"], 0.5)
        self.assertAlmostEqual(concentration["top5_fraction"], 1.0)


class AbsZCollectorTests(unittest.TestCase):
    def test_collector_uses_full_strict_lower_and_never_changes_scores(self):
        collector = FullPromptAbsZCollector(layer_count=2, head_count=2)
        mu = torch.zeros((2, 2), dtype=torch.float32)
        sigma = torch.ones((2, 2), dtype=torch.float32)
        collector.set_prompt_context(
            prompt_length=20,
            excluded_selection_position=0,
            mu=mu,
            sigma=sigma,
        )
        first = antisymmetric_scores(20, 2.0, heads=2)
        second = antisymmetric_scores(20, 3.0, heads=2)
        self.assertIs(collector.apply(0, first), first)
        self.assertIs(collector.apply(1, second), second)
        abs_z, runtime = collector.result()
        self.assertEqual(abs_z.numel(), 2 * 2 * 190)
        self.assertTrue(torch.equal(abs_z[: 2 * 190], torch.full((380,), 2.0)))
        self.assertTrue(torch.equal(abs_z[2 * 190 :], torch.full((380,), 3.0)))
        self.assertEqual(runtime["strict_lower_elements_per_head"], 190)
        self.assertEqual(runtime["pre_score_nonfinite_count"], 0)
        self.assertEqual(runtime["z_nonfinite_count"], 0)

    def test_missing_layer_and_nonfinite_scores_fail_fast(self):
        collector = FullPromptAbsZCollector(layer_count=2, head_count=1)
        collector.set_prompt_context(
            prompt_length=3,
            excluded_selection_position=0,
            mu=torch.zeros((2, 1), dtype=torch.float32),
            sigma=torch.ones((2, 1), dtype=torch.float32),
        )
        scores = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        collector.apply(0, scores)
        with self.assertRaisesRegex(ReferenceInterventionRuntimeError, "every"):
            collector.result()
        collector.clear()
        collector.set_prompt_context(
            prompt_length=3,
            excluded_selection_position=0,
            mu=torch.zeros((2, 1), dtype=torch.float32),
            sigma=torch.ones((2, 1), dtype=torch.float32),
        )
        scores[0, 0, 2, 0] = torch.inf
        with self.assertRaisesRegex(ReferenceInterventionRuntimeError, "NaN or infinity"):
            collector.apply(0, scores)


class DiagnosticEntrypointTests(unittest.TestCase):
    def test_default_is_read_only_and_has_no_tau_argument(self):
        args = parse_args([])
        self.assertEqual(args.stage, "cpu-preflight")
        self.assertIsNone(args.output_dir)
        self.assertFalse(hasattr(args, "tau"))
        self.assertIsNone(validate_request(args))

    def test_run_requires_new_output_directory(self):
        args = parse_args(["--stage", "run"])
        with self.assertRaisesRegex(ValueError, "output-dir"):
            validate_request(args)


class TinyLlamaNoInterventionTests(unittest.TestCase):
    def test_patched_forward_matches_unpatched_forward_exactly(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config).half().eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.inference_mode():
            baseline = model.model(
                input_ids=input_ids, use_cache=False, return_dict=True
            ).last_hidden_state
        with patch_llama_abs_z_collection(
            model, layer_count=2, head_count=2
        ) as collector:
            collector.set_prompt_context(
                prompt_length=4,
                excluded_selection_position=0,
                mu=torch.zeros((2, 2), dtype=torch.float32),
                sigma=torch.ones((2, 2), dtype=torch.float32),
            )
            with torch.inference_mode():
                observed = model.model(
                    input_ids=input_ids, use_cache=False, return_dict=True
                ).last_hidden_state
            abs_z, runtime = collector.result()
        torch.testing.assert_close(observed, baseline, atol=0, rtol=0)
        self.assertEqual(abs_z.numel(), 2 * 2 * 6)
        self.assertTrue(runtime["all_finite"])


if __name__ == "__main__":
    unittest.main()
