import math
import unittest

import torch

from scripts.prepare_gcg_dynamic_asym_reference import parse_args, validate_request
from utils.gcg_dynamic_asym_intervention import (
    NUMERICAL_SIGMA_FLOOR,
    DynamicAsymmetryReference,
    DynamicAsymmetryController,
    FullPromptAsymmetryMomentCollector,
    compute_equal_prompt_reference_statistics,
    full_prompt_asymmetry_moments,
    patch_llama_dynamic_asymmetry_attention,
    validate_tau,
)
from utils.reference_asym_score_intervention import (
    ReferenceInterventionRuntimeError,
)


def scores_with_constant_lower(length: int, value: float, heads: int = 1):
    scores = torch.zeros((1, heads, length, length), dtype=torch.float32)
    rows, cols = torch.tril_indices(length, length, offset=-1)
    scores[:, :, rows, cols] = value
    scores[:, :, cols, rows] = -value
    return scores


class FullPromptReferenceTests(unittest.TestCase):
    def test_full_strict_lower_includes_positions_outside_tail16(self):
        scores = torch.zeros((1, 2, 20, 20), dtype=torch.float32)
        scores[0, 0, 2, 0] = 6.0
        means, mean_squares, count = full_prompt_asymmetry_moments(scores)
        self.assertEqual(count, 20 * 19 // 2)
        self.assertEqual(tuple(means.shape), (2,))
        self.assertAlmostEqual(means[0].item(), 3.0 / count)
        self.assertAlmostEqual(mean_squares[0].item(), 9.0 / count)
        self.assertEqual(means[1].item(), 0.0)

    def test_variable_length_prompts_are_equal_weighted(self):
        short_m, short_q, short_count = full_prompt_asymmetry_moments(
            scores_with_constant_lower(2, 1.0)
        )
        long_m, long_q, long_count = full_prompt_asymmetry_moments(
            scores_with_constant_lower(5, 3.0)
        )
        self.assertEqual(short_count, 1)
        self.assertEqual(long_count, 10)
        stats = compute_equal_prompt_reference_statistics(
            torch.stack((short_m, long_m)).view(2, 1, 1),
            torch.stack((short_q, long_q)).view(2, 1, 1),
            expected_prompt_count=2,
        )
        self.assertEqual(stats["mu"].item(), 2.0)
        self.assertEqual(stats["mean_q"].item(), 5.0)
        self.assertEqual(stats["variance"].item(), 1.0)
        self.assertEqual(stats["sigma"].item(), 1.0)
        pooled_element_mean = (1.0 * short_count + 3.0 * long_count) / (
            short_count + long_count
        )
        self.assertNotEqual(stats["mu"].item(), pooled_element_mean)

    def test_negative_variance_roundoff_is_clamped_but_material_error_fails(self):
        means = torch.ones((2, 1, 1), dtype=torch.float64)
        tiny_roundoff = torch.full((2, 1, 1), 1.0 - 1e-15, dtype=torch.float64)
        stats = compute_equal_prompt_reference_statistics(
            means, tiny_roundoff, expected_prompt_count=2
        )
        self.assertEqual(stats["variance"].item(), 0.0)
        self.assertEqual(stats["sigma"].item(), 0.0)
        self.assertEqual(stats["clamped_negative_variance_count"], 1)
        materially_invalid = torch.full((2, 1, 1), 0.9, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "mean-square"):
            compute_equal_prompt_reference_statistics(
                means, materially_invalid, expected_prompt_count=2
            )

    def test_nonfinite_reference_inputs_fail(self):
        means = torch.zeros((2, 1, 1), dtype=torch.float64)
        squares = torch.ones_like(means)
        means[0, 0, 0] = math.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            compute_equal_prompt_reference_statistics(
                means, squares, expected_prompt_count=2
            )

    def test_collector_requires_every_layer_once(self):
        collector = FullPromptAsymmetryMomentCollector(layer_count=2, head_count=1)
        collector.set_prompt_context(prompt_length=4)
        scores = scores_with_constant_lower(4, 2.0)
        self.assertIs(collector.apply(0, scores), scores)
        with self.assertRaisesRegex(ReferenceInterventionRuntimeError, "every"):
            collector.result()
        collector.apply(1, scores)
        means, mean_squares, count = collector.result()
        self.assertEqual(tuple(means.shape), (2, 1))
        self.assertTrue(torch.equal(means, torch.full_like(means, 2.0)))
        self.assertTrue(torch.equal(mean_squares, torch.full_like(mean_squares, 4.0)))
        self.assertEqual(count, 6)


class DynamicAsymmetryControllerTests(unittest.TestCase):
    def make_controller(self, *, tau=0.5, alpha=1.0, sigma_value=1.0):
        return DynamicAsymmetryController(
            mu=torch.zeros((2, 3), dtype=torch.float32),
            sigma=torch.full((2, 3), sigma_value, dtype=torch.float32),
            tau=tau,
            alpha=alpha,
        )

    def test_selected_pair_preserves_sym_and_alpha_one_reaches_mu(self):
        controller = self.make_controller()
        scores = torch.zeros((1, 3, 20, 20), dtype=torch.float32)
        # (2,0) is outside the historical tail16 block (absolute rows 4..19).
        scores[0, 1, 2, 0] = 4.0
        before = scores.clone()
        controller.set_prompt_context(prompt_length=20)
        result = controller.apply(0, scores)
        before_sym = (before + before.transpose(-1, -2)) / 2
        after_sym = (result + result.transpose(-1, -2)) / 2
        torch.testing.assert_close(after_sym, before_sym, atol=0, rtol=0)
        after_asym = (result - result.transpose(-1, -2)) / 2
        self.assertEqual(after_asym[0, 1, 2, 0].item(), 0.0)
        self.assertEqual(result[0, 1, 2, 0].item(), 2.0)
        self.assertEqual(result[0, 1, 0, 2].item(), 2.0)
        summary = controller.summary()
        self.assertEqual(summary["strict_lower_element_count_per_head"], 190)
        self.assertEqual(summary["selected_element_intervention_count"], 1)
        self.assertEqual(summary["max_symmetric_component_error_float32"], 0.0)
        self.assertEqual(summary["max_asym_expected_error_float32"], 0.0)
        self.assertEqual(summary["max_selected_distance_to_mu_after_float32"], 0.0)
        self.assertTrue(summary["all_scores_finite"])

    def test_alpha_half_moves_selected_asym_halfway(self):
        controller = self.make_controller(alpha=0.5)
        scores = torch.zeros((1, 3, 5, 5), dtype=torch.float32)
        scores[0, 0, 4, 1] = 4.0
        before_asym = (scores[0, 0, 4, 1] - scores[0, 0, 1, 4]) / 2
        controller.set_prompt_context(prompt_length=5)
        result = controller.apply(0, scores)
        after_asym = (result[0, 0, 4, 1] - result[0, 0, 1, 4]) / 2
        self.assertEqual(before_asym.item(), 2.0)
        self.assertEqual(after_asym.item(), 1.0)

    def test_threshold_is_strictly_greater_than_tau(self):
        controller = self.make_controller(tau=2.0)
        scores = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
        scores[0, 0, 2, 0] = 4.0  # A_asym=2, exactly tau: do not select.
        controller.set_prompt_context(prompt_length=3)
        result = controller.apply(0, scores)
        self.assertTrue(torch.equal(result, scores))
        self.assertEqual(controller.summary()["selected_element_intervention_count"], 0)

    def test_decode_steps_are_never_intervened(self):
        controller = self.make_controller()
        prefill = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        prefill[0, 0, 3, 0] = 4.0
        controller.set_prompt_context(prompt_length=4)
        controller.apply(0, prefill)
        decode = torch.randn((1, 3, 5, 5), dtype=torch.float32)
        self.assertIs(controller.apply(0, decode), decode)
        summary = controller.summary()
        self.assertEqual(summary["later_forward_skip_count"], 1)
        self.assertEqual(summary["later_forward_intervention_count"], 0)

    def test_complete_prefill_invariant_requires_every_layer(self):
        controller = self.make_controller(tau=10.0)
        scores = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
        controller.set_prompt_context(prompt_length=3)
        controller.apply(0, scores)
        with self.assertRaisesRegex(ReferenceInterventionRuntimeError, "every layer"):
            controller.assert_complete_prefill()
        controller.apply(1, scores)
        controller.assert_complete_prefill()

    def test_alpha_zero_is_exact_prefill_bypass(self):
        controller = self.make_controller(tau=0.0, alpha=0.0)
        scores = torch.randn((1, 3, 3, 3), dtype=torch.float32)
        controller.set_prompt_context(prompt_length=3)
        self.assertIs(controller.apply(0, scores), scores)
        summary = controller.summary()
        self.assertEqual(summary["alpha_zero_bypass_count"], 1)
        self.assertEqual(summary["selected_element_intervention_count"], 0)
        self.assertFalse(summary["first_forward_intervened"])

    def test_zero_sigma_uses_numerical_floor_without_nonfinite_values(self):
        controller = self.make_controller(tau=0.0, sigma_value=0.0)
        scores = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
        scores[0, 2, 2, 0] = 2.0
        controller.set_prompt_context(prompt_length=3)
        result = controller.apply(0, scores)
        self.assertTrue(bool(torch.isfinite(result).all()))
        summary = controller.summary()
        self.assertEqual(summary["sigma_floor_head_count"], 6)
        self.assertEqual(summary["numerical_sigma_floor"], NUMERICAL_SIGMA_FLOOR)
        self.assertTrue(math.isfinite(summary["max_absolute_z_score"]))

    def test_float16_realization_is_finite_and_cast_error_is_bounded(self):
        controller = self.make_controller(tau=0.1, alpha=1.0)
        scores = torch.zeros((1, 3, 3, 3), dtype=torch.float16)
        scores[0, 0, 2, 0] = 1.1
        controller.set_prompt_context(prompt_length=3)
        result = controller.apply(0, scores)
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(bool(torch.isfinite(result).all()))
        summary = controller.summary()
        self.assertLessEqual(
            summary["max_symmetric_component_error_after_cast"],
            summary["max_score_cast_quantization_error"],
        )

    def test_nonfinite_scores_fail(self):
        controller = self.make_controller()
        scores = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
        scores[0, 0, 2, 0] = math.inf
        controller.set_prompt_context(prompt_length=3)
        with self.assertRaisesRegex(ReferenceInterventionRuntimeError, "NaN or infinity"):
            controller.apply(0, scores)

    def test_tau_has_no_implicit_default_and_must_be_nonnegative(self):
        self.assertEqual(validate_tau(0.0), 0.0)
        for invalid in (-1.0, math.nan, math.inf):
            with self.assertRaises((TypeError, ValueError)):
                validate_tau(invalid)
        with self.assertRaises(TypeError):
            DynamicAsymmetryController(
                mu=torch.zeros((1, 1), dtype=torch.float32),
                sigma=torch.ones((1, 1), dtype=torch.float32),
            )


class ReferenceEntrypointTests(unittest.TestCase):
    def test_default_is_read_only_cpu_preflight_and_has_no_tau(self):
        args = parse_args([])
        self.assertEqual(args.stage, "cpu-preflight")
        self.assertIsNone(args.output_dir)
        self.assertFalse(hasattr(args, "tau"))
        self.assertIsNone(validate_request(args))

    def test_build_requires_new_output_directory(self):
        args = parse_args(["--stage", "build-reference"])
        with self.assertRaisesRegex(ValueError, "output-dir"):
            validate_request(args)


class TinyLlamaIntegrationTests(unittest.TestCase):
    def test_all_prefill_layers_are_patched_and_later_forward_is_skipped(self):
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
        reference = DynamicAsymmetryReference(
            mu=torch.zeros((2, 2), dtype=torch.float32),
            sigma=torch.ones((2, 2), dtype=torch.float32),
            metadata={},
        )
        with patch_llama_dynamic_asymmetry_attention(
            model,
            reference=reference,
            tau=0.0,
            alpha=1.0,
        ) as controller:
            controller.set_prompt_context(prompt_length=4)
            with torch.inference_mode():
                prefill = model.model(
                    input_ids=torch.tensor([[1, 2, 3, 4]]),
                    use_cache=False,
                    return_dict=True,
                )
                later = model.model(
                    input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                    use_cache=False,
                    return_dict=True,
                )
            controller.assert_complete_prefill()
            summary = controller.summary()
        self.assertEqual(tuple(prefill.last_hidden_state.shape), (1, 4, 16))
        self.assertEqual(tuple(later.last_hidden_state.shape), (1, 5, 16))
        self.assertEqual(summary["prefill_layer_count"], 2)
        self.assertEqual(summary["intervention_call_count"], 2)
        self.assertEqual(summary["later_forward_skip_count"], 2)
        self.assertEqual(summary["later_forward_intervention_count"], 0)
        self.assertEqual(summary["max_symmetric_component_error_float32"], 0.0)
        self.assertTrue(summary["all_scores_finite"])


if __name__ == "__main__":
    unittest.main()
