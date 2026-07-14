import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
)

from utils.asym_score_intervention import (
    patch_llama_attention_scores,
    suppress_antisymmetric_component,
    temporarily_disable_default_dynamic_cache,
)
from scripts.evaluate_autodan_asym_score_intervention import (
    EXPECTED_ELEMENT_HEADS,
    PAPER_HEADS,
    align_historical_baseline,
    build_condition_plan,
    classify_transition,
    compare_historical_baseline,
    load_headsets,
    summarize_results,
)


def symmetric_component(values):
    return (values + values.transpose(-1, -2)) / 2


def antisymmetric_component(values):
    return (values - values.transpose(-1, -2)) / 2


class PureScoreInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.scores = torch.randn(2, 4, 5, 5, dtype=torch.float64)

    def test_beta_zero_is_exact_identity_without_input_mutation(self):
        before = self.scores.clone()
        modified, diagnostics = suppress_antisymmetric_component(
            self.scores, [1, 3], 0.0, return_diagnostics=True
        )
        self.assertTrue(torch.equal(modified, before))
        self.assertTrue(torch.equal(self.scores, before))
        self.assertEqual(diagnostics["intervention_identity_max_abs_error"], 0.0)
        self.assertEqual(diagnostics["unselected_head_max_abs_error"], 0.0)

    def test_beta_one_removes_selected_asymmetry(self):
        modified, diagnostics = suppress_antisymmetric_component(
            self.scores, [1, 3], 1.0, return_diagnostics=True
        )
        post_asym = antisymmetric_component(modified[:, [1, 3]])
        self.assertTrue(torch.allclose(post_asym, torch.zeros_like(post_asym)))
        self.assertAlmostEqual(diagnostics["observed_asym_scale"], 0.0)
        self.assertAlmostEqual(diagnostics["expected_asym_scale"], 0.0)

    def test_beta_half_halves_asymmetry_and_preserves_symmetry(self):
        selected = self.scores[:, [0, 2]]
        modified, diagnostics = suppress_antisymmetric_component(
            self.scores, [0, 2], 0.5, return_diagnostics=True
        )
        modified_selected = modified[:, [0, 2]]
        torch.testing.assert_close(
            antisymmetric_component(modified_selected),
            0.5 * antisymmetric_component(selected),
        )
        torch.testing.assert_close(
            symmetric_component(modified_selected),
            symmetric_component(selected),
        )
        self.assertAlmostEqual(diagnostics["observed_asym_scale"], 0.5)
        self.assertAlmostEqual(diagnostics["expected_asym_scale"], 0.5)
        self.assertLessEqual(diagnostics["sym_component_max_abs_error"], 1e-15)

    def test_unselected_heads_diagonal_batch_dtype_and_device_are_preserved(self):
        modified = suppress_antisymmetric_component(self.scores, [1], 0.75)
        self.assertEqual(modified.shape, self.scores.shape)
        self.assertEqual(modified.dtype, self.scores.dtype)
        self.assertEqual(modified.device, self.scores.device)
        self.assertTrue(torch.equal(modified[:, [0, 2, 3]], self.scores[:, [0, 2, 3]]))
        self.assertTrue(
            torch.equal(
                torch.diagonal(modified[:, 1], dim1=-2, dim2=-1),
                torch.diagonal(self.scores[:, 1], dim1=-2, dim2=-1),
            )
        )

    def test_symmetric_input_is_unchanged_for_every_beta(self):
        symmetric = symmetric_component(self.scores)
        for beta in (0.0, 0.25, 0.5, 1.0):
            modified = suppress_antisymmetric_component(symmetric, [0, 1], beta)
            torch.testing.assert_close(modified, symmetric)

    def test_autograd_path_is_retained(self):
        scores = self.scores.clone().requires_grad_(True)
        modified = suppress_antisymmetric_component(scores, [0, 2], 0.5)
        modified.square().sum().backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_intervention_identity(self):
        beta = 0.3
        modified = suppress_antisymmetric_component(self.scores, [1], beta)
        selected = self.scores[:, [1]]
        expected = selected - beta * antisymmetric_component(selected)
        torch.testing.assert_close(modified[:, [1]], expected)

    def test_rejects_beta_outside_closed_unit_interval(self):
        for beta in (-0.01, 1.01, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "0.0 <= beta <= 1.0"):
                suppress_antisymmetric_component(self.scores, [0], beta)

    def test_rejects_non_square_duplicate_and_out_of_range_heads(self):
        with self.assertRaisesRegex(ValueError, "query_length == key_length"):
            suppress_antisymmetric_component(torch.randn(1, 2, 3, 4), [0], 0.5)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            suppress_antisymmetric_component(self.scores, [1, 1], 0.5)
        with self.assertRaisesRegex(ValueError, "outside"):
            suppress_antisymmetric_component(self.scores, [4], 0.5)


class PatchContextTests(unittest.TestCase):
    def make_tiny_model(self):
        config = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        attention = LlamaAttention(config=config, layer_idx=0).eval()
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(self_attn=attention)]
            )
        )
        return model, attention

    def test_beta_zero_patched_forward_matches_original_and_records_diagnostics(self):
        model, attention = self.make_tiny_model()
        hidden_states = torch.randn(2, 5, 32)
        position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        original = attention(
            hidden_states,
            position_ids=position_ids,
            use_cache=False,
        )[0]
        with patch_llama_attention_scores(model, [(0, 1)], 0.0) as diagnostics:
            patched = attention(
                hidden_states,
                position_ids=position_ids,
                use_cache=False,
            )[0]
            summary = diagnostics.summary()
        self.assertTrue(torch.equal(original, patched))
        self.assertEqual(summary["call_count"], 1)
        self.assertEqual(summary["query_lengths"], [5])
        self.assertEqual(summary["key_lengths"], [5])
        self.assertEqual(summary["selected_head_count"], 1)
        self.assertEqual(summary["aggregate_observed_asym_scale"], 1.0)
        self.assertEqual(summary["max_intervention_identity_error"], 0.0)

    def test_context_restores_forward_normally(self):
        model, attention = self.make_tiny_model()
        original_function = attention.forward.__func__
        self.assertNotIn("forward", attention.__dict__)
        with patch_llama_attention_scores(model, [(0, 1)], 0.5):
            self.assertIn("forward", attention.__dict__)
            self.assertIsNot(attention.forward.__func__, original_function)
        self.assertNotIn("forward", attention.__dict__)
        self.assertIs(attention.forward.__func__, original_function)

    def test_context_restores_forward_after_exception(self):
        model, attention = self.make_tiny_model()
        original_function = attention.forward.__func__
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with patch_llama_attention_scores(model, [(0, 2)], 0.5):
                raise RuntimeError("synthetic failure")
        self.assertNotIn("forward", attention.__dict__)
        self.assertIs(attention.forward.__func__, original_function)

    def test_patched_forward_rejects_use_cache(self):
        model, attention = self.make_tiny_model()
        hidden_states = torch.randn(1, 3, 32)
        position_ids = torch.arange(3).unsqueeze(0)
        with patch_llama_attention_scores(model, [(0, 0)], 0.5):
            with self.assertRaisesRegex(ValueError, "use_cache=False"):
                attention(
                    hidden_states,
                    position_ids=position_ids,
                    use_cache=True,
                )

    def test_patched_forward_still_rejects_past_key_value(self):
        model, attention = self.make_tiny_model()
        hidden_states = torch.randn(1, 3, 32)
        position_ids = torch.arange(3).unsqueeze(0)
        with patch_llama_attention_scores(model, [(0, 0)], 0.5):
            with self.assertRaisesRegex(ValueError, "past_key_value"):
                attention(
                    hidden_states,
                    position_ids=position_ids,
                    past_key_value=object(),
                    use_cache=False,
                )


class DynamicCacheDisableContextTests(unittest.TestCase):
    def test_temporarily_disables_and_restores_normal_exit(self):
        model = SimpleNamespace(_supports_cache_class=True)
        with temporarily_disable_default_dynamic_cache(model) as state:
            self.assertFalse(model._supports_cache_class)
            self.assertTrue(
                state["generation_model_supports_cache_class_original"]
            )
        self.assertTrue(model._supports_cache_class)
        self.assertTrue(
            state["generation_model_supports_cache_class_restored"]
        )

    def test_restores_after_exception(self):
        model = SimpleNamespace(_supports_cache_class=True)
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with temporarily_disable_default_dynamic_cache(model):
                self.assertFalse(model._supports_cache_class)
                raise RuntimeError("synthetic failure")
        self.assertTrue(model._supports_cache_class)

    def test_nested_contexts_restore_each_observed_state(self):
        model = SimpleNamespace(_supports_cache_class=True)
        with temporarily_disable_default_dynamic_cache(model):
            self.assertFalse(model._supports_cache_class)
            with temporarily_disable_default_dynamic_cache(model):
                self.assertFalse(model._supports_cache_class)
            self.assertFalse(model._supports_cache_class)
        self.assertTrue(model._supports_cache_class)

    def test_rejects_missing_or_non_boolean_attribute(self):
        with self.assertRaisesRegex(AttributeError, "lacks"):
            with temporarily_disable_default_dynamic_cache(SimpleNamespace()):
                pass
        with self.assertRaisesRegex(TypeError, "must be a bool"):
            with temporarily_disable_default_dynamic_cache(
                SimpleNamespace(_supports_cache_class=1)
            ):
                pass


class TinyLlamaGenerationIntegrationTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(42)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
            attention_dropout=0.0,
            use_cache=False,
        )
        config._attn_implementation = "eager"
        return LlamaForCausalLM(config).eval()

    def setUp(self):
        self.model = self.make_model()
        self.input_ids = torch.tensor([[1, 5, 6, 7]])
        self.generation_kwargs = {
            "use_cache": False,
            "do_sample": False,
            "max_new_tokens": 3,
            "pad_token_id": 0,
        }

    def test_no_intervention_generation_receives_no_past_key_values(self):
        seen_past = []

        def capture_past(_module, _args, kwargs):
            seen_past.append(kwargs.get("past_key_values"))

        hook = self.model.register_forward_pre_hook(
            capture_past, with_kwargs=True
        )
        original = self.model._supports_cache_class
        try:
            with temporarily_disable_default_dynamic_cache(self.model):
                self.assertFalse(self.model._supports_cache_class)
                output = self.model.generate(
                    self.input_ids, **self.generation_kwargs
                )
        finally:
            hook.remove()
        self.assertEqual(output.shape[-1], self.input_ids.shape[-1] + 3)
        self.assertTrue(seen_past)
        self.assertTrue(all(past is None for past in seen_past))
        self.assertEqual(self.model._supports_cache_class, original)

    def test_patched_generation_has_square_growing_scores_and_restores_flag(self):
        original = self.model._supports_cache_class
        had_instance_attribute = "_supports_cache_class" in vars(self.model)
        with patch_llama_attention_scores(
            self.model, [(0, 1)], 0.5
        ) as diagnostics:
            with temporarily_disable_default_dynamic_cache(self.model):
                output = self.model.generate(
                    self.input_ids, **self.generation_kwargs
                )
            summary = diagnostics.summary()
        self.assertEqual(output.shape[-1], self.input_ids.shape[-1] + 3)
        self.assertGreater(summary["call_count"], 0)
        self.assertEqual(summary["query_lengths"], [4, 5, 6])
        self.assertEqual(summary["query_lengths"], summary["key_lengths"])
        self.assertTrue(summary["all_score_matrices_square"])
        self.assertEqual(self.model._supports_cache_class, original)
        self.assertEqual(
            "_supports_cache_class" in vars(self.model), had_instance_attribute
        )

    def test_beta_zero_generation_and_argmax_match_no_intervention(self):
        with torch.inference_mode():
            baseline_logits = self.model(
                input_ids=self.input_ids, use_cache=False
            ).logits
        with temporarily_disable_default_dynamic_cache(self.model):
            baseline_output = self.model.generate(
                self.input_ids, **self.generation_kwargs
            )

        with patch_llama_attention_scores(
            self.model, [(0, 1)], 0.0
        ) as diagnostics:
            with torch.inference_mode():
                patched_logits = self.model(
                    input_ids=self.input_ids, use_cache=False
                ).logits
            diagnostics.clear()
            with temporarily_disable_default_dynamic_cache(self.model):
                patched_output = self.model.generate(
                    self.input_ids, **self.generation_kwargs
                )
            summary = diagnostics.summary()

        self.assertTrue(torch.equal(baseline_output, patched_output))
        self.assertTrue(torch.equal(baseline_logits, patched_logits))
        self.assertEqual(
            baseline_logits[:, -1].argmax(dim=-1).item(),
            patched_logits[:, -1].argmax(dim=-1).item(),
        )
        self.assertEqual(summary["max_intervention_identity_error"], 0.0)
        self.assertTrue(summary["all_score_matrices_square"])
        self.assertEqual(summary["query_lengths"], [4, 5, 6])


class EvaluationPipelineHelperTests(unittest.TestCase):
    def head_payload(self):
        return {
            "consistent_positive_top": [
                {"layer": layer, "head": head}
                for layer, head in EXPECTED_ELEMENT_HEADS["element_positive"]
            ],
            "consistent_negative_top": [
                {"layer": layer, "head": head}
                for layer, head in EXPECTED_ELEMENT_HEADS["element_negative"]
            ],
            "consistent_absolute_top": [
                {"layer": layer, "head": head}
                for layer, head in EXPECTED_ELEMENT_HEADS["element_absolute"]
            ],
        }

    def test_loads_and_validates_all_headsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heads.json"
            path.write_text(json.dumps(self.head_payload()))
            headsets = load_headsets(path)
        self.assertEqual(headsets["paper"], PAPER_HEADS)
        self.assertEqual(
            headsets["element_positive"],
            EXPECTED_ELEMENT_HEADS["element_positive"],
        )

    def test_elementwise_head_mismatch_fails_fast(self):
        payload = self.head_payload()
        payload["consistent_positive_top"][0] = {"layer": 0, "head": 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heads.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "differs from the fixed"):
                load_headsets(path)

    def test_condition_plan_has_regenerated_baseline_and_one_beta_zero_control(self):
        headsets = {"paper": PAPER_HEADS, **EXPECTED_ELEMENT_HEADS}
        plan = build_condition_plan(
            ["paper", "element_positive"], [0.0, 0.5, 1.0], headsets
        )
        self.assertEqual(plan[0]["condition"], "no_intervention")
        self.assertFalse(plan[0]["patched"])
        self.assertEqual(plan[1]["condition"], "beta_zero_control")
        self.assertTrue(plan[1]["patched"])
        self.assertEqual(plan[1]["beta"], 0.0)
        self.assertEqual(
            [item["condition"] for item in plan[2:]],
            [
                "paper_beta_0.5",
                "paper_beta_1",
                "element_positive_beta_0.5",
                "element_positive_beta_1",
            ],
        )

    def test_transition_and_summary_use_new_baseline(self):
        self.assertEqual(classify_transition(True, True), "remained_success")
        self.assertEqual(classify_transition(True, False), "defended")
        self.assertEqual(classify_transition(False, True), "regressed")
        self.assertEqual(classify_transition(False, False), "remained_failure")
        rows = []
        for index, (baseline, current) in enumerate(
            [(True, True), (True, False), (False, True), (False, False)]
        ):
            rows.append(
                {
                    "source_index": index,
                    "baseline_success": baseline,
                    "success": current,
                    "transition": classify_transition(baseline, current),
                    "generation_failed": False,
                    "response_length_tokens": 2,
                    "score_intervention_call_count": 3,
                    "mean_observed_asym_scale": 0.5,
                    "max_intervention_identity_error": 0.0,
                    "max_symmetric_component_error": 0.0,
                    "max_unselected_head_error": 0.0,
                }
            )
        spec = {
            "condition": "paper_beta_0.5",
            "headset": "paper",
            "layer_head_pairs": PAPER_HEADS,
            "beta": 0.5,
        }
        summary = summarize_results(rows, spec)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["asr"], 0.5)
        self.assertEqual(summary["defended_count"], 1)
        self.assertEqual(summary["regressed_count"], 1)
        self.assertEqual(summary["defense_success_rate"], 0.5)
        self.assertEqual(summary["regression_rate"], 0.5)
        self.assertEqual(summary["mean_observed_asym_scale"], 0.5)

    def test_historical_baseline_is_aligned_and_compared_only_as_auxiliary(self):
        records = [
            {"source_index": 2, "prompt": "p2"},
            {"source_index": 5, "prompt": "p5"},
        ]
        historical = [
            {"source_index": 2, "prompt": "p2", "success": True},
            {"source_index": 5, "prompt": "p5", "success": False},
        ]
        aligned = align_historical_baseline(historical, records, "historical")
        regenerated = [
            {
                "source_index": 2,
                "success": False,
                "generation_failed": False,
            },
            {
                "source_index": 5,
                "success": False,
                "generation_failed": False,
            },
        ]
        comparison = compare_historical_baseline(regenerated, aligned)
        self.assertEqual(
            comparison["role"], "auxiliary_only_not_transition_reference"
        )
        self.assertEqual(comparison["historical_success_count"], 1)
        self.assertEqual(comparison["regenerated_success_count"], 0)
        self.assertEqual(comparison["agreement_rate"], 0.5)
        self.assertEqual(
            comparison["historical_true_regenerated_false_indices"], [2]
        )


if __name__ == "__main__":
    unittest.main()
