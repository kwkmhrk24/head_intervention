import ast
import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts import analyze_gcg_dynamic_asym_selected_positions as analysis
from scripts import prepare_gcg_dynamic_asym_reference as builder
from utils import autodan_dynamic_asym_position_comparison as comparison
from utils.gcg_dynamic_asym_intervention import (
    REFERENCE_SCHEMA_VERSION, compute_equal_prompt_reference_statistics,
    validate_dynamic_reference_payload,
)
from utils.gcg_dynamic_asym_selection_diagnostics import (
    patch_llama_abs_z_collection, patch_llama_multi_reference_z_collection,
)


class OriginalReferenceTests(unittest.TestCase):
    def payload(self):
        means = torch.zeros((20, 32, 32), dtype=torch.float64)
        squares = torch.ones_like(means)
        indices = list(builder.AUTODAN_SELECTION_INDICES)
        return {"schema_version": REFERENCE_SCHEMA_VERSION,
                **compute_equal_prompt_reference_statistics(means, squares),
                "metadata": {
                    "reference_condition": builder.AUTODAN_ORIGINAL_CONDITION,
                    "prompt_weighting": "equal_weight_per_prompt",
                    "position_scope": "full_prompt_strict_lower_i_gt_j",
                    "component": "A_asym=(S-S^T)/2", "prompt_count": 20,
                    "prompt_format": builder.AUTODAN_OFFICIAL_FORMAT,
                    "jailbreak_reference_data_used": False,
                    "original_prompt_hashes_verified": True,
                    "selection_dataset_indices": indices,
                    "evaluation_dataset_indices": [i for i in range(130) if i not in indices],
                    "original_input_sha256": "a" * 64,
                    "original_prompt_records": [{"dataset_index": i, "original_prompt_sha256": "b" * 64}
                                                for i in indices],
                }}

    def test_original_reference_moments_and_provenance(self):
        payload = self.payload()
        validate_dynamic_reference_payload(payload)
        self.assertTrue(torch.equal(payload["sigma"], torch.ones((32, 32))))

    def test_rejects_jailbreak_leakage_wrong_format_and_split(self):
        changes = [
            {"jailbreak_reference_data_used": True},
            {"original_prompt_hashes_verified": False},
            {"prompt_format": "current"}, {"component": "A_sym=(S+S^T)/2"},
            {"original_input_sha256": "invalid"}, {"original_prompt_records": []},
            {"evaluation_dataset_indices": list(range(110))},
            {"selection_dataset_indices": [1] * 20},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                payload = self.payload()
                payload["metadata"].update(change)
                validate_dynamic_reference_payload(payload)

    def test_original_loader_uses_corresponding_originals_not_jailbreaks(self):
        indices = list(builder.AUTODAN_SELECTION_INDICES)
        from utils.artifact_io import sha256_text
        originals = [{"prompt": f"original {i}"} for i in range(130)]
        raw = [{"user prompt": f"jailbreak {i}"} for i in range(130)]
        saved = {
            "selection_indices": indices, "selection_count": 20, "evaluation_count": 110,
            "evaluation_indices": [i for i in range(130) if i not in indices],
            "format": builder.AUTODAN_OFFICIAL_FORMAT,
            "source_provenance": {k: builder.EXPECTED_AUTODAN_SELECTION_MANIFEST_SHA256
                                  for k in ("derived_original_sha256", "advbench_sha256", "autodan_sha256")},
            "records": [{"original_prompt_sha256": sha256_text(r["prompt"])} for r in originals],
        }
        with patch.object(builder, "sha256_file", return_value=builder.EXPECTED_AUTODAN_SELECTION_MANIFEST_SHA256), \
             patch.object(builder, "read_json", return_value=saved), \
             patch.object(builder, "read_jsonl", return_value=originals), \
             patch.object(builder, "validate_autodan_sources", return_value=(raw, originals, [], saved)):
            records, _ = builder.load_autodan_selection20_records("manifest", original_input="originals")
            self.assertEqual([r["prompt"] for r in records], [originals[i]["prompt"] for i in indices])
            self.assertTrue(all("autodan_prompt_sha256" not in r for r in records))
            broken = copy.deepcopy(originals)
            broken[indices[0]]["prompt"] = raw[indices[0]]["user prompt"]
            with patch.object(builder, "read_jsonl", return_value=broken), self.assertRaises(ValueError):
                builder.load_autodan_selection20_records("manifest", original_input="originals")

    def test_no_symmetric_original_condition_or_preflight_gpu_claim(self):
        for argv in (["--source", "autodan-selection20-original", "--component", "symmetric"],
                     ["--sandbox-exempt-gpu"]):
            with self.assertRaises(ValueError):
                builder.validate_request(builder.parse_args(argv))


class SharedPrefillTests(unittest.TestCase):
    def model(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=32)
        config._attn_implementation = "eager"
        return LlamaForCausalLM(config).half().eval()

    def test_shared_collection_exactly_matches_two_separate_collections_and_output(self):
        model = self.model()
        ids = torch.tensor([[1, 2, 3, 4]])
        references = {
            "B": SimpleNamespace(mu=torch.zeros(2, 2), sigma=torch.ones(2, 2)),
            "C": SimpleNamespace(mu=torch.full((2, 2), 0.3), sigma=torch.full((2, 2), 0.7)),
        }
        with torch.inference_mode():
            baseline = model.model(input_ids=ids, use_cache=False).last_hidden_state
        expected = {}
        for name, reference in references.items():
            with patch_llama_abs_z_collection(model, layer_count=2, head_count=2) as collector:
                collector.set_prompt_context(prompt_length=4, excluded_selection_position=0,
                                             mu=reference.mu, sigma=reference.sigma)
                with torch.inference_mode():
                    model.model(input_ids=ids, use_cache=False)
                expected[name] = collector.result(signed=True)
        with patch_llama_multi_reference_z_collection(model, references=references, prompt_length=4) as collectors:
            with torch.inference_mode():
                observed = model.model(input_ids=ids, use_cache=False).last_hidden_state
            for name, collector in collectors.items():
                signed_z, diagnostics = collector.result(signed=True)
                torch.testing.assert_close(signed_z, expected[name][0], rtol=0, atol=0)
                self.assertEqual(diagnostics, expected[name][1])
        torch.testing.assert_close(observed, baseline, rtol=0, atol=0)
        with torch.inference_mode():
            restored = model.model(input_ids=ids, use_cache=False).last_hidden_state
        torch.testing.assert_close(restored, baseline, rtol=0, atol=0)

    def test_empty_references_rejected_and_patches_restored_after_exception(self):
        model = self.model()
        forwards = [layer.self_attn.forward for layer in model.model.layers]
        with self.assertRaises(ValueError):
            with patch_llama_multi_reference_z_collection(model, references={}, prompt_length=4):
                pass
        with self.assertRaises(RuntimeError):
            with patch_llama_multi_reference_z_collection(
                model, references={"B": SimpleNamespace(mu=torch.zeros(2, 2), sigma=torch.ones(2, 2))}, prompt_length=4,
            ):
                raise RuntimeError("test")
        self.assertEqual([layer.self_attn.forward for layer in model.model.layers], forwards)


class PositionComparisonTests(unittest.TestCase):
    def test_overlap_uses_layer_head_and_position_identity(self):
        b = [{"dataset_index": 0, "prompt_length": 4, "selected_indices": [12, 13, 14]}]
        c = [{"dataset_index": 0, "prompt_length": 4, "selected_indices": [13, 14, 15]}]
        result = comparison.compare_selected_positions(b, c)
        self.assertEqual(result["intersection_total"], 2)
        self.assertEqual(result["prompt_equal_jaccard"]["mean"], 0.5)
        self.assertEqual(result["prompt_equal_overlap_fraction_of_B"]["mean"], 2 / 3)
        self.assertEqual(result["per_prompt"][0]["C_minus_B_selected_count"], 0)
        for broken in ([{**c[0], "dataset_index": 1}], [{**c[0], "prompt_length": 5}],
                       [{**c[0], "selected_indices": [1, 1]}]):
            with self.assertRaises(ValueError):
                comparison.compare_selected_positions(b, broken)

    def test_shared_workflow_never_generates_intervenes_or_calls_lm_head(self):
        tree = ast.parse(inspect.getsource(comparison.run_comparison))
        attrs = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
        for forbidden in ("generate", "lm_head", "patch_llama_stage2_dynamic_asymmetry_attention"):
            self.assertNotIn(forbidden, attrs)
        prefill_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr == "model"
                         and isinstance(node.func.value, ast.Name) and node.func.value.id == "model"]
        self.assertEqual(len(prefill_calls), 1)

    def test_generic_writer_supports_evaluation110_and_records_no_prior_for_C(self):
        metrics, coordinates = analysis.position_metrics(4, torch.tensor([192]))
        rows = [{"dataset_index": i, **metrics} for i in range(110)]
        decoded = [{"dataset_index": i, "selected_coordinates": coordinates} for i in range(110)]
        with tempfile.TemporaryDirectory() as directory, patch.object(analysis, "plot_heatmap"):
            summary = analysis.save_analysis(
                Path(directory) / "analysis", rows, decoded, {"dataset": "AutoDAN evaluation110"},
                expected_total=110, expected_prompt_count=110, coordinate_filename="selected_positions.jsonl",
                validation={"prior_selected_z_match": None, "prior_counts_and_layer_counts_match": None},
            )
            self.assertEqual(summary["prompt_count"], 110)
            self.assertIsNone(summary["validation"]["prior_selected_z_match"])
            self.assertTrue((Path(directory) / "analysis/selected_positions.jsonl").is_file())

    def test_runner_dispatch_and_required_reference(self):
        args = analysis.parse_args(["--dataset", "autodan110-reference-comparison", "--stage", "run", "--output-dir", "/tmp/unused"])
        with self.assertRaisesRegex(ValueError, "autodan-original-reference"):
            comparison.run_comparison(args)
        with patch.object(comparison, "run_comparison") as run:
            analysis.main(["--dataset", "autodan110-reference-comparison"])
            run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
