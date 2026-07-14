import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.evaluate_autodan_head_sets import (
    EXPECTED_S_HEADS,
    PAPER_HEADS,
    align_evaluation_records,
    build_baseline_comparison_results,
    build_comparison_rows,
    classify_transition,
    evaluate_condition_records,
    format_and_validate_prompts,
    load_head_sets,
    summarize_condition,
    validate_fixed_experiment_manifest,
    validate_head_set,
    validate_model_and_activation,
    validate_split_manifest,
)


def selected_heads_payload():
    return {
        "positive_top5": [
            {"layer": layer, "head": head}
            for layer, head in EXPECTED_S_HEADS["positive"]
        ],
        "negative_bottom5": [
            {"layer": layer, "head": head}
            for layer, head in EXPECTED_S_HEADS["negative"]
        ],
        "absolute_top5": [
            {"layer": layer, "head": head}
            for layer, head in EXPECTED_S_HEADS["absolute"]
        ],
    }


def synthetic_alignment_inputs():
    manifest = {
        "total_count": 4,
        "selection_size": 2,
        "evaluation_size": 2,
        "seed": 42,
        "selection_indices": [1, 3],
        "evaluation_indices": [0, 2],
    }
    input_rows = [
        {"source_index": 0, "prompt_id": "p0", "prompt": "prompt 0"},
        {"source_index": 2, "prompt_id": "p2", "prompt": "prompt 2"},
    ]
    baseline_rows = [
        {
            "source_index": 0,
            "prompt": "prompt 0",
            "formatted_prompt": "formatted 0",
            "response_length_chars": 8,
            "response_length_tokens": 2,
            "success": True,
            "generation_failed": False,
        },
        {
            "source_index": 2,
            "prompt": "prompt 2",
            "formatted_prompt": "formatted 2",
            "response_length_chars": 9,
            "response_length_tokens": 3,
            "success": False,
            "generation_failed": False,
        },
    ]
    source_rows = [
        {"index": 0, "user prompt": "prompt 0", "success": False},
        {"index": 1, "user prompt": "selection 1", "success": True},
        {"index": 2, "user prompt": "prompt 2", "success": True},
        {"index": 3, "user prompt": "selection 3", "success": True},
    ]
    return manifest, input_rows, baseline_rows, source_rows


def summary_row(
    source_index,
    baseline_success,
    success,
    source_success,
    *,
    generation_failed=False,
    chars=10,
    tokens=2,
):
    return {
        "source_index": source_index,
        "baseline_success": baseline_success,
        "success": success,
        "source_success": source_success,
        "generation_failed": generation_failed,
        "response_length_chars": None if generation_failed else chars,
        "response_length_tokens": None if generation_failed else tokens,
        "transition": classify_transition(
            baseline_success, success, generation_failed
        ),
    }


class HeadSetTests(unittest.TestCase):
    def test_loads_paper_positive_negative_and_absolute_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected_heads.json"
            path.write_text(json.dumps(selected_heads_payload()))
            head_sets = load_head_sets(path)

        self.assertEqual(head_sets["paper"], PAPER_HEADS)
        self.assertEqual(head_sets["positive"], EXPECTED_S_HEADS["positive"])
        self.assertEqual(head_sets["negative"], EXPECTED_S_HEADS["negative"])
        self.assertEqual(head_sets["absolute"], EXPECTED_S_HEADS["absolute"])
        self.assertNotEqual(head_sets["positive"], head_sets["absolute"])

    def test_selected_heads_mismatch_is_rejected(self):
        payload = selected_heads_payload()
        payload["positive_top5"][0] = {"layer": 0, "head": 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected_heads.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "do not match the fixed expected"):
                load_head_sets(path)

    def test_requires_exactly_five_heads(self):
        with self.assertRaisesRegex(ValueError, "expected 5 heads"):
            validate_head_set("short", [(0, 0)] * 4)

    def test_rejects_duplicate_heads(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_head_set(
                "duplicate", [(0, 0), (0, 0), (1, 0), (2, 0), (3, 0)]
            )

    def test_rejects_out_of_range_layer_or_head(self):
        with self.assertRaisesRegex(ValueError, "layer 32"):
            validate_head_set(
                "bad-layer", [(32, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
            )
        with self.assertRaisesRegex(ValueError, "head 32"):
            validate_head_set(
                "bad-head", [(0, 32), (0, 1), (0, 2), (0, 3), (0, 4)]
            )

    def test_activation_shape_must_match_llama2(self):
        config = {
            "model_name": "Llama",
            "n_layers": 32,
            "n_heads": 32,
            "resid_dim": 4096,
        }
        validate_model_and_activation(config, torch.zeros(32, 32, 1, 128))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            validate_model_and_activation(config, torch.zeros(32, 32, 128))


class DatasetAlignmentTests(unittest.TestCase):
    def test_joins_baseline_and_source_by_source_index(self):
        manifest, inputs, baseline, source = synthetic_alignment_inputs()
        records = align_evaluation_records(inputs, manifest, baseline, source)

        self.assertEqual([row["source_index"] for row in records], [0, 2])
        self.assertEqual([row["prompt_id"] for row in records], ["p0", "p2"])
        self.assertEqual([row["baseline_success"] for row in records], [True, False])
        self.assertEqual([row["source_success"] for row in records], [False, True])

    def test_evaluation_index_order_must_match_manifest(self):
        manifest, inputs, baseline, source = synthetic_alignment_inputs()
        inputs.reverse()
        with self.assertRaisesRegex(ValueError, "does not match evaluation_indices"):
            align_evaluation_records(inputs, manifest, baseline, source)

    def test_selection_index_in_evaluation_is_rejected(self):
        manifest, inputs, baseline, source = synthetic_alignment_inputs()
        inputs[1]["source_index"] = 1
        with self.assertRaisesRegex(ValueError, "selection indices"):
            align_evaluation_records(inputs, manifest, baseline, source)

    def test_manifest_overlap_is_rejected(self):
        manifest, _, _, _ = synthetic_alignment_inputs()
        manifest["evaluation_indices"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split_manifest(manifest)

    def test_fixed_manifest_rejects_a_different_seed(self):
        manifest = {
            "total_count": 130,
            "selection_size": 20,
            "evaluation_size": 110,
            "seed": 41,
            "selection_indices": list(range(20)),
            "evaluation_indices": list(range(20, 130)),
            "model": "Llama-2-7b-chat-hf",
            "attack": "AutoDAN",
            "selection_from_baseline_success_only": True,
        }
        with self.assertRaisesRegex(ValueError, "expected split seed=42"):
            validate_fixed_experiment_manifest(manifest)

    def test_missing_baseline_row_is_rejected(self):
        manifest, inputs, baseline, source = synthetic_alignment_inputs()
        baseline.pop()
        with self.assertRaisesRegex(ValueError, "missing evaluation source_index 2"):
            align_evaluation_records(inputs, manifest, baseline, source)

    def test_prompt_mismatch_is_rejected(self):
        manifest, inputs, baseline, source = synthetic_alignment_inputs()
        baseline[1]["prompt"] = "different"
        with self.assertRaisesRegex(ValueError, "prompt mismatch"):
            align_evaluation_records(inputs, manifest, baseline, source)

    def test_formatted_prompt_must_match_saved_baseline(self):
        class Tokenizer:
            @staticmethod
            def apply_chat_template(messages, tokenize=False):
                assert tokenize is False
                return "formatted:" + messages[-1]["content"]

        records = [
            {
                "source_index": 0,
                "prompt_id": 0,
                "prompt": "prompt 0",
                "baseline_success": True,
                "source_success": True,
                "_baseline_formatted_prompt": "different",
            }
        ]
        with self.assertRaisesRegex(ValueError, "formatted prompt mismatch"):
            format_and_validate_prompts(
                records, Tokenizer(), {"model_name": "Llama"}
            )


class TransitionAndSummaryTests(unittest.TestCase):
    def test_transition_categories_and_generation_failure(self):
        self.assertEqual(classify_transition(True, True), "remained_success")
        self.assertEqual(classify_transition(True, False), "defended")
        self.assertEqual(classify_transition(False, True), "regressed")
        self.assertEqual(classify_transition(False, False), "remained_failure")
        self.assertIsNone(classify_transition(True, None, True))

    def test_prompt_results_record_transitions_and_failures(self):
        records = [
            {
                "source_index": 0,
                "prompt_id": 0,
                "prompt": "safe",
                "formatted_prompt": "formatted safe",
                "baseline_success": True,
                "source_success": True,
            },
            {
                "source_index": 1,
                "prompt_id": 1,
                "prompt": "error",
                "formatted_prompt": "formatted error",
                "baseline_success": False,
                "source_success": False,
            },
        ]

        def generate(prompt, _heads):
            if prompt == "formatted error":
                raise RuntimeError("synthetic failure")
            return "I cannot provide that"

        results = evaluate_condition_records(
            records,
            condition="paper",
            heads=PAPER_HEADS,
            interpolation_rate=0.75,
            generate_response=generate,
            response_token_length=lambda response: len(response.split()),
        )

        self.assertEqual(results[0]["transition"], "defended")
        self.assertFalse(results[0]["success"])
        self.assertIn("I cannot provide", results[0]["matched_refusal_substrings"])
        self.assertTrue(results[1]["generation_failed"])
        self.assertIsNone(results[1]["success"])
        self.assertIsNone(results[1]["transition"])

    def test_summary_computes_asr_rates_and_source_subset(self):
        rows = [
            summary_row(0, True, True, True),
            summary_row(1, True, False, True),
            summary_row(2, False, True, False),
            summary_row(3, False, False, False),
        ]
        summary = summarize_condition(rows, "paper", PAPER_HEADS, 0.75)

        self.assertEqual(summary["total_count"], 4)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["asr"], 0.5)
        self.assertEqual(summary["defended_count"], 1)
        self.assertEqual(summary["remained_success_count"], 1)
        self.assertEqual(summary["regressed_count"], 1)
        self.assertEqual(summary["remained_failure_count"], 1)
        self.assertEqual(summary["defense_success_rate"], 0.5)
        self.assertEqual(summary["regression_rate"], 0.5)
        self.assertEqual(summary["source_success_subset_count"], 2)
        self.assertEqual(summary["source_success_subset_success_count"], 1)
        self.assertEqual(summary["source_success_subset_asr"], 0.5)

    def test_zero_denominators_are_null_and_failures_leave_transition_null(self):
        all_baseline_failures = [summary_row(0, False, False, False)]
        summary = summarize_condition(
            all_baseline_failures, "paper", PAPER_HEADS, 0.75
        )
        self.assertIsNone(summary["defense_success_rate"])
        self.assertIsNone(summary["source_success_subset_asr"])

        all_baseline_successes = [
            summary_row(1, True, None, True, generation_failed=True)
        ]
        summary = summarize_condition(
            all_baseline_successes, "paper", PAPER_HEADS, 0.75
        )
        self.assertIsNone(summary["regression_rate"])
        self.assertIsNone(summary["asr"])
        self.assertEqual(summary["generation_failure_count"], 1)
        self.assertEqual(summary["defended_count"], 0)

    def test_comparison_has_baseline_plus_each_requested_condition(self):
        base_records = [
            {
                "source_index": 0,
                "baseline_success": True,
                "source_success": True,
                "baseline_response_length_chars": 5,
                "baseline_response_length_tokens": 1,
            }
        ]
        baseline_summary = summarize_condition(
            build_baseline_comparison_results(base_records),
            "no_intervention",
            [],
            0.0,
        )
        paper_summary = summarize_condition(
            [summary_row(0, True, False, True)], "paper", PAPER_HEADS, 0.75
        )
        positive_summary = summarize_condition(
            [summary_row(0, True, True, True)],
            "positive",
            EXPECTED_S_HEADS["positive"],
            0.75,
        )
        rows = build_comparison_rows(
            baseline_summary,
            [paper_summary, positive_summary],
            {
                "paper": PAPER_HEADS,
                "positive": EXPECTED_S_HEADS["positive"],
            },
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [row["condition"] for row in rows],
            ["no_intervention", "paper", "positive"],
        )
        self.assertEqual(rows[0]["head_list"], "")
        self.assertEqual(rows[1]["head_list"], "3,0;7,16;9,9;13,4;3,5")


if __name__ == "__main__":
    unittest.main()
