import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate_autodan_baseline import (
    build_run_metadata,
    evaluate_prompt_records,
    get_git_metadata,
    prepare_prompt_records,
    summarize_results,
    write_jsonl,
)


def source_row(index, prompt, success):
    return {"index": index, "user prompt": prompt, "success": success}


class AutoDANBaselineDiagnosticsTests(unittest.TestCase):
    def test_prepare_joins_source_labels_and_preserves_indices(self):
        input_rows = [
            {"source_index": 10, "prompt_id": "p10", "prompt": "prompt 10"},
            {"source_index": 12, "prompt_id": "p12", "prompt": "prompt 12"},
        ]
        source_rows = [
            source_row(10, "prompt 10", True),
            source_row(12, "prompt 12", False),
        ]

        prepared = prepare_prompt_records(
            input_rows,
            source_rows,
            "input.jsonl",
            "source.json",
        )

        self.assertEqual(len(prepared), len(input_rows))
        self.assertEqual([row["source_index"] for row in prepared], [10, 12])
        self.assertEqual([row["prompt_id"] for row in prepared], ["p10", "p12"])
        self.assertEqual([row["source_success"] for row in prepared], [True, False])

    def test_prepare_rejects_prompt_mismatch(self):
        with self.assertRaisesRegex(ValueError, "prompt mismatch"):
            prepare_prompt_records(
                [{"prompt": "different"}],
                [source_row(0, "source prompt", True)],
                "input.jsonl",
                "source.json",
            )

    def test_prompt_results_save_response_success_and_generation_failure(self):
        records = [
            {"source_index": 0, "prompt_id": 0, "prompt": "safe", "source_success": True},
            {"source_index": 1, "prompt_id": 1, "prompt": "refuse", "source_success": False},
            {"source_index": 2, "prompt_id": 2, "prompt": "error", "source_success": True},
        ]

        def generate(formatted_prompt):
            if formatted_prompt == "formatted:error":
                raise RuntimeError("synthetic generation failure")
            responses = {
                "formatted:safe": ("Requested answer", 2),
                "formatted:refuse": ("I cannot provide that", 4),
            }
            return responses[formatted_prompt]

        results = evaluate_prompt_records(
            records,
            format_prompt=lambda prompt: f"formatted:{prompt}",
            generate_response=generate,
        )

        self.assertEqual(len(results), len(records))
        self.assertEqual(results[0]["response"], "Requested answer")
        self.assertTrue(results[0]["success"])
        self.assertTrue(results[0]["success_agreement"])
        self.assertEqual(results[1]["response"], "I cannot provide that")
        self.assertFalse(results[1]["success"])
        self.assertIn("I cannot provide", results[1]["matched_refusal_substrings"])
        self.assertTrue(results[2]["generation_failed"])
        self.assertIsNone(results[2]["response"])
        self.assertIsNone(results[2]["success"])
        self.assertIn("RuntimeError", results[2]["failure_reason"])

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "results.jsonl"
            write_jsonl(output_path, results)
            saved = [json.loads(line) for line in output_path.read_text().splitlines()]
        self.assertEqual(len(saved), len(records))
        self.assertEqual(saved[0]["response"], "Requested answer")
        self.assertTrue(saved[0]["success"])
        self.assertEqual(saved[2]["source_index"], 2)

    def test_summary_counts_agreement_and_mismatch_indices(self):
        records = [
            {"source_index": 0, "prompt_id": 0, "prompt": "a", "source_success": True},
            {"source_index": 1, "prompt_id": 1, "prompt": "b", "source_success": False},
            {"source_index": 2, "prompt_id": 2, "prompt": "c", "source_success": True},
            {"source_index": 3, "prompt_id": 3, "prompt": "d", "source_success": False},
        ]
        responses = {
            "a": ("I cannot provide that", 4),
            "b": ("answer b", 2),
            "c": ("answer c", 2),
            "d": ("I'm sorry", 2),
        }
        results = evaluate_prompt_records(records, lambda prompt: prompt, responses.__getitem__)

        summary = summarize_results(results)

        self.assertEqual(summary["total_count"], 4)
        self.assertEqual(summary["source_success_count"], 2)
        self.assertEqual(summary["current_success_count"], 2)
        self.assertEqual(summary["both_success_count"], 1)
        self.assertEqual(summary["source_only_success_count"], 1)
        self.assertEqual(summary["current_only_success_count"], 1)
        self.assertEqual(summary["both_failure_count"], 1)
        self.assertEqual(summary["agreement_rate"], 0.5)
        self.assertEqual(summary["source_true_current_false_indices"], [0])
        self.assertEqual(summary["source_false_current_true_indices"], [1])

    def test_generation_failure_is_explicit_and_excluded_from_agreement(self):
        records = [
            {"source_index": 7, "prompt_id": 7, "prompt": "ok", "source_success": True},
            {"source_index": 8, "prompt_id": 8, "prompt": "error", "source_success": False},
        ]

        def generate(prompt):
            if prompt == "error":
                raise ValueError("failed")
            return "answer", 1

        summary = summarize_results(
            evaluate_prompt_records(records, lambda prompt: prompt, generate)
        )

        self.assertEqual(summary["evaluated_count"], 1)
        self.assertEqual(summary["generation_failure_count"], 1)
        self.assertEqual(summary["generation_failure_indices"], [8])
        self.assertEqual(summary["agreement_rate"], 1.0)

    def test_metadata_records_generation_and_no_intervention_conditions(self):
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="llama", _commit_hash=None),
            generation_config=SimpleNamespace(
                max_length=4096,
                top_k=50,
                num_beams=1,
                eos_token_id=2,
                pad_token_id=0,
                bos_token_id=1,
            ),
        )
        tokenizer = SimpleNamespace(chat_template="synthetic template")
        git_metadata = {
            "git_commit": "abc123",
            "git_commit_available": True,
            "git_commit_failure_reason": None,
            "git_worktree_dirty": False,
        }

        metadata = build_run_metadata(
            model_path="model",
            model=model,
            tokenizer=tokenizer,
            model_config={"model_name": "Llama", "name_or_path": "upstream"},
            input_file="input.jsonl",
            source_labels="source.json",
            total_count=3,
            max_new_tokens=32,
            seed=42,
            system_prompt=None,
            command="python script.py",
            git_metadata=git_metadata,
        )

        self.assertEqual(metadata["max_new_tokens"], 32)
        self.assertFalse(metadata["do_sample"])
        self.assertIsNone(metadata["temperature"])
        self.assertEqual(metadata["top_k"], 50)
        self.assertEqual(metadata["seed"], 42)
        self.assertFalse(metadata["intervention_enabled"])
        self.assertFalse(metadata["intervention_hooks_installed"])
        self.assertEqual(metadata["git_commit"], "abc123")

    def test_missing_git_commit_is_recorded_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = get_git_metadata(Path(directory))

        self.assertFalse(metadata["git_commit_available"])
        self.assertIsNone(metadata["git_commit"])
        self.assertTrue(metadata["git_commit_failure_reason"])


if __name__ == "__main__":
    unittest.main()
