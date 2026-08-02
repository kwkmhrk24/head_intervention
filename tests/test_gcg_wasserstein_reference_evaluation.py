import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import evaluate_gcg_wasserstein_reference_intervention as evaluator
from utils.gcg_wasserstein_reference_evaluation import (
    HeadDiagnosticAccumulator,
    classify_transition,
    compare_alpha_zero_to_baseline,
    ensure_output_absent,
    load_evaluation_records,
    summarize_transitions,
    validate_result_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = (
    REPO_ROOT
    / "data/outputs/gcg_wasserstein_reference/llama2/seed42_selection20/"
    "codex_gpu_20260802T061114Z"
)
JAILBREAK_INPUT = REPO_ROOT / "data/derived/llama2/gcg/jailbreak_prompts.jsonl"
BASELINE_RESULTS = (
    REPO_ROOT
    / "data/outputs/multidataset_asym_score_intervention/"
    "beta05_full_20260722_041303/llama_gcg/no_intervention/"
    "prompt_results.jsonl"
)


class EvaluationSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.split = load_evaluation_records(
            split_manifest_path=ARTIFACT_DIR / "split_manifest.json",
            jailbreak_input=JAILBREAK_INPUT,
            baseline_results=BASELINE_RESULTS,
        )

    def test_only_evaluation_140_are_loaded(self):
        self.assertEqual(len(self.records), 140)
        self.assertTrue(all(record["split"] == "evaluation" for record in self.records))
        selection = set(self.split["selection_row_positions"])
        evaluation = {record["row_position"] for record in self.records}
        self.assertFalse(selection & evaluation)
        self.assertEqual(len(selection | evaluation), 160)

    def test_row_position_and_dataset_index_alignment(self):
        self.assertEqual(self.records[0]["row_position"], 0)
        self.assertEqual(self.records[0]["dataset_index"], 60)
        self.assertEqual(self.records[-1]["row_position"], 159)
        self.assertEqual(self.records[-1]["dataset_index"], 219)
        self.assertEqual(
            len({record["dataset_index"] for record in self.records}),
            140,
        )

    def test_baseline_success_and_failure_counts(self):
        successes = sum(record["baseline_success"] for record in self.records)
        self.assertEqual(successes, 54)
        self.assertEqual(len(self.records) - successes, 86)

    def test_fixed_preflight_rows_are_evaluation(self):
        by_row = {record["row_position"]: record for record in self.records}
        selected = [by_row[row] for row in (0, 2, 9, 14, 15)]
        self.assertEqual(
            [record["dataset_index"] for record in selected],
            [60, 62, 69, 74, 75],
        )
        self.assertEqual({record["baseline_success"] for record in selected}, {False, True})


class TransitionTests(unittest.TestCase):
    def test_four_transition_categories(self):
        self.assertEqual(classify_transition(True, False), "newly_defended")
        self.assertEqual(classify_transition(True, True), "still_attack_success")
        self.assertEqual(classify_transition(False, True), "newly_worsened")
        self.assertEqual(classify_transition(False, False), "still_defended")

    def test_transition_sum_attack_success_and_asr(self):
        pairs = ((True, False), (True, True), (False, True), (False, False))
        rows = [
            {
                "dataset_index": index,
                "baseline_success": baseline,
                "success": intervention,
                "generation_failed": False,
                "transition": classify_transition(baseline, intervention),
            }
            for index, (baseline, intervention) in enumerate(pairs)
        ]
        summary = summarize_transitions(rows, expected_count=4)
        self.assertEqual(summary["transition_count"], 4)
        self.assertEqual(summary["attack_success_count"], 2)
        self.assertEqual(summary["asr"], 0.5)
        self.assertEqual(summary["net_defended"], 0)

    def test_alpha_zero_expected_table(self):
        records, _ = load_evaluation_records(
            split_manifest_path=ARTIFACT_DIR / "split_manifest.json",
            jailbreak_input=JAILBREAK_INPUT,
            baseline_results=BASELINE_RESULTS,
        )
        rows = [
            {
                **record,
                "response": record["baseline_response"],
                "success": record["baseline_success"],
                "generation_failed": record["baseline_generation_failed"],
                "transition": classify_transition(
                    record["baseline_success"], record["baseline_success"]
                ),
            }
            for record in records
        ]
        summary = summarize_transitions(rows, expected_count=140)
        self.assertEqual(summary["newly_defended"], 0)
        self.assertEqual(summary["still_attack_success"], 54)
        self.assertEqual(summary["newly_worsened"], 0)
        self.assertEqual(summary["still_defended"], 86)
        self.assertEqual(summary["attack_success_count"], 54)
        self.assertAlmostEqual(summary["asr"], 54 / 140)
        self.assertTrue(compare_alpha_zero_to_baseline(rows)["passed"])


class ResultValidationTests(unittest.TestCase):
    def setUp(self):
        self.expected = [
            {"dataset_index": i, "row_position": i, "prompt": f"prompt-{i}"}
            for i in range(140)
        ]
        self.rows = [dict(record) for record in self.expected]

    def test_duplicate_dataset_index_is_rejected(self):
        self.rows[-1]["dataset_index"] = self.rows[0]["dataset_index"]
        with self.assertRaisesRegex(ValueError, "duplicate dataset indices"):
            validate_result_rows(self.rows, self.expected)

    def test_incomplete_140_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            validate_result_rows(self.rows[:-1], self.expected)

    def test_existing_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                ensure_output_absent(directory)


class DiagnosticAggregationTests(unittest.TestCase):
    def test_distribution_and_error_statistics(self):
        accumulator = HeadDiagnosticAccumulator([(3, 4)], alpha=0.5)
        summary = {
            "selected_heads": [{"layer": 3, "head": 4}],
            "attention_forward_call_count": 2,
            "intervention_call_count": 2,
            "short_sequence_skip_count": 0,
            "all_scores_finite": True,
            "per_head": [
                {
                    "layer": 3,
                    "head": 4,
                    "short_sequence_skip_count": 0,
                }
            ],
        }

        def record(value):
            return {
                "layer": 3,
                "head": 4,
                "sequence_length": 20,
                "finite": True,
                "mean_absolute_delta": value,
                "max_absolute_delta": value,
                "intervened_to_current_norm_ratio": value,
                "current_asym_frobenius_norm": value,
                "intervened_asym_frobenius_norm": value,
                "max_constructed_delta_antisymmetry_error_float32": 0.0,
                "max_delta_antisymmetry_error_float32": value / 100,
                "max_delta_antisymmetry_error": value / 10,
                "max_symmetric_component_error_float32": value / 200,
                "max_symmetric_component_error": value / 20,
                "max_diagonal_change": 0.0,
                "max_unselected_head_change": 0.0,
                "max_outside_tail_change": 0.0,
            }

        accumulator.add_prompt(
            dataset_index=60,
            controller_summary=summary,
            head_metric_records=[record(1.0), record(3.0)],
        )
        result = accumulator.summary()
        head = result["per_head"][0]
        self.assertEqual(head["call_count"], 2)
        self.assertEqual(head["mean_absolute_delta"]["min"], 1.0)
        self.assertEqual(head["mean_absolute_delta"]["median"], 2.0)
        self.assertEqual(head["mean_absolute_delta"]["mean"], 2.0)
        self.assertAlmostEqual(head["mean_absolute_delta"]["p95"], 2.9)
        self.assertEqual(head["mean_absolute_delta"]["max"], 3.0)
        self.assertEqual(head["max_float16_delta_antisymmetry_error"], 0.3)
        self.assertTrue(result["all_finite"])


class GenerationGateTests(unittest.TestCase):
    def test_artifact_hash_failure_precedes_model_loading(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            evaluator,
            "load_validated_reference_artifacts",
            side_effect=ValueError("artifact verification selection hash mismatch"),
        ), mock.patch.object(evaluator, "load_model_and_tokenizer") as load_model:
            with self.assertRaisesRegex(ValueError, "selection hash mismatch"):
                evaluator.main(["--output-dir", str(Path(directory) / "new")])
            load_model.assert_not_called()

    def test_top5_failure_precedes_model_loading(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            evaluator,
            "load_validated_reference_artifacts",
            side_effect=ValueError("top5 heads mismatch"),
        ), mock.patch.object(evaluator, "load_model_and_tokenizer") as load_model:
            with self.assertRaisesRegex(ValueError, "top5 heads mismatch"):
                evaluator.main(["--output-dir", str(Path(directory) / "new")])
            load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
