import math
import json
import tempfile
import unittest
from pathlib import Path

import torch

from utils.gcg_reference_intervention_diagnostics import (
    aggregate_logit_steps,
    build_head_subset_conditions,
    common_prefix_length,
    compare_token_sequences,
    compute_logit_step_metrics,
    first_different_position,
    summarize_response_changes,
    token_edit_distance,
    validate_head_subset,
    validate_logit_diagnostic_record,
)
from scripts.diagnose_gcg_reference_intervention_logits import (
    finalize_run_manifest,
)


EXPECTED_TOP5 = ((31, 22), (13, 4), (5, 30), (9, 18), (27, 24))


class ResponseComparisonTests(unittest.TestCase):
    def test_exact_response_token_sequence(self):
        result = compare_token_sequences([1, 2, 3], [1, 2, 3])
        self.assertTrue(result["response_equal"])
        self.assertEqual(result["common_prefix_token_count"], 3)
        self.assertIsNone(result["first_different_token_position"])
        self.assertEqual(result["token_edit_distance"], 0)

    def test_common_prefix_and_first_difference(self):
        self.assertEqual(common_prefix_length([1, 2, 3], [1, 2, 9]), 2)
        self.assertEqual(first_different_position([1, 2, 3], [1, 2, 9]), 2)
        self.assertEqual(first_different_position([1, 2], [1, 2, 3]), 2)

    def test_token_edit_distance(self):
        self.assertEqual(token_edit_distance([1, 2, 3], [1, 4, 3]), 1)
        self.assertEqual(token_edit_distance([1, 2], [1, 2, 3]), 1)
        self.assertEqual(token_edit_distance([], [1, 2]), 2)

    def test_response_change_summary(self):
        rows = [
            {
                "dataset_index": 60,
                "response_equal": True,
                "baseline_success": False,
                "intervention_success": False,
                "baseline_token_count": 4,
                "intervention_token_count": 4,
            },
            {
                "dataset_index": 61,
                "response_equal": False,
                "baseline_success": False,
                "intervention_success": False,
                "baseline_token_count": 4,
                "intervention_token_count": 5,
            },
            {
                "dataset_index": 62,
                "response_equal": False,
                "baseline_success": True,
                "intervention_success": False,
                "baseline_token_count": 4,
                "intervention_token_count": 4,
            },
        ]
        result = summarize_response_changes(rows)
        self.assertEqual(result["response_equal_count"], 1)
        self.assertEqual(result["response_changed_count"], 2)
        self.assertEqual(result["success_changed_count"], 1)
        self.assertEqual(result["success_same_response_changed_count"], 1)
        self.assertEqual(result["response_token_count_changed_count"], 1)

    def test_response_summary_rejects_duplicate_index(self):
        row = {
            "dataset_index": 60,
            "response_equal": True,
            "baseline_success": False,
            "intervention_success": False,
            "baseline_token_count": 4,
            "intervention_token_count": 4,
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_response_changes([row, row])


class HeadSubsetTests(unittest.TestCase):
    def test_single_and_leave_one_out_sets(self):
        conditions = build_head_subset_conditions(EXPECTED_TOP5)
        self.assertEqual(len(conditions), 11)
        self.assertEqual(conditions[0]["name"], "full_top5")
        singles = [item for item in conditions if item["name"].startswith("single_")]
        without = [item for item in conditions if item["name"].startswith("without_")]
        self.assertEqual(len(singles), 5)
        self.assertEqual(len(without), 5)
        self.assertTrue(all(len(item["selected_heads"]) == 1 for item in singles))
        self.assertTrue(all(len(item["selected_heads"]) == 4 for item in without))
        self.assertEqual(
            {item["selected_heads"][0] for item in singles},
            set(EXPECTED_TOP5),
        )
        self.assertEqual(
            {item["omitted_head"] for item in without},
            set(EXPECTED_TOP5),
        )

    def test_head_subsets_have_no_duplicates(self):
        for condition in build_head_subset_conditions(EXPECTED_TOP5):
            heads = condition["selected_heads"]
            self.assertEqual(len(heads), len(set(heads)))

    def test_unexpected_head_rejected(self):
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_head_subset(((0, 0),), expected_heads=EXPECTED_TOP5)

    def test_duplicate_head_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_head_subset(
                (EXPECTED_TOP5[0], EXPECTED_TOP5[0]),
                expected_heads=EXPECTED_TOP5,
            )


class LogitMetricTests(unittest.TestCase):
    def test_identical_logits_have_zero_kl_and_equal_argmax(self):
        logits = torch.tensor([1.0, 3.0, -2.0])
        result = compute_logit_step_metrics(logits, logits.clone(), step=0)
        self.assertEqual(result["max_absolute_logit_difference"], 0.0)
        self.assertEqual(result["kl_baseline_to_intervention"], 0.0)
        self.assertTrue(result["argmax_equal"])
        self.assertEqual(result["baseline_top1_token_id"], 1)

    def test_argmax_and_margin(self):
        baseline = torch.tensor([4.0, 2.0, 1.0])
        intervention = torch.tensor([1.0, 5.0, 3.0])
        result = compute_logit_step_metrics(baseline, intervention, step=2)
        self.assertFalse(result["argmax_equal"])
        self.assertEqual(result["baseline_top1_top2_margin"], 2.0)
        self.assertEqual(result["intervention_top1_top2_margin"], 2.0)
        self.assertEqual(result["margin_change"], 0.0)

    def test_step_aggregation(self):
        first = compute_logit_step_metrics(
            torch.tensor([3.0, 1.0]), torch.tensor([3.0, 1.0]), step=0
        )
        second = compute_logit_step_metrics(
            torch.tensor([3.0, 1.0]), torch.tensor([1.0, 3.0]), step=1
        )
        result = aggregate_logit_steps([first, second])
        self.assertEqual(result["generation_step_count"], 2)
        self.assertEqual(result["argmax_change_step_count"], 1)
        self.assertEqual(result["first_argmax_change_step"], 1)
        self.assertTrue(result["all_finite"])

    def test_teacher_forced_and_free_running_schemas_are_separate(self):
        step = compute_logit_step_metrics(
            torch.tensor([2.0, 1.0]), torch.tensor([2.0, 1.0]), step=0
        )
        record = {
            "diagnostic_mode": "teacher_forced_common_prefix",
            "dataset_index": 60,
            "alpha": 0.1,
            "steps": [step],
            "summary": aggregate_logit_steps([step]),
            "baseline_response_token_ids": [0],
            "intervention_response_token_ids": [0],
        }
        validate_logit_diagnostic_record(
            record, mode="teacher_forced_common_prefix"
        )
        with self.assertRaisesRegex(ValueError, "mode/schema"):
            validate_logit_diagnostic_record(record, mode="free_running")

    def test_nan_and_infinity_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "NaN or infinity"
            ):
                compute_logit_step_metrics(
                    torch.tensor([0.0, value]),
                    torch.tensor([0.0, 1.0]),
                    step=0,
                )

    def test_shape_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            compute_logit_step_metrics(
                torch.zeros(2),
                torch.zeros(3),
                step=0,
            )


class RunManifestTests(unittest.TestCase):
    def test_pending_manifest_is_finalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_manifest.json"
            path.write_text(
                json.dumps({"status": "response_analysis_complete_gpu_pending"}),
                encoding="utf-8",
            )
            summary = {
                "created_at": "2026-08-02T00:00:00+00:00",
                "execution_git_head": "abc",
                "gpu": "test GPU",
                "torch_version": "test torch",
                "torch_cuda_version": "test CUDA",
                "diagnostic_dataset_indices": [182],
                "artifact_verified": True,
                "generation_failure_count": 0,
                "nan_or_inf_count": 0,
            }
            finalize_run_manifest(path, diagnosis_summary=summary)
            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["gpu"], "test GPU")
            self.assertFalse((Path(temporary) / ".run_manifest.json.tmp").exists())

    def test_complete_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_manifest.json"
            original = {"status": "complete", "sentinel": 42}
            path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pending state"):
                finalize_run_manifest(path, diagnosis_summary={})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                original,
            )


if __name__ == "__main__":
    unittest.main()
