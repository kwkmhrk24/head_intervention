import math
import unittest

import torch

from utils.gcg_reference_factorization_diagnostics import (
    ALPHA,
    BASELINE_SUCCESS_COUNTS,
    EXPECTED_TOP5,
    GROUP_COUNTS,
    REFERENCE_FIELDS,
    SINGLE_HEAD,
    build_paired_original_records,
    classify_factorization_causes,
    factorization_conditions,
    reference_for_condition,
    summarize_condition_logits,
    summarize_condition_results,
    summarize_reference_distances,
    validate_condition_definitions,
    validate_paired_original_values,
)
from utils.gcg_wasserstein_reference import sha256_text


def aligned_inputs():
    selection_indices = list(range(60, 80))
    holdout_indices = list(range(48))
    groups = {
        "selection20": [
            {
                "dataset_index": index,
                "row_position": index - 60,
                "prompt_id": f"s{index}",
            }
            for index in selection_indices
        ],
        "holdout48": [
            {
                "dataset_index": index,
                "row_position": index,
                "prompt_id": f"h{index}",
            }
            for index in holdout_indices
        ],
    }
    selection_original = [
        {"index": index, "prompt": f"selection original {index}"}
        for index in selection_indices
    ]
    advbench = [
        {"index": index, "prompt": f"holdout original {index}"}
        for index in holdout_indices
    ]
    split = {
        "records": [
            {
                "dataset_index": row["index"],
                "split": "selection",
                "original_prompt_sha256": sha256_text(row["prompt"]),
            }
            for row in selection_original
        ]
    }
    holdout = {
        "records": [
            {
                "dataset_index": row["index"],
                "included": True,
                "original_prompt": row["prompt"],
                "original_prompt_sha256": sha256_text(row["prompt"]),
            }
            for row in advbench
        ]
    }
    return groups, selection_original, advbench, split, holdout


def metric(value=0.1):
    return {
        "max_absolute_logit_difference": value,
        "mean_absolute_logit_difference": value / 2,
        "kl_baseline_to_intervention": value / 100,
        "baseline_top1_top2_margin": 1.0,
        "intervention_top1_top2_margin": 1.0 - value,
        "margin_change": -value,
    }


def condition_rows():
    rows = []
    for group, count in GROUP_COUNTS.items():
        baseline_success_count = BASELINE_SUCCESS_COUNTS[group]
        for condition_position, condition in enumerate(factorization_conditions()):
            for position in range(count):
                baseline_success = position < baseline_success_count
                success = baseline_success
                changed = False
                if condition["name"] == "head13_4_global_mean" and position == 0:
                    success = False
                    changed = True
                if condition["name"] == "head13_4_paired_original_oracle" and position < 2:
                    success = False
                    changed = True
                rows.append(
                    {
                        "dataset_group": group,
                        "condition": condition["name"],
                        "dataset_index": position,
                        "baseline_success": baseline_success,
                        "success": success,
                        "transition": (
                            "newly_defended"
                            if baseline_success and not success
                            else "still_attack_success"
                            if baseline_success and success
                            else "newly_worsened"
                            if success
                            else "still_defended"
                        ),
                        "response_equal": not changed,
                        "first_different_token_position": 0 if changed else None,
                        "first_token_argmax_equal": not changed,
                        "first_token_logit_metrics": metric(
                            0.1 + condition_position / 10
                        ),
                        "generation_failed": False,
                        "empty_response": False,
                        "oracle_reference": condition["oracle_reference"],
                    }
                )
    return rows


def reference_rows():
    rows = []
    for group, count in GROUP_COUNTS.items():
        for condition_position, condition in enumerate(factorization_conditions()):
            for position in range(count):
                for layer, head in condition["heads"]:
                    value = 1.0 + condition_position + position / 100
                    rows.append(
                        {
                            "dataset_group": group,
                            "condition": condition["name"],
                            "dataset_index": position,
                            "layer": layer,
                            "head": head,
                            **{field: value for field in REFERENCE_FIELDS},
                        }
                    )
    return rows


class PairedOriginalTests(unittest.TestCase):
    def test_dataset_index_alignment_and_order(self):
        groups, selection, advbench, split, holdout = aligned_inputs()
        records = build_paired_original_records(
            groups,
            selection_original_rows=selection,
            advbench_rows=advbench,
            selection_split_manifest=split,
            holdout_manifest=holdout,
        )
        self.assertEqual(len(records), 68)
        self.assertEqual(records[0]["dataset_group"], "selection20")
        self.assertEqual(records[20]["dataset_group"], "holdout48")
        self.assertEqual(records[-1]["dataset_index"], 47)

    def test_dataset_index_or_hash_mismatch_is_rejected(self):
        groups, selection, advbench, split, holdout = aligned_inputs()
        selection[0]["prompt"] = "modified"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            build_paired_original_records(
                groups,
                selection_original_rows=selection,
                advbench_rows=advbench,
                selection_split_manifest=split,
                holdout_manifest=holdout,
            )

    def test_paired_values_shape_dtype_and_finite(self):
        groups, selection, advbench, split, holdout = aligned_inputs()
        records = build_paired_original_records(
            groups,
            selection_original_rows=selection,
            advbench_rows=advbench,
            selection_split_manifest=split,
            holdout_manifest=holdout,
        )
        values = torch.zeros((68, 32, 32, 120), dtype=torch.float32)
        validate_paired_original_values(values, records)
        self.assertEqual(tuple(values[0].shape), (32, 32, 120))
        values[0, 0, 0, 0] = math.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            validate_paired_original_values(values, records)


class ConditionSwitchTests(unittest.TestCase):
    def test_fixed_head_and_reference_switches(self):
        conditions = factorization_conditions()
        validate_condition_definitions(conditions)
        self.assertEqual(tuple(conditions[0]["heads"]), EXPECTED_TOP5)
        self.assertEqual(tuple(conditions[1]["heads"]), SINGLE_HEAD)
        global_reference = torch.zeros((32, 32, 120), dtype=torch.float32)
        paired_reference = torch.ones((32, 32, 120), dtype=torch.float32)
        self.assertIs(
            reference_for_condition(
                conditions[0],
                global_reference=global_reference,
                paired_reference=paired_reference,
            ),
            global_reference,
        )
        self.assertIs(
            reference_for_condition(
                conditions[2],
                global_reference=global_reference,
                paired_reference=paired_reference,
            ),
            paired_reference,
        )

    def test_oracle_metadata_is_required(self):
        conditions = list(factorization_conditions())
        conditions[2] = {**conditions[2], "deployable_defense": True}
        with self.assertRaisesRegex(ValueError, "oracle metadata"):
            validate_condition_definitions(conditions)


class SummaryTests(unittest.TestCase):
    def test_transition_and_first_token_aggregation(self):
        summary = summarize_condition_results(condition_rows())
        self.assertEqual(len(summary["conditions"]), 8)
        selection_b = next(
            row
            for row in summary["conditions"]
            if row["dataset_group"] == "selection20"
            and row["condition"] == "head13_4_global_mean"
        )
        self.assertEqual(selection_b["newly_defended"], 1)
        self.assertEqual(selection_b["response_changed_count"], 1)
        self.assertAlmostEqual(selection_b["asr"], 19 / 20)

    def test_response_change_after_token_zero_is_rejected(self):
        rows = condition_rows()
        rows[0]["response_equal"] = False
        rows[0]["first_token_argmax_equal"] = False
        rows[0]["first_different_token_position"] = 1
        with self.assertRaisesRegex(ValueError, "later-forward intervention"):
            summarize_condition_results(rows)

    def test_logit_summary_and_nonfinite_rejection(self):
        rows = condition_rows()
        summary = summarize_condition_logits(rows)
        self.assertEqual(len(summary["conditions"]), 8)
        rows[0]["first_token_logit_metrics"][
            "kl_baseline_to_intervention"
        ] = math.inf
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            summarize_condition_logits(rows)

    def test_reference_distance_summary_global_and_oracle(self):
        summary = summarize_reference_distances(reference_rows())
        self.assertEqual(len(summary["conditions"]), 24)
        self.assertEqual(len(summary["contrasts"]), 12)
        first = summary["contrasts"][0]
        self.assertAlmostEqual(
            first["paired_minus_global"][
                "current_to_reference_frobenius_distance"
            ]["mean"],
            2.0,
        )

    def test_requested_cause_classification(self):
        summary = summarize_condition_results(condition_rows())
        result = classify_factorization_causes(summary)
        self.assertEqual(len(result["groups"]), 2)
        self.assertTrue(
            all(
                row["improved_conditions"]
                == [
                    "head13_4_global_mean",
                    "head13_4_paired_original_oracle",
                ]
                for row in result["groups"]
            )
        )
        self.assertTrue(
            all("Layer 13 Head 4" in row["classification"] for row in result["groups"])
        )


if __name__ == "__main__":
    unittest.main()
