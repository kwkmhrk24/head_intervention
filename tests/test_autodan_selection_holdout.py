import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analyze_autodan_selection_holdout import (
    aggregate_head_ranking,
    bootstrap_head_stability,
    build_holdout_split,
    build_selection_metric_pairs,
    read_selected_metric_csv,
    select_ranked_heads,
    subset_rows,
    validate_prompt_pairs_and_labels,
)


def synthetic_prompt_inputs(total_count=130, success_count=90):
    original = [{"prompt": f"original-{index}"} for index in range(total_count)]
    jailbreak = [{"prompt": f"jailbreak-{index}"} for index in range(total_count)]
    baseline = [
        {
            "index": index,
            "user prompt": f"jailbreak-{index}",
            "success": index < success_count,
        }
        for index in range(total_count)
    ]
    return original, jailbreak, baseline


def synthetic_prompt_metrics():
    deltas = {
        (0, 0): [0.40, 0.50, 0.30, 0.40],
        (0, 1): [-0.80, -0.70, -0.90, -0.80],
        (0, 2): [0.20, 0.10, 0.30, 0.20],
    }
    rows = []
    for source_index in range(4):
        for (layer, head), values in deltas.items():
            original = 1.0 + head * 0.1
            jailbreak = original + values[source_index]
            rows.append(
                {
                    "source_index": source_index,
                    "layer": layer,
                    "head": head,
                    "original_asym_ratio": original,
                    "jailbreak_asym_ratio": jailbreak,
                    "delta_asym_ratio": values[source_index],
                }
            )
    return pd.DataFrame(rows)


class AutoDANSelectionHoldoutTests(unittest.TestCase):
    def test_prompt_pair_annotation_preserves_line_index_alignment(self):
        original, jailbreak, baseline = synthetic_prompt_inputs()
        annotated_original, annotated_jailbreak, successes, has_responses = (
            validate_prompt_pairs_and_labels(original, jailbreak, baseline)
        )
        self.assertEqual(len(annotated_original), 130)
        self.assertEqual(len(annotated_jailbreak), 130)
        self.assertEqual(successes, list(range(90)))
        self.assertFalse(has_responses)
        for index, (original_row, jailbreak_row) in enumerate(
            zip(annotated_original, annotated_jailbreak)
        ):
            self.assertEqual(original_row["index"], index)
            self.assertEqual(original_row["prompt_id"], index)
            self.assertEqual(original_row["source_index"], index)
            self.assertEqual(jailbreak_row["index"], index)
            self.assertEqual(jailbreak_row["prompt_id"], index)
            self.assertEqual(jailbreak_row["source_index"], index)

    def test_prompt_mismatch_fails_instead_of_silent_pairing(self):
        original, jailbreak, baseline = synthetic_prompt_inputs()
        baseline[12]["user prompt"] = "wrong"
        with self.assertRaisesRegex(ValueError, "does not match jailbreak prompt"):
            validate_prompt_pairs_and_labels(original, jailbreak, baseline)

    def test_split_is_disjoint_complete_success_only_and_reproducible(self):
        successes = list(range(84))
        selection_a, evaluation_a = build_holdout_split(130, successes, 20, 42)
        selection_b, evaluation_b = build_holdout_split(130, successes, 20, 42)
        selection_other, _ = build_holdout_split(130, successes, 20, 43)

        self.assertEqual(selection_a, selection_b)
        self.assertEqual(evaluation_a, evaluation_b)
        self.assertEqual(
            selection_a,
            [
                3,
                4,
                11,
                13,
                14,
                17,
                25,
                27,
                28,
                29,
                31,
                35,
                53,
                54,
                64,
                69,
                71,
                74,
                81,
                83,
            ],
        )
        self.assertNotEqual(selection_a, selection_other)
        self.assertEqual(len(selection_a), 20)
        self.assertEqual(len(evaluation_a), 110)
        self.assertFalse(set(selection_a).intersection(evaluation_a))
        self.assertEqual(set(selection_a).union(evaluation_a), set(range(130)))
        self.assertTrue(set(selection_a).issubset(successes))

    def test_split_subsets_keep_original_jailbreak_pair_order(self):
        original, jailbreak, baseline = synthetic_prompt_inputs()
        annotated_original, annotated_jailbreak, successes, _ = (
            validate_prompt_pairs_and_labels(original, jailbreak, baseline)
        )
        selection, evaluation = build_holdout_split(130, successes, 20, 42)
        selection_original = subset_rows(annotated_original, selection)
        selection_jailbreak = subset_rows(annotated_jailbreak, selection)
        evaluation_original = subset_rows(annotated_original, evaluation)
        evaluation_jailbreak = subset_rows(annotated_jailbreak, evaluation)

        self.assertEqual(
            [row["source_index"] for row in selection_original], selection
        )
        self.assertEqual(
            [row["source_index"] for row in selection_jailbreak], selection
        )
        self.assertEqual(
            [row["source_index"] for row in evaluation_original], evaluation
        )
        self.assertEqual(
            [row["source_index"] for row in evaluation_jailbreak], evaluation
        )

    def test_split_fails_when_fewer_than_twenty_successes_exist(self):
        with self.assertRaisesRegex(ValueError, "Not enough baseline-success"):
            build_holdout_split(130, list(range(19)), 20, 42)

    def test_positive_negative_and_absolute_rankings_remain_separate(self):
        ranking = aggregate_head_ranking(synthetic_prompt_metrics())
        selected = select_ranked_heads(ranking, top_k=1)

        self.assertEqual(
            tuple(selected["positive"][["layer", "head"]].iloc[0]), (0, 0)
        )
        self.assertEqual(
            tuple(selected["negative"][["layer", "head"]].iloc[0]), (0, 1)
        )
        self.assertEqual(
            tuple(selected["absolute"][["layer", "head"]].iloc[0]), (0, 1)
        )
        positive_row = ranking[(ranking["layer"] == 0) & (ranking["head"] == 0)].iloc[0]
        absolute_row = ranking[(ranking["layer"] == 0) & (ranking["head"] == 1)].iloc[0]
        self.assertEqual(int(positive_row["positive_rank"]), 1)
        self.assertEqual(int(absolute_row["absolute_rank"]), 1)

    def test_selection_metric_pairing_computes_jailbreak_minus_original(self):
        original = []
        jailbreak = []
        for source_index in (4, 9):
            for head in range(3):
                original.append(
                    {
                        "source_index": source_index,
                        "layer": 0,
                        "head": head,
                        "asym_ratio": 0.2 + head,
                    }
                )
                jailbreak.append(
                    {
                        "source_index": source_index,
                        "layer": 0,
                        "head": head,
                        "asym_ratio": 0.3 + head,
                    }
                )
        paired = build_selection_metric_pairs(
            pd.DataFrame(original),
            pd.DataFrame(jailbreak),
            [4, 9],
            expected_layers=None,
            expected_heads_per_layer=None,
        )
        self.assertEqual(len(paired), 6)
        self.assertTrue((paired["delta_asym_ratio"].round(12) == 0.1).all())

    def test_metric_reader_ignores_evaluation_metric_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            fieldnames = [
                "prompt_id",
                "source_index",
                "layer",
                "head",
                "window_size",
                "asym_ratio",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "prompt_id": 0,
                            "source_index": "",
                            "layer": 0,
                            "head": 0,
                            "window_size": 16,
                            "asym_ratio": 0.2,
                        },
                        {
                            "prompt_id": 1,
                            "source_index": "",
                            "layer": 0,
                            "head": 0,
                            "window_size": 16,
                            "asym_ratio": 0.3,
                        },
                        {
                            "prompt_id": 2,
                            "source_index": "",
                            "layer": "evaluation-not-read",
                            "head": "evaluation-not-read",
                            "window_size": "evaluation-not-read",
                            "asym_ratio": "evaluation-not-read",
                        },
                    ]
                )
            selected = read_selected_metric_csv(str(path), [0, 1], 16)
            self.assertEqual(len(selected), 2)
            self.assertEqual(set(selected["source_index"]), {0, 1})

    def test_metric_reader_accepts_equivalent_legacy_skew_ratio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "prompt_id",
                        "source_index",
                        "layer",
                        "head",
                        "window_size",
                        "skew_ratio",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "prompt_id": 0,
                        "source_index": "",
                        "layer": 0,
                        "head": 0,
                        "window_size": 16,
                        "skew_ratio": 0.25,
                    }
                )
            selected = read_selected_metric_csv(str(path), [0], 16)
            self.assertAlmostEqual(float(selected.iloc[0]["asym_ratio"]), 0.25)

    def test_bootstrap_iteration_rates_and_seed_are_reproducible(self):
        prompt_metrics = synthetic_prompt_metrics()
        ranking = aggregate_head_ranking(prompt_metrics)
        base = select_ranked_heads(ranking, top_k=1)
        stability_a, jaccard_a, summary_a = bootstrap_head_stability(
            prompt_metrics, base, iterations=40, seed=42, top_k=1
        )
        stability_b, jaccard_b, summary_b = bootstrap_head_stability(
            prompt_metrics, base, iterations=40, seed=42, top_k=1
        )

        self.assertEqual(len(jaccard_a), 40)
        self.assertEqual(summary_a["bootstrap_iterations"], 40)
        self.assertFalse(summary_a["evaluation_metrics_used"])
        for condition in ("positive", "negative", "absolute"):
            rates = stability_a[f"{condition}_selection_rate"]
            self.assertTrue(rates.between(0, 1).all())
            self.assertEqual(
                int(stability_a[f"{condition}_selection_count"].sum()), 40
            )
        pd.testing.assert_frame_equal(stability_a, stability_b)
        pd.testing.assert_frame_equal(jaccard_a, jaccard_b)
        self.assertEqual(summary_a, summary_b)


if __name__ == "__main__":
    unittest.main()
