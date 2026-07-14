import unittest

import numpy as np
import pandas as pd

from scripts.analyze_autodan_elementwise_head_metrics import (
    bootstrap_consistent_stability,
    bootstrap_prompt_indices,
    build_overlap_outputs,
    build_top_head_outputs,
    compute_elementwise_metrics,
    compute_structural_metrics,
    jaccard_similarity,
    metrics_dataframe,
    strict_lower_positions,
    validate_delta_identity,
    validate_selection_indices,
)


class ShapeAndIndexTests(unittest.TestCase):
    def test_tail16_strict_lower_last_row_and_lags(self):
        rows, columns = strict_lower_positions(16)
        self.assertEqual(len(rows), 120)
        self.assertEqual(int(np.sum(rows == 15)), 15)
        self.assertEqual(sorted(np.unique(rows - columns).tolist()), list(range(1, 16)))

    def test_shape_mismatch_is_rejected(self):
        rows, columns = strict_lower_positions(16)
        original = np.zeros((2, 1, 1, 120), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "selection shapes differ"):
            compute_elementwise_metrics(
                original,
                np.zeros((3, 1, 1, 120), dtype=np.float32),
                original,
                rows,
                columns,
            )

    def test_delta_identity_and_nonfinite_values(self):
        original = np.ones((2, 1, 1, 120), dtype=np.float32)
        jailbreak = original + 2
        delta = jailbreak - original
        self.assertEqual(validate_delta_identity(original, jailbreak, delta), 0.0)
        bad = delta.copy()
        bad[0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "delta identity"):
            validate_delta_identity(original, jailbreak, bad)
        bad = delta.copy()
        bad[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            validate_delta_identity(original, jailbreak, bad)

    def test_selection_indices_extract_twenty_and_reject_duplicate_or_range(self):
        indices = list(range(20))
        validated = validate_selection_indices(indices, total_count=130)
        values = np.arange(130 * 2).reshape(130, 2)
        self.assertEqual(values[validated].shape, (20, 2))

        duplicate = list(range(19)) + [18]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_selection_indices(duplicate, total_count=130)
        out_of_range = list(range(19)) + [130]
        with self.assertRaisesRegex(ValueError, "out of range"):
            validate_selection_indices(out_of_range, total_count=130)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.columns = strict_lower_positions(16)

    def test_signed_mass_rate_consistency_last_row_and_lag_metrics(self):
        # h0 is always +2; h1 is always -3; h2 changes prompt direction.
        delta = np.zeros((4, 1, 3, 120), dtype=np.float32)
        delta[:, 0, 0, :] = 2
        delta[:, 0, 1, :] = -3
        delta[:2, 0, 2, :] = 1
        delta[2:, 0, 2, :] = -1
        original = np.full(delta.shape, 10, dtype=np.float32)
        jailbreak = original + delta

        metrics, position_mean, lag_signed, lag_absolute = (
            compute_elementwise_metrics(
                original,
                jailbreak,
                delta,
                self.rows,
                self.columns,
                consistency_threshold=0.75,
            )
        )

        self.assertAlmostEqual(metrics["signed_mean"][0, 0], 2)
        self.assertAlmostEqual(metrics["signed_median"][0, 0], 2)
        self.assertAlmostEqual(metrics["positive_mass"][0, 0], 2)
        self.assertAlmostEqual(metrics["negative_mass"][0, 0], 0)
        self.assertAlmostEqual(metrics["absolute_mass"][0, 0], 2)
        self.assertAlmostEqual(metrics["positive_rate"][0, 0], 1)
        self.assertAlmostEqual(metrics["negative_rate"][0, 0], 0)
        self.assertAlmostEqual(metrics["sign_imbalance"][0, 0], 1)
        self.assertAlmostEqual(metrics["strong_positive_position_rate"][0, 0], 1)

        self.assertAlmostEqual(metrics["signed_mean"][0, 1], -3)
        self.assertAlmostEqual(metrics["negative_mass"][0, 1], 3)
        self.assertAlmostEqual(metrics["absolute_mass"][0, 1], 3)
        self.assertAlmostEqual(metrics["negative_rate"][0, 1], 1)
        self.assertAlmostEqual(metrics["sign_imbalance"][0, 1], -1)
        self.assertAlmostEqual(metrics["strong_negative_position_rate"][0, 1], 1)

        self.assertAlmostEqual(metrics["positive_mass"][0, 2], 0.5)
        self.assertAlmostEqual(metrics["negative_mass"][0, 2], 0.5)
        self.assertAlmostEqual(metrics["absolute_mass"][0, 2], 1)
        self.assertAlmostEqual(metrics["consistent_positive"][0, 2], 0)
        self.assertAlmostEqual(metrics["consistent_negative"][0, 2], 0)
        self.assertAlmostEqual(metrics["consistent_absolute"][0, 2], 0)
        self.assertAlmostEqual(metrics["strong_positive_position_rate"][0, 2], 0)
        self.assertAlmostEqual(metrics["strong_negative_position_rate"][0, 2], 0)

        np.testing.assert_allclose(
            metrics["absolute_mass"],
            metrics["positive_mass"] + metrics["negative_mass"],
        )
        np.testing.assert_allclose(
            metrics["consistent_absolute"],
            metrics["consistent_positive"] + metrics["consistent_negative"],
        )
        self.assertAlmostEqual(metrics["last_row_signed_mean"][0, 0], 2)
        self.assertAlmostEqual(metrics["last_row_positive_mass"][0, 0], 2)
        self.assertAlmostEqual(metrics["last_row_negative_mass"][0, 1], 3)
        self.assertAlmostEqual(metrics["last_row_absolute_mass"][0, 1], 3)
        self.assertAlmostEqual(metrics["last_row_consistent_positive"][0, 0], 2)
        self.assertAlmostEqual(metrics["last_row_consistent_negative"][0, 1], 3)
        self.assertAlmostEqual(metrics["last_row_consistent_absolute"][0, 2], 0)

        self.assertEqual(position_mean.shape, (1, 3, 120))
        self.assertEqual(lag_signed.shape, (1, 3, 15))
        self.assertEqual(lag_absolute.shape, (1, 3, 15))
        np.testing.assert_allclose(lag_signed[0, 0], 2)
        np.testing.assert_allclose(lag_absolute[0, 1], 3)
        self.assertAlmostEqual(metrics["lag_profile_absolute"][0, 0], 2)
        self.assertAlmostEqual(metrics["lag_profile_mass"][0, 1], 3)

    def test_sign_flip_cosine_and_pearson_constant_vector_policy(self):
        original = np.zeros((2, 1, 3, 120), dtype=np.float32)
        pattern = np.linspace(-2, 2, 120, dtype=np.float32)
        original[:, 0, 0, :] = pattern
        jailbreak = original.copy()
        jailbreak[:, 0, 0, :] *= 2

        # Both zero vectors use cosine 1 and have invalid Pearson correlation.
        # Exactly one zero vector uses cosine 0.
        jailbreak[:, 0, 2, :] = 1
        structural = compute_structural_metrics(original, jailbreak)

        self.assertAlmostEqual(structural["mean_cosine_similarity"][0, 0], 1)
        self.assertAlmostEqual(structural["mean_cosine_distance"][0, 0], 0)
        self.assertAlmostEqual(structural["mean_pearson_correlation"][0, 0], 1)
        self.assertEqual(structural["valid_pearson_prompt_count"][0, 0], 2)
        self.assertAlmostEqual(structural["sign_flip_rate"][0, 0], 0)

        self.assertAlmostEqual(structural["mean_cosine_similarity"][0, 1], 1)
        self.assertTrue(np.isnan(structural["mean_pearson_correlation"][0, 1]))
        self.assertEqual(structural["valid_pearson_prompt_count"][0, 1], 0)
        self.assertAlmostEqual(structural["mean_cosine_similarity"][0, 2], 0)
        self.assertAlmostEqual(structural["sign_flip_rate"][0, 2], 1)

    def test_np_sign_zero_is_part_of_sign_flip_definition(self):
        original = np.zeros((1, 1, 1, 120), dtype=np.float32)
        jailbreak = original.copy()
        jailbreak[..., :60] = 1
        structural = compute_structural_metrics(original, jailbreak)
        self.assertAlmostEqual(structural["sign_flip_rate"][0, 0], 0.5)


class BootstrapTests(unittest.TestCase):
    def test_prompt_resampling_shape_and_seed_are_reproducible(self):
        first = bootstrap_prompt_indices(20, 100, 42)
        second = bootstrap_prompt_indices(20, 100, 42)
        other = bootstrap_prompt_indices(20, 100, 43)
        self.assertEqual(first.shape, (100, 20))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        self.assertTrue(((first >= 0) & (first < 20)).all())

    def test_inclusion_probability_and_jaccard_summary(self):
        delta = np.zeros((4, 2, 2, 120), dtype=np.float32)
        delta[:, 0, 0, :] = 5
        delta[:, 0, 1, :] = -4
        delta[:, 1, 0, :] = 1
        inclusion_a, summary_a, samples_a = bootstrap_consistent_stability(
            delta, top_k=1, sample_count=50, seed=42
        )
        inclusion_b, summary_b, samples_b = bootstrap_consistent_stability(
            delta, top_k=1, sample_count=50, seed=42
        )

        pd.testing.assert_frame_equal(inclusion_a, inclusion_b)
        self.assertEqual(summary_a, summary_b)
        np.testing.assert_array_equal(samples_a, samples_b)
        probability_columns = [
            column for column in inclusion_a if column.endswith("inclusion_probability")
        ]
        self.assertTrue(
            ((inclusion_a[probability_columns] >= 0)
             & (inclusion_a[probability_columns] <= 1)).all().all()
        )
        for metric in ("consistent_positive", "consistent_negative", "consistent_absolute"):
            stats = summary_a["metrics"][metric]["jaccard_similarity"]
            self.assertEqual(stats["mean"], 1.0)
            self.assertEqual(stats["std"], 0.0)
            self.assertEqual(stats["q05"], 1.0)
            self.assertEqual(stats["median"], 1.0)
            self.assertEqual(stats["q95"], 1.0)


class RankingAndOverlapTests(unittest.TestCase):
    def synthetic_metrics(self):
        shape = (2, 2)
        metrics = {}
        for index, name in enumerate(
            [
                "signed_mean", "signed_median", "positive_mass", "negative_mass",
                "absolute_mass", "positive_rate", "negative_rate", "sign_imbalance",
                "consistent_positive", "consistent_negative", "consistent_absolute",
                "strong_positive_position_rate", "strong_negative_position_rate",
                "last_row_signed_mean", "last_row_positive_mass",
                "last_row_negative_mass", "last_row_absolute_mass",
                "last_row_consistent_positive", "last_row_consistent_negative",
                "last_row_consistent_absolute", "lag_profile_absolute",
                "lag_profile_mass", "mean_cosine_similarity", "mean_cosine_distance",
                "sign_flip_rate", "mean_pearson_correlation",
                "valid_pearson_prompt_count",
            ]
        ):
            metrics[name] = np.arange(4, dtype=float).reshape(shape) + index
        metrics["signed_mean"] = np.array([[3.0, -5.0], [1.0, -2.0]])
        metrics["negative_mass"] = np.array([[0.0, 9.0], [2.0, 4.0]])
        return metrics

    def test_negative_mass_descending_signed_bottom_ascending_and_mapping(self):
        frame = metrics_dataframe(self.synthetic_metrics())
        payload, top_csv = build_top_head_outputs(frame, top_n=4)

        negative_mass = payload["negative_mass_top"]
        self.assertEqual((negative_mass[0]["layer"], negative_mass[0]["head"]), (0, 1))
        self.assertEqual([row["value"] for row in negative_mass], [9.0, 4.0, 2.0, 0.0])
        signed_bottom = payload["signed_mean_negative_bottom"]
        self.assertEqual((signed_bottom[0]["layer"], signed_bottom[0]["head"]), (0, 1))
        self.assertEqual([row["value"] for row in signed_bottom], [-5.0, -2.0, 1.0, 3.0])
        for ranking in top_csv["ranking"].unique():
            subset = top_csv[top_csv["ranking"] == ranking]
            self.assertFalse(subset.duplicated(["layer", "head"]).any())

    def test_overlap_and_jaccard(self):
        frame = metrics_dataframe(self.synthetic_metrics())
        references = {"paper": [(1, 1), (0, 0)]}
        overlap, payload = build_overlap_outputs(frame, references, top_sizes=(1, 2))
        row = overlap[
            (overlap["metric"] == "consistent_positive")
            & (overlap["new_head_count"] == 2)
        ].iloc[0]
        self.assertEqual(row["overlap_count"], 1)
        self.assertAlmostEqual(row["jaccard_similarity"], 1 / 3)
        self.assertEqual(len(payload["comparisons"]), 6)
        self.assertAlmostEqual(jaccard_similarity({1, 2}, {2, 3}), 1 / 3)


if __name__ == "__main__":
    unittest.main()
