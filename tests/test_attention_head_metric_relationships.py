import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_attention_head_metric_relationships import (
    add_absolute_columns,
    aggregate_attention_metrics,
    attach_aie_and_flags,
    build_element_metrics,
    build_layer_summary,
    correlation_long,
    layer_center,
    pearson_correlation,
    prepare_output_dir,
    spearman_correlation,
    validate_aggregation_consistency,
    validate_head_grid,
)
from scripts.plot_attention_head_metric_relationships import (
    heatmap_pivot,
    prepare_outputs,
)


def synthetic_inputs():
    original_rows = []
    jailbreak_rows = []
    per_head_rows = []
    for layer in range(32):
        for head in range(32):
            lower_means = []
            jailbreak_lower_means = []
            positive_rates = []
            jailbreak_positive_rates = []
            for prompt_id in range(2):
                lower_mean = layer * 0.01 + head * 0.001 + prompt_id * 0.0001
                lower_delta = 0.02 - layer * 0.0002 + head * 0.00003
                positive_rate = 0.2 + layer * 0.002 + head * 0.0005
                positive_delta = 0.03 - layer * 0.0001
                base = {
                    "prompt_id": prompt_id,
                    "source_index": np.nan,
                    "layer": layer,
                    "head": head,
                    "window_size": 16,
                    "sym_ratio": 0.5 + layer * 0.001,
                    "asym_ratio": 0.4 - layer * 0.001,
                    "asym_lower_mean": lower_mean,
                    "asym_lower_abs_mean": abs(lower_mean) + 0.5,
                    "asym_lower_pos_rate": positive_rate,
                    "sym_norm_per_token": 2.0 + head * 0.01,
                    "asym_norm_per_token": 1.5 + layer * 0.01,
                }
                jailbreak = dict(base)
                jailbreak.update(
                    {
                        "sym_ratio": base["sym_ratio"] + 0.01,
                        "asym_ratio": base["asym_ratio"] - 0.015,
                        "asym_lower_mean": lower_mean + lower_delta,
                        "asym_lower_abs_mean": base["asym_lower_abs_mean"] + 0.005,
                        "asym_lower_pos_rate": positive_rate + positive_delta,
                        "sym_norm_per_token": base["sym_norm_per_token"] + 0.02,
                        "asym_norm_per_token": base["asym_norm_per_token"] - 0.01,
                    }
                )
                original_rows.append(base)
                jailbreak_rows.append(jailbreak)
                lower_means.append(lower_mean)
                jailbreak_lower_means.append(lower_mean + lower_delta)
                positive_rates.append(positive_rate)
                jailbreak_positive_rates.append(positive_rate + positive_delta)
            original_std = 1.0 + layer * 0.01
            std_shift = head * 0.001 - 0.01
            per_head_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "value_count": 240,
                    "original_mean": np.mean(lower_means),
                    "jailbreak_mean": np.mean(jailbreak_lower_means),
                    "delta_mean": np.mean(np.asarray(jailbreak_lower_means) - lower_means),
                    "original_std": original_std,
                    "jailbreak_std": original_std + std_shift,
                    "delta_std": 0.25 + head * 0.001,
                    "original_positive_rate": np.mean(positive_rates),
                    "jailbreak_positive_rate": np.mean(jailbreak_positive_rates),
                    "delta_positive_rate": 0.4 + head * 0.001,
                }
            )
    return (
        pd.DataFrame(per_head_rows),
        pd.DataFrame(original_rows),
        pd.DataFrame(jailbreak_rows),
    )


def synthetic_merged():
    per_head, original, jailbreak = synthetic_inputs()
    element = build_element_metrics(per_head)
    attention, _, _ = aggregate_attention_metrics(original, jailbreak)
    merged = element.merge(attention, on=["layer", "head"], validate="one_to_one")
    aie = np.linspace(-1, 1, 1024, dtype=np.float32).reshape(32, 32)
    aie[13, 4] = 2.0
    merged = attach_aie_and_flags(merged, aie, top_k=20)
    return add_absolute_columns(merged), aie


class AttentionHeadMetricRelationshipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.per_head, cls.original, cls.jailbreak = synthetic_inputs()
        cls.merged, cls.aie = synthetic_merged()

    def test_prompt_level_join_uses_prompt_layer_head(self):
        aggregated, prompt_count, window_size = aggregate_attention_metrics(
            self.original, self.jailbreak
        )
        self.assertEqual(len(aggregated), 1024)
        self.assertEqual(prompt_count, 2)
        self.assertEqual(window_size, 16)

    def test_source_index_nan_does_not_prevent_join(self):
        self.assertTrue(self.original["source_index"].isna().all())
        aggregated, _, _ = aggregate_attention_metrics(self.original, self.jailbreak)
        self.assertEqual(len(aggregated), 1024)

    def test_1024_head_grid_is_unique(self):
        layers, heads = validate_head_grid(self.merged)
        self.assertEqual(len(self.merged), 1024)
        self.assertEqual(layers, list(range(32)))
        self.assertEqual(heads, list(range(32)))

    def test_element_std_shift_definition(self):
        element = build_element_metrics(self.per_head)
        np.testing.assert_allclose(
            element["element_std_shift"],
            element["element_jailbreak_std"] - element["element_original_std"],
        )
        self.assertFalse(
            np.allclose(element["element_std_shift"], element["element_delta_std"])
        )

    def test_element_positive_rate_shift_definition(self):
        element = build_element_metrics(self.per_head)
        np.testing.assert_allclose(
            element["element_positive_rate_shift"],
            element["element_jailbreak_positive_rate"]
            - element["element_original_positive_rate"],
        )

    def test_aie_layer_head_mapping(self):
        row = self.merged[
            (self.merged["layer"] == 13) & (self.merged["head"] == 4)
        ].iloc[0]
        self.assertEqual(row.aie, self.aie[13, 4])

    def test_signed_and_absolute_ranks(self):
        self.assertEqual(
            int(self.merged.loc[self.merged["aie_rank"].idxmin(), "aie_rank"]), 1
        )
        self.assertEqual(
            self.merged.loc[self.merged["aie_rank"] == 1, "head_label"].iloc[0],
            "L13H4",
        )
        expected_abs = int(np.argmax(np.abs(self.aie)))
        expected_layer, expected_head = np.unravel_index(expected_abs, self.aie.shape)
        actual = self.merged.loc[self.merged["abs_aie_rank"] == 1].iloc[0]
        self.assertEqual(
            (actual["layer"], actual["head"]), (expected_layer, expected_head)
        )

    def test_paper_head_flag(self):
        self.assertEqual(int(self.merged["paper_head"].sum()), 5)
        self.assertTrue(
            bool(
                self.merged.loc[
                    (self.merged["layer"] == 13) & (self.merged["head"] == 4),
                    "paper_head",
                ].iloc[0]
            )
        )

    def test_layer_summary(self):
        summary = build_layer_summary(self.merged)
        self.assertEqual(len(summary), 32)
        self.assertTrue(summary["head_count"].eq(32).all())

    def test_layer_centered_means_are_zero(self):
        metrics = ["element_delta_mean", "element_std_shift", "aie"]
        centered = layer_center(self.merged, metrics)
        means = centered.groupby("layer")[metrics].mean()
        self.assertTrue(np.allclose(means, 0, atol=1e-7))

    def test_known_pearson_and_spearman(self):
        x = [1, 2, 3, 4]
        y = [2, 4, 6, 8]
        self.assertAlmostEqual(pearson_correlation(x, y), 1.0)
        self.assertAlmostEqual(spearman_correlation(x, y), 1.0)
        correlation = correlation_long(
            pd.DataFrame({"x": x, "y": y}), ["x", "y"], "spearman"
        )
        self.assertEqual(set(correlation["n"]), {4})

    def test_heatmap_pivot_layer_head_placement(self):
        values = heatmap_pivot(self.merged, "element_delta_mean")
        expected = self.merged.loc[
            (self.merged["layer"] == 7) & (self.merged["head"] == 16),
            "element_delta_mean",
        ].iloc[0]
        self.assertEqual(values[7, 16], expected)

    def test_existing_outputs_require_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            prepare_output_dir(path, overwrite=False)
            (path / "metadata.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output_dir(path, overwrite=False)
            prepare_output_dir(path, overwrite=True)

            (path / "heatmap_aie.png").write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                prepare_outputs(path, overwrite=False)

    def test_mean_consistency_passes_and_mismatch_fails(self):
        validate_aggregation_consistency(self.merged)
        broken = self.merged.copy()
        broken.loc[0, "element_delta_mean"] += 0.1
        with self.assertRaisesRegex(ValueError, "Aggregation consistency failed"):
            validate_aggregation_consistency(broken)

    def test_positive_rate_consistency_passes_and_mismatch_fails(self):
        validate_aggregation_consistency(self.merged)
        broken = self.merged.copy()
        broken.loc[0, "element_positive_rate_shift"] += 0.1
        with self.assertRaisesRegex(ValueError, "Aggregation consistency failed"):
            validate_aggregation_consistency(broken)


if __name__ == "__main__":
    unittest.main()
