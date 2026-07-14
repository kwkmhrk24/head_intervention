import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.analyze_attention_score_all_heads_distributions import (
    compute_mean_distributions,
    mean_over_layer_heads,
    prepare_line_indexed_prompt_pairs,
    require_tail_window,
    strict_lower_positions,
    validate_output_directories,
)
from scripts.plot_attention_score_all_heads_distributions import (
    common_histogram_bin_edges,
    compute_mean_quantile_rows,
    histogram_counts,
    plot_histogram_set,
    save_mean_ecdf,
    save_mean_quantile_plot,
    symmetric_delta_bin_edges,
    write_mean_quantile_summary,
)


class AllHeadAttentionDistributionTests(unittest.TestCase):
    def test_prompt_ids_and_source_indices_use_row_numbers(self):
        original_rows = [
            {"prompt": "original zero", "source_index": 99},
            {"prompt": "original one", "index": 88},
        ]
        jailbreak_rows = [
            {"prompt": "jailbreak zero", "source_index": 77},
            {"prompt": "jailbreak one", "index": 66},
        ]
        with patch(
            "scripts.analyze_attention_score_all_heads_distributions.read_prompt_rows",
            side_effect=[original_rows, jailbreak_rows],
        ):
            pairs = prepare_line_indexed_prompt_pairs(
                "original.jsonl",
                "jailbreak.jsonl",
                max_prompts=None,
            )
        self.assertEqual([pair["prompt_id"] for pair in pairs], [0, 1])
        self.assertEqual([pair["source_index"] for pair in pairs], [0, 1])

    def test_strict_lower_positions_use_row_major_order(self):
        rows, columns = strict_lower_positions(4)
        np.testing.assert_array_equal(rows, [1, 2, 2, 3, 3, 3])
        np.testing.assert_array_equal(columns, [0, 0, 1, 0, 1, 2])
        self.assertEqual(len(rows), 4 * 3 // 2)

    def test_raw_and_mean_delta_identities(self):
        original = np.arange(2 * 3 * 6, dtype=np.float32).reshape(2, 3, 6)
        head_offsets = np.arange(2 * 3, dtype=np.float32).reshape(2, 3, 1)
        position_offsets = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
        jailbreak = original + head_offsets + position_offsets
        raw_delta = jailbreak - original

        mean_original, mean_jailbreak, mean_delta = compute_mean_distributions(
            original,
            jailbreak,
            raw_delta,
        )
        np.testing.assert_allclose(
            mean_original,
            original.mean(axis=(0, 1), dtype=np.float64),
        )
        np.testing.assert_allclose(
            mean_jailbreak,
            jailbreak.mean(axis=(0, 1), dtype=np.float64),
        )
        np.testing.assert_allclose(mean_delta, mean_jailbreak - mean_original)
        np.testing.assert_allclose(
            mean_delta,
            raw_delta.mean(axis=(0, 1), dtype=np.float64),
        )
        np.testing.assert_allclose(mean_over_layer_heads(raw_delta), mean_delta)
        self.assertTrue(np.isfinite(raw_delta).all())

    def test_short_sequence_is_an_explicit_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "condition=original.*prompt_id=3.*seq_len=15.*tail_tokens=16",
        ):
            require_tail_window(
                condition="original",
                input_file="original.jsonl",
                prompt_id=3,
                source_index=3,
                seq_len=15,
                tail_tokens=16,
            )

    def test_common_histogram_edges_and_counts_cover_all_values(self):
        original = np.asarray(
            [[[-2.0, -1.0, 0.0], [0.5, 1.0, 2.0]]],
            dtype=np.float32,
        )
        jailbreak = np.asarray(
            [[[-1.5, -0.5, 0.25], [1.5, 2.5, 3.0]]],
            dtype=np.float32,
        )
        edges = common_histogram_bin_edges(original, jailbreak, bins=5)
        self.assertEqual(edges[0], -2.0)
        self.assertEqual(edges[-1], 3.0)
        original_counts = histogram_counts(original, edges)
        jailbreak_counts = histogram_counts(jailbreak, edges)
        self.assertEqual(int(original_counts.sum()), original.size)
        self.assertEqual(int(jailbreak_counts.sum()), jailbreak.size)

    def test_delta_histogram_edges_are_symmetric_and_cover_all_values(self):
        delta = np.asarray([[[-4.0, -0.5, 0.0, 1.0, 3.0]]], dtype=np.float32)
        edges = symmetric_delta_bin_edges(delta, bins=8)
        self.assertEqual(edges[0], -4.0)
        self.assertEqual(edges[-1], 4.0)
        counts = histogram_counts(delta, edges)
        self.assertEqual(int(counts.sum()), delta.size)

    def test_existing_targets_require_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            mean_dir = Path(temp_dir) / "mean"
            validate_output_directories(raw_dir, mean_dir, overwrite=False)
            (raw_dir / "metadata.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                validate_output_directories(raw_dir, mean_dir, overwrite=False)
            validate_output_directories(raw_dir, mean_dir, overwrite=True)

    def test_synthetic_plot_outputs_use_every_value(self):
        original = np.arange(2 * 2 * 3 * 6, dtype=np.float32).reshape(2, 2, 3, 6)
        jailbreak = original * 0.8 + 0.5
        delta = jailbreak - original
        raw_arrays = {
            "original": original,
            "jailbreak": jailbreak,
            "delta": delta,
        }
        mean_arrays = {
            condition: values.mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
            for condition, values in raw_arrays.items()
        }
        metadata = {
            "window_size": 4,
            "processed_prompt_count": 2,
            "layer_head_count": 6,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            raw_dir = base / "raw"
            mean_dir = base / "mean"
            raw_dir.mkdir()
            mean_dir.mkdir()
            plot_histogram_set(
                raw_dir,
                raw_arrays,
                metadata,
                bins=10,
                title_label="Synthetic Raw",
                write_bin_csvs=True,
            )
            plot_histogram_set(
                mean_dir,
                mean_arrays,
                metadata,
                bins=10,
                title_label="Synthetic Mean",
                write_bin_csvs=False,
            )
            save_mean_ecdf(
                mean_dir / "ecdf_original_vs_jailbreak.png",
                mean_arrays["original"],
                mean_arrays["jailbreak"],
                metadata,
            )
            quantile_rows = compute_mean_quantile_rows(
                mean_arrays,
                metadata,
                [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
            )
            write_mean_quantile_summary(
                mean_dir / "quantile_summary.csv",
                quantile_rows,
            )
            save_mean_quantile_plot(
                mean_dir / "quantile_comparison.png",
                quantile_rows,
                metadata,
            )

            expected_raw = {
                "histogram_original_vs_jailbreak.png",
                "histogram_delta.png",
                "histogram_original_vs_jailbreak_bins.csv",
                "histogram_delta_bins.csv",
            }
            expected_mean = {
                "histogram_original_vs_jailbreak.png",
                "histogram_delta.png",
                "ecdf_original_vs_jailbreak.png",
                "quantile_comparison.png",
                "quantile_summary.csv",
            }
            self.assertTrue(expected_raw.issubset(path.name for path in raw_dir.iterdir()))
            self.assertTrue(expected_mean.issubset(path.name for path in mean_dir.iterdir()))
            for filename in expected_raw:
                self.assertGreater((raw_dir / filename).stat().st_size, 0)
            for filename in expected_mean:
                self.assertGreater((mean_dir / filename).stat().st_size, 0)

            with (raw_dir / "histogram_original_vs_jailbreak_bins.csv").open(
                newline="",
                encoding="utf-8",
            ) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(sum(int(row["original_count"]) for row in rows), original.size)
            self.assertEqual(
                sum(int(row["jailbreak_count"]) for row in rows),
                jailbreak.size,
            )
            self.assertEqual(len(quantile_rows), 7)


if __name__ == "__main__":
    unittest.main()
