import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.analyze_attention_score_all_heads_distributions import (
    apply_short_sequence_policy,
    build_common_metadata,
    compute_mean_distributions,
    main as run_all_heads_main,
    mean_over_layer_heads,
    prepare_line_indexed_prompt_pairs,
    require_tail_window,
    strict_lower_positions,
    validate_prompt_axis,
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
from scripts.analyze_attention_scores import select_rows_from_prompt_manifest


class AllHeadAttentionDistributionTests(unittest.TestCase):
    def test_prompt_ids_use_rows_and_source_indices_are_preserved(self):
        original_rows = [
            {"prompt": "original zero", "source_index": 99},
            {"prompt": "original one", "index": 88},
        ]
        jailbreak_rows = [
            {"prompt": "jailbreak zero", "source_index": 99},
            {"prompt": "jailbreak one", "index": 88},
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
        self.assertEqual([pair["source_index"] for pair in pairs], [99, 88])

    def test_source_index_mismatch_is_rejected(self):
        with patch(
            "scripts.analyze_attention_score_all_heads_distributions.read_prompt_rows",
            side_effect=[
                [{"prompt": "original", "index": 10}],
                [{"prompt": "jailbreak", "index": 11}],
            ],
        ):
            with self.assertRaisesRegex(ValueError, "source_index mismatch"):
                prepare_line_indexed_prompt_pairs(
                    "original.jsonl", "jailbreak.jsonl", max_prompts=None
                )

    @staticmethod
    def _pairs(count=4):
        return [
            {
                "prompt_id": index,
                "input_row_index": index,
                "source_index": 60 + index,
                "original_source_index": 60 + index,
                "jailbreak_source_index": 60 + index,
                "original_prompt": f"o{index}",
                "jailbreak_prompt": f"j{index}",
            }
            for index in range(count)
        ]

    def test_exclude_pair_policy_handles_each_short_sequence_case(self):
        included, manifest = apply_short_sequence_policy(
            self._pairs(),
            original_seq_lengths=[16, 15, 16, 15],
            jailbreak_seq_lengths=[16, 16, 15, 15],
            tail_tokens=16,
            policy="exclude_pair",
            expected_included_prompts=1,
        )
        self.assertEqual([pair["source_index"] for pair in included], [60])
        self.assertEqual([row["included"] for row in manifest], [True, False, False, False])
        self.assertEqual(
            [row["exclusion_reason"] for row in manifest[1:]],
            [
                "original_shorter_than_tail_tokens",
                "jailbreak_shorter_than_tail_tokens",
                "original_and_jailbreak_shorter_than_tail_tokens",
            ],
        )
        self.assertEqual(manifest[0]["raw_prompt_index"], 0)
        self.assertEqual([row["raw_prompt_index"] for row in manifest[1:]], ["", "", ""])
        validate_prompt_axis(included, manifest, raw_prompt_count=1)

    def test_error_policy_rejects_short_pair(self):
        with self.assertRaisesRegex(ValueError, "Sequence shorter than tail window"):
            apply_short_sequence_policy(
                self._pairs(1),
                original_seq_lengths=[15],
                jailbreak_seq_lengths=[20],
                tail_tokens=16,
                policy="error",
            )

    def test_prompt_axis_count_must_match_manifest(self):
        included, manifest = apply_short_sequence_policy(
            self._pairs(2), [16, 16], [16, 16], 16, "exclude_pair"
        )
        with self.assertRaisesRegex(ValueError, "counts differ"):
            validate_prompt_axis(included, manifest, raw_prompt_count=1)

    def test_enhanced_metadata_records_pairing_and_model_structure(self):
        pairs = self._pairs(2)
        included, manifest = apply_short_sequence_policy(
            pairs, [16, 15], [20, 20], 16, "exclude_pair"
        )
        args = SimpleNamespace(
            model="/models/Mistral-7B-Instruct-v0.2",
            original_prompts="original.jsonl",
            jailbreak_prompts="data/gcg/jailbreak.jsonl",
            attack_name="GCG",
            short_sequence_policy="exclude_pair",
            system_prompt=None,
        )
        model = SimpleNamespace(
            config=SimpleNamespace(num_key_value_heads=8, hidden_size=4096)
        )
        tokenizer = SimpleNamespace(chat_template="template")
        metadata = build_common_metadata(
            args=args,
            model_config={"model_name": "Mistral"},
            input_pair_count=2,
            included_pairs=included,
            manifest_rows=manifest,
            tail_tokens=16,
            elements_per_head=120,
            num_layers=32,
            num_attention_heads=32,
            row_positions=np.asarray([1]),
            col_positions=np.asarray([0]),
            execution_datetime="2026-01-01T00:00:00+00:00",
            tokenizer=tokenizer,
            model=model,
            source_commit="abc123",
        )
        self.assertEqual(metadata["input_pair_count"], 2)
        self.assertEqual(metadata["included_pair_count"], 1)
        self.assertEqual(metadata["excluded_pair_count"], 1)
        self.assertEqual(metadata["excluded_source_indices"], [61])
        self.assertEqual(metadata["num_key_value_heads"], 8)
        self.assertEqual(metadata["head_dim"], 128)
        self.assertTrue(metadata["gqa"])
        self.assertEqual(metadata["prompt_manifest_path"], "prompt_manifest.csv")
        self.assertEqual(metadata["attack_name"], "GCG")
        self.assertEqual(metadata["source_commit"], "abc123")
        self.assertIn("torch_version", metadata)
        self.assertIn("transformers_version", metadata)

    def test_attention_metrics_can_reuse_included_manifest_rows(self):
        rows = [
            {"prompt": "zero", "index": 60},
            {"prompt": "one", "index": 61},
            {"prompt": "two", "index": 62},
        ]
        columns = [
            "raw_prompt_index",
            "input_row_index",
            "prompt_id",
            "source_index",
            "original_source_index",
            "jailbreak_source_index",
            "included",
        ]
        manifest = [
            [0, 0, 0, 60, 60, 60, True],
            ["", 1, 1, 61, 61, 61, False],
            [1, 2, 2, 62, 62, 62, True],
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prompt_manifest.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerows(manifest)
            selected = select_rows_from_prompt_manifest(rows, path, "original")
        self.assertEqual([row["raw_prompt_index"] for row in selected], [0, 1])
        self.assertEqual([row["source_index"] for row in selected], ["60", "62"])

    def test_strict_lower_positions_use_row_major_order(self):
        rows, columns = strict_lower_positions(4)
        np.testing.assert_array_equal(rows, [1, 2, 2, 3, 3, 3])
        np.testing.assert_array_equal(columns, [0, 0, 1, 0, 1, 2])
        self.assertEqual(len(rows), 4 * 3 // 2)
        tail16_rows, _ = strict_lower_positions(16)
        self.assertEqual(len(tail16_rows), 120)

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

    def test_main_output_stage_routes_pairs_and_writes_all_summaries(self):
        class DummyModel:
            device = torch.device("cpu")
            config = SimpleNamespace(
                num_key_value_heads=3,
                hidden_size=12,
                head_dim=4,
            )

            def eval(self):
                return self

        def fake_extract(**kwargs):
            value = 1.0 if kwargs["condition"] == "original" else 2.0
            destination = kwargs["destination"]
            destination[...] = value
            mean = np.mean(destination, axis=(0, 1), dtype=np.float64)
            return 20, mean

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            raw_dir = base / "raw"
            mean_dir = base / "mean"
            args = SimpleNamespace(
                model="synthetic-model",
                original_prompts="original.jsonl",
                jailbreak_prompts="jailbreak.jsonl",
                tail_tokens=16,
                raw_output_dir=str(raw_dir),
                mean_output_dir=str(mean_dir),
                max_prompts=None,
                short_sequence_policy="exclude_pair",
                expected_included_prompts=1,
                attack_name="GCG",
                system_prompt=None,
                overwrite=False,
            )
            prompt_rows = [
                {"prompt": "normal", "index": 60},
                {"prompt": "short", "index": 166},
            ]
            model_config = {
                "model_name": "Llama",
                "n_layers": 2,
                "n_heads": 3,
            }
            with (
                patch(
                    "scripts.analyze_attention_score_all_heads_distributions.get_args",
                    return_value=args,
                ),
                patch(
                    "scripts.analyze_attention_score_all_heads_distributions.read_prompt_rows",
                    side_effect=[prompt_rows, prompt_rows],
                ),
                patch(
                    "scripts.analyze_attention_score_all_heads_distributions.load_model_and_tokenizer",
                    return_value=(
                        DummyModel(),
                        SimpleNamespace(chat_template="synthetic-template"),
                        model_config,
                    ),
                ),
                patch(
                    "scripts.analyze_attention_score_all_heads_distributions.get_attention_helpers",
                    return_value=(None, None),
                ),
                patch(
                    "scripts.analyze_attention_score_all_heads_distributions.measure_pair_sequence_lengths",
                    return_value=([20, 15], [20, 35]),
                ),
                patch(
                    "scripts.analyze_attention_score_all_heads_distributions.extract_condition_into_memmap",
                    side_effect=fake_extract,
                ),
                patch(
                    "scripts.analyze_attention_score_all_heads_distributions.get_source_commit",
                    return_value="synthetic-commit",
                ),
            ):
                run_all_heads_main()

            required_raw = {
                "raw_distribution_summary.csv",
                "per_head_summary.csv",
                "prompt_manifest.csv",
                "metadata.json",
            }
            required_mean = {
                "all_heads_mean_distribution_summary.csv",
                "prompt_level_all_heads_mean_summary.csv",
                "metadata.json",
            }
            self.assertTrue(required_raw.issubset(path.name for path in raw_dir.iterdir()))
            self.assertTrue(required_mean.issubset(path.name for path in mean_dir.iterdir()))

            with (raw_dir / "metadata.json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["input_pair_count"], 2)
            self.assertEqual(metadata["included_pair_count"], 1)
            self.assertEqual(metadata["excluded_pair_count"], 1)
            self.assertEqual(metadata["array_shape"], [1, 2, 3, 120])

            with (mean_dir / "prompt_level_all_heads_mean_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                prompt_summary = list(csv.DictReader(handle))
            self.assertEqual(len(prompt_summary), 1)
            self.assertEqual(prompt_summary[0]["prompt_id"], "0")
            self.assertEqual(prompt_summary[0]["source_index"], "60")

            for filename in (
                raw_dir / "raw_distribution_summary.csv",
                mean_dir / "all_heads_mean_distribution_summary.csv",
            ):
                with filename.open(newline="", encoding="utf-8") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 3)

            raw_original = np.load(raw_dir / "raw_original_values.npy")
            raw_jailbreak = np.load(raw_dir / "raw_jailbreak_values.npy")
            raw_delta = np.load(raw_dir / "raw_delta_values.npy")
            np.testing.assert_allclose(raw_delta, raw_jailbreak - raw_original)

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
