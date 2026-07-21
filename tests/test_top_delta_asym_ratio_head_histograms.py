import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.plot_top_delta_asym_ratio_head_histograms import (
    build_summary_row,
    extract_head_arrays,
    infer_plot_labels,
    main,
    read_head_metrics,
    reject_existing_targets,
    select_signed_heads,
    validate_delta,
    validate_prompt_manifest,
)


def synthetic_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "layer": [0, 0, 1, 1, 2, 2],
            "head": [0, 1, 0, 1, 0, 1],
            "delta_asym_ratio": [9.0, 3.0, 1.0, -2.0, -20.0, 0.5],
            "aie": [-100.0, -50.0, 200.0, 150.0, 300.0, 250.0],
            "paper_head": [False, False, False, True, False, False],
        }
    )


def synthetic_arrays() -> dict:
    original = np.arange(2 * 3 * 2 * 3, dtype=np.float32).reshape(2, 3, 2, 3)
    jailbreak = original + np.float32(2.0)
    return {
        "original": original,
        "jailbreak": jailbreak,
        "delta": jailbreak - original,
    }


class TopDeltaAsymRatioHistogramTests(unittest.TestCase):
    def test_increase_top_k_is_signed_descending(self):
        selected = select_signed_heads(
            synthetic_metrics(),
            "delta_asym_ratio",
            top_k=2,
            bottom_k=2,
        )
        increase = selected[selected["group"] == "increase_top2"]
        self.assertEqual(increase[["layer", "head"]].values.tolist(), [[0, 0], [0, 1]])
        self.assertEqual(increase["delta_asym_ratio"].tolist(), [9.0, 3.0])

    def test_decrease_bottom_k_is_signed_ascending(self):
        selected = select_signed_heads(
            synthetic_metrics(),
            "delta_asym_ratio",
            top_k=2,
            bottom_k=2,
        )
        decrease = selected[selected["group"] == "decrease_top2"]
        self.assertEqual(decrease[["layer", "head"]].values.tolist(), [[2, 0], [1, 1]])
        self.assertEqual(decrease["delta_asym_ratio"].tolist(), [-20.0, -2.0])

    def test_selection_is_not_absolute_value_ranking(self):
        selected = select_signed_heads(
            synthetic_metrics(),
            "delta_asym_ratio",
            top_k=1,
            bottom_k=1,
        )
        increase = selected.iloc[0]
        decrease = selected.iloc[1]
        self.assertEqual((increase["layer"], increase["head"]), (0, 0))
        self.assertEqual((decrease["layer"], decrease["head"]), (2, 0))

    def test_selected_heads_do_not_overlap(self):
        selected = select_signed_heads(
            synthetic_metrics(),
            "delta_asym_ratio",
            top_k=3,
            bottom_k=3,
        )
        self.assertEqual(len(selected), 6)
        self.assertFalse(selected.duplicated(["layer", "head"]).any())

    def test_extracts_requested_layer_head_and_flattened_size(self):
        arrays = synthetic_arrays()
        extracted = extract_head_arrays(arrays, layer=1, head=1, expected_shape=(2, 3))
        np.testing.assert_array_equal(extracted["original"], arrays["original"][:, 1, 1, :])
        self.assertEqual(extracted["original"].shape, (2, 3))
        self.assertEqual(extracted["original"].size, 6)

    def test_delta_validation_accepts_match_and_rejects_mismatch(self):
        extracted = extract_head_arrays(
            synthetic_arrays(),
            layer=1,
            head=1,
            expected_shape=(2, 3),
        )
        self.assertEqual(validate_delta(extracted, layer=1, head=1), 0.0)
        extracted["delta"][0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "Raw delta mismatch"):
            validate_delta(extracted, layer=1, head=1)

    def test_nan_and_inf_are_rejected(self):
        for invalid in (np.nan, np.inf):
            arrays = synthetic_arrays()
            arrays["original"][0, 1, 1, 0] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "NaN/inf"):
                    extract_head_arrays(
                        arrays,
                        layer=1,
                        head=1,
                        expected_shape=(2, 3),
                    )

    def test_duplicate_head_metric_rows_are_rejected(self):
        frame = synthetic_metrics()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate layer/head"):
                read_head_metrics(path, "delta_asym_ratio")

    def test_only_selected_metric_is_required(self):
        frame = synthetic_metrics()[["layer", "head", "delta_asym_ratio"]]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            frame.to_csv(path, index=False)
            loaded = read_head_metrics(path, "delta_asym_ratio")
        self.assertEqual(len(loaded), 6)

    def test_nonfinite_selection_metric_is_rejected(self):
        frame = synthetic_metrics()
        frame.loc[0, "delta_asym_ratio"] = np.inf
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "NaN/inf"):
                read_head_metrics(path, "delta_asym_ratio")

    def test_existing_target_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "histogram.png"
            target.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                reject_existing_targets([target], overwrite=False)
            reject_existing_targets([target], overwrite=True)

    def test_summary_checks_mean_delta_identity(self):
        selected_frame = select_signed_heads(
            synthetic_metrics(),
            "delta_asym_ratio",
            top_k=1,
            bottom_k=1,
        )
        selected = next(selected_frame.iloc[[0]].itertuples(index=False))
        arrays = extract_head_arrays(
            synthetic_arrays(),
            layer=0,
            head=0,
            expected_shape=(2, 3),
        )
        summary = build_summary_row(
            selected,
            arrays,
            "delta_asym_ratio",
            prompt_count=2,
            values_per_prompt=3,
            delta_max_abs_error=0.0,
            output_directory="increase_top1_layer0_head0",
        )
        self.assertEqual(summary["total_values"], 6)
        self.assertAlmostEqual(
            summary["jailbreak_mean"] - summary["original_mean"],
            summary["delta_mean"],
        )
        for name in ("q01", "q05", "q25", "q50", "q75", "q95", "q99"):
            self.assertIn(f"delta_{name}", summary)

    def test_manifest_matches_dynamic_raw_prompt_axis(self):
        manifest = pd.DataFrame(
            {
                "raw_prompt_index": [0, np.nan, 1],
                "input_row_index": [0, 1, 2],
                "prompt_id": [0, 1, 2],
                "source_index": [60, 61, 62],
                "original_source_index": [60, 61, 62],
                "jailbreak_source_index": [60, 61, 62],
                "included": [True, False, True],
                "exclusion_reason": ["", "original_shorter_than_tail_tokens", ""],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            manifest.to_csv(raw_dir / "prompt_manifest.csv", index=False)
            metadata = {
                "prompt_manifest_path": "prompt_manifest.csv",
                "input_pair_count": 3,
                "excluded_pair_count": 1,
            }
            counts = validate_prompt_manifest(raw_dir, metadata, prompt_count=2)
        self.assertEqual(counts, (2, 1))

    def test_gcg_plot_label_comes_from_metadata_or_cli(self):
        metadata = {
            "model_name": "Llama",
            "attack_name": "gcg",
            "jailbreak_prompts_path": "data/derived/llama2/gcg/jailbreak.jsonl",
        }
        self.assertEqual(infer_plot_labels(metadata), ("Llama", "GCG"))
        self.assertEqual(
            infer_plot_labels(metadata, "CustomModel", "GCG"),
            ("CustomModel", "GCG"),
        )

    def test_plot_cli_smoke_without_aie_or_paper_columns(self):
        arrays = synthetic_arrays()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            raw_dir = base / "raw"
            output_dir = base / "plots"
            raw_dir.mkdir()
            for condition, values in arrays.items():
                np.save(raw_dir / f"raw_{condition}_values.npy", values)
            pd.DataFrame(
                {
                    "raw_prompt_index": [0, 1, np.nan],
                    "input_row_index": [0, 1, 2],
                    "prompt_id": [0, 1, 2],
                    "source_index": [60, 61, 62],
                    "original_source_index": [60, 61, 62],
                    "jailbreak_source_index": [60, 61, 62],
                    "included": [True, True, False],
                    "exclusion_reason": ["", "", "original_shorter_than_tail_tokens"],
                }
            ).to_csv(raw_dir / "prompt_manifest.csv", index=False)
            metadata = {
                "processed_prompt_count": 2,
                "included_pair_count": 2,
                "excluded_pair_count": 1,
                "input_pair_count": 3,
                "window_size": 3,
                "elements_per_head": 3,
                "num_layers": 3,
                "num_attention_heads": 2,
                "array_shape": [2, 3, 2, 3],
                "array_axes": ["prompt", "layer", "head", "strict_lower_position"],
                "delta_definition": "jailbreak - original at paired prompt/layer/head/position",
                "prompt_manifest_path": "prompt_manifest.csv",
                "model_name": "Llama",
                "attack_name": "GCG",
            }
            (raw_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            metrics_path = base / "metrics.csv"
            synthetic_metrics()[["layer", "head", "delta_asym_ratio"]].to_csv(
                metrics_path, index=False
            )
            argv = [
                "plot",
                "--head_metrics_csv",
                str(metrics_path),
                "--raw_output_dir",
                str(raw_dir),
                "--metric",
                "delta_asym_ratio",
                "--top_k",
                "1",
                "--bottom_k",
                "1",
                "--output_dir",
                str(output_dir),
                "--bins",
                "8",
                "--dpi",
                "50",
            ]
            with patch("sys.argv", argv):
                main()
            pngs = list(output_dir.rglob("*.png"))
            self.assertEqual(len(pngs), 4)
            self.assertTrue(all(path.stat().st_size > 0 for path in pngs))
            summary = pd.read_csv(output_dir / "selection_summary.csv")
            self.assertTrue(summary["included_pair_count"].eq(2).all())
            self.assertTrue(summary["excluded_pair_count"].eq(1).all())


if __name__ == "__main__":
    unittest.main()
