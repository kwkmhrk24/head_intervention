import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_attention_head_metric_relationships import ATTENTION_METRICS
from scripts.analyze_matched_attention_head_metrics import (
    COMPARISON_METRICS,
    aggregate_matched_attention,
    aggregate_raw_head_metrics,
    build_attack_metrics,
    build_autodan_mapping,
    build_full_vs_matched_comparison,
    build_gcg_mapping,
    build_layer_outputs,
    build_manifest_mapping,
    build_joined_metrics,
    build_metadata,
    build_metric_comparisons,
    build_topk_overlap,
    compute_common_mapping,
    extract_head_subset,
    prepare_matched_attention_metrics,
    read_fullset_head_metrics,
    summarize_explicit_success_labels,
    validate_output_dir,
    validate_unique_mapping,
)


def raw_arrays(prompt_count=3, layers=2, heads=3, positions=4):
    original = (
        np.arange(prompt_count * layers * heads * positions, dtype=np.float32)
        .reshape(prompt_count, layers, heads, positions)
        / np.float32(10.0)
        - np.float32(5.0)
    )
    offsets = np.linspace(-0.5, 0.75, positions, dtype=np.float32)
    jailbreak = original + offsets
    return {
        "original": original,
        "jailbreak": jailbreak,
        "delta": jailbreak - original,
    }


def mapping(source_indices, raw_indices=None):
    if raw_indices is None:
        raw_indices = range(len(source_indices))
    return pd.DataFrame(
        {
            "raw_prompt_index": list(raw_indices),
            "input_row_index": list(raw_indices),
            "source_index": list(source_indices),
            "mapping_basis": "synthetic",
            "mapping_verified": True,
        }
    )


def attention_frame(source_indices, jailbreak=False):
    rows = []
    for source_index in source_indices:
        for layer in range(32):
            for head in range(32):
                base = source_index * 1e-4 + layer * 1e-3 + head * 1e-5
                row = {
                    "prompt_id": source_index,
                    "source_index": source_index,
                    "layer": layer,
                    "head": head,
                    "window_size": 16,
                }
                for metric_index, metric in enumerate(ATTENTION_METRICS):
                    value = base + metric_index * 0.01
                    if jailbreak:
                        value += (layer + head + source_index + metric_index) * 1e-5
                    row[metric] = value
                rows.append(row)
    return pd.DataFrame(rows)


class MatchedAttentionHeadMetricTests(unittest.TestCase):
    def test_common_source_intersection_and_expected_sequence(self):
        common = compute_common_mapping(
            mapping([0, 1, 2, 3]),
            mapping([2, 3, 4], raw_indices=[0, 1, 2]),
            expected_count=2,
            expected_indices=[2, 3],
        )
        self.assertEqual(common["source_index"].tolist(), [2, 3])
        self.assertEqual(common["raw_prompt_index_autodan"].tolist(), [2, 3])
        self.assertEqual(common["raw_prompt_index_gcg"].tolist(), [0, 1])

    def test_duplicate_and_missing_mappings_are_rejected(self):
        duplicate = mapping([1, 1])
        with self.assertRaisesRegex(ValueError, "duplicate source_index"):
            validate_unique_mapping(duplicate, "test")
        with self.assertRaisesRegex(ValueError, "Expected common source count"):
            compute_common_mapping(mapping([0, 1]), mapping([1, 2]), 2)
        with self.assertRaisesRegex(ValueError, "expected sequence"):
            compute_common_mapping(
                mapping([0, 1]), mapping([0, 1]), 2, expected_indices=[1, 2]
            )

    def test_autodan_mapping_requires_exact_verified_row_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = [{"prompt": "original zero"}, {"prompt": "original one"}]
            jailbreak = [{"prompt": "attack zero"}, {"prompt": "attack one"}]
            source = [
                {"index": 0, "prompt": "attack zero"},
                {"index": 1, "prompt": "attack one"},
            ]
            labels = [
                {"index": 0, "user prompt": "attack zero", "success": True},
                {"index": 1, "user prompt": "attack one", "success": False},
            ]
            advbench = [
                {"index": 0, "prompt": "original zero"},
                {"index": 1, "prompt": "original one"},
            ]
            paths = {}
            for name, rows in (
                ("original", original),
                ("jailbreak", jailbreak),
                ("source", source),
                ("advbench", advbench),
            ):
                path = root / f"{name}.jsonl"
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                paths[name] = path
            labels_path = root / "labels.json"
            labels_path.write_text(json.dumps(labels), encoding="utf-8")
            result, details = build_autodan_mapping(
                2,
                {
                    "prompt_id_definition": "zero-based JSONL row index",
                    "source_index_definition": "zero-based JSONL row index",
                },
                paths["original"],
                paths["jailbreak"],
                paths["source"],
                labels_path,
                paths["advbench"],
                "validated_row_index",
            )
            self.assertEqual(result["source_index"].tolist(), [0, 1])
            self.assertEqual(details["success_true_count"], 1)
            self.assertNotIn("prompt", result.columns)
            source[1]["index"] = 9
            paths["source"].write_text(
                "".join(json.dumps(row) + "\n" for row in source),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "indices"):
                build_autodan_mapping(
                    2,
                    {
                        "prompt_id_definition": "zero-based JSONL row index",
                        "source_index_definition": "zero-based JSONL row index",
                    },
                    paths["original"],
                    paths["jailbreak"],
                    paths["source"],
                    labels_path,
                    paths["advbench"],
                    "validated_row_index",
                )

    def test_gcg_manifest_builds_explicit_included_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.csv"
            pd.DataFrame(
                {
                    "raw_prompt_index": [0, np.nan, 1],
                    "input_row_index": [0, 1, 2],
                    "source_index": [60, 166, 61],
                    "original_source_index": [60, 166, 61],
                    "jailbreak_source_index": [60, 166, 61],
                    "included": [True, False, True],
                    "exclusion_reason": ["", "short", ""],
                }
            ).to_csv(path, index=False)
            result, details = build_gcg_mapping(path, raw_prompt_count=2)
        self.assertEqual(result["source_index"].tolist(), [60, 61])
        self.assertEqual(result["raw_prompt_index"].tolist(), [0, 1])
        self.assertEqual(details["excluded_source_indices"], [166])

    def test_both_attack_manifests_compute_dynamic_common_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for name, sources in (("a", [49, 50, 94, 95]), ("b", [50, 94, 95, 96])):
                path = root / f"{name}.csv"
                included = [source != 94 for source in sources]
                raw_indices = []
                next_raw = 0
                for keep in included:
                    raw_indices.append(next_raw if keep else np.nan)
                    next_raw += int(keep)
                pd.DataFrame(
                    {
                        "raw_prompt_index": raw_indices,
                        "input_row_index": range(len(sources)),
                        "source_index": sources,
                        "original_source_index": sources,
                        "jailbreak_source_index": sources,
                        "included": included,
                        "exclusion_reason": ["" if keep else "short" for keep in included],
                    }
                ).to_csv(path, index=False)
                paths.append(path)
            first, _ = build_manifest_mapping(paths[0], 3, "AutoDAN")
            second, _ = build_manifest_mapping(paths[1], 3, "GCG")
            common = compute_common_mapping(first, second, expected_count=2)
        self.assertEqual(common.source_index.tolist(), [50, 95])
        self.assertNotIn(94, common.source_index.tolist())

    def test_fullset_reader_and_dynamic_layer_comparison(self):
        frame = pd.DataFrame(
            {
                "layer": [0, 0, 1, 1],
                "head": [0, 1, 0, 1],
                **{
                    metric: [1.0, -2.0, 3.0, 1.0]
                    for metric in COMPARISON_METRICS
                },
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "merged.csv"
            frame.to_csv(path, index=False)
            first = read_fullset_head_metrics(path, "A")
        second = first.copy()
        for metric in COMPARISON_METRICS:
            second[metric] *= 0.5
        summary, correlations = build_layer_outputs({"A": first, "B": second})
        self.assertEqual(len(summary), 4)
        self.assertTrue(summary.head_count.eq(2).all())
        self.assertTrue(np.isfinite(correlations[["pearson", "spearman"]]).all().all())

    def test_explicit_success_counts_do_not_infer_missing_definition(self):
        unknown = summarize_explicit_success_labels(None, {1, 2}, {2})
        self.assertEqual(unknown["status"], "unknown")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.json"
            path.write_text(
                json.dumps(
                    [
                        {"index": 0, "success": None},
                        {"index": 1, "success": True},
                        {"index": 2, "success": False},
                    ]
                ),
                encoding="utf-8",
            )
            result = summarize_explicit_success_labels(path, {1, 2}, {2})
        self.assertEqual(result["full"], {"success": 1, "failure": 1})
        self.assertEqual(result["matched"], {"success": 0, "failure": 1})

    def test_memory_mapped_subset_aggregation_matches_numpy_ddof_zero(self):
        arrays = raw_arrays()
        with tempfile.TemporaryDirectory() as temp_dir:
            mmap_arrays = {}
            for condition, values in arrays.items():
                path = Path(temp_dir) / f"{condition}.npy"
                np.save(path, values)
                mmap_arrays[condition] = np.load(
                    path, mmap_mode="r", allow_pickle=False
                )
            result, max_error = aggregate_raw_head_metrics(mmap_arrays, [0, 2])
        self.assertEqual(max_error, 0.0)
        self.assertEqual(result["element_value_count"].unique().tolist(), [8])
        row = result[(result["layer"] == 1) & (result["head"] == 2)].iloc[0]
        original = arrays["original"][[0, 2], 1, 2, :]
        jailbreak = arrays["jailbreak"][[0, 2], 1, 2, :]
        self.assertAlmostEqual(row.original_element_mean, original.mean(), places=6)
        self.assertAlmostEqual(row.jailbreak_element_mean, jailbreak.mean(), places=6)
        self.assertAlmostEqual(
            row.element_delta_mean, (jailbreak - original).mean(), places=6
        )
        self.assertAlmostEqual(
            row.element_std_shift,
            jailbreak.astype(np.float64).std(ddof=0)
            - original.astype(np.float64).std(ddof=0),
            places=7,
        )
        self.assertAlmostEqual(
            row.element_positive_rate_shift,
            (jailbreak > 0).mean() - (original > 0).mean(),
            places=7,
        )

    def test_nonfinite_and_raw_delta_mismatch_are_rejected(self):
        arrays = raw_arrays()
        arrays["original"][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN/inf"):
            aggregate_raw_head_metrics(arrays, [0])
        arrays = raw_arrays()
        arrays["delta"][0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "delta mismatch"):
            aggregate_raw_head_metrics(arrays, [0])

    def test_matched_attention_mean_and_delta_identity(self):
        sources = [60, 61]
        source_mapping = mapping(sources)
        original = prepare_matched_attention_metrics(
            attention_frame(sources),
            source_mapping,
            set(sources),
            "original",
            "source_index",
        )
        jailbreak = prepare_matched_attention_metrics(
            attention_frame(sources, jailbreak=True),
            source_mapping,
            set(sources),
            "jailbreak",
            "source_index",
        )
        result = aggregate_matched_attention(original, jailbreak)
        self.assertEqual(len(result), 1024)
        identity = (
            result["jailbreak_asym_ratio"]
            - result["original_asym_ratio"]
            - result["delta_asym_ratio"]
        )
        self.assertLess(float(identity.abs().max()), 1e-12)

    def test_prompt_id_mapping_handles_missing_source_index(self):
        sources = [60]
        frame = attention_frame(sources).drop(columns="source_index")
        frame.insert(1, "source_index", np.nan)
        frame["prompt_id"] = 0
        result = prepare_matched_attention_metrics(
            frame,
            mapping(sources),
            set(sources),
            "AutoDAN",
            "prompt_id_mapping",
        )
        self.assertEqual(result["source_index"].unique().tolist(), [60])

    def test_signed_top_bottom_overlap_and_jaccard(self):
        auto = pd.DataFrame(
            {
                "layer": [0, 0, 1, 1],
                "head": [0, 1, 0, 1],
                **{
                    metric: [4.0, 2.0, -1.0, -3.0]
                    for metric in COMPARISON_METRICS
                },
            }
        )
        gcg = auto.copy()
        for metric in COMPARISON_METRICS:
            gcg[metric] = [5.0, -2.0, 1.0, -4.0]
        overlap = build_topk_overlap(auto, gcg, k_values=[1, 2])
        increase_one = overlap[
            (overlap.metric == "delta_asym_ratio")
            & (overlap.direction == "increase")
            & (overlap.k == 1)
        ].iloc[0]
        decrease_one = overlap[
            (overlap.metric == "delta_asym_ratio")
            & (overlap.direction == "decrease")
            & (overlap.k == 1)
        ].iloc[0]
        self.assertEqual(increase_one.overlap_count, 1)
        self.assertEqual(increase_one.jaccard, 1.0)
        self.assertEqual(decrease_one.common_heads, "L1H1")

    def test_histogram_subset_has_only_requested_indices_and_element_count(self):
        arrays = raw_arrays(prompt_count=4, positions=5)
        subset = extract_head_subset(arrays, [1, 3], layer=1, head=2)
        self.assertEqual(subset["original"].shape, (2, 5))
        self.assertEqual(subset["original"].size, 10)
        np.testing.assert_array_equal(
            subset["original"], arrays["original"][[1, 3], 1, 2, :]
        )

    def test_attack_join_and_metric_comparisons_are_finite(self):
        raw, _ = aggregate_raw_head_metrics(raw_arrays(layers=2, heads=2), [0, 1])
        attention = pd.DataFrame(
            {
                "layer": [0, 0, 1, 1],
                "head": [0, 1, 0, 1],
                "original_asym_ratio": [0.1, 0.2, 0.3, 0.4],
                "jailbreak_asym_ratio": [0.15, 0.1, 0.5, 0.35],
                "delta_asym_ratio": [0.05, -0.1, 0.2, -0.05],
            }
        )
        auto = build_attack_metrics(raw, attention)
        for metric_index, metric in enumerate(COMPARISON_METRICS, start=1):
            auto[metric] = (
                np.arange(len(auto), dtype=np.float64) + metric_index
            ) * (1e-3 * metric_index)
        gcg = auto.copy()
        for metric in COMPARISON_METRICS:
            gcg[metric] = gcg[metric] * 0.5 + np.arange(len(gcg)) * 1e-4
        joined = build_joined_metrics(auto, gcg)
        correlations, signs = build_metric_comparisons(joined)
        self.assertEqual(len(correlations), 4)
        self.assertEqual(len(signs), 4)
        self.assertTrue(np.isfinite(correlations[["pearson", "spearman"]]).all().all())

    def test_full_vs_matched_comparison_reports_changes(self):
        correlations = pd.DataFrame(
            [
                {
                    "metric": metric,
                    "pearson": 0.2,
                    "spearman": 0.3,
                    "autodan_mean": 0.1,
                    "gcg_mean": 0.2,
                }
                for metric in COMPARISON_METRICS
            ]
        )
        matched_correlations = correlations.copy()
        matched_correlations["pearson"] += 0.1
        matched_correlations["spearman"] += 0.2
        signs = pd.DataFrame(
            {"metric": COMPARISON_METRICS, "sign_agreement_rate": [0.5] * 4}
        )
        matched_signs = signs.copy()
        matched_signs["sign_agreement_rate"] = 0.6
        topk = pd.DataFrame(
            [
                {
                    "metric": metric,
                    "direction": direction,
                    "k": k,
                    "overlap_count": 1,
                }
                for metric in COMPARISON_METRICS
                for direction in ("increase", "decrease")
                for k in (3, 10)
            ]
        )
        matched_topk = topk.copy()
        matched_topk["overlap_count"] = 2
        layers = pd.DataFrame(
            [
                {"metric": f"{metric}_mean", "pearson": 0.4, "spearman": 0.5}
                for metric in COMPARISON_METRICS
            ]
        )
        matched_layers = layers.copy()
        matched_layers["pearson"] += 0.1
        result = build_full_vs_matched_comparison(
            correlations,
            signs,
            topk,
            layers,
            matched_correlations,
            matched_signs,
            matched_topk,
            matched_layers,
        )
        self.assertEqual(len(result), 4)
        self.assertTrue(np.allclose(result.pearson_change, 0.1))
        self.assertTrue(result.increase_top3_overlap_change.eq(1).all())

    def test_existing_output_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                validate_output_dir(output, overwrite=False)

    def test_metadata_contains_required_provenance_without_prompt_text(self):
        common = pd.DataFrame(
            {
                "source_index": [60, 61],
                "raw_prompt_index_autodan": [60, 61],
                "input_row_index_autodan": [60, 61],
                "raw_prompt_index_gcg": [0, 1],
                "input_row_index_gcg": [0, 1],
            }
        )
        args = argparse.Namespace(
            autodan_raw_dir="auto",
            gcg_raw_dir="gcg",
            autodan_original_metrics="ao.csv",
            autodan_jailbreak_metrics="aj.csv",
            gcg_original_metrics="go.csv",
            gcg_jailbreak_metrics="gj.csv",
            gcg_manifest="manifest.csv",
            autodan_original_prompts="aop.jsonl",
            autodan_jailbreak_prompts="ajp.jsonl",
            autodan_source_data="source.jsonl",
            autodan_labels="labels.json",
            advbench="advbench.jsonl",
            output_dir="output",
            autodan_mapping_mode="validated_row_index",
            top_k=3,
            expected_common_count=2,
            bins=10,
            dpi=50,
            overwrite=False,
            dry_run_mapping=False,
            model_name="Mistral-7B-Instruct-v0.2",
            attack_a_name="AutoDAN",
            attack_b_name="GCG",
            autodan_manifest="auto_manifest.csv",
            autodan_merged_metrics="auto_merged.csv",
            gcg_merged_metrics="gcg_merged.csv",
            gcg_success_labels=None,
        )
        metadata = build_metadata(
            args,
            1.25,
            common,
            {"array_shape": [2, 2, 2, 4], "source_commit": "a"},
            {"array_shape": [2, 2, 2, 4], "source_commit": "g"},
            {"mapping_basis": "validated"},
            {"mapping_basis": "manifest"},
            {"AutoDAN": 0.0, "GCG": 0.0},
        )
        self.assertEqual(metadata["analysis_type"], "matched")
        self.assertEqual(metadata["model"], "Mistral-7B-Instruct-v0.2")
        self.assertEqual(metadata["std_ddof"], 0)
        self.assertIn("success_definition_limitation", metadata)
        self.assertIn("execution_datetime_utc", metadata)


if __name__ == "__main__":
    unittest.main()
