import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import analyze_gcg_dynamic_asym_selected_positions as analysis
from utils.gcg_wasserstein_reference import strict_lower_indices


class PositionMetricTests(unittest.TestCase):
    def test_flat_identity_decode_and_even_geometry(self):
        metrics, coordinates = analysis.position_metrics(
            4, torch.tensor([12, 35]), layer_count=3, head_count=2,
        )
        self.assertEqual(coordinates, [[1, 0, 1, 0], [2, 1, 3, 2]])
        self.assertEqual(metrics["layer_counts"], [0, 1, 1])
        self.assertEqual(metrics["total_candidate_count"], 24)
        regions = list(metrics["regions"].values())
        self.assertEqual([r["candidate_count"] for r in regions], [4, 16, 4])
        self.assertEqual([r["selected_count"] for r in regions], [1, 0, 1])
        self.assertEqual(regions[0]["enrichment"], 3)
        self.assertEqual(regions[0]["selection_rate"], 0.25)

    def test_odd_length_half_boundary_and_selected_self(self):
        lower = strict_lower_indices(5)
        indices = torch.tensor([
            10 + int(torch.nonzero((lower[0] == i) & (lower[1] == j))[0])
            for i, j in [(2, 1), (3, 2), (4, 3)]
        ])
        metrics, _ = analysis.position_metrics(5, indices, layer_count=2, head_count=1)
        regions = list(metrics["regions"].values())
        self.assertEqual([r["candidate_count"] for r in regions], [3, 6, 1])
        self.assertEqual([r["selected_count"] for r in regions], [1, 1, 1])

    def test_bins_use_i_over_n_not_n_minus_one_and_mask_empty_candidates(self):
        metrics, _ = analysis.position_metrics(10, torch.tensor([89]), layer_count=2, head_count=1)
        cell = metrics["position_bins"][98]
        self.assertEqual((cell["query_bin"], cell["key_bin"]), (9, 8))
        self.assertEqual(cell["selected_count"], 1)
        self.assertEqual(cell["enrichment"], 45)
        self.assertEqual(metrics["position_bins"][99]["candidate_count"], 0)
        self.assertIsNone(metrics["position_bins"][99]["selection_rate"])
        self.assertIsNone(metrics["position_bins"][1]["selected_share"])

    def test_bin_and_region_conservation_with_intra_bin_diagonal_candidates(self):
        for n in (2, 3, 4, 5, 11, 37, 60):
            with self.subTest(n=n):
                elements = n * (n - 1) // 2
                indices = torch.arange(elements, 3 * elements)
                metrics, _ = analysis.position_metrics(n, indices, layer_count=3, head_count=1)
                for groups in (metrics["regions"].values(), metrics["position_bins"]):
                    groups = list(groups)
                    self.assertEqual(sum(r["candidate_count"] for r in groups), 2 * elements)
                    self.assertEqual(sum(r["selected_count"] for r in groups), 2 * elements)
                    self.assertAlmostEqual(sum(r["selected_share"] or 0 for r in groups), 1)
                    self.assertTrue(all(r["enrichment"] in (None, 1.0) for r in groups))
                if n == 37:
                    self.assertGreater(metrics["position_bins"][0]["candidate_count"], 0)

    def test_invalid_layer0_duplicate_out_of_range_empty_and_dtype(self):
        for indices in (torch.tensor([0]), torch.tensor([6, 6]), torch.tensor([12]),
                        torch.tensor([], dtype=torch.int64), torch.tensor([6.0])):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                analysis.position_metrics(4, indices, layer_count=2, head_count=1)
        with self.assertRaises(ValueError):
            analysis.position_metrics(1, torch.tensor([6]))

    def test_zero_candidate_null_and_invalid_divisions(self):
        metrics = analysis.count_metrics(0, 0, 100, 10)
        self.assertIsNone(metrics["enrichment"])
        for values in ((0, 1, 100, 10), (1, 0, 0, 10), (1, 0, 10, 0), (-1, 0, 10, 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                analysis.count_metrics(*values)


class PromptEqualTests(unittest.TestCase):
    def test_prompt_equal_is_not_element_pooling(self):
        first, _ = analysis.position_metrics(4, torch.tensor([6]), layer_count=2, head_count=1)
        lower = strict_lower_indices(8)
        indices = torch.nonzero(lower[1] >= 4).flatten() + 28
        second, _ = analysis.position_metrics(8, indices, layer_count=2, head_count=1)
        summary = analysis.aggregate_positions([
            {"dataset_index": 1, **first}, {"dataset_index": 2, **second},
        ])
        ee = summary["regions"][analysis.REGIONS[0]]
        self.assertEqual(ee["prompt_equal"]["selected_share"]["mean"], 0.5)
        self.assertEqual(ee["pooled_reference_only"]["selected_share"], 1 / 7)
        self.assertEqual(ee["prompt_equal"]["selected_share"]["std"], 0.5)
        self.assertEqual(ee["enrichment_gt_1_prompt_count"], 1)
        self.assertEqual(ee["prompt_equal"]["selected_share"]["median"], 0.5)
        self.assertEqual(summary["position_bins"][9]["prompt_equal"]["enrichment"]["valid_prompt_count"], 0)
        json.dumps(summary, allow_nan=False)

    def test_duplicate_prompt_and_nonfinite_metrics_rejected(self):
        metrics, _ = analysis.position_metrics(4, torch.tensor([6]), layer_count=2, head_count=1)
        rows = [{"dataset_index": 1, **metrics}] * 2
        with self.assertRaises(ValueError):
            analysis.aggregate_positions(rows)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                analysis.metric_summary([value])


class AnalysisArtifactTests(unittest.TestCase):
    def test_cli_exposes_no_selection_or_generation_controls(self):
        args = analysis.parse_args([])
        self.assertEqual(args.stage, "cpu-preflight")
        for key in ("alpha", "rate", "replacement_mode", "dataset", "component", "bins"):
            self.assertFalse(hasattr(args, key))

    def test_cpu_preflight_rejects_outputs_and_replay_rejects_gpu_flag(self):
        for argv in (["--output-dir", "new"], ["--sandbox-exempt-gpu"],
                     ["--coordinates-from", "old", "--sandbox-exempt-gpu"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                analysis.main(argv)

    def test_csv_is_immutable_and_carries_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            rows = [{"candidate_count": 0, "enrichment": None, "provenance": '{"git_head":"toy"}'}]
            analysis.write_csv(path, rows)
            with path.open() as handle:
                written = list(csv.DictReader(handle))[0]
            self.assertEqual(written["enrichment"], "")
            self.assertEqual(json.loads(written["provenance"])["git_head"], "toy")
            with self.assertRaises(FileExistsError):
                analysis.write_csv(path, rows)

    def test_heatmap_orientation_and_png_provenance(self):
        from PIL import Image
        metrics, _ = analysis.position_metrics(11, torch.tensor([109]), layer_count=2, head_count=1)
        summary = analysis.aggregate_positions([{ "dataset_index": 1, **metrics}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heatmap.png"
            # Inspect the plotting call, in addition to writing a real PNG.
            from matplotlib.axes import Axes
            original = Axes.imshow
            observed = []

            def capture(ax, values, **kwargs):
                observed.append((values, kwargs))
                return original(ax, values, **kwargs)

            with patch.object(Axes, "imshow", capture):
                analysis.plot_heatmap(path, summary["position_bins"], "enrichment", {"git_head": "toy"})
            with Image.open(path) as image:
                self.assertEqual(json.loads(image.info["Provenance"])["git_head"], "toy")
            values, options = observed[0]
            self.assertEqual(options["origin"], "upper")
            self.assertEqual(options["extent"], (0, 1, 1, 0))
            self.assertTrue(values.mask[0, 9])
            self.assertGreater(values[9, 8], 0)

    def test_attention_collection_never_calls_generate_or_lm_head(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(analysis.collect_positions))
        names = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
        self.assertNotIn("generate", names)
        self.assertNotIn("lm_head", names)
        self.assertNotIn("patch_llama_stage2_dynamic_asymmetry_attention", names)
        prefill_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr == "model"
                         and isinstance(node.func.value, ast.Name) and node.func.value.id == "model"]
        self.assertEqual(len(prefill_calls), 1)

    def test_full_cpu_artifact_write_and_no_overwrite(self):
        from scripts.evaluate_gcg_pure_symmetric_autodan_transfer import verify_artifact_manifest
        from PIL import Image
        rows, coordinates = [], []
        # Synthetic counts sum to the frozen production total; no dataset or model access.
        for position in range(100):
            count = 915 if position < 99 else 946
            indices = torch.arange(32 * 946, 32 * 946 + count)
            metrics, decoded = analysis.position_metrics(44, indices)
            rows.append({"dataset_index": 220 + position, **metrics})
            coordinates.append({"dataset_index": 220 + position, "selected_coordinates": decoded})
        provenance = {"git_head": "toy", "reference_sha256": "toy", "prompt_count": 100}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            summary = analysis.save_analysis(output, rows, coordinates, provenance)
            self.assertEqual(summary["total_selected_count"], 91531)
            verify_artifact_manifest(output)
            self.assertEqual(len((output / "per_prompt_region_metrics.jsonl").read_text().splitlines()), 100)
            self.assertEqual(len((output / "position_bin_10x10.csv").read_text().splitlines()), 101)
            for name in ("selected_share", "enrichment", "selection_rate"):
                with Image.open(output / f"{name}_heatmap.png") as image:
                    self.assertEqual(json.loads(image.info["Provenance"]), provenance)
            for filename in ("summary.json", "run_manifest.json", "artifact_manifest.json"):
                json.loads((output / filename).read_text(),
                           parse_constant=lambda value: self.fail(f"non-standard JSON {value}"))
            with self.assertRaises(FileExistsError):
                analysis.save_analysis(output, rows, coordinates, provenance)

    def test_coordinate_replay_identity_and_validation(self):
        from scripts.diagnose_gcg_dynamic_asym_selection import tensor_sha256
        indices = torch.tensor([192])  # n=4, Layer 1, Head 0, position (1,0).
        layers = [0, 1] + [0] * 30
        rows = [{"dataset_index": index, "prompt_id": str(index), "prompt_length": 4,
                 "selected_indices": indices.tolist(), "selected_indices_sha256": tensor_sha256(indices),
                 "selected_z": {}, "layer_counts": layers} for index in range(220, 320)]
        prior = {index: {"prompt_length": 4, "selected_count": 1, "selected_z": {},
                         "diagnostics": {"planned_layer_counts": layers}} for index in range(220, 320)}
        with patch.object(analysis, "collect_positions", side_effect=AssertionError("GPU must not be called")):
            metrics, decoded = analysis.analyze_coordinates(rows, prior)
        self.assertEqual(len(metrics), 100)
        self.assertEqual(decoded[0]["selected_coordinates"], [[1, 0, 1, 0]])
        with self.assertRaises(ValueError):
            analysis.analyze_coordinates(rows[:-1], prior)
        rows[0]["selected_indices_sha256"] = "invalid"
        with self.assertRaises(ValueError):
            analysis.analyze_coordinates(rows, prior)


if __name__ == "__main__":
    unittest.main()
