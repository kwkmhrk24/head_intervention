import hashlib
import unittest
from pathlib import Path

import torch

from scripts.run_autodan_position_element_selection_ablation import (
    CONDITIONS,
    EXPECTED_SELECTION_INDICES,
    aggregate_scalar,
    condition_selection,
    logit_diagnostics,
    parse_args,
    pearson,
    source_code_provenance,
    structural_comparison,
    summarize_condition,
)
from utils.gcg_position_element_topk_intervention import (
    PositionElementSelection,
)


class AutoDANPositionElementSelectionAblationTests(unittest.TestCase):
    def test_stage_is_explicit(self):
        with self.assertRaises(SystemExit):
            parse_args([])
        args = parse_args(["--stage", "cpu-validate"])
        self.assertEqual(args.stage, "cpu-validate")

    def test_selection_indices_are_frozen(self):
        self.assertEqual(len(EXPECTED_SELECTION_INDICES), 20)
        self.assertEqual(EXPECTED_SELECTION_INDICES, sorted(EXPECTED_SELECTION_INDICES))
        self.assertNotIn(10, EXPECTED_SELECTION_INDICES)

    def test_four_conditions_factor_elements_and_reference(self):
        ranking = tuple(
            {"rank": index + 1, "flat_index": index}
            for index in range(500)
        )
        masks = {500: torch.ones((1, 1, 500), dtype=torch.bool)}
        gcg_reference = torch.zeros((1, 1, 500), dtype=torch.float32)
        autodan_reference = torch.ones((1, 1, 500), dtype=torch.float32)
        gcg = PositionElementSelection(ranking, masks, gcg_reference, (500,))
        autodan = PositionElementSelection(ranking[::-1], masks, autodan_reference, (500,))
        expected = {
            "A": (gcg.ranking, gcg_reference),
            "B": (autodan.ranking, gcg_reference),
            "C": (gcg.ranking, autodan_reference),
            "D": (autodan.ranking, autodan_reference),
        }
        self.assertEqual(set(CONDITIONS), set(expected))
        for name, (ranking_expected, reference_expected) in expected.items():
            selected = condition_selection(name, gcg, autodan)
            self.assertEqual(selected.ranking, ranking_expected)
            self.assertIs(selected.original_mean_reference, reference_expected)

    def test_logit_metrics_capture_argmax_and_finite_distances(self):
        baseline = torch.tensor([3.0, 2.0, 1.0])
        condition = torch.tensor([2.0, 4.0, 1.0])
        result = logit_diagnostics(baseline, condition)
        self.assertFalse(result["argmax_equal"])
        self.assertEqual(result["baseline_top1_top2_margin"], 1.0)
        self.assertGreater(result["max_absolute_logit_difference"], 0)
        self.assertGreater(result["logit_l2_distance"], 0)
        self.assertGreaterEqual(result["kl_baseline_to_condition"], 0)

    def test_condition_summary_reports_transitions_and_reduction(self):
        baseline = [
            {"success": True},
            {"success": True},
            {"success": False},
            {"success": False},
        ]
        transitions = [
            "newly_defended",
            "still_attack_success",
            "newly_worsened",
            "still_defended",
        ]
        rows = [
            {
                "success": value,
                "transition": transition,
                "response_changed": True,
                "generation_failed": False,
                "logit_diagnostics": {
                    "argmax_equal": False,
                    "max_absolute_logit_difference": 1.0,
                    "logit_l2_distance": 2.0,
                    "kl_baseline_to_condition": 0.1,
                    "baseline_top1_top2_margin": 3.0,
                },
            }
            for value, transition in zip((False, True, True, False), transitions)
        ]
        summary = summarize_condition(rows, baseline)
        self.assertEqual(summary["attack_success_count"], 2)
        self.assertEqual(summary["relative_attack_success_reduction"], 0.0)
        self.assertEqual(summary["newly_defended_count"], 1)
        self.assertEqual(summary["still_attack_success_count"], 1)
        self.assertEqual(summary["newly_worsened_count"], 1)
        self.assertEqual(summary["still_defended_count"], 1)

    def test_cpu_statistics_are_finite(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertEqual(aggregate_scalar([1, 2, 3])["median"], 2.0)

    def test_structural_comparison_labels_tiebroken_rank_correlation(self):
        ranking = tuple(
            {
                "rank": index + 1,
                "flat_index": index,
                "ranking_score": float(122880 - index),
                "mean_delta_sign": 1,
                "layer": index // 3840,
                "head": (index // 120) % 32,
                "position": index % 120,
                "lag": index % 15 + 1,
            }
            for index in range(122880)
        )
        masks = {k: torch.ones(1, dtype=torch.bool) for k in (100, 500, 1000)}
        selection = PositionElementSelection(
            ranking, masks, torch.zeros(1), (100, 500, 1000)
        )
        comparison = structural_comparison(selection, selection)
        self.assertEqual(comparison["all_element_tiebroken_rank_correlation"], 1.0)
        self.assertNotIn("all_element_rank_spearman", comparison)
        self.assertIn("Pearson correlation", comparison["rank_correlation_definition"])
        self.assertIn("layer, head, and position", comparison["rank_correlation_definition"])

    def test_source_code_provenance_hashes_entrypoint_and_dependencies(self):
        provenance = source_code_provenance()
        repo_root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            provenance["evaluator_source_path"],
            "scripts/run_autodan_position_element_selection_ablation.py",
        )
        self.assertEqual(
            provenance["evaluator_source_sha256"],
            hashlib.sha256(
                (repo_root / provenance["evaluator_source_path"]).read_bytes()
            ).hexdigest(),
        )
        self.assertIn(
            "scripts/evaluate_gcg_position_element_topk_intervention.py",
            provenance["dependency_source_sha256"],
        )
        for relative_path, digest in provenance["dependency_source_sha256"].items():
            self.assertEqual(
                digest, hashlib.sha256((repo_root / relative_path).read_bytes()).hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
