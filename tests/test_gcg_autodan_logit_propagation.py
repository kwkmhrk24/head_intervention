import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scripts.diagnose_gcg_autodan_logit_propagation import (
    METRICS,
    RELATIONS,
    build_summary_payloads,
    classify_direction,
    decision_metrics,
    distribution,
    load_fixed_existing_diagnostic,
    parse_args,
    selected_score_metrics,
    summarize_existing,
)


class LogitPropagationDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def _synthetic_rows():
        rows = []
        for dataset, count in (("GCG", 6), ("AutoDAN", 6)):
            for index in range(count):
                gcg = dataset == "GCG"
                score_l2 = (10.0 if gcg else 20.0) + index / 10
                logit_l2 = (20.0 if gcg else 4.0) + index / 10
                row = {
                    "dataset": dataset,
                    "dataset_index": (
                        220 + index if gcg else 10 if index == 0 else index
                    ),
                    "baseline_attack_success": index < 3,
                    "intervention_attack_success": index < (2 if gcg else 3),
                    "argmax_changed": gcg and index < 2,
                    "baseline_margin": (1.0 if gcg else 5.0) + index / 100,
                    "max_absolute_logit_difference": (
                        2.0 if gcg else 0.4
                    ) + index / 100,
                    "logit_l2": logit_l2,
                    "kl_baseline_to_intervention": (
                        0.2 if gcg else 0.01
                    ) + index / 1000,
                    "baseline_top1_delta_logit": -index / 100,
                    "intervention_residual_margin": (
                        0.5 if gcg else 5.0
                    ) + index / 100,
                    "boundary_progress": (
                        0.5 if gcg else 0.0
                    ) + index / 1000,
                    "selected_score_change_mean_abs": score_l2 / 500**0.5,
                    "selected_score_change_rms": score_l2 / 500**0.5,
                    "selected_score_change_max_abs": score_l2 / 4,
                    "selected_score_change_l2": score_l2,
                    "selected_reference_before_mean_abs": score_l2 / 500**0.5,
                    "logit_l2_per_selected_score_change_l2": logit_l2 / score_l2,
                    "max_logit_change_per_selected_score_change_l2": (
                        (2.0 if gcg else 0.4) + index / 100
                    )
                    / score_l2,
                }
                self_missing = set(METRICS) - set(row)
                if self_missing:
                    raise AssertionError(f"missing synthetic metrics: {self_missing}")
                rows.append(row)
        return rows

    def test_summarize_existing_is_cpu_only_and_reproduces_hypotheses(self):
        rows = self._synthetic_rows()
        gcg, autodan, comparison = build_summary_payloads(rows)
        source = {
            "source_dir": "/fixed/source",
            "artifact_manifest_sha256": "a" * 64,
            "per_prompt_diagnostics_sha256": "b" * 64,
            "run_manifest": {
                "model": "/model",
                "model_revision": "revision",
                "gcg_prompt_format": "saved IDs",
                "autodan_prompt_format": "official_fastchat_34062e9",
                "metric_definitions": {"bootstrap": "fixed"},
            },
            "gcg_summary": gcg,
            "autodan_summary": autodan,
            "comparison_summary": comparison,
        }
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                source_diagnostic_dir="/fixed/source",
                output_dir=str(Path(temporary) / "summary"),
            )
            with mock.patch(
                "scripts.diagnose_gcg_autodan_logit_propagation."
                "load_fixed_existing_diagnostic",
                return_value=(rows, source),
            ), mock.patch.object(
                torch.cuda,
                "is_available",
                side_effect=AssertionError("summarization must not query CUDA"),
            ):
                result = summarize_existing(args)
            self.assertFalse(result["gpu_used"])
            self.assertEqual(
                result["hypotheses"],
                {
                    "H1_larger_AutoDAN_baseline_margin": "supported",
                    "H2_smaller_AutoDAN_selected_attention_change": "not supported",
                    "H3_weaker_AutoDAN_attention_to_logit_propagation": "supported",
                    "H4_sufficient_logit_change_but_weaker_boundary_progress": "not supported",
                },
            )
            output = Path(result["output_dir"])
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "gcg_summary.json",
                    "autodan_summary.json",
                    "comparison_summary.json",
                    "run_manifest.json",
                    "artifact_manifest.json",
                },
            )
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertTrue(manifest["cpu_only_resummarization"])
            self.assertFalse(manifest["gpu_forward_performed"])

    def test_summarize_existing_rejects_wrong_fixed_manifest_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "artifact_manifest.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                load_fixed_existing_diagnostic(source)

    def test_summarize_existing_stage_requires_output(self):
        with self.assertRaises(SystemExit):
            parse_args(["--stage", "summarize-existing"])
        args = parse_args(
            [
                "--stage",
                "summarize-existing",
                "--output-dir",
                "/tmp/summary",
            ]
        )
        self.assertEqual(args.stage, "summarize-existing")

    def test_all_requested_metric_relations_are_frozen(self):
        self.assertEqual(
            RELATIONS,
            (
                ("baseline_margin", "max_absolute_logit_difference"),
                ("baseline_margin", "logit_l2"),
                ("baseline_margin", "boundary_progress"),
                ("selected_score_change_l2", "logit_l2"),
                ("selected_score_change_l2", "max_absolute_logit_difference"),
            ),
        )

    def test_boundary_progress_uses_strongest_post_intervention_competitor(self):
        baseline = torch.tensor([3.0, 2.0, 0.0])
        intervention = torch.tensor([2.5, 2.6, 0.0])
        metrics = decision_metrics(baseline, intervention)
        self.assertEqual(metrics["baseline_top1_token_id"], 0)
        self.assertEqual(metrics["baseline_top2_token_id"], 1)
        self.assertEqual(metrics["intervention_top1_token_id"], 1)
        self.assertTrue(metrics["argmax_changed"])
        self.assertAlmostEqual(metrics["baseline_margin"], 1.0)
        self.assertAlmostEqual(metrics["intervention_residual_margin"], -0.1, places=6)
        self.assertAlmostEqual(metrics["boundary_progress"], 1.1, places=6)
        self.assertAlmostEqual(metrics["baseline_top1_delta_logit"], -0.5)

    def test_selected_score_metrics_use_realized_after_and_reject_zero_denominator(self):
        before = [0.0] * 500
        reference = [2.0] * 500
        after = [1.0] * 500
        metrics = selected_score_metrics(before, reference, after, 10.0, 2.0)
        self.assertEqual(metrics["selected_score_change_mean_abs"], 1.0)
        self.assertEqual(metrics["selected_score_change_rms"], 1.0)
        self.assertAlmostEqual(metrics["selected_score_change_l2"], 500**0.5)
        self.assertEqual(metrics["selected_reference_before_mean_abs"], 2.0)
        self.assertAlmostEqual(
            metrics["logit_l2_per_selected_score_change_l2"], 10 / 500**0.5
        )
        with self.assertRaisesRegex(ValueError, "zero"):
            selected_score_metrics(before, reference, before, 10.0, 2.0)

    def test_summary_and_hypothesis_direction_are_deterministic(self):
        first = distribution([1.0, 2.0, 3.0, 4.0])
        second = distribution([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(first, second)
        self.assertEqual(first["median"], 2.5)
        self.assertEqual(first["q25"], 1.75)
        self.assertEqual(first["q75"], 3.25)
        self.assertEqual(
            classify_direction({"bootstrap_95_ci": [1.0, 2.0]}, "higher"),
            "supported",
        )
        self.assertEqual(
            classify_direction({"bootstrap_95_ci": [-2.0, -1.0]}, "higher"),
            "not supported",
        )
        self.assertEqual(
            classify_direction({"bootstrap_95_ci": [-1.0, 1.0]}, "higher"),
            "inconclusive",
        )


if __name__ == "__main__":
    unittest.main()
