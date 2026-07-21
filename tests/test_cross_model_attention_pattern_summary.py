import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.summarize_cross_model_attention_patterns import (
    METRICS,
    build_attack_specific_vs_consensus,
    build_consensus,
    build_cross_model_pattern_comparison,
    build_full_matched_stability,
    build_hypothesis_evaluations,
    build_intervention_candidates,
    build_layer_consensus,
    build_topk_shift_summary,
    build_within_condition_outputs,
    prepare_output_dir,
    quadrant_labels,
    read_head_metrics,
)


def head_frame(layers=2, heads=3, scale=1.0):
    rows = []
    for layer in range(layers):
        for head in range(heads):
            base = (layer * heads + head - 2.5) * scale
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "head_label": f"L{layer}H{head}",
                    "element_value_count": 40,
                    "delta_asym_ratio": base,
                    "element_delta_mean": base * 0.5 + 0.1,
                    "element_std_shift": base * 0.25 - 0.05,
                    "element_positive_rate_shift": base * 0.1,
                    "original_element_std": 1.0,
                    "jailbreak_element_std": 1.0 + base * 0.25 - 0.05,
                    "original_positive_rate": 0.4,
                    "jailbreak_positive_rate": 0.4 + base * 0.1,
                }
            )
    return pd.DataFrame(rows)


class CrossModelAttentionPatternSummaryTests(unittest.TestCase):
    def test_dynamic_condition_relationships_and_quadrants(self):
        conditions = {
            "model_a_attack_x_full": {
                "model": "Model-A",
                "attack": "Attack-X",
                "scope": "full",
                "frame": head_frame(),
            }
        }
        relationships, quadrants = build_within_condition_outputs(conditions)
        self.assertEqual(len(relationships), 4)
        self.assertEqual(quadrants["count"].sum(), 6)
        self.assertTrue(np.isfinite(relationships[["pearson", "spearman", "slope"]]).all().all())

    def test_quadrant_definition_includes_zero(self):
        labels = quadrant_labels([1, -1, -1, 1, 0], [1, 1, -1, -1, 2])
        self.assertEqual(labels.tolist(), ["Q1", "Q2", "Q3", "Q4", "zero"])

    def test_topk_shift_summary_uses_signed_ranking(self):
        frame = head_frame(layers=10, heads=5)
        conditions = {
            "c": {"model": "M", "attack": "A", "scope": "matched", "frame": frame}
        }
        result = build_topk_shift_summary(conditions)
        increase = result[(result.direction == "increase") & (result.k == 3)].iloc[0]
        decrease = result[(result.direction == "decrease") & (result.k == 3)].iloc[0]
        self.assertGreater(increase.element_std_shift_mean, 0)
        self.assertLess(decrease.element_std_shift_mean, 0)

    def test_full_matched_join_and_rank_stability(self):
        full = head_frame()
        matched = full.copy()
        matched["delta_asym_ratio"] += np.linspace(-0.1, 0.1, len(matched))
        summary, topk = build_full_matched_stability(full, matched, "M", "A")
        self.assertEqual(len(summary), 8)
        self.assertEqual(len(topk), 40)
        self.assertTrue(summary.spearman.between(-1, 1).all())

    def test_consensus_filters_sign_and_uses_worst_rank(self):
        auto = head_frame()
        gcg = head_frame(scale=0.8)
        gcg.loc[0, "delta_asym_ratio"] *= -1
        result = build_consensus(auto, gcg, "M", "increase")
        self.assertTrue((result.autodan_delta_asym_ratio > 0).all())
        self.assertTrue((result.gcg_delta_asym_ratio > 0).all())
        self.assertTrue(result.worst_rank.is_monotonic_increasing)
        self.assertIn("consensus_percentile_rank", result)

    def test_layer_normalization_never_joins_head_ids_across_models(self):
        by_model = {
            "A": {"AutoDAN": head_frame(), "GCG": head_frame(scale=0.8)},
            "B": {"AutoDAN": head_frame(), "GCG": head_frame(scale=1.2)},
        }
        result = build_layer_consensus(by_model)
        self.assertEqual(len(result), 4)
        self.assertNotIn("head", result.columns)
        self.assertEqual(sorted(result.normalized_layer_depth.unique()), [0.0, 1.0])

    def test_read_rejects_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            duplicate = pd.concat([head_frame(), head_frame().iloc[[0]]])
            duplicate.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_head_metrics(path, "test", "matched")
            frame = head_frame()
            frame.loc[0, "delta_asym_ratio"] = np.nan
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "NaN/inf"):
                read_head_metrics(path, "test", "matched")

    def test_attack_specific_and_intervention_schema(self):
        auto = head_frame(layers=4, heads=5)
        gcg = head_frame(layers=4, heads=5, scale=0.8)
        consensus = {
            ("M", direction): build_consensus(auto, gcg, "M", direction)
            for direction in ("increase", "decrease")
        }
        specs = {
            "m_a": {"model": "M", "attack": "AutoDAN", "scope": "matched", "frame": auto},
            "m_g": {"model": "M", "attack": "GCG", "scope": "matched", "frame": gcg},
        }
        comparison = build_attack_specific_vs_consensus(specs, consensus)
        self.assertIn("consensus_worst_rank", comparison)
        full = auto.copy()
        full["paper_head"] = False
        full.loc[0, "paper_head"] = True
        full["aie"] = np.linspace(1.0, 0.0, len(full))
        full["aie_rank"] = np.arange(1, len(full) + 1)
        candidates = build_intervention_candidates(specs, consensus, {"M": full})
        required = {
            "model", "candidate_type", "source_attack", "direction", "k",
            "layer", "head", "rank", "selection_metric", "selection_value",
            "attack_consistency", "notes",
        }
        self.assertTrue(required.issubset(candidates.columns))
        self.assertEqual(
            set(candidates[candidates.candidate_type == "existing_aie"].model), {"M"}
        )
        self.assertEqual(
            set(candidates[candidates.candidate_type == "paper_heads"].model), {"M"}
        )

    def test_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "output"
            path.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_output_dir(path, overwrite=False)

    def test_hypothesis_output_has_all_five_judgments(self):
        conditions = {
            f"{model}_{attack}": {
                "model": model,
                "attack": attack,
                "scope": "matched",
                "frame": head_frame(layers=10, heads=5),
            }
            for model in ("Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.2")
            for attack in ("AutoDAN", "GCG")
        }
        relationships, _ = build_within_condition_outputs(conditions)
        topk = build_topk_shift_summary(conditions)
        overlap = pd.DataFrame(
            {
                "metric": ["delta_asym_ratio", "delta_asym_ratio"],
                "direction": ["increase", "decrease"],
                "k": [10, 10],
                "overlap_count": [1, 2],
            }
        )
        layer = build_layer_consensus(
            {
                model: {"AutoDAN": head_frame(), "GCG": head_frame(scale=0.8)}
                for model in ("Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.2")
            }
        )
        result = build_hypothesis_evaluations(
            relationships,
            topk,
            {"Llama-2-7b-chat-hf": overlap, "Mistral-7B-Instruct-v0.2": overlap},
            layer,
        )
        self.assertEqual(result.hypothesis.tolist(), ["H1", "H2", "H3", "H4", "H5"])
        self.assertTrue(result.judgment.isin({"support", "partial support", "not supported", "insufficient evidence"}).all())

        keyed_conditions = {
            f"{slug}_{attack.lower()}_{scope}": {
                "model": model,
                "attack": attack,
                "scope": scope,
                "frame": head_frame(layers=2, heads=3, scale=scale),
            }
            for slug, model in (
                ("llama2", "Llama-2-7b-chat-hf"),
                ("mistral", "Mistral-7B-Instruct-v0.2"),
            )
            for attack, scale in (("AutoDAN", 1.0), ("GCG", 0.8))
            for scope in ("full", "matched")
        }
        consensus = {
            (model, direction): build_consensus(
                keyed_conditions[f"{slug}_autodan_matched"]["frame"],
                keyed_conditions[f"{slug}_gcg_matched"]["frame"],
                model,
                direction,
            )
            for slug, model in (
                ("llama2", "Llama-2-7b-chat-hf"),
                ("mistral", "Mistral-7B-Instruct-v0.2"),
            )
            for direction in ("increase", "decrease")
        }
        comparison = build_cross_model_pattern_comparison(
            keyed_conditions, consensus, layer, result
        )
        self.assertEqual(
            set(comparison.record_type),
            {
                "attack_head_relationship",
                "matched_layer_relationship",
                "attack_consistent_head_prevalence",
                "hypothesis_evaluation",
            },
        )
        prevalence = comparison[
            comparison.record_type == "attack_consistent_head_prevalence"
        ]
        self.assertTrue(prevalence["value"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
