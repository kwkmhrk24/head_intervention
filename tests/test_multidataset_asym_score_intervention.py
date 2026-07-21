import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralAttention

from scripts.evaluate_asym_score_intervention import (
    build_condition_plan,
    parse_betas,
    parse_conditions,
)
from utils.asym_score_dataset import (
    apply_evaluation_split,
    load_dataset_records,
    normalize_dataset_records,
)
from utils.asym_score_evaluation import (
    attach_baseline_transitions,
    summarize_prompt_results,
)
from utils.asym_score_headsets import (
    PAPER_HEADS,
    load_consensus_headsets,
    load_headsets,
    validate_model_specific_paper_heads,
)
from utils.asym_score_intervention import (
    patch_attention_scores,
    patch_mistral_attention_scores,
    suppress_antisymmetric_component,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CSV = REPO_ROOT / (
    "data/outputs/attention_score_element_distributions/"
    "cross_model_attention_pattern_summary/intervention_candidate_head_sets.csv"
)
EXPECTED_CONSENSUS = {
    "llama": {
        "consensus_increase": [(4, 27), (5, 0), (0, 26), (27, 24), (10, 25)],
        "consensus_decrease": [(13, 5), (17, 23), (10, 15), (21, 31), (22, 12)],
    },
    "mistral": {
        "consensus_increase": [(12, 21), (10, 18), (7, 30), (19, 19), (6, 16)],
        "consensus_decrease": [(21, 20), (21, 22), (5, 26), (21, 4), (29, 31)],
    },
}


class DatasetAdapterTests(unittest.TestCase):
    def test_four_real_input_schemas_and_counts(self):
        cases = [
            ("llama", "autodan", "data/derived/llama2/autodan/jailbreak_prompts.jsonl", 130, 0),
            ("llama", "gcg", "data/raw/llama2/GCG_llama2.jsonl", 160, 0),
            ("mistral", "autodan", "data/raw/mistral/AutoDAN_mistral.jsonl", 100, 0),
            ("mistral", "gcg", "data/raw/mistral/GCG_mistral.jsonl", 100, 0),
        ]
        for family, attack, relative, count, first_index in cases:
            records, manifest = load_dataset_records(
                REPO_ROOT / relative, model_family=family, attack=attack
            )
            self.assertIsNone(manifest)
            self.assertEqual(len(records), count)
            self.assertEqual(records[0]["source_index"], first_index)
            self.assertEqual(
                set(records[0]),
                {"prompt_id", "source_index", "prompt", "attack", "model", "raw_index"},
            )

    def test_llama_autodan_holdout_preserves_manifest_order_and_ids(self):
        records, manifest = load_dataset_records(
            REPO_ROOT / "data/derived/llama2/autodan/holdout_seed42/evaluation_jailbreak_prompts.jsonl",
            model_family="llama",
            attack="autodan",
            split_manifest=REPO_ROOT / "data/derived/llama2/autodan/holdout_seed42/split_manifest.json",
        )
        self.assertEqual(len(records), 110)
        self.assertEqual([row["source_index"] for row in records], manifest["evaluation_indices"])
        self.assertEqual(records[0]["prompt_id"], records[0]["source_index"])

    def test_generated_ids_are_deterministic(self):
        rows = [{"prompt": "synthetic alpha"}, {"prompt": "synthetic beta"}]
        first = normalize_dataset_records(rows, model_family="mistral", attack="autodan")
        second = normalize_dataset_records(rows, model_family="mistral", attack="autodan")
        self.assertEqual(first, second)
        self.assertEqual(first[1]["prompt_id"], "mistral:autodan:1")

    def test_duplicate_and_missing_prompt_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate source_index"):
            normalize_dataset_records(
                [{"index": 2, "prompt": "a"}, {"index": 2, "prompt": "b"}],
                model_family="llama",
                attack="gcg",
            )
        with self.assertRaisesRegex(ValueError, "non-empty prompt"):
            normalize_dataset_records([{"index": 1}], model_family="llama", attack="gcg")

    def test_invalid_split_is_rejected(self):
        records = normalize_dataset_records(
            [{"prompt": "a"}, {"prompt": "b"}], model_family="llama", attack="autodan"
        )
        manifest = {
            "selection_indices": [0], "evaluation_indices": [0], "total_count": 2,
            "selection_size": 1, "evaluation_size": 1,
        }
        with self.assertRaisesRegex(ValueError, "overlap"):
            apply_evaluation_split(records, manifest)


class HeadLoaderTests(unittest.TestCase):
    def test_model_specific_paper_heads(self):
        self.assertEqual(PAPER_HEADS["llama"], [(3, 0), (7, 16), (9, 9), (13, 4), (3, 5)])
        self.assertEqual(PAPER_HEADS["mistral"], [(16, 14), (31, 14), (15, 14), (30, 3), (16, 31)])
        with self.assertRaisesRegex(ValueError, "do not match mistral"):
            validate_model_specific_paper_heads("mistral", PAPER_HEADS["llama"])

    def test_real_candidate_csv_exact_ranked_heads(self):
        for family in ("llama", "mistral"):
            loaded = load_consensus_headsets(CANDIDATE_CSV, model_family=family)
            for condition, expected in EXPECTED_CONSENSUS[family].items():
                self.assertEqual(loaded[condition]["heads"], expected)
                self.assertEqual(
                    [row["rank"] for row in loaded[condition]["ranked_heads"]],
                    [1, 2, 3, 4, 5],
                )

    def test_csv_not_required_for_paper_and_required_for_consensus(self):
        loaded = load_headsets(model_family="llama", conditions=["paper"], candidate_csv=None)
        self.assertEqual(loaded["paper"]["heads"], PAPER_HEADS["llama"])
        with self.assertRaisesRegex(ValueError, "--candidate-csv"):
            load_headsets(model_family="llama", conditions=["consensus_decrease"], candidate_csv=None)

    def test_duplicate_candidate_head_is_rejected(self):
        with CANDIDATE_CSV.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        chosen = [row for row in rows if row["model"] == "Llama-2-7b-chat-hf" and row["candidate_type"] in {"attack_consistent_increase", "attack_consistent_decrease"} and row["k"] == "5"]
        increase = [row for row in chosen if row["direction"] == "increase"]
        increase[1]["layer"] = increase[0]["layer"]
        increase[1]["head"] = increase[0]["head"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(chosen)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_consensus_headsets(path, model_family="llama")

    def test_condition_plan_is_ordered_and_model_specific(self):
        conditions = parse_conditions("paper,consensus_decrease")
        headsets = load_headsets(model_family="mistral", conditions=conditions, candidate_csv=CANDIDATE_CSV)
        plan = build_condition_plan(conditions, parse_betas("0.5"), headsets)
        self.assertEqual(
            [row["condition"] for row in plan],
            ["no_intervention", "beta_zero_control", "paper_beta_0.5", "consensus_decrease_beta_0.5"],
        )
        self.assertEqual(plan[1]["heads"], PAPER_HEADS["mistral"])


class MistralScoreAdapterTests(unittest.TestCase):
    def make_attention(self):
        config = MistralConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        attention = MistralAttention(config=config, layer_idx=0).eval()
        model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]))
        return model, attention

    def test_beta_zero_matches_original_with_gqa(self):
        torch.manual_seed(42)
        model, attention = self.make_attention()
        hidden = torch.randn(2, 5, 32)
        positions = torch.arange(5).unsqueeze(0).expand(2, -1)
        original = attention(hidden, position_ids=positions, use_cache=False)[0]
        with patch_mistral_attention_scores(model, [(0, 1)], 0.0) as diagnostics:
            patched = attention(hidden, position_ids=positions, use_cache=False)[0]
            summary = diagnostics.summary()
        self.assertTrue(torch.equal(original, patched))
        self.assertEqual(attention.num_key_value_groups, 2)
        self.assertEqual(summary["query_lengths"], [5])
        self.assertEqual(summary["key_lengths"], [5])
        self.assertEqual(summary["max_intervention_identity_error"], 0.0)

    def test_cache_and_invalid_heads_are_rejected_and_forward_restored(self):
        model, attention = self.make_attention()
        original = attention.forward.__func__
        hidden = torch.randn(1, 3, 32)
        positions = torch.arange(3).unsqueeze(0)
        with patch_attention_scores(model, "mistral", [(0, 0)], 0.5):
            with self.assertRaisesRegex(ValueError, "use_cache=False"):
                attention(hidden, position_ids=positions, use_cache=True)
        self.assertIs(attention.forward.__func__, original)
        with self.assertRaisesRegex(ValueError, "head index out of range"):
            with patch_mistral_attention_scores(model, [(0, 4)], 0.5):
                pass

    def test_non_square_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "query_length == key_length"):
            suppress_antisymmetric_component(torch.randn(1, 4, 2, 3), [0], 0.5)


class EvaluationCoreTests(unittest.TestCase):
    def test_transitions_empty_responses_and_failure_denominator(self):
        baseline = [
            {"source_index": 0, "success": True},
            {"source_index": 1, "success": False},
            {"source_index": 2, "success": True},
        ]
        current = [
            {"source_index": 0, "success": False, "response": "refusal", "generation_failed": False, "response_length_tokens": 1},
            {"source_index": 1, "success": True, "response": "", "generation_failed": False, "response_length_tokens": 0},
            {"source_index": 2, "success": None, "response": None, "generation_failed": True, "response_length_tokens": None},
        ]
        rows = attach_baseline_transitions(current, baseline)
        self.assertEqual([row["transition"] for row in rows], ["defended", "regressed", None])
        summary = summarize_prompt_results(rows, condition="synthetic", heads=[(0, 0)], beta=0.5)
        self.assertEqual(summary["evaluated_count"], 2)
        self.assertEqual(summary["asr_denominator"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["asr"], 0.5)
        self.assertEqual(summary["empty_response_count"], 1)
        self.assertEqual(summary["generation_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
