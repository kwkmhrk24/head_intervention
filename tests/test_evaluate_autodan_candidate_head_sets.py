import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate_autodan_head_sets import (
    EXPECTED_S_HEADS,
    PAPER_HEADS,
    build_metadata,
    load_candidate_head_sets,
    load_head_sets,
    parse_args,
    parse_conditions,
    resolve_head_sets,
)
from scripts.export_intervention_head_configs import build_export_payload


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CSV = (
    REPO_ROOT
    / "data/outputs/attention_score_element_distributions"
    / "cross_model_attention_pattern_summary"
    / "intervention_candidate_head_sets.csv"
)
LEGACY_HEADS = (
    REPO_ROOT
    / "data/outputs/llama2/autodan/holdout_seed42"
    / "head_selection_tail16/selected_heads.json"
)
SOURCE_COMMIT = "89d241a1c9cb5f1147b7ae127967ba8b95ed7613"
EXPECTED_INCREASE = [(4, 27), (5, 0), (0, 26), (27, 24), (10, 25)]
EXPECTED_DECREASE = [(13, 5), (17, 23), (10, 15), (21, 31), (22, 12)]


def candidate_payload():
    return build_export_payload(
        CANDIDATE_CSV,
        model="Llama-2-7b-chat-hf",
        candidate_types=[
            "attack_consistent_increase",
            "attack_consistent_decrease",
        ],
        source_attack="AutoDAN+GCG",
        k=5,
        source_commit=SOURCE_COMMIT,
        expected_num_layers=32,
        expected_num_heads=32,
    )


def write_payload(directory, payload):
    path = Path(directory) / "candidate_heads.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class CandidateLoaderTests(unittest.TestCase):
    def test_legacy_loader_is_unchanged(self):
        head_sets = load_head_sets(LEGACY_HEADS)
        self.assertEqual(set(head_sets), {"paper", "positive", "negative", "absolute"})
        self.assertEqual(head_sets["paper"], PAPER_HEADS)
        self.assertEqual(head_sets["positive"], EXPECTED_S_HEADS["positive"])

    def test_existing_cli_defaults_remain_legacy_only(self):
        args = parse_args(
            [
                "--model", "model",
                "--input-file", "input.jsonl",
                "--split-manifest", "manifest.json",
                "--selected-heads", "legacy.json",
                "--baseline-results", "baseline.jsonl",
                "--source-labels", "source.json",
                "--intervention-activation", "vector.pt",
                "--output-dir", "output",
            ]
        )
        self.assertEqual(args.conditions, "paper,positive,negative,absolute")
        self.assertIsNone(args.candidate_heads)

    def test_candidate_conditions_are_parsed(self):
        self.assertEqual(
            parse_conditions("paper,candidate_consensus_decrease"),
            ["paper", "candidate_consensus_decrease"],
        )

    def test_loads_candidate_increase_and_decrease(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_payload(directory, candidate_payload())
            head_sets, metadata = load_candidate_head_sets(path)

        self.assertEqual(head_sets["candidate_consensus_increase"], EXPECTED_INCREASE)
        self.assertEqual(head_sets["candidate_consensus_decrease"], EXPECTED_DECREASE)
        self.assertEqual(metadata["candidate_consensus_increase"]["direction"], "increase")
        self.assertEqual(metadata["candidate_consensus_decrease"]["direction"], "decrease")

    def test_candidate_condition_requires_candidate_file(self):
        with self.assertRaisesRegex(ValueError, "--candidate-heads is required"):
            resolve_head_sets(
                LEGACY_HEADS,
                None,
                ["paper", "candidate_consensus_decrease"],
            )

    def test_legacy_and_candidate_conditions_resolve_together(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_payload(directory, candidate_payload())
            head_sets, metadata = resolve_head_sets(
                LEGACY_HEADS,
                path,
                ["paper", "candidate_consensus_decrease"],
            )

        self.assertEqual(head_sets["paper"], PAPER_HEADS)
        self.assertEqual(head_sets["candidate_consensus_decrease"], EXPECTED_DECREASE)
        self.assertIn("candidate_consensus_increase", head_sets)
        self.assertIn("candidate_consensus_decrease", metadata)

    def test_direction_condition_mismatch_is_rejected(self):
        payload = candidate_payload()
        payload["candidate_sets"]["candidate_consensus_increase"]["direction"] = "decrease"
        with tempfile.TemporaryDirectory() as directory:
            path = write_payload(directory, payload)
            with self.assertRaisesRegex(ValueError, "direction must be 'increase'"):
                load_candidate_head_sets(path)

    def test_duplicate_candidate_head_is_rejected(self):
        payload = candidate_payload()
        rows = payload["candidate_sets"]["candidate_consensus_decrease"]["heads"]
        rows[1]["layer"] = rows[0]["layer"]
        rows[1]["head"] = rows[0]["head"]
        with tempfile.TemporaryDirectory() as directory:
            path = write_payload(directory, payload)
            with self.assertRaisesRegex(ValueError, "duplicate head"):
                load_candidate_head_sets(path)

    def test_candidate_k_mismatch_is_rejected(self):
        payload = candidate_payload()
        payload["candidate_sets"]["candidate_consensus_decrease"]["k"] = 4
        with tempfile.TemporaryDirectory() as directory:
            path = write_payload(directory, payload)
            with self.assertRaisesRegex(ValueError, "k must be integer 5"):
                load_candidate_head_sets(path)


class CandidateMetadataTests(unittest.TestCase):
    @staticmethod
    def metadata_inputs():
        args = SimpleNamespace(
            model="model",
            input_file="input.jsonl",
            split_manifest="manifest.json",
            baseline_results="baseline.jsonl",
            source_labels="source.json",
            intervention_activation="vector.pt",
            interpolation_rate=0.75,
            selected_heads=str(LEGACY_HEADS),
        )
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="llama", _commit_hash=None),
            generation_config=SimpleNamespace(
                num_beams=1,
                max_length=20,
                eos_token_id=2,
                pad_token_id=0,
                bos_token_id=1,
            ),
        )
        tokenizer = SimpleNamespace(chat_template="template")
        model_config = {
            "model_name": "Llama",
            "name_or_path": "model",
            "n_layers": 32,
            "n_heads": 32,
            "resid_dim": 4096,
        }
        manifest = {"selection_size": 20, "evaluation_size": 110, "seed": 42}
        return args, model, tokenizer, model_config, manifest

    def test_candidate_metadata_contains_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_payload(directory, candidate_payload())
            head_sets, candidate_metadata = load_candidate_head_sets(path)
            args, model, tokenizer, model_config, manifest = self.metadata_inputs()
            metadata = build_metadata(
                args=args,
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                manifest=manifest,
                condition="candidate_consensus_decrease",
                heads=head_sets["candidate_consensus_decrease"],
                evaluated_prompt_count=1,
                command="synthetic",
                git_metadata={"git_commit": SOURCE_COMMIT},
                candidate_metadata=candidate_metadata[
                    "candidate_consensus_decrease"
                ],
            )

        self.assertEqual(
            metadata["selection_family"],
            "delta_asym_ratio_cross_attack_consensus",
        )
        self.assertEqual(
            metadata["selection_method"],
            "attack_consistent_consensus_decrease_k5",
        )
        self.assertEqual(metadata["direction"], "decrease")
        self.assertEqual(metadata["k"], 5)
        self.assertTrue(metadata["attack_consistency"])
        self.assertEqual(metadata["source_commit"], SOURCE_COMMIT)
        self.assertEqual(metadata["heads"], [{"layer": l, "head": h} for l, h in EXPECTED_DECREASE])

    def test_legacy_metadata_remains_without_candidate_fields(self):
        args, model, tokenizer, model_config, manifest = self.metadata_inputs()
        metadata = build_metadata(
            args=args,
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            manifest=manifest,
            condition="paper",
            heads=PAPER_HEADS,
            evaluated_prompt_count=1,
            command="synthetic",
            git_metadata={"git_commit": SOURCE_COMMIT},
        )
        self.assertEqual(
            metadata["selection_method"],
            "fixed Llama paper heads from evaluate.py",
        )
        for field in (
            "selection_family",
            "candidate_type",
            "source_attack",
            "candidate_head_file",
        ):
            self.assertNotIn(field, metadata)


if __name__ == "__main__":
    unittest.main()
