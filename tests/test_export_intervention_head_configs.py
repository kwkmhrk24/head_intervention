import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_intervention_head_configs import (
    build_export_payload,
    candidate_set_id,
    export_candidate_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CSV = (
    REPO_ROOT
    / "data/outputs/attention_score_element_distributions"
    / "cross_model_attention_pattern_summary"
    / "intervention_candidate_head_sets.csv"
)
SOURCE_COMMIT = "89d241a1c9cb5f1147b7ae127967ba8b95ed7613"
MODEL = "Llama-2-7b-chat-hf"
CANDIDATE_TYPES = ["attack_consistent_increase", "attack_consistent_decrease"]
SOURCE_ATTACK = "AutoDAN+GCG"
EXPECTED_INCREASE = [(4, 27), (5, 0), (0, 26), (27, 24), (10, 25)]
EXPECTED_DECREASE = [(13, 5), (17, 23), (10, 15), (21, 31), (22, 12)]
CSV_COLUMNS = [
    "model",
    "candidate_type",
    "source_attack",
    "direction",
    "k",
    "layer",
    "head",
    "rank",
    "selection_metric",
    "selection_value",
    "attack_consistency",
]


def build_real_payload():
    return build_export_payload(
        CANDIDATE_CSV,
        model=MODEL,
        candidate_types=CANDIDATE_TYPES,
        source_attack=SOURCE_ATTACK,
        k=5,
        source_commit=SOURCE_COMMIT,
        expected_num_layers=32,
        expected_num_heads=32,
    )


def synthetic_rows():
    rows = []
    for direction, pairs in (
        ("increase", [(0, 0), (1, 1)]),
        ("decrease", [(2, 2), (3, 3)]),
    ):
        for rank, (layer, head) in enumerate(pairs, start=1):
            rows.append(
                {
                    "model": MODEL,
                    "candidate_type": f"attack_consistent_{direction}",
                    "source_attack": SOURCE_ATTACK,
                    "direction": direction,
                    "k": "2",
                    "layer": str(layer),
                    "head": str(head),
                    "rank": str(rank),
                    "selection_metric": "synthetic consensus rank",
                    "selection_value": str(rank / 10),
                    "attack_consistency": "True",
                }
            )
    return rows


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_synthetic_payload(path, *, layers=32, heads=32):
    return build_export_payload(
        path,
        model=MODEL,
        candidate_types=CANDIDATE_TYPES,
        source_attack=SOURCE_ATTACK,
        k=2,
        source_commit=SOURCE_COMMIT,
        expected_num_layers=layers,
        expected_num_heads=heads,
    )


class RealCandidateExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = build_real_payload()

    def test_real_increase_and_decrease_heads_match_expected(self):
        sets = self.payload["candidate_sets"]
        increase = [
            (row["layer"], row["head"])
            for row in sets["candidate_consensus_increase"]["heads"]
        ]
        decrease = [
            (row["layer"], row["head"])
            for row in sets["candidate_consensus_decrease"]["heads"]
        ]
        self.assertEqual(increase, EXPECTED_INCREASE)
        self.assertEqual(decrease, EXPECTED_DECREASE)

    def test_heads_are_rank_ordered_and_selection_values_are_numeric(self):
        for candidate_set in self.payload["candidate_sets"].values():
            self.assertEqual(
                [row["rank"] for row in candidate_set["heads"]],
                [1, 2, 3, 4, 5],
            )
            self.assertTrue(
                all(isinstance(row["selection_value"], float) for row in candidate_set["heads"])
            )

    def test_source_csv_sha256_matches_bytes(self):
        expected = hashlib.sha256(CANDIDATE_CSV.read_bytes()).hexdigest()
        self.assertEqual(self.payload["source_csv_sha256"], expected)
        self.assertEqual(
            expected,
            "93a5db26d36e0df48a36a3658f3124a6c0e0696e73533ba3d55f7868f3708c63",
        )

    def test_candidate_set_id_is_deterministic(self):
        first = candidate_set_id(
            MODEL, "attack_consistent_decrease", SOURCE_ATTACK, 5
        )
        second = candidate_set_id(
            MODEL, "attack_consistent_decrease", SOURCE_ATTACK, 5
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            "llama_2_7b_chat_hf_attack_consistent_decrease_autodan_gcg_k5",
        )


class CandidateValidationTests(unittest.TestCase):
    def test_k_mismatch_fails_and_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.json"
            with self.assertRaisesRegex(ValueError, "expected 4 matching rows"):
                export_candidate_config(
                    output,
                    candidate_csv=CANDIDATE_CSV,
                    model=MODEL,
                    candidate_types=CANDIDATE_TYPES,
                    source_attack=SOURCE_ATTACK,
                    k=4,
                    source_commit=SOURCE_COMMIT,
                    expected_num_layers=32,
                    expected_num_heads=32,
                )
            self.assertFalse(output.exists())

    def test_missing_rank_is_rejected(self):
        rows = synthetic_rows()
        rows[1]["rank"] = "3"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "ranks must be exactly"):
                build_synthetic_payload(path)

    def test_duplicate_head_is_rejected(self):
        rows = synthetic_rows()
        rows[1]["layer"] = rows[0]["layer"]
        rows[1]["head"] = rows[0]["head"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "duplicate layer/head"):
                build_synthetic_payload(path)

    def test_out_of_range_layer_and_head_are_rejected(self):
        for field in ("layer", "head"):
            rows = synthetic_rows()
            rows[0][field] = "32"
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "input.csv"
                write_csv(path, rows)
                with self.assertRaisesRegex(ValueError, f"{field} 32 is outside"):
                    build_synthetic_payload(path)

    def test_false_attack_consistency_is_rejected(self):
        rows = synthetic_rows()
        rows[0]["attack_consistency"] = "False"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "attack_consistency"):
                build_synthetic_payload(path)

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.json"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                export_candidate_config(
                    output,
                    candidate_csv=CANDIDATE_CSV,
                    model=MODEL,
                    candidate_types=CANDIDATE_TYPES,
                    source_attack=SOURCE_ATTACK,
                    k=5,
                    source_commit=SOURCE_COMMIT,
                    expected_num_layers=32,
                    expected_num_heads=32,
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")

    def test_successful_atomic_export_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.json"
            payload = export_candidate_config(
                output,
                candidate_csv=CANDIDATE_CSV,
                model=MODEL,
                candidate_types=CANDIDATE_TYPES,
                source_attack=SOURCE_ATTACK,
                k=5,
                source_commit=SOURCE_COMMIT,
                expected_num_layers=32,
                expected_num_heads=32,
            )
            self.assertEqual(json.loads(output.read_text()), payload)
            self.assertEqual(list(Path(directory).iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
