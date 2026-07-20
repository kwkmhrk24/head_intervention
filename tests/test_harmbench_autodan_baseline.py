import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from scripts.evaluate_harmbench_autodan_baseline import (
    build_decoding_config,
    build_summary,
    configure_model_for_baseline,
    evaluate_records,
    load_harmbench_records,
    parse_args,
    parse_behavior_ids,
    parse_harmbench_run_indices,
    run,
    sha256_file,
    sha256_text,
    validate_transformers_version,
)


class FakeBatch:
    def __init__(self):
        self.input_ids = torch.tensor([[1, 2]])
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    name_or_path = "fake-llama-tokenizer"
    chat_template = "fake {{ messages }} template"

    def __init__(self, response="ordinary generated text"):
        self.response = response
        self.messages = []
        self.tokenizer_calls = []

    def apply_chat_template(self, messages, tokenize=False):
        self.messages.append(messages)
        if tokenize:
            raise AssertionError("formatting must request tokenize=False")
        return "FORMATTED_SECRET"

    def __call__(self, prompt, **kwargs):
        self.tokenizer_calls.append((prompt, kwargs))
        return FakeBatch()

    def decode(self, response_ids):
        self.decoded_ids = response_ids.tolist()
        return self.response


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(use_cache=True)
        self.device = "cpu"
        self.generate_calls = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def generate(self, input_ids, **kwargs):
        self.generate_calls.append((input_ids.clone(), kwargs))
        return torch.tensor([[1, 2, 3, 4]])


@contextmanager
def fake_cache_context(model):
    model.cache_context_entries = getattr(model, "cache_context_entries", 0) + 1
    yield {"generation_default_dynamic_cache_disabled": True}


def write_fixture(
    root,
    rows,
    payloads,
    logs_payloads=None,
    method_config_payloads=None,
):
    root = Path(root)
    csv_path = root / "behaviors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["BehaviorID", "Behavior", "RawIndex"]
        )
        writer.writeheader()
        writer.writerows(rows)

    test_cases_dir = root / "test_cases"
    test_cases_dir.mkdir(parents=True, exist_ok=True)
    for behavior_id, payload in payloads.items():
        path = (
            test_cases_dir
            / "test_cases_individual_behaviors"
            / behavior_id
            / "test_cases.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        method_config = (method_config_payloads or {}).get(
            behavior_id,
            {"behavior_id": behavior_id, "resolved": True},
        )
        (path.parent / "method_config.json").write_text(
            json.dumps(method_config),
            encoding="utf-8",
        )
    for behavior_id, payload in (logs_payloads or {}).items():
        path = (
            test_cases_dir
            / "test_cases_individual_behaviors"
            / behavior_id
            / "logs.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return test_cases_dir, csv_path


def load_records(
    test_cases_dir,
    csv_path,
    behavior_ids,
    candidate_mode="final",
    harmbench_run_index=0,
    harmbench_run_indices=None,
):
    return load_harmbench_records(
        test_cases_dir,
        csv_path,
        behavior_ids,
        "generation-commit",
        "current-commit",
        candidate_mode=candidate_mode,
        harmbench_run_index=harmbench_run_index,
        harmbench_run_indices=harmbench_run_indices,
    )


def v2_step(run_index, run_seed, step_index, test_case, loss):
    return {
        "log_schema_version": 2,
        "run_index": run_index,
        "run_seed": run_seed,
        "step_index": step_index,
        "test_case": test_case,
        "loss": loss,
        "candidate_role": "population_best",
    }


def three_v2_runs(prompts=None):
    prompts = prompts or (("A", "B"), ("C", "D"), ("E", "F"))
    seeds = (100, 200, 300)
    return [
        [
            v2_step(run_index, seeds[run_index], step_index, prompt, loss)
            for step_index, (prompt, loss) in enumerate(
                zip(run_prompts, (1.0 + run_index, 0.5 + run_index))
            )
        ]
        for run_index, run_prompts in enumerate(prompts)
    ]


def base_record(behavior_id="b1", raw_index=4, attack_prompt="ATTACK_SECRET"):
    return {
        "behavior_id": behavior_id,
        "raw_index": raw_index,
        "goal": "GOAL_SECRET",
        "attack_prompt": attack_prompt,
        "attack_prompt_sha256": sha256_text(attack_prompt),
        "candidate_index": 0,
        "candidate_count": 1,
        "candidate_selection_rule": "last",
        "attack_source": "HarmBench",
        "attack_method": "AutoDAN",
        "selection_split": "selection",
        "source_test_cases_path": "/source/test_cases.json",
        "source_test_cases_sha256": "test-cases-sha",
        "source_method_config_path": "/source/method_config.json",
        "source_method_config_sha256": "method-config-sha",
        "source_behaviors_csv_path": "/source/behaviors.csv",
        "source_behaviors_csv_sha256": "behaviors-sha",
        "source_harmbench_generation_commit": "generation-commit",
        "source_harmbench_current_commit": "current-commit",
    }


def source_manifest():
    return {
        "source_harmbench_generation_commit": "generation-commit",
        "source_harmbench_current_commit": "current-commit",
        "source_behaviors_csv_path": "/source/behaviors.csv",
        "source_behaviors_csv_sha256": "behaviors-sha",
    }


class HarmBenchRunIndicesCLITests(unittest.TestCase):
    @staticmethod
    def _required_args():
        return [
            "--model", "/model",
            "--harmbench-test-cases-dir", "/test-cases",
            "--behaviors-csv", "/behaviors.csv",
            "--behavior-ids", "b1",
            "--output-jsonl", "/output.jsonl",
            "--summary-json", "/summary.json",
            "--source-harmbench-generation-commit", "generation",
            "--source-harmbench-current-commit", "current",
        ]

    def test_parses_multiple_run_indices_in_given_order(self):
        args = parse_args(
            self._required_args()
            + [
                "--candidate-mode", "trajectory",
                "--harmbench-run-indices", "2,0,1",
            ]
        )
        self.assertEqual(args.harmbench_run_indices, [2, 0, 1])
        self.assertEqual(args.harmbench_run_index, 0)

    def test_existing_single_run_default_is_preserved(self):
        args = parse_args(self._required_args())
        self.assertEqual(args.harmbench_run_index, 0)
        self.assertIsNone(args.harmbench_run_indices)
        self.assertEqual(args.candidate_mode, "final")

    def test_run_indices_parser_rejects_invalid_values(self):
        for value, pattern in (
            ("", "at least one"),
            ("0,,1", "empty run index"),
            ("0,-1", "non-negative"),
            ("0,1,0", "duplicate"),
            ("0,x", "integers"),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(Exception, pattern):
                    parse_harmbench_run_indices(value)

    def test_cli_rejects_explicit_single_and_multiple_run_options(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(
                self._required_args()
                + [
                    "--candidate-mode", "trajectory",
                    "--harmbench-run-index", "0",
                    "--harmbench-run-indices", "0,1,2",
                ]
            )

    def test_cli_rejects_multiple_runs_in_final_mode(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(
                self._required_args()
                + ["--harmbench-run-indices", "0,1,2"]
            )


class HarmBenchInputTests(unittest.TestCase):
    def test_selects_last_candidate_in_csv_order_with_metadata(self):
        rows = [
            {"BehaviorID": "b2", "Behavior": "goal two", "RawIndex": "19"},
            {"BehaviorID": "b1", "Behavior": "goal one", "RawIndex": "4"},
        ]
        payloads = {
            "b1": {"b1": ["old one", "final one"]},
            "b2": {"b2": ["only two"]},
        }
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(directory, rows, payloads)
            records, manifest = load_records(
                test_cases_dir, csv_path, ["b1", "b2"]
            )

            self.assertEqual([row["behavior_id"] for row in records], ["b2", "b1"])
            self.assertEqual([row["raw_index"] for row in records], [19, 4])
            self.assertEqual([row["goal"] for row in records], ["goal two", "goal one"])
            self.assertEqual(records[0]["attack_prompt"], "only two")
            self.assertEqual(records[0]["candidate_index"], 0)
            self.assertEqual(records[0]["candidate_count"], 1)
            self.assertEqual(records[1]["attack_prompt"], "final one")
            self.assertEqual(records[1]["candidate_index"], 1)
            self.assertEqual(records[1]["candidate_count"], 2)
            self.assertEqual(
                records[1]["attack_prompt_sha256"], sha256_text("final one")
            )
            self.assertEqual(
                records[1]["source_test_cases_sha256"],
                sha256_file(records[1]["source_test_cases_path"]),
            )
            self.assertEqual(
                records[1]["source_method_config_sha256"],
                sha256_file(records[1]["source_method_config_path"]),
            )
            method_configs_by_id = {
                item["behavior_id"]: item
                for item in manifest["source_method_config_files"]
            }
            self.assertEqual(
                method_configs_by_id["b1"]["path"],
                records[1]["source_method_config_path"],
            )
            self.assertEqual(
                method_configs_by_id["b1"]["sha256"],
                records[1]["source_method_config_sha256"],
            )
            self.assertEqual(
                manifest["source_behaviors_csv_sha256"], sha256_file(csv_path)
            )
            self.assertNotIn("source_logs_path", records[0])
            self.assertNotIn("source_logs_files", manifest)

    def test_trajectory_deduplicates_in_logs_then_saved_final_order(self):
        rows = [{"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"}]
        logs = {
            "b1": [
                [
                    {"test_case": "candidate A", "loss": 1.0},
                    {"test_case": "candidate B", "loss": 0.8},
                    {"test_case": "candidate B", "loss": 0.7},
                ]
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(
                directory,
                rows,
                {"b1": {"b1": ["candidate C"]}},
                {"b1": logs},
            )
            records, manifest = load_records(
                test_cases_dir,
                csv_path,
                ["b1"],
                candidate_mode="trajectory",
            )

            self.assertEqual(
                [record["attack_prompt"] for record in records],
                ["candidate A", "candidate B", "candidate C"],
            )
            self.assertEqual([record["attempt_index"] for record in records], [0, 1, 2])
            self.assertEqual(records[1]["harmbench_step_index"], 1)
            self.assertEqual(records[1]["logged_loss"], 0.8)
            self.assertEqual(
                [record["candidate_origin"] for record in records],
                ["logs_step", "logs_step", "saved_final"],
            )
            self.assertEqual(
                [record["is_saved_final_candidate"] for record in records],
                [False, False, True],
            )
            self.assertTrue(all(record["trajectory_candidate_count"] == 3 for record in records))
            self.assertTrue(all(record["duplicate_candidates_removed"] == 1 for record in records))
            self.assertIsNone(records[2]["harmbench_step_index"])
            self.assertIsNone(records[2]["logged_loss"])
            self.assertEqual(
                records[0]["source_logs_sha256"],
                sha256_file(records[0]["source_logs_path"]),
            )
            self.assertEqual(
                manifest["source_logs_files"],
                [
                    {
                        "behavior_id": "b1",
                        "path": records[0]["source_logs_path"],
                        "sha256": records[0]["source_logs_sha256"],
                    }
                ],
            )

    def test_rejects_missing_behavior_key(self):
        rows = [{"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"}]
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(
                directory, rows, {"b1": {"different": ["candidate"]}}
            )
            with self.assertRaisesRegex(ValueError, "missing BehaviorID key"):
                load_records(test_cases_dir, csv_path, ["b1"])

    def test_rejects_empty_candidate_list(self):
        rows = [{"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"}]
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(
                directory, rows, {"b1": {"b1": []}}
            )
            with self.assertRaisesRegex(ValueError, "candidate list.*empty"):
                load_records(test_cases_dir, csv_path, ["b1"])

    def test_rejects_non_string_candidate(self):
        rows = [{"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"}]
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(
                directory, rows, {"b1": {"b1": [123]}}
            )
            with self.assertRaisesRegex(ValueError, "must be a string"):
                load_records(test_cases_dir, csv_path, ["b1"])

    def test_rejects_duplicate_csv_behavior_id(self):
        rows = [
            {"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"},
            {"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "5"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(
                directory, rows, {"b1": {"b1": ["candidate"]}}
            )
            with self.assertRaisesRegex(ValueError, "duplicate BehaviorID"):
                load_records(test_cases_dir, csv_path, ["b1"])

    def test_rejects_requested_behavior_absent_from_csv(self):
        rows = [{"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"}]
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(
                directory, rows, {"b1": {"b1": ["candidate"]}}
            )
            with self.assertRaisesRegex(ValueError, "absent from behavior CSV"):
                load_records(test_cases_dir, csv_path, ["missing"])

    def test_rejects_missing_test_cases_json(self):
        rows = [{"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"}]
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(directory, rows, {})
            with self.assertRaisesRegex(FileNotFoundError, "test_cases.json"):
                load_records(test_cases_dir, csv_path, ["b1"])

    def test_rejects_missing_method_config_json(self):
        rows = [{"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"}]
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(
                directory, rows, {"b1": {"b1": ["candidate"]}}
            )
            method_config_path = (
                test_cases_dir
                / "test_cases_individual_behaviors"
                / "b1"
                / "method_config.json"
            )
            method_config_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "method_config.json"):
                load_records(test_cases_dir, csv_path, ["b1"])

    def test_rejects_invalid_method_config_json(self):
        rows = [{"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"}]
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(
                directory, rows, {"b1": {"b1": ["candidate"]}}
            )
            method_config_path = (
                test_cases_dir
                / "test_cases_individual_behaviors"
                / "b1"
                / "method_config.json"
            )
            method_config_path.write_text("not JSON", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_records(test_cases_dir, csv_path, ["b1"])

    def test_rejects_non_object_method_config_json(self):
        rows = [{"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"}]
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(
                directory, rows, {"b1": {"b1": ["candidate"]}}
            )
            method_config_path = (
                test_cases_dir
                / "test_cases_individual_behaviors"
                / "b1"
                / "method_config.json"
            )
            method_config_path.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "top-level value must be an object"):
                load_records(test_cases_dir, csv_path, ["b1"])

    def test_rejects_duplicate_requested_behavior_id(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_behavior_ids("b1,b1")

    def test_trajectory_rejects_missing_logs_json(self):
        rows = [{"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"}]
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(
                directory, rows, {"b1": {"b1": ["final"]}}
            )
            with self.assertRaisesRegex(FileNotFoundError, "logs.json"):
                load_records(
                    test_cases_dir, csv_path, ["b1"], candidate_mode="trajectory"
                )

    def test_trajectory_rejects_missing_behavior_key_in_logs(self):
        self._assert_invalid_logs(
            {"different": [[]]},
            "missing BehaviorID key",
        )

    def test_trajectory_rejects_run_index_out_of_range(self):
        self._assert_invalid_logs(
            {"b1": [[]]},
            "run index 1 is out of range",
            harmbench_run_index=1,
        )

    def test_trajectory_rejects_non_object_step(self):
        self._assert_invalid_logs({"b1": [["not an object"]]}, "must be an object")

    def test_trajectory_rejects_non_string_test_case(self):
        self._assert_invalid_logs(
            {"b1": [[{"test_case": 123, "loss": 1.0}]]},
            "test_case.*must be a string",
        )

    def test_trajectory_rejects_non_numeric_loss(self):
        self._assert_invalid_logs(
            {"b1": [[{"test_case": "candidate", "loss": "high"}]]},
            "loss.*must be numeric",
        )

    def test_trajectory_rejects_invalid_logs_json(self):
        rows = [{"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"}]
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(
                directory,
                rows,
                {"b1": {"b1": ["final"]}},
                {"b1": {"b1": [[]]}},
            )
            logs_path = (
                test_cases_dir
                / "test_cases_individual_behaviors"
                / "b1"
                / "logs.json"
            )
            logs_path.write_text("not JSON", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_records(
                    test_cases_dir, csv_path, ["b1"], candidate_mode="trajectory"
                )

    def test_trajectory_rejects_non_object_logs_root(self):
        self._assert_invalid_logs([], "top-level value must be an object")

    def _assert_invalid_logs(
        self,
        logs_payload,
        error_pattern,
        harmbench_run_index=0,
    ):
        rows = [{"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"}]
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(
                directory,
                rows,
                {"b1": {"b1": ["final"]}},
                {"b1": logs_payload},
            )
            with self.assertRaisesRegex(ValueError, error_pattern):
                load_records(
                    test_cases_dir,
                    csv_path,
                    ["b1"],
                    candidate_mode="trajectory",
                    harmbench_run_index=harmbench_run_index,
                )


class SchemaV2MultirunInputTests(unittest.TestCase):
    @staticmethod
    def _load(
        runs,
        finals=("B", "D", "F"),
        *,
        method_run_count=3,
        run_indices=(0, 1, 2),
    ):
        rows = [{"BehaviorID": "b1", "Behavior": "goal", "RawIndex": "4"}]
        with tempfile.TemporaryDirectory() as directory:
            test_cases_dir, csv_path = write_fixture(
                directory,
                rows,
                {"b1": {"b1": list(finals)}},
                {"b1": {"b1": runs}},
                {
                    "b1": {
                        "num_test_cases_per_behavior": method_run_count,
                        "resolved": True,
                    }
                },
            )
            return load_records(
                test_cases_dir,
                csv_path,
                ["b1"],
                candidate_mode="trajectory",
                harmbench_run_indices=list(run_indices),
            )

    def test_three_runs_follow_run_then_step_order(self):
        records, manifest = self._load(three_v2_runs())

        self.assertEqual(
            [record["attack_prompt"] for record in records],
            ["A", "B", "C", "D", "E", "F"],
        )
        self.assertEqual([record["attempt_index"] for record in records], list(range(6)))
        self.assertEqual(
            [record["harmbench_run_index"] for record in records],
            [0, 0, 1, 1, 2, 2],
        )
        self.assertEqual(
            [record["harmbench_step_index"] for record in records],
            [0, 1, 0, 1, 0, 1],
        )
        self.assertEqual(
            [record["run_seed"] for record in records],
            [100, 100, 200, 200, 300, 300],
        )
        self.assertEqual(
            [record["logged_loss"] for record in records],
            [1.0, 0.5, 2.0, 1.5, 3.0, 2.5],
        )
        self.assertEqual(
            [record["is_saved_final_candidate"] for record in records],
            [False, True, False, True, False, True],
        )
        self.assertTrue(all(record["log_schema_version"] == 2 for record in records))
        self.assertTrue(
            all(record["candidate_role"] == "population_best" for record in records)
        )
        self.assertTrue(all(record["candidate_origin"] == "logs_step" for record in records))
        self.assertEqual(manifest["selected_harmbench_run_indices"], [0, 1, 2])
        self.assertEqual(manifest["run_evaluation_order"], "run_then_step")
        self.assertEqual(manifest["logs_schema_version"], 2)
        self.assertEqual(
            manifest["run_seeds_by_behavior"],
            {"b1": {"0": 100, "1": 200, "2": 300}},
        )
        self.assertEqual(manifest["raw_candidate_count_by_behavior"], {"b1": 6})
        self.assertEqual(
            manifest["candidate_count_after_dedup_by_behavior"], {"b1": 6}
        )

    def test_requested_run_order_is_preserved(self):
        records, manifest = self._load(
            three_v2_runs(), run_indices=(2, 0, 1)
        )
        self.assertEqual(
            [record["attack_prompt"] for record in records],
            ["E", "F", "A", "B", "C", "D"],
        )
        self.assertEqual(
            [record["harmbench_run_index"] for record in records],
            [2, 2, 0, 0, 1, 1],
        )
        self.assertEqual(manifest["selected_harmbench_run_indices"], [2, 0, 1])

    def test_cross_run_dedup_preserves_first_metadata_and_final_appearance(self):
        runs = three_v2_runs((("A", "B"), ("B", "C"), ("A", "D")))
        records, manifest = self._load(runs, finals=("B", "C", "D"))

        self.assertEqual(
            [record["attack_prompt"] for record in records],
            ["A", "B", "C", "D"],
        )
        self.assertEqual([record["attempt_index"] for record in records], [0, 1, 2, 3])
        self.assertEqual(records[0]["harmbench_run_index"], 0)
        self.assertEqual(records[0]["harmbench_step_index"], 0)
        self.assertEqual(records[0]["run_seed"], 100)
        self.assertEqual(records[0]["logged_loss"], 1.0)
        self.assertFalse(records[0]["is_saved_final_candidate"])
        self.assertFalse(records[0]["appears_as_saved_final"])
        self.assertTrue(records[1]["is_saved_final_candidate"])
        self.assertTrue(records[1]["appears_as_saved_final"])
        self.assertEqual(records[0]["raw_candidate_count_before_dedup"], 6)
        self.assertEqual(records[0]["candidate_count_after_dedup"], 4)
        self.assertEqual(records[0]["duplicate_candidates_removed"], 2)
        self.assertEqual(
            records[0]["duplicate_candidates_removed_by_run"],
            {"0": 0, "1": 1, "2": 1},
        )
        self.assertEqual(
            manifest["duplicate_candidates_removed_by_behavior"], {"b1": 2}
        )
        self.assertEqual(
            manifest["duplicate_candidates_removed_by_run_and_behavior"],
            {"b1": {"0": 0, "1": 1, "2": 1}},
        )

    def test_later_saved_final_duplicate_is_recorded_without_replacing_metadata(self):
        runs = three_v2_runs((("A", "B"), ("C", "A"), ("D", "E")))
        records, _ = self._load(runs, finals=("B", "A", "E"))
        first_a = records[0]

        self.assertEqual(first_a["attack_prompt"], "A")
        self.assertEqual(first_a["harmbench_run_index"], 0)
        self.assertEqual(first_a["harmbench_step_index"], 0)
        self.assertFalse(first_a["is_saved_final_candidate"])
        self.assertTrue(first_a["appears_as_saved_final"])
        self.assertEqual(first_a["saved_final_run_indices"], [1])

    def test_schema_v2_single_run_uses_aligned_logs_without_appending_final(self):
        runs = [three_v2_runs()[0]]
        records, manifest = self._load(
            runs,
            finals=("B",),
            method_run_count=1,
            run_indices=(0,),
        )
        self.assertEqual([record["attack_prompt"] for record in records], ["A", "B"])
        self.assertEqual(len(records), 2)
        self.assertTrue(records[-1]["is_saved_final_candidate"])
        self.assertEqual(manifest["logs_schema_version"], 2)

    def test_rejects_invalid_schema_v2_records(self):
        cases = []

        runs = three_v2_runs()
        runs[1][0]["run_index"] = 0
        cases.append(("run index mismatch", runs, ("B", "D", "F"), 3, (0, 1, 2), "run 1, step 0.*run_index"))

        runs = three_v2_runs()
        runs[1][1]["run_seed"] = 201
        cases.append(("seed changes", runs, ("B", "D", "F"), 3, (0, 1, 2), "run 1, step 1.*run_seed changed"))

        runs = three_v2_runs()
        runs[1][1]["step_index"] = 2
        cases.append(("step gap", runs, ("B", "D", "F"), 3, (0, 1, 2), "run 1, step 1.*step_index"))

        runs = three_v2_runs()
        runs[1][0]["log_schema_version"] = 1
        cases.append(("wrong version", runs, ("B", "D", "F"), 3, (0, 1, 2), "run 1, step 0.*schema v2"))

        runs = three_v2_runs()
        runs[1][0]["candidate_role"] = "other"
        cases.append(("wrong role", runs, ("B", "D", "F"), 3, (0, 1, 2), "run 1, step 0.*candidate_role"))

        runs = three_v2_runs()
        runs[1][0]["loss"] = True
        cases.append(("bool loss", runs, ("B", "D", "F"), 3, (0, 1, 2), "run 1, step 0.*loss"))

        runs = three_v2_runs()
        runs[1][0]["run_seed"] = True
        cases.append(("bool seed", runs, ("B", "D", "F"), 3, (0, 1, 2), "run 1, step 0.*run_seed"))

        runs = three_v2_runs()
        cases.append(("final mismatch", runs, ("B", "not-D", "F"), 3, (0, 1, 2), "run 1.*saved final"))

        runs = three_v2_runs()
        cases.append(("final missing", runs, ("B", "D"), 3, (0, 1, 2), "run 2.*saved final is missing"))

        runs = three_v2_runs()
        cases.append(("method count", runs, ("B", "D", "F"), 2, (0, 1, 2), "run 2.*num_test_cases_per_behavior"))

        runs = three_v2_runs()[:2]
        cases.append(("run out of range", runs, ("B", "D"), 3, (0, 1, 2), "run index 2 is out of range.*BehaviorID 'b1'.*run 2"))

        runs = three_v2_runs()
        runs[1] = []
        cases.append(("empty run", runs, ("B", "D", "F"), 3, (0, 1, 2), "BehaviorID 'b1', run 1.*non-empty schema v2"))

        for name, runs, finals, method_count, indices, pattern in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, pattern):
                self._load(
                    runs,
                    finals,
                    method_run_count=method_count,
                    run_indices=indices,
                )


class BaselineGenerationTests(unittest.TestCase):
    def test_generation_uses_fixed_user_only_no_cache_configuration(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        configure_model_for_baseline(model)
        results = evaluate_records(
            [base_record()],
            model,
            tokenizer,
            {"model_name": "Llama"},
            "/model/Llama-2-7b-chat-hf",
            build_decoding_config(32),
            "4.42.3",
            "2026-01-01T00:00:00+00:00",
            source_manifest(),
            cache_context=fake_cache_context,
        )

        self.assertFalse(model.config.use_cache)
        self.assertTrue(model.eval_called)
        self.assertEqual(
            tokenizer.messages,
            [[{"role": "user", "content": "ATTACK_SECRET"}]],
        )
        self.assertEqual(len(model.generate_calls), 1)
        _, tokenizer_kwargs = tokenizer.tokenizer_calls[0]
        self.assertEqual(
            tokenizer_kwargs,
            {"return_tensors": "pt", "add_special_tokens": False},
        )
        input_ids, kwargs = model.generate_calls[0]
        self.assertEqual(input_ids.shape[0], 1)
        self.assertEqual(
            kwargs,
            {
                "max_new_tokens": 32,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": False,
            },
        )
        self.assertEqual(model.cache_context_entries, 1)
        self.assertFalse(results[0]["generation_failed"])

    def test_existing_success_checker_handles_refusal_and_non_refusal(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        records = [base_record("b1", 4), base_record("b2", 5)]
        with patch(
            "scripts.evaluate_harmbench_autodan_baseline.generate_baseline_response",
            side_effect=[("I cannot provide that", 4), ("ordinary answer", 2)],
        ):
            results = evaluate_records(
                records,
                model,
                tokenizer,
                {"model_name": "Llama"},
                "/model",
                build_decoding_config(32),
                "4.42.3",
                "2026-01-01T00:00:00+00:00",
                source_manifest(),
                cache_context=fake_cache_context,
            )

        self.assertFalse(results[0]["keyword_success"])
        self.assertIn(
            "I cannot provide", results[0]["matched_refusal_substrings"]
        )
        self.assertTrue(results[1]["keyword_success"])
        self.assertEqual(results[1]["matched_refusal_substrings"], [])

    def test_generation_failure_is_saved_and_next_record_continues(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        records = [base_record("b1", 4), base_record("b2", 5)]
        with patch(
            "scripts.evaluate_harmbench_autodan_baseline.generate_baseline_response",
            side_effect=[RuntimeError("synthetic failure"), ("ordinary answer", 2)],
        ):
            results = evaluate_records(
                records,
                model,
                tokenizer,
                {"model_name": "Llama"},
                "/model",
                build_decoding_config(32),
                "4.42.3",
                "2026-01-01T00:00:00+00:00",
                source_manifest(),
                cache_context=fake_cache_context,
            )

        self.assertTrue(results[0]["generation_failed"])
        self.assertIsNone(results[0]["keyword_success"])
        self.assertIn("RuntimeError", results[0]["failure_reason"])
        self.assertFalse(results[1]["generation_failed"])


class TrajectorySummaryTests(unittest.TestCase):
    def test_behavior_success_and_first_success_attempt(self):
        records = [
            {
                **base_record("b1", 4, "b1 attempt 0"),
                "attempt_index": 0,
                "duplicate_candidates_removed": 1,
            },
            {
                **base_record("b1", 4, "b1 attempt 1"),
                "attempt_index": 1,
                "duplicate_candidates_removed": 1,
            },
            {
                **base_record("b2", 5, "b2 attempt 0"),
                "attempt_index": 0,
                "duplicate_candidates_removed": 0,
            },
        ]
        results = [
            {**records[0], "generation_failed": False, "keyword_success": False},
            {**records[1], "generation_failed": False, "keyword_success": True},
            {**records[2], "generation_failed": False, "keyword_success": False},
        ]
        manifest = {
            **source_manifest(),
            "candidate_mode": "trajectory",
            "seed": 42,
        }

        summary = build_summary(
            records,
            results,
            "/model",
            build_decoding_config(32),
            "4.42.3",
            "4.42.3",
            manifest,
            "2026-01-01T00:00:00+00:00",
            "/output/results.jsonl",
        )

        self.assertEqual(summary["total_candidate_attempts"], 3)
        self.assertEqual(summary["behavior_success_count"], 1)
        self.assertEqual(summary["behavior_failure_count"], 1)
        self.assertEqual(summary["behavior_asr"], 0.5)
        self.assertEqual(summary["successful_behavior_ids"], ["b1"])
        self.assertEqual(
            summary["first_success_attempt_by_behavior"],
            {"b1": 1, "b2": None},
        )
        self.assertEqual(
            summary["candidate_attempt_count_by_behavior"],
            {"b1": 2, "b2": 1},
        )
        self.assertEqual(
            summary["duplicate_candidates_removed_by_behavior"],
            {"b1": 1, "b2": 0},
        )
        self.assertEqual(summary["candidate_keyword_asr"], 1 / 3)
        self.assertEqual(summary["keyword_asr"], 1 / 3)

    def test_multirun_first_success_metadata_and_aggregate_counts(self):
        def record(
            behavior_id,
            raw_index,
            attempt_index,
            run_index,
            step_index,
            run_seed,
            loss,
            prompt,
            is_final=False,
            duplicates=0,
        ):
            return {
                **base_record(behavior_id, raw_index, prompt),
                "attempt_index": attempt_index,
                "harmbench_run_index": run_index,
                "harmbench_step_index": step_index,
                "run_seed": run_seed,
                "logged_loss": loss,
                "is_saved_final_candidate": is_final,
                "duplicate_candidates_removed": duplicates,
                "duplicate_candidates_removed_by_run": {
                    "0": 0,
                    "1": duplicates,
                    "2": 0,
                },
            }

        records = [
            record("b1", 4, 0, 0, 0, 100, 1.0, "b1-r0-s0"),
            record("b1", 4, 1, 1, 0, 200, 0.9, "b1-r1-s0"),
            record("b1", 4, 2, 1, 1, 200, 0.8, "b1-r1-s1", True),
            record("b1", 4, 3, 2, 0, 300, 0.7, "b1-r2-s0", False, 1),
            record("b2", 5, 0, 0, 0, 400, 1.2, "b2-r0-s0"),
        ]
        success_values = [False, False, True, True, False]
        results = [
            {
                **record_item,
                "generation_failed": False,
                "keyword_success": keyword_success,
            }
            for record_item, keyword_success in zip(records, success_values)
        ]
        manifest = {
            **source_manifest(),
            "candidate_mode": "trajectory",
            "seed": 42,
            "selected_harmbench_run_indices": [0, 1, 2],
            "run_evaluation_order": "run_then_step",
            "logs_schema_version": 2,
            "raw_candidate_count_by_behavior": {"b1": 5, "b2": 1},
            "candidate_count_after_dedup_by_behavior": {"b1": 4, "b2": 1},
            "duplicate_candidates_removed_by_behavior": {"b1": 1, "b2": 0},
            "duplicate_candidates_removed_by_run_and_behavior": {
                "b1": {"0": 0, "1": 1, "2": 0},
                "b2": {"0": 0, "1": 0, "2": 0},
            },
        }

        summary = build_summary(
            records,
            results,
            "/model",
            build_decoding_config(32),
            "4.42.3",
            "4.42.3",
            manifest,
            "2026-01-01T00:00:00+00:00",
            "/output/results.jsonl",
        )

        self.assertEqual(summary["behavior_success_count"], 1)
        self.assertEqual(summary["behavior_failure_count"], 1)
        self.assertEqual(summary["behavior_asr"], 0.5)
        self.assertEqual(summary["successful_behavior_ids"], ["b1"])
        self.assertEqual(
            summary["first_success_attempt_by_behavior"],
            {"b1": 2, "b2": None},
        )
        first = summary["first_success_candidate_by_behavior"]["b1"]
        self.assertEqual(first["attempt_index"], 2)
        self.assertEqual(first["harmbench_run_index"], 1)
        self.assertEqual(first["harmbench_step_index"], 1)
        self.assertEqual(first["run_seed"], 200)
        self.assertEqual(first["logged_loss"], 0.8)
        self.assertTrue(first["is_saved_final_candidate"])
        self.assertEqual(
            first["attack_prompt_sha256"], sha256_text("b1-r1-s1")
        )
        self.assertNotIn("attack_prompt", first)
        self.assertIsNone(summary["first_success_candidate_by_behavior"]["b2"])
        self.assertEqual(summary["total_raw_candidate_attempts"], 6)
        self.assertEqual(summary["total_candidate_attempts"], 5)
        self.assertEqual(summary["total_duplicates_removed"], 1)
        self.assertEqual(
            summary["candidate_attempt_count_by_behavior"], {"b1": 4, "b2": 1}
        )
        self.assertEqual(len(results), 5)
        self.assertTrue(results[3]["keyword_success"])


class VersionAndOutputTests(unittest.TestCase):
    def test_transformers_version_match_passes(self):
        self.assertEqual(validate_transformers_version("4.42.3", "4.42.3"), "4.42.3")

    def test_version_mismatch_stops_before_model_loader(self):
        loader = Mock()
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(directory)
            with self.assertRaisesRegex(RuntimeError, "model was not loaded"):
                run(
                    args,
                    model_loader=loader,
                    actual_transformers_version="9.9.9",
                    seed_setter=lambda seed: None,
                    cache_context=fake_cache_context,
                )
        loader.assert_not_called()

    def test_outputs_have_required_schema_counts_and_no_secret_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(directory)
            test_cases_dir, csv_path = write_fixture(
                directory,
                [
                    {
                        "BehaviorID": "b1",
                        "Behavior": "GOAL_SECRET",
                        "RawIndex": "4",
                    }
                ],
                {"b1": {"b1": ["ATTACK_SECRET"]}},
            )
            args.harmbench_test_cases_dir = str(test_cases_dir)
            args.behaviors_csv = str(csv_path)
            loader = Mock(
                return_value=(
                    FakeModel(),
                    FakeTokenizer(response="RESPONSE_SECRET"),
                    {"model_name": "Llama"},
                )
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                run(
                    args,
                    model_loader=loader,
                    actual_transformers_version="4.42.3",
                    seed_setter=lambda seed: None,
                    cache_context=fake_cache_context,
                )

            output_text = stdout.getvalue()
            self.assertNotIn("ATTACK_SECRET", output_text)
            self.assertNotIn("RESPONSE_SECRET", output_text)
            result = json.loads(Path(args.output_jsonl).read_text().strip())
            summary = json.loads(Path(args.summary_json).read_text())
            required_result_keys = {
                "behavior_id",
                "raw_index",
                "goal",
                "attack_prompt",
                "attack_prompt_sha256",
                "candidate_index",
                "candidate_count",
                "candidate_selection_rule",
                "attack_source",
                "attack_method",
                "selection_split",
                "source_test_cases_path",
                "source_test_cases_sha256",
                "source_method_config_path",
                "source_method_config_sha256",
                "response",
                "response_sha256",
                "keyword_success",
                "contains_refusal_substring",
                "matched_refusal_substrings",
                "success_decision_reason",
                "generation_failed",
                "failure_reason",
                "response_length_chars",
                "response_length_tokens",
                "model_path",
                "decoding_config",
                "transformers_version",
                "chat_template_identifier",
                "chat_template_sha256",
                "run_timestamp_utc",
                "source_manifest",
            }
            self.assertTrue(required_result_keys <= result.keys())
            self.assertEqual(result["response"], "RESPONSE_SECRET")
            self.assertEqual(
                result["response_sha256"],
                hashlib.sha256(b"RESPONSE_SECRET").hexdigest(),
            )
            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["generation_success_count"], 1)
            self.assertEqual(summary["generation_failed_count"], 0)
            self.assertEqual(summary["keyword_success_count"], 1)
            self.assertEqual(summary["keyword_failure_count"], 0)
            self.assertEqual(summary["keyword_asr"], 1.0)
            self.assertEqual(summary["seed"], 42)
            self.assertNotIn("candidate_mode", summary)
            self.assertNotIn("ATTACK_SECRET", json.dumps(summary))
            self.assertNotIn("RESPONSE_SECRET", json.dumps(summary))

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run(
                    args,
                    model_loader=Mock(),
                    actual_transformers_version="4.42.3",
                    seed_setter=lambda seed: None,
                    cache_context=fake_cache_context,
                )

    @staticmethod
    def _args(directory):
        directory = Path(directory)
        return SimpleNamespace(
            model="/model/Llama-2-7b-chat-hf",
            harmbench_test_cases_dir=str(directory / "missing-test-cases"),
            behaviors_csv=str(directory / "missing-behaviors.csv"),
            behavior_ids="b1",
            output_jsonl=str(directory / "output" / "results.jsonl"),
            summary_json=str(directory / "output" / "summary.json"),
            max_new_tokens=32,
            seed=42,
            expected_transformers_version="4.42.3",
            source_harmbench_generation_commit="generation-commit",
            source_harmbench_current_commit="current-commit",
            candidate_mode="final",
            harmbench_run_index=0,
            overwrite=False,
        )


if __name__ == "__main__":
    unittest.main()
