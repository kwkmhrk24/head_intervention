import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from scripts.evaluate_harmbench_autodan_baseline import (
    build_decoding_config,
    configure_model_for_baseline,
    evaluate_records,
    load_harmbench_records,
    parse_behavior_ids,
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


def write_fixture(root, rows, payloads):
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
        (path.parent / "method_config.json").write_text(
            json.dumps({"behavior_id": behavior_id, "resolved": True}),
            encoding="utf-8",
        )
    return test_cases_dir, csv_path


def load_records(test_cases_dir, csv_path, behavior_ids):
    return load_harmbench_records(
        test_cases_dir,
        csv_path,
        behavior_ids,
        "generation-commit",
        "current-commit",
    )


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
            overwrite=False,
        )


if __name__ == "__main__":
    unittest.main()
