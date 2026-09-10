import copy
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.evaluate_gcg_position_element_topk_intervention import (
    format_records,
    parse_args as parse_eval_args,
)
from scripts.generate_harmbench_gcg_independent_test import (
    validate_synthetic_candidates,
)
from utils.artifact_io import read_json
from utils.harmbench_gcg_independent_test import (
    CandidateAlignmentError,
    atomic_write_json_exclusive,
    audit_run_outputs,
    build_initial_prompt_layout,
    example_output_path,
    filter_aligned_candidates,
    load_fast_tokenizer,
    load_or_skip_completed_example,
    load_source_records,
    sample_control_isolated,
    save_completed_example,
    save_failure_attempt,
    validate_formatted_evaluation_identity,
    validate_fixed_config,
    validate_input_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/gcg_independent_test_v1/attack_config.json"
INPUT_MANIFEST = REPO_ROOT / "configs/gcg_independent_test_v1/input_manifest.json"


def completed_payload(record, layout, fingerprint, *, updates=500):
    return {
        "schema_version": 1,
        "status": "complete",
        "dataset_index": record["dataset_index"],
        "behavior_id": record["behavior_id"],
        "record_sha256": record["record_sha256"],
        "behavior": record["behavior"],
        "target": record["target"],
        "example_seed": record["example_seed"],
        **fingerprint,
        "control_token_count": 20,
        "final_suffix_text": layout.suffix_text,
        "final_suffix_token_ids": list(layout.control_ids),
        "test_case": layout.test_case,
        "formatted_prompt": layout.formatted_prompt,
        "non_target_input_ids": list(layout.non_target_input_ids),
        "target_token_ids": list(layout.target_ids),
        "ranges": layout.ranges_payload(),
        "termination_reason": "max_updates_reached",
        "termination_step_zero_based": updates - 1,
        "updates_completed": updates,
        "final_loss": 1.0,
        "all_losses": [1.0] * updates,
        "valid_candidate_counts": [511] * updates,
        "rejected_candidate_counts": [1] * updates,
        "early_stop_check_history": [],
        "effective_candidate_batch_size": 128,
        "temperature_effective": False,
    }


class IndependentManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(CONFIG)
        cls.manifest = read_json(INPUT_MANIFEST)

    def test_fixed_config_and_exact_100_record_mapping(self):
        validate_fixed_config(self.config)
        records = validate_input_manifest(self.manifest, self.config)
        self.assertEqual(len(records), 100)
        self.assertEqual(records[0]["dataset_index"], 220)
        self.assertEqual(records[-1]["dataset_index"], 319)
        self.assertEqual(records[0]["behavior_id"], "advbench_221")
        self.assertEqual(records[-1]["behavior_id"], "advbench_320")
        self.assertEqual([row["example_seed"] for row in records], list(range(262, 362)))
        self.assertEqual(len({row["behavior"] for row in records}), 100)
        self.assertEqual(len({row["target"] for row in records}), 100)

    def test_manifest_tampering_and_config_drift_fail(self):
        bad_manifest = copy.deepcopy(self.manifest)
        bad_manifest["records_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "records hash mismatch"):
            validate_input_manifest(bad_manifest, self.config)
        bad_config = copy.deepcopy(self.config)
        bad_config["optimization"]["search_width"] = 256
        with self.assertRaisesRegex(ValueError, "search_width"):
            validate_fixed_config(bad_config)

    def test_cross_split_target_duplicate_is_retained(self):
        records = load_source_records(self.config)
        self.assertIn(294, [row["dataset_index"] for row in records])
        duplicate = self.manifest["known_cross_split_target_duplicates"]
        self.assertEqual(
            duplicate,
            [
                {
                    "dataset_index": 294,
                    "same_target_as_dataset_index": 32,
                    "action": "retain",
                }
            ],
        )

    def test_evaluator_accepts_independent_group_without_changing_defaults(self):
        defaults = parse_eval_args([])
        self.assertEqual(defaults.dataset_groups, ["selection20", "holdout48"])
        selected = parse_eval_args(
            ["--dataset-groups", "independent100", "--k-values", "500"]
        )
        self.assertEqual(selected.dataset_groups, ["independent100"])
        self.assertEqual(selected.k_values, [500])


class TokenContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(CONFIG)
        cls.records = validate_input_manifest(read_json(INPUT_MANIFEST), cls.config)
        try:
            cls.tokenizer = load_fast_tokenizer(cls.config)
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest(str(exc))
        initial = cls.tokenizer(
            cls.config["optimization"]["initial_control"],
            add_special_tokens=False,
        )["input_ids"]
        cls.layout = build_initial_prompt_layout(
            cls.tokenizer,
            behavior=cls.records[0]["behavior"],
            target=cls.records[0]["target"],
            control_ids=initial,
        )

    def test_initial_layout_has_exact_ranges_and_evaluation_ids(self):
        layout = self.layout
        self.assertEqual(len(layout.control_ids), 20)
        self.assertEqual(layout.cache_range[1], layout.control_range[0])
        self.assertEqual(layout.control_range[1], layout.after_control_range[0])
        self.assertEqual(layout.after_control_range[1], layout.target_range[0])
        actual = self.tokenizer(
            layout.formatted_prompt, add_special_tokens=False
        )["input_ids"]
        self.assertEqual(actual, list(layout.non_target_input_ids))
        self.assertNotIn("<<SYS>>", layout.formatted_prompt)

    def test_top500_evaluator_reconstructs_identical_non_target_ids(self):
        records = format_records(
            [
                {
                    "dataset_index": 220,
                    "prompt": self.layout.test_case,
                    "saved_formatted_prompt": self.layout.formatted_prompt,
                    "saved_non_target_input_ids": list(
                        self.layout.non_target_input_ids
                    ),
                }
            ],
            self.tokenizer,
        )
        validate_formatted_evaluation_identity(records)
        records[0]["saved_non_target_input_ids"][-1] = -1
        with self.assertRaisesRegex(ValueError, "input ID mismatch"):
            validate_formatted_evaluation_identity(records)

    def test_whitespace_symbol_filter_and_zero_valid_failure(self):
        result = validate_synthetic_candidates(self.tokenizer, self.layout)
        self.assertTrue(result["zero_valid_failed_explicitly"])
        self.assertGreaterEqual(result["valid_synthetic_candidate_count"], 2)
        invalid = [list(self.layout.control_ids[:-1])]
        invalid_tensor = torch.tensor(invalid, dtype=torch.long)
        with self.assertRaises(ValueError):
            filter_aligned_candidates(self.tokenizer, self.layout, invalid_tensor)

    def test_isolated_generator_repeats_without_global_rng(self):
        control = torch.tensor(self.layout.control_ids)
        gradient = torch.arange(20 * 512, dtype=torch.float32).reshape(20, 512)
        first = sample_control_isolated(
            control,
            gradient,
            64,
            topk=32,
            not_allowed_tokens=None,
            generator=torch.Generator().manual_seed(262),
        )
        torch.manual_seed(9999)
        second = sample_control_isolated(
            control,
            gradient,
            64,
            topk=32,
            not_allowed_tokens=None,
            generator=torch.Generator().manual_seed(262),
        )
        different = sample_control_isolated(
            control,
            gradient,
            64,
            topk=32,
            not_allowed_tokens=None,
            generator=torch.Generator().manual_seed(263),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))


class ImmutableOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(CONFIG)
        cls.record = validate_input_manifest(
            read_json(INPUT_MANIFEST), cls.config
        )[0]
        try:
            cls.tokenizer = load_fast_tokenizer(cls.config)
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest(str(exc))
        ids = cls.tokenizer(
            cls.config["optimization"]["initial_control"],
            add_special_tokens=False,
        )["input_ids"]
        cls.layout = build_initial_prompt_layout(
            cls.tokenizer,
            behavior=cls.record["behavior"],
            target=cls.record["target"],
            control_ids=ids,
        )
        cls.fingerprint = {
            "config_sha256": "a" * 64,
            "input_manifest_sha256": "b" * 64,
            "adapter_sha256": "c" * 64,
        }

    def test_atomic_save_resume_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            payload = completed_payload(
                self.record, self.layout, self.fingerprint
            )
            path = save_completed_example(
                run_dir,
                payload,
                self.record,
                self.fingerprint,
                tokenizer=self.tokenizer,
            )
            self.assertEqual(path, example_output_path(run_dir, "advbench_221"))
            resumed = load_or_skip_completed_example(
                run_dir,
                self.record,
                self.fingerprint,
                tokenizer=self.tokenizer,
            )
            self.assertEqual(resumed["status"], "complete")
            with self.assertRaises(FileExistsError):
                save_completed_example(
                    run_dir,
                    payload,
                    self.record,
                    self.fingerprint,
                    tokenizer=self.tokenizer,
                )
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_condition_mismatch_stops_and_failure_attempts_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            payload = completed_payload(
                self.record, self.layout, self.fingerprint
            )
            save_completed_example(
                run_dir, payload, self.record, self.fingerprint
            )
            bad = dict(self.fingerprint)
            bad["config_sha256"] = "d" * 64
            with self.assertRaisesRegex(ValueError, "config_sha256"):
                load_or_skip_completed_example(run_dir, self.record, bad)
            first = save_failure_attempt(
                run_dir, self.record, self.fingerprint, RuntimeError("one")
            )
            second = save_failure_attempt(
                run_dir, self.record, self.fingerprint, RuntimeError("two")
            )
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists() and second.exists())
            self.assertEqual(read_json(first)["status"], "processing_failed")
            self.assertEqual(read_json(first)["example_seed"], 262)
            self.assertEqual(read_json(second)["attempt"], 2)

    def test_missing_extra_duplicate_and_filename_mismatch_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            audit = audit_run_outputs(
                run_dir,
                [self.record],
                self.fingerprint,
                require_complete=False,
            )
            self.assertEqual(audit["missing_count"], 1)
            with self.assertRaisesRegex(ValueError, "missing 1"):
                audit_run_outputs(
                    run_dir,
                    [self.record],
                    self.fingerprint,
                    require_complete=True,
                )
            payload = completed_payload(
                self.record, self.layout, self.fingerprint
            )
            wrong = run_dir / "examples" / "wrong_name.json"
            atomic_write_json_exclusive(wrong, payload)
            with self.assertRaisesRegex(ValueError, "filename mismatch"):
                audit_run_outputs(
                    run_dir,
                    [self.record],
                    self.fingerprint,
                    require_complete=False,
                )


if __name__ == "__main__":
    unittest.main()
