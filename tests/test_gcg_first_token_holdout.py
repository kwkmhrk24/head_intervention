import copy
import tempfile
import unittest
from pathlib import Path

from utils.gcg_first_token_holdout import (
    EVALUATION_ROLE,
    HOLDOUT_COUNT,
    HOLDOUT_INDICES,
    build_condition_summary,
    build_holdout_manifest,
    semantic_audit_summary,
    validate_alpha_zero_gate,
    validate_first_token_condition,
    validate_holdout_manifest,
    validate_semantic_audit,
    write_holdout_manifest_artifacts,
)
from utils.gcg_wasserstein_reference import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ARCHIVE = REPO_ROOT / "data/raw/llama2/llama2_GCG.json"
ADVBENCH = REPO_ROOT / "data/raw/advbench.jsonl"


def make_condition_rows(*, alpha=0.5, baseline_success_count=20):
    rows = []
    for dataset_index in HOLDOUT_INDICES:
        baseline_success = len(rows) < baseline_success_count
        rows.append(
            {
                "dataset_index": dataset_index,
                "alpha": alpha,
                "baseline_success": baseline_success,
                "baseline_generation_failed": False,
                "baseline_response": "same",
                "success": baseline_success,
                "generation_failed": False,
                "response": "same",
                "empty_response": False,
                "transition": (
                    "still_attack_success" if baseline_success else "still_defended"
                ),
                "response_equal": True,
                "baseline_token_count": 2,
                "intervention_token_count": 2,
                "first_different_token_position": None,
                "token_edit_distance": 0,
                "first_token_argmax_equal": True,
                "intervention_call_count": 5 if alpha else 0,
            }
        )
    return rows


def make_diagnostics(prompt_count=HOLDOUT_COUNT):
    return {
        "total_intervention_count": prompt_count * 5,
        "later_forward_intervention_count": 0,
        "all_scores_finite": True,
        "per_head_intervention_count": [
            {"layer": layer, "head": head, "count": prompt_count}
            for layer, head in ((31, 22), (13, 4), (5, 30), (9, 18), (27, 24))
        ],
    }


class HoldoutManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_holdout_manifest(
            raw_archive_path=RAW_ARCHIVE,
            advbench_path=ADVBENCH,
            created_at="fixed",
        )

    def test_fixed_indices_and_counts(self):
        included = [row for row in self.manifest["records"] if row["included"]]
        self.assertEqual(len(included), 48)
        self.assertEqual(tuple(row["dataset_index"] for row in included), HOLDOUT_INDICES)
        self.assertNotIn(10, HOLDOUT_INDICES)
        self.assertNotIn(13, HOLDOUT_INDICES)

    def test_duplicate_ambiguity_is_preserved(self):
        excluded = [row for row in self.manifest["records"] if not row["included"]]
        self.assertEqual(len(excluded), 4)
        self.assertEqual([row["dataset_index"] for row in excluded], [10, 10, 13, 13])
        self.assertTrue(all("ambiguous_duplicate" in row["exclusion_reason"] for row in excluded))

    def test_independence_metadata_is_explicit(self):
        self.assertEqual(self.manifest["evaluation_role"], EVALUATION_ROLE)
        self.assertFalse(self.manifest["fully_unseen_across_all_prior_research"])
        self.assertTrue(self.manifest["unused_for_current_gcg_head_selection"])
        self.assertTrue(self.manifest["original_behaviors_previously_used_in_autodan_research"])

    def test_manifest_validates_to_unique_runtime_records(self):
        runtime = validate_holdout_manifest(
            self.manifest,
            raw_archive_path=RAW_ARCHIVE,
            advbench_path=ADVBENCH,
        )
        self.assertEqual(len(runtime), HOLDOUT_COUNT)
        self.assertEqual(len({row["dataset_index"] for row in runtime}), HOLDOUT_COUNT)
        self.assertEqual(runtime[0]["raw_file_position"], 10)

    def test_record_hash_tampering_is_rejected(self):
        tampered = copy.deepcopy(self.manifest)
        tampered["records"][0]["GCG_prompt"] += "tamper"
        with self.assertRaisesRegex(ValueError, "records hash"):
            validate_holdout_manifest(
                tampered,
                raw_archive_path=RAW_ARCHIVE,
                advbench_path=ADVBENCH,
            )

    def test_written_manifest_hash_is_stable_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_holdout_manifest_artifacts(self.manifest, output)
            first = sha256_file(output / "holdout_manifest.json")
            second = sha256_file(output / "holdout_manifest.json")
            self.assertEqual(first, second)
            with self.assertRaises(FileExistsError):
                write_holdout_manifest_artifacts(self.manifest, output)


class TransitionTests(unittest.TestCase):
    def test_four_way_transition_and_baseline_subset(self):
        rows = make_condition_rows(baseline_success_count=20)
        rows[0].update(success=False, transition="newly_defended")
        rows[20].update(success=True, transition="newly_worsened")
        summary = build_condition_summary(rows, alpha=0.5, diagnostics=make_diagnostics())
        self.assertEqual(summary["newly_defended"], 1)
        self.assertEqual(summary["still_attack_success"], 19)
        self.assertEqual(summary["newly_worsened"], 1)
        self.assertEqual(summary["still_defended"], 27)
        self.assertEqual(summary["attack_success_count"], 20)
        self.assertEqual(summary["asr"], 20 / 48)
        self.assertEqual(summary["net_defended"], 0)
        self.assertEqual(summary["baseline_success_subset"]["newly_defended_rate"], 1 / 20)

    def test_transition_total_requires_48(self):
        with self.assertRaisesRegex(ValueError, "expected 48"):
            build_condition_summary(
                make_condition_rows()[:-1], alpha=0.5, diagnostics=make_diagnostics(47)
            )


class EquivalenceAndTimingTests(unittest.TestCase):
    def test_alpha_zero_exact_gate(self):
        rows = make_condition_rows(alpha=0.0)
        for row in rows:
            row.pop("intervention_call_count")
            row["diagnostics"] = {"total_intervention_count": 0}
        comparison = validate_alpha_zero_gate(rows)
        self.assertTrue(comparison["passed"])
        rows[0]["response"] = "different"
        with self.assertRaisesRegex(RuntimeError, "refusing positive-alpha"):
            validate_alpha_zero_gate(rows)

    def test_alpha_zero_nonzero_intervention_is_rejected(self):
        rows = make_condition_rows(alpha=0.0)
        rows[0]["intervention_call_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "nonzero intervention"):
            validate_alpha_zero_gate(rows)

    def test_first_token_counts_and_later_forward(self):
        rows = make_condition_rows()
        diagnostics = make_diagnostics()
        validate_first_token_condition(rows, diagnostics, expected_count=48)
        diagnostics["later_forward_intervention_count"] = 1
        with self.assertRaisesRegex(ValueError, "later forward"):
            validate_first_token_condition(rows, diagnostics, expected_count=48)

    def test_response_change_must_begin_at_token_zero(self):
        rows = make_condition_rows()
        rows[0].update(response_equal=False, first_different_token_position=1)
        with self.assertRaisesRegex(ValueError, "after token 0"):
            validate_first_token_condition(rows, make_diagnostics(), expected_count=48)


class SemanticAuditTests(unittest.TestCase):
    def test_exact_changed_response_coverage(self):
        changed = [
            {
                "dataset_index": 0,
                "alpha": 0.5,
                "baseline_success": True,
                "success": False,
                "transition": "newly_defended",
                "matched_refusal_substrings": ["I cannot"],
            }
        ]
        audit = [
            {
                "dataset_index": 0,
                "alpha": 0.5,
                "semantic_classification": "genuine_refusal",
                "harmful_actionable_content_present": False,
                "audit_confidence": "high",
                "audit_reason": "Declines the request and provides no steps.",
            }
        ]
        validated = validate_semantic_audit(audit, changed)
        self.assertEqual(validated[0]["semantic_classification"], "genuine_refusal")
        self.assertEqual(
            semantic_audit_summary(validated)["classification_counts"]["genuine_refusal"],
            1,
        )
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_semantic_audit([], changed)


if __name__ == "__main__":
    unittest.main()
