import json
import tempfile
import unittest
from pathlib import Path

from utils.intervention_transitions import (
    PromptOutcome,
    jaccard_counts,
    matched_attack_rows,
    mcnemar_exact_p,
    pair_outcomes,
    paired_bootstrap_ci,
    summarize_transitions,
    validate_existing_summary,
    write_outputs,
)


def outcomes(values):
    return [PromptOutcome(str(index), index, value) for index, value in enumerate(values)]


class CrossRunTransitionTests(unittest.TestCase):
    def test_transition_reaggregation_and_identities(self):
        paired = pair_outcomes(
            outcomes([True, False, True, False]),
            outcomes([False, True, True, False]),
        )
        self.assertEqual(
            [row["transition"] for row in paired],
            ["defended", "regressed", "remained_success", "remained_failure"],
        )
        summary = summarize_transitions(paired)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["baseline_success"], 2)
        self.assertEqual(summary["intervention_success"], 2)
        self.assertEqual(summary["net_defended"], 0)

    def test_jaccard(self):
        self.assertEqual(jaccard_counts({1, 2}, {2, 3}), (1, 3, 1 / 3))
        self.assertEqual(jaccard_counts(set(), set()), (0, 0, 1.0))

    def test_matched_source_join(self):
        auto = [
            {"source_index": 1, "transition": "defended"},
            {"source_index": 2, "transition": "regressed"},
            {"source_index": 3, "transition": "remained_failure"},
        ]
        gcg = [
            {"source_index": 2, "transition": "regressed"},
            {"source_index": 3, "transition": "defended"},
            {"source_index": 4, "transition": "defended"},
        ]
        row = matched_attack_rows("model", "paper", auto, gcg)
        self.assertEqual(row["matched_source_count"], 2)
        self.assertEqual(row["regressed_in_both"], 1)
        self.assertEqual(row["defended_gcg_only"], 1)
        self.assertEqual(row["same_transition_count"], 1)

    def test_mcnemar_exact_p_value(self):
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)
        self.assertEqual(mcnemar_exact_p(2, 2), 1.0)
        self.assertAlmostEqual(mcnemar_exact_p(8, 1), 0.0390625)

    def test_paired_bootstrap_is_deterministic(self):
        paired = pair_outcomes(
            outcomes([True, True, False, False, True]),
            outcomes([False, True, True, False, False]),
        )
        first = paired_bootstrap_ci(paired, seed=42, resamples=500)
        second = paired_bootstrap_ci(paired, seed=42, resamples=500)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])

    def test_summary_consistency(self):
        paired = pair_outcomes(outcomes([True, False]), outcomes([False, False]))
        summary = summarize_transitions(paired)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "total_count": 2,
                        "baseline_success_count": 1,
                        "success_count": 0,
                        "asr": 0.0,
                        "defended_count": 1,
                        "regressed_count": 0,
                        "remained_success_count": 0,
                        "remained_failure_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            validate_existing_summary(path, summary)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["defended_count"] = 0
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "summary mismatch"):
                validate_existing_summary(path, summary)

    def test_output_directory_overwrite_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                write_outputs(
                    Path(directory),
                    {
                        "transition_summary": [],
                        "prompt_ids": [],
                        "condition_overlap": [],
                        "matched_attack": [],
                        "statistical_tests": [],
                    },
                )


if __name__ == "__main__":
    unittest.main()
