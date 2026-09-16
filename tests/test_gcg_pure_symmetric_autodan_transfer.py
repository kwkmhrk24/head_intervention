import unittest

from scripts.evaluate_gcg_pure_symmetric_autodan_transfer import (
    ALPHA,
    ASYM_TRANSFER_SUCCESS_COUNT,
    BASELINE_SUCCESS_COUNT,
    INTERVENTION_TIMING,
    INTERVENTION_VARIANT,
    K,
    MALFORMED_INDEX,
    TOTAL_COUNT,
    comparison_summary,
    normalize_canonical_baseline_rows,
    parse_args,
    validate_asym_transfer_rows,
)


def canonical_fixture_rows():
    raw = []
    baseline = []
    for index in range(TOTAL_COUNT):
        prompt = f"attack prompt {index}"
        if index == MALFORMED_INDEX:
            prompt += " [REPLACE]"
        success = index < BASELINE_SUCCESS_COUNT
        raw.append(
            {
                "dataset_index": index,
                "prompt_id": f"autodan_{index}",
                "prompt": prompt,
            }
        )
        baseline.append(
            {
                "dataset_index": index,
                "prompt_id": f"autodan_{index}",
                "contains_unreplaced_placeholder": index == MALFORMED_INDEX,
                "prompt": prompt,
                "official_formatted_prompt": f"[INST] {prompt} [/INST]",
                "official_input_ids": [1, index + 2],
                "official_prompt_length": 2,
                "official_format_success": success,
                "official_format_response": f"response {index}",
                "official_response_token_ids": [index + 100],
                "generation_failed": False,
            }
        )
    return raw, baseline


class PureSymmetricAutoDANTransferTests(unittest.TestCase):
    def test_cli_is_frozen_to_reuse_only_entrypoint(self):
        args = parse_args([])
        self.assertEqual(args.stage, "cpu-validate")
        self.assertIsNone(args.output_dir)
        self.assertEqual(K, 500)
        self.assertEqual(ALPHA, 1.0)
        self.assertEqual(INTERVENTION_TIMING, "first_token_only")
        self.assertEqual(INTERVENTION_VARIANT, "sym_pure_pairwise_delta")

    def test_canonical_baseline_normalization_preserves_index10(self):
        raw, source = canonical_fixture_rows()
        rows = normalize_canonical_baseline_rows(source, raw)
        self.assertEqual(len(rows), TOTAL_COUNT)
        self.assertEqual(sum(row["success"] for row in rows), 49)
        self.assertIn("[REPLACE]", rows[MALFORMED_INDEX]["prompt"])
        self.assertEqual(rows[MALFORMED_INDEX]["input_ids"], [1, 12])

    def test_canonical_baseline_rejects_replaced_index10(self):
        raw, source = canonical_fixture_rows()
        raw[MALFORMED_INDEX]["prompt"] = "replacement was applied"
        source[MALFORMED_INDEX]["prompt"] = "replacement was applied"
        with self.assertRaisesRegex(ValueError, "placeholder text mismatch"):
            normalize_canonical_baseline_rows(source, raw)

    def test_canonical_baseline_rejects_success_count_drift(self):
        raw, source = canonical_fixture_rows()
        source[BASELINE_SUCCESS_COUNT]["official_format_success"] = True
        with self.assertRaisesRegex(ValueError, "success count must equal 49"):
            normalize_canonical_baseline_rows(source, raw)

    def test_asym_transfer_requires_exact_baseline_identity(self):
        raw, source = canonical_fixture_rows()
        baseline = normalize_canonical_baseline_rows(source, raw)
        reproduced = []
        asym = []
        for index, row in enumerate(baseline):
            reproduced.append(
                {
                    "dataset_index": index,
                    "input_ids": row["input_ids"],
                    "response": row["response"],
                    "success": row["success"],
                }
            )
            asym.append(
                {
                    "dataset_index": index,
                    "input_ids": row["input_ids"],
                    "baseline_response": row["response"],
                    "baseline_success": row["success"],
                    "success": index < ASYM_TRANSFER_SUCCESS_COUNT,
                    "k": K,
                    "alpha": ALPHA,
                    "intervention_timing": INTERVENTION_TIMING,
                    "generation_failed": False,
                }
            )
        validate_asym_transfer_rows(asym, reproduced, baseline)
        asym[0]["input_ids"] = [999]
        with self.assertRaisesRegex(ValueError, "input ID mismatch"):
            validate_asym_transfer_rows(asym, reproduced, baseline)

    def test_comparison_keeps_three_conditions_distinct(self):
        pure = {
            "prompt_count": TOTAL_COUNT,
            "attack_success_count": 41,
            "asr": 41 / TOTAL_COUNT,
            "intervention_variant": INTERVENTION_VARIANT,
        }
        comparison = comparison_summary(pure)
        conditions = comparison["conditions"]
        self.assertEqual(
            [row["condition"] for row in conditions],
            [
                "baseline",
                "gcg_asym_top500_transfer",
                "gcg_pure_sym_top500_transfer",
            ],
        )
        self.assertEqual(conditions[0]["attack_success_count"], 49)
        self.assertEqual(conditions[1]["attack_success_count"], 49)
        self.assertEqual(conditions[2]["attack_success_count"], 41)


if __name__ == "__main__":
    unittest.main()
