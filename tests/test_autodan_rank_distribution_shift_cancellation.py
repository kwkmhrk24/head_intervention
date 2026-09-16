import unittest

import torch

from scripts.evaluate_autodan_rank_distribution_shift_cancellation import (
    EXPECTED_BASELINE_SUCCESS,
    parse_args,
    verify_baseline_reproduction,
)
from utils.gcg_rank_distribution_shift_cancellation import (
    ALL_RANK_COUNT,
    ALPHA_DEFAULT,
    METHOD,
)


class FrozenConfigurationTests(unittest.TestCase):
    def test_cli_has_fixed_method_and_no_alpha_or_dataset_override(self):
        args = parse_args(["--shift-artifact-dir", "shift"])
        self.assertEqual(ALPHA_DEFAULT, 1.0)
        self.assertEqual(ALL_RANK_COUNT, 122880)
        self.assertEqual(METHOD, "rank_distribution_shift_cancellation_v1")
        self.assertFalse(hasattr(args, "alpha"))
        self.assertFalse(hasattr(args, "dataset_groups"))
        with self.assertRaises(SystemExit):
            parse_args(["--shift-artifact-dir", "shift", "--alpha", "0.5"])

    def test_baseline_gate_requires_exact_saved_response(self):
        input_ids = torch.tensor([[1, 2, 3]])
        baseline = [
            {
                "dataset_index": 7,
                "prompt": "prompt",
                "formatted_prompt": "formatted",
                "response": "response",
                "success": False,
                "response_token_count": 32,
                "generation_failed": False,
                "input_ids": input_ids,
            }
        ]
        prior = [
            {
                "source_index": 7,
                "prompt": "prompt",
                "formatted_prompt": "formatted",
                "response": "response",
                "success": False,
                "response_length_tokens": 32,
            }
        ]
        verify_baseline_reproduction(baseline, prior, expected_count=1)
        prior[0]["response"] = "different"
        with self.assertRaisesRegex(RuntimeError, "refusing intervention"):
            verify_baseline_reproduction(baseline, prior, expected_count=1)

    def test_frozen_expected_baseline_is_13(self):
        self.assertEqual(EXPECTED_BASELINE_SUCCESS, 13)


if __name__ == "__main__":
    unittest.main()
