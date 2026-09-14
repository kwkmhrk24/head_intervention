import unittest

from scripts.prepare_gcg_independent_derived import (
    EXPECTED_INDICES,
    parse_args,
    require_exact_indices,
)


class IndependentDerivedTests(unittest.TestCase):
    def test_defaults_are_read_only_validation(self):
        args = parse_args([])
        self.assertEqual(args.stage, "validate")
        self.assertTrue(args.output_dir.endswith("independent_advbench_220_319"))

    def test_exact_index_sequence_is_required(self):
        rows = [{"index": index} for index in EXPECTED_INDICES]
        require_exact_indices(rows, "index")
        for invalid in (rows[:-1], rows[::-1], rows[:-1] + [rows[-2]]):
            with self.subTest(count=len(invalid)), self.assertRaises(ValueError):
                require_exact_indices(invalid, "index")


if __name__ == "__main__":
    unittest.main()
