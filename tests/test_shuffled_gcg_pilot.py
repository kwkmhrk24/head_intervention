import unittest
from collections import Counter
from unittest.mock import patch

from scripts.evaluate_shuffled_gcg_pilot import (
    MODEL, permute_suffix, prepare_samples, token_span, validate_permutation,
)


class TokenInvariantTests(unittest.TestCase):
    def test_duplicate_token_counts_and_both_flanks_survive(self):
        ids = [1, 99, 4, 4, 7, 8, 8, 2, 98]
        shuffled, _ = permute_suffix(ids, 2, 7, 123, special_ids=[1, 2])
        self.assertEqual(shuffled[:2], ids[:2])
        self.assertEqual(shuffled[7:], ids[7:])
        self.assertEqual(Counter(shuffled[2:7]), Counter(ids[2:7]))
        self.assertNotEqual(shuffled[2:7], ids[2:7])
        self.assertEqual(permute_suffix(ids, 2, 7, 123)[0], shuffled)

    def test_identity_draw_is_retried(self):
        calls = []

        def draw(values):
            calls.append(1)
            if len(calls) > 1:
                values.reverse()

        with patch('random.Random.shuffle', side_effect=draw):
            result, attempts = permute_suffix([1, 4, 7, 2], 1, 3, 123)
        self.assertEqual(attempts, 2)
        self.assertEqual(result, [1, 7, 4, 2])

    def test_impossible_shuffle_and_special_token_are_rejected(self):
        for ids, special in [([1, 4, 4, 2], []), ([1, 4, 2, 3], [2])]:
            with self.assertRaises(ValueError):
                permute_suffix(ids, 1, 3, 1, special)

    def test_tampering_with_prefix_template_counts_or_order_is_rejected(self):
        original = [1, 4, 7, 2]
        for changed in ([9, 7, 4, 2], [1, 7, 4, 9], [1, 7, 7, 2], original, [1, 7, 4, 4, 2]):
            with self.assertRaises(ValueError):
                validate_permutation(original, list(changed), 1, 3)

    def test_crossing_original_or_template_token_is_rejected(self):
        self.assertEqual(token_span([(0, 4), (4, 7), (7, 9), (9, 12)], 4, 9), (1, 3))
        for offsets in ([(0, 5), (5, 9), (9, 12)], [(0, 4), (4, 10), (10, 12)]):
            with self.assertRaisesRegex(ValueError, 'boundary'):
                token_span(offsets, 4, 9)


@unittest.skipUnless((MODEL/'tokenizer.json').exists(), 'local research tokenizer unavailable')
class LegacyDataPreflightTests(unittest.TestCase):
    def test_five_exact_legacy_inputs_without_retokenizing_shuffle(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
        selected, rows, exclusions = prepare_samples(tokenizer)
        self.assertEqual(exclusions, [])
        self.assertEqual(selected, [68, 73, 91, 96, 208])
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertLess(row['dataset_index'], 220)
            self.assertEqual(row['suffix_token_count'], 20)
            self.assertTrue(row['historical_success'])
            start, end = row['suffix_token_range']
            self.assertEqual(row['gcg_input_ids'][end:], [518, 29914, 25580, 29962])
            validate_permutation(row['gcg_input_ids'], row['shuffled_input_ids'], start, end)


if __name__ == '__main__':
    unittest.main()
