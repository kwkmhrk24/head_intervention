import copy
import unittest
from unittest.mock import patch

from scripts import build_shuffled_gcg_comparison as comparison
from scripts import evaluate_shuffled_gcg_pilot as pilot


@unittest.skipUnless((pilot.MODEL/'tokenizer.json').exists(), 'local research tokenizer unavailable')
class ComparisonDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(pilot.MODEL, use_fast=True, local_files_only=True)
        cls.samples = comparison.build_samples(cls.tokenizer)

    def test_all_selection20_have_exact_three_condition_boundaries(self):
        from utils.gcg_first_token_reference_ablation import EXPECTED_SELECTION_DATASET_INDICES
        self.assertEqual(tuple(s['dataset_index'] for s in self.samples), EXPECTED_SELECTION_DATASET_INDICES)
        for sample in self.samples:
            comparison.validate_sample(sample)
            self.assertEqual(sample['suffix_token_count'], 20)
            self.assertLess(sample['dataset_index'], 220)
            for condition in comparison.CONDITIONS:
                item = sample['conditions'][condition]
                self.assertEqual(item['input_ids'][slice(*item['token_ranges']['template_prefix'])], [1, 518, 25580, 29962])
                self.assertEqual(item['input_ids'][slice(*item['token_ranges']['template_closing'])], [518, 29914, 25580, 29962])
                self.assertEqual(item['final_input_token_id'], 29962)

    def test_original_mutation_bad_ranges_and_query_position_fail(self):
        for mutation in ('original', 'range', 'query'):
            sample = copy.deepcopy(self.samples[0])
            if mutation == 'original':
                sample['conditions']['original']['input_ids'][5] += 1
            elif mutation == 'range':
                sample['conditions']['gcg']['token_ranges']['suffix'][0] += 1
            else:
                sample['conditions']['original']['final_input_token_position'] -= 1
            with self.assertRaises(ValueError):
                comparison.validate_sample(sample)

    def test_pilot_reuse_requires_exact_inputs_and_seed(self):
        reused = comparison.load_pilot_reuse(self.samples)
        self.assertEqual(len(reused), 10)
        self.assertEqual({r['dataset_index'] for r in reused}, {68, 73, 91, 96, 208})
        modified = copy.deepcopy(self.samples)
        modified[0]['shuffle_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'seed mismatch'):
            comparison.load_pilot_reuse(modified)

    def test_reuse_rejects_wrong_saved_automatic_label(self):
        reused = copy.deepcopy(comparison.load_pilot_reuse(self.samples))
        reused[0]['success'] = not reused[0]['success']
        with patch.object(pilot, 'read_jsonl', return_value=reused):
            with self.assertRaisesRegex(ValueError, 'label mismatch'):
                comparison.load_pilot_reuse(self.samples)

    def test_result_validation_rejects_missing_inputs_labels_and_fake_human_review(self):
        rows = []
        for sample in self.samples:
            for condition in comparison.CONDITIONS:
                response = sample['historical_gcg_response'] if condition == 'gcg' else "I'm sorry"
                rows.append(dict(dataset_index=sample['dataset_index'], condition=condition,
                    input_ids=sample['conditions'][condition]['input_ids'], response=response,
                    success=condition == 'gcg', generation_failed=False, empty_response=False,
                    response_token_count=1, response_token_ids=[123],
                    human_jailbreak_success=None, human_review_status='not_reviewed'))
        comparison.validate_results(self.samples, rows)
        for mutation in ('missing', 'input', 'label', 'human', 'gcg_response'):
            modified = copy.deepcopy(rows)
            if mutation == 'missing':
                modified.pop()
            elif mutation == 'input':
                modified[0]['input_ids'][0] += 1
            elif mutation == 'label':
                modified[0]['success'] = True
            elif mutation == 'human':
                modified[0]['human_jailbreak_success'] = False
            else:
                modified[1]['response'] = 'Another non-refusal response'
            with self.assertRaises(ValueError):
                comparison.validate_results(self.samples, modified)


if __name__ == '__main__':
    unittest.main()
