import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.gcg_shuffled_metrics import METRICS,calculate_metrics,paired_differences,summarize


class MetricDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.scores=np.zeros((1,2,26,26),dtype=np.float32)
        self.original=[2,5];self.suffix=[5,25]

    def calc(self,scores=None):
        return calculate_metrics(self.scores if scores is None else scores,self.original,self.suffix,25)

    def test_uniform_softmax_includes_template_keys_and_zero_ratio_undefined(self):
        values,checks=self.calc()
        np.testing.assert_allclose(values['M_U'],20/26)
        np.testing.assert_allclose(values['M_O'],3/26)
        np.testing.assert_allclose(values['D'],np.log(20/3))
        self.assertTrue(np.isnan(values['R_asym']).all())
        self.assertEqual(checks['zero_block_count'],2)
        self.assertTrue((values['F']==0).all())

    def test_squared_ratio_and_full_unmasked_frobenius_reference(self):
        block=np.zeros((20,20));block[0,0]=3;block[0,1]=2;block[1,0]=-2
        self.scores[:,:,5:25,5:25]=block
        values,checks=self.calc()
        np.testing.assert_allclose(values['F'],np.sqrt(17))
        np.testing.assert_allclose(values['F_sym'],3)
        np.testing.assert_allclose(values['F_asym'],np.sqrt(8))
        np.testing.assert_allclose(values['R_asym'],8/17)
        self.assertLess(checks['energy_identity_max_relative_error'],1e-14)
        self.assertEqual(checks['reconstruction_max_error'],0)

    def test_extreme_logits_are_stable_and_underflow_is_explicit(self):
        self.scores[:,:,25,2:5]=-10000
        self.scores[:,:,25,5:25]=10000
        values,checks=self.calc()
        np.testing.assert_allclose(values['M_U'],1)
        np.testing.assert_allclose(values['D'],20000+np.log(20/3))
        self.assertTrue(np.isfinite(values['D']).all())
        self.assertEqual(checks['softmax_region_underflow_count'],2)

    def test_permutation_invariance_does_not_require_equal_matrices(self):
        rng=np.random.default_rng(7)
        self.scores[:,:,5:25,5:25]=rng.normal(size=(1,2,20,20))
        before,_=self.calc()
        permutation=rng.permutation(20)
        after_scores=self.scores.copy()
        block=self.scores[:,:,5:25,5:25]
        after_scores[:,:,5:25,5:25]=block[:,:,permutation,:][:,:,:,permutation]
        self.assertFalse(np.array_equal(self.scores,after_scores))
        after,_=self.calc(after_scores)
        for metric in METRICS:np.testing.assert_allclose(before[metric],after[metric],rtol=1e-12,atol=1e-12)

    def test_wrong_query_bad_suffix_and_nonfinite_inputs_rejected(self):
        for query,suffix in [(24,[5,25]),(25,[5,24])]:
            with self.assertRaises(ValueError):calculate_metrics(self.scores,self.original,suffix,query)
        self.scores[0,0,0,0]=np.inf
        with self.assertRaises(ValueError):self.calc()


class PairingTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for index,delta in [(10,1),(11,2),(12,-3)]:
            for condition in ('gcg','shuffled'):
                for head in (0,1):
                    value=10+head+(delta if condition=='gcg' else 0)
                    rows.append(dict(dataset_index=index,condition=condition,layer=0,head=head,primary_18=True,
                                     **{m:float(value) for m in METRICS}))
        return pd.DataFrame(rows).sample(frac=1,random_state=5).reset_index(drop=True)

    def test_keyed_pairs_summary_median_sd_and_counts(self):
        frame=self.fixture();paired=paired_differences(frame)
        summary=summarize(frame,paired,[10,11,12])
        row=summary[(summary.metric=='F')&(summary['head']==0)].iloc[0]
        self.assertEqual(row.delta_mean,0)
        self.assertEqual(row.delta_median,1)
        self.assertAlmostEqual(row.delta_std,np.std([1,2,-3],ddof=1))
        self.assertEqual(row.gcg_larger_count,2)
        self.assertEqual(row.shuffled_larger_count,1)
        self.assertEqual(len(paired),6)

    def test_missing_or_duplicate_pair_is_rejected(self):
        frame=self.fixture()
        for changed in (frame.iloc[1:],pd.concat([frame,frame.iloc[:1]])):
            with self.assertRaises(ValueError):paired_differences(changed)

    def test_undefined_ratio_remains_missing_with_valid_counts_and_blank_csv(self):
        frame=self.fixture();frame.loc[(frame.dataset_index==10)&(frame.condition=='gcg'),'R_asym']=np.nan
        summary=summarize(frame,paired_differences(frame),[10,11,12])
        self.assertTrue((summary[summary.metric=='R_asym'].pair_valid_count==2).all())
        self.assertTrue((summary[summary.metric=='F'].pair_valid_count==3).all())
        with tempfile.TemporaryDirectory() as tmp:
            frame.to_csv(Path(tmp)/'metrics.csv',index=False)
            saved=pd.read_csv(Path(tmp)/'metrics.csv')
            self.assertEqual(saved.R_asym.isna().sum(),2)


class SourceAndFigureTests(unittest.TestCase):
    def test_existing_input_correspondence_and_no_model_load(self):
        from scripts.analyze_gcg_shuffled_four_metrics import DATASET,inspect_inputs
        if not DATASET.exists():self.skipTest('research artifacts unavailable')
        with patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=AssertionError('no model')):
            dataset,primary,references=inspect_inputs()
        self.assertEqual(len(dataset),20);self.assertEqual(len(primary),18);self.assertEqual(len(references),60)
        self.assertNotIn(91,primary);self.assertNotIn(123,primary)
        self.assertTrue(all(r['dataset_index']<220 for r in references))


if __name__=='__main__':
    unittest.main()
