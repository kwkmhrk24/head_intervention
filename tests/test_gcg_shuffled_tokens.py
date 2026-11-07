import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils.gcg_shuffled_metrics import METRICS,calculate_metrics
from utils.gcg_shuffled_token_metrics import (
    CLOSING_IDS,closing_queries,matrix_diagnostics,reduce_views,sample_summary,select_heads,
)


class MatrixContributionTests(unittest.TestCase):
    def test_hand_computable_energy_norm_and_ratio_contributions(self):
        g=np.zeros((1,20,20));s=g.copy()
        g[0,0,0]=s[0,0,0]=3;g[0,0,1]=2;g[0,1,0]=-2
        fields,scalars=matrix_diagnostics(g,s)
        self.assertAlmostEqual(fields['delta_square'].sum(),8)
        self.assertAlmostEqual(fields['F_delta_contribution'].sum(),np.sqrt(17)-3)
        self.assertNotAlmostEqual(fields['delta_square'].sum(),scalars[0]['F'][0]-scalars[1]['F'][0])
        self.assertAlmostEqual(fields['delta_normalized_asym_square'].sum(),8/17)
        self.assertAlmostEqual((scalars[0]['R_sym']+scalars[0]['R_asym'])[0],1)
        np.testing.assert_allclose(fields['raw_gcg'],fields['sym_gcg']+fields['asym_gcg'])

    def test_derive_then_head_mean_differs_from_square_mean_and_preserves_cancellation(self):
        g=np.zeros((2,20,20));s=g.copy();g[:,0,0]=[2,-2]
        fields,_=matrix_diagnostics(g,s);views=[dict(members=[0,1])]
        self.assertEqual(reduce_views(fields['delta_raw'],views)[0,0,0],0)
        self.assertEqual(reduce_views(fields['abs_delta_raw'],views)[0,0,0],2)
        self.assertEqual(reduce_views(fields['square_gcg'],views)[0,0,0],4)
        self.assertEqual(reduce_views(fields['raw_gcg'],views)[0,0,0]**2,0)

    def test_bad_shape_nonfinite_and_zero_ratio_are_not_silently_imputed(self):
        with self.assertRaises(ValueError):matrix_diagnostics(np.zeros((1,19,20)),np.zeros((1,19,20)))
        x=np.zeros((1,20,20));x[0,0,0]=np.nan
        with self.assertRaises(ValueError):matrix_diagnostics(x,x)
        _,scalars=matrix_diagnostics(np.zeros((1,20,20)),np.zeros((1,20,20)))
        self.assertTrue(np.isnan(scalars[0]['R_asym']).all())
        with self.assertRaises(ValueError):sample_summary(np.array([[np.nan],[0.]]),[0,1])


class QueryTests(unittest.TestCase):
    def fixture(self):
        ids=list(range(26));ids[22:26]=CLOSING_IDS
        item=dict(input_ids=ids,token_ranges=dict(template_prefix=[0,1],original=[1,2],suffix=[2,22],template_closing=[22,26]))
        return np.zeros((32,32,26,26),dtype=np.float32),item

    def test_four_queries_causal_denominator_and_existing_q4(self):
        scores,item=self.fixture();scores[:,:,22,23:]=10000
        result=closing_queries(scores,item)
        np.testing.assert_allclose(result['metrics'][0,:,0],[20/23,20/24,20/25,20/26])
        np.testing.assert_allclose(result['metrics'][0,:,1],[1/23,1/24,1/25,1/26])
        np.testing.assert_allclose(result['metrics'][0,:,2],np.log(20))
        self.assertTrue((result['weight'][:,0,23:]==0).all())
        old,_=calculate_metrics(scores,[1,2],[2,22],25)
        np.testing.assert_allclose(result['metrics'][:,3,0].reshape(32,32),old['M_U'])
        np.testing.assert_allclose(result['metrics'][:,3,2].reshape(32,32),old['D'])
        self.assertNotAlmostEqual(result['metrics'][0,:,0].mean(),20/((23+24+25+26)/4))

    def test_mismatched_template_or_region_and_nan_fail(self):
        scores,item=self.fixture();item['input_ids'][-1]=2
        with self.assertRaises(ValueError):closing_queries(scores,item)
        scores,item=self.fixture();item['token_ranges']['suffix']=[2,21]
        with self.assertRaises(ValueError):closing_queries(scores,item)
        scores,item=self.fixture();scores[0,0,0,0]=np.inf
        with self.assertRaises(ValueError):closing_queries(scores,item)


class SelectionAndSampleTests(unittest.TestCase):
    def fixture(self,small=False):
        rows=[]
        for metric in METRICS:
            for i in range(1024):
                value=1. if i<512 else -1.
                if small:value=1. if i==0 else -1. if i==1 else 0.
                rows.append(dict(metric=metric,layer=i//32,head=i%32,delta_mean=value))
        return pd.DataFrame(rows)

    def test_signed_top_bottom_ties_and_duplicate_computation_elimination(self):
        selected,views=select_heads(self.fixture())
        self.assertEqual(len(selected),60)
        self.assertEqual(len([v for v in views if v['kind']=='individual']),10)
        part=selected[(selected.metric=='M_U')&(selected.direction=='increase')]
        self.assertEqual(list(part['head']),list(range(5)))
        self.assertTrue((selected[selected.direction=='decrease'].delta_mean<0).all())

    def test_fewer_than_five_and_missing_grid(self):
        selected,views=select_heads(self.fixture(small=True))
        self.assertEqual(len(selected),12)
        self.assertEqual(len([v for v in views if v['kind']=='individual']),2)
        with self.assertRaises(ValueError):select_heads(self.fixture().iloc[1:])

    def test_consistency_counts_mean_median_sd_and_subset(self):
        x=np.array([[[1.,-2.]],[[2.,2.]],[[100.,1.]]])
        primary=sample_summary(x,[0,1]);full=sample_summary(x,[0,1,2])
        np.testing.assert_allclose(primary['mean'],[[1.5,0]])
        np.testing.assert_allclose(primary['mean_absolute'],[[1.5,2.]])
        self.assertEqual(primary['positive_count'][0,1],1)
        self.assertEqual(primary['negative_count'][0,1],1)
        self.assertGreater(full['std'][0,0],primary['std'][0,0])
        self.assertEqual(full['median'][0,0],2)

    def test_real_selection_is_60_memberships_45_unique_heads(self):
        path=Path(__file__).resolve().parents[1]/'data/outputs/gcg_shuffled_four_metrics/legacy_selection20_20261107T060324Z/summary_18.csv'
        if not path.exists():self.skipTest('research summary unavailable')
        selected,views=select_heads(pd.read_csv(path))
        self.assertEqual(len(selected),60)
        self.assertEqual(len([v for v in views if v['kind']=='individual']),45)


if __name__=='__main__':unittest.main()
