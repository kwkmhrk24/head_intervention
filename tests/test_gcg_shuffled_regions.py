import unittest
import numpy as np

from utils.gcg_shuffled_region_metrics import block_norm,region_metrics,paired_stats,safe_ratio


class RegionDefinitionTests(unittest.TestCase):
    def fixture(self):
        scores=np.zeros((2,27,27),dtype=np.float32)
        scores[:,1:3,1:3]=[[3,4],[0,0]]
        scores[:,3:23,3:23]=3
        scores[:,3:23,1:3]=2
        return scores,dict(original=[1,3],suffix=[3,23])

    def test_hand_norm_rms_counts_and_rectangular_normalization(self):
        scores,ranges=self.fixture();values,counts=region_metrics(scores,ranges)
        np.testing.assert_allclose(values['F_OO_full'],5)
        np.testing.assert_allclose(values['RMS_OO_full'],2.5)
        np.testing.assert_allclose(values['F_UU_full'],60)
        np.testing.assert_allclose(values['RMS_UU_full'],3)
        np.testing.assert_allclose(values['F_UO_full'],2*np.sqrt(40))
        np.testing.assert_allclose(values['RMS_UO_full'],2)
        np.testing.assert_allclose(values['R_UU_OO_full'],1.2)
        np.testing.assert_allclose(values['R_UU_UO_full'],1.5)
        self.assertEqual(counts['OO_full'],4);self.assertEqual(counts['OO_causal'],3)
        self.assertEqual(counts['UU_full'],400);self.assertEqual(counts['UU_causal'],210)
        self.assertEqual(counts['UO_full'],40);self.assertEqual(counts['UO_causal'],40)

    def test_causal_rms_divides_by_visible_count_not_zero_padded_size(self):
        scores,ranges=self.fixture();values,_=region_metrics(scores,ranges)
        np.testing.assert_allclose(values['F_UU_causal'],3*np.sqrt(210))
        np.testing.assert_allclose(values['RMS_UU_causal'],3)
        self.assertNotAlmostEqual(values['RMS_UU_causal'][0],3*np.sqrt(210)/20)
        np.testing.assert_allclose(values['F_OO_causal'],3)
        np.testing.assert_allclose(values['RMS_OO_causal'],np.sqrt(3))

    def test_future_upper_elements_are_full_only_and_uo_always_visible(self):
        scores,ranges=self.fixture();b=np.ones((20,20));b[np.triu_indices(20,1)]=2
        scores[:,3:23,3:23]=b;values,_=region_metrics(scores,ranges)
        np.testing.assert_allclose(values['F_UU_full'],np.sqrt(970))
        np.testing.assert_allclose(values['F_UU_causal'],np.sqrt(210))
        np.testing.assert_allclose(values['RMS_UU_causal'],1)
        np.testing.assert_array_equal(values['RMS_UO_causal'],values['RMS_UO_full'])

    def test_zero_denominator_is_undefined_not_epsilon(self):
        scores,ranges=self.fixture();scores[:,1:3,1:3]=0
        values,_=region_metrics(scores,ranges)
        self.assertTrue(np.isnan(values['R_UU_OO_full']).all())
        self.assertTrue(np.isfinite(values['R_UU_UO_full']).all())
        self.assertTrue(np.isnan(safe_ratio(np.array([1.]),np.array([0.]))).all())

    def test_invalid_range_shape_suffix_and_nonfinite_fail(self):
        scores,ranges=self.fixture()
        for wrong in [dict(original=[1,4],suffix=[3,23]),dict(original=[1,3],suffix=[3,22])]:
            with self.assertRaises(ValueError):region_metrics(scores,wrong)
        with self.assertRaises(ValueError):region_metrics(scores[...,:-1],ranges)
        scores[:,1,1]=np.inf
        with self.assertRaises(ValueError):region_metrics(scores,ranges)
        with self.assertRaises(ValueError):block_norm(np.zeros((2,2)),np.zeros((2,2),dtype=bool))

    def test_original_invariance_requires_real_correspondence(self):
        from scripts.analyze_gcg_shuffled_regions import check_original
        x=np.arange(8,dtype=np.float32).reshape(2,2,2)
        diff,reference=check_original(x,x.copy(),x.copy())
        self.assertTrue((diff==0).all());self.assertTrue((reference==0).all())
        changed=x.copy();changed[0,0,0]+=1
        with self.assertRaises(AssertionError):check_original(x,changed,x)


class AggregationTests(unittest.TestCase):
    def test_per_head_ratios_and_norms_precede_head_average(self):
        scores=np.zeros((2,27,27),dtype=np.float32)
        scores[:,1:3,1:3]=np.array([1,4])[:,None,None]
        scores[:,3:23,3:23]=np.array([2,4])[:,None,None]
        scores[:,3:23,1:3]=1
        values,_=region_metrics(scores,dict(original=[1,3],suffix=[3,23]))
        self.assertEqual(values['R_UU_OO_full'].mean(),1.5)
        self.assertNotEqual(values['R_UU_OO_full'].mean(),values['RMS_UU_full'].mean()/values['RMS_OO_full'].mean())
        blocks=np.stack([np.ones((20,20)),-np.ones((20,20))])
        self.assertEqual(np.linalg.norm(blocks,axis=(-2,-1)).mean(),20)
        self.assertEqual(np.linalg.norm(blocks.mean(axis=0)),0)

    def test_paired_statistics_sd_absolute_counts_and_undefined_validity(self):
        values=np.zeros((3,2,1,2));values[:,0,0,0]=[1,2,-3];values[:,0,0,1]=[np.nan,2,4]
        stats=paired_stats(values,[0,1,2])
        self.assertEqual(stats['delta_mean'][0,0],0)
        self.assertEqual(stats['delta_median'][0,0],1)
        self.assertAlmostEqual(stats['delta_std'][0,0],np.std([1,2,-3],ddof=1))
        self.assertEqual(stats['delta_mean_absolute'][0,0],2)
        self.assertEqual(stats['gcg_larger_count'][0,0],2);self.assertEqual(stats['shuffled_larger_count'][0,0],1)
        self.assertEqual(stats['pair_valid_count'][0,1],2);self.assertEqual(stats['delta_mean'][0,1],3)

    def test_numerical_equivalence_and_subset_are_preserved(self):
        values=np.ones((3,2,1,1));values[2,0,0,0]=100
        primary=paired_stats(values,[0,1]);full=paired_stats(values,[0,1,2])
        self.assertEqual(primary['equal_count'][0,0],2);self.assertEqual(primary['delta_mean'][0,0],0)
        self.assertGreater(full['delta_mean'][0,0],0)


if __name__=='__main__':unittest.main()
