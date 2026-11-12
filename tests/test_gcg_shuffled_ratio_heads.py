"""Hand-checkable CSV diagnostics; no tensors/models or Score files."""
import unittest
import numpy as np
import pandas as pd
from utils.gcg_shuffled_common import paired_stats
from utils.gcg_shuffled_ratio_heads import complete_heads,direction_statistics,align_pairs,analyze_heads,fixed_group_tables


class RatioHeadTests(unittest.TestCase):
    def fixture(self):
        rows=[];pairs=[]
        for i in (68,70):
            for h in range(1024):
                g=2. if i==68 else 1.
                rows.extend([dict(dataset_index=i,layer=h//32,head=h%32,condition='gcg',primary18=True,R_UU_OO_full=g),
                             dict(dataset_index=i,layer=h//32,head=h%32,condition='shuffled',primary18=True,R_UU_OO_full=1.)])
                pairs.append(dict(dataset_index=i,layer=h//32,head=h%32,delta_R_UU_OO_full=g-1.))
        return pd.DataFrame(rows),pd.DataFrame(pairs)

    def test_direction_mean_sign_and_majority_can_differ(self):
        # One large positive sample vs two negative: mean +, majority -.
        d=np.array([[4.,0.],[-1.,0.],[-1.,0.]])
        s=np.full_like(d,2.);g=s+d
        r=direction_statistics(d,g,s)
        self.assertEqual(r['positive_samples'][0],1)
        self.assertEqual(r['negative_samples'][0],2)
        self.assertEqual(r['mean_direction'][0],1)
        self.assertEqual(r['mean_direction_agreement'][0],1/3)
        self.assertEqual(r['majority_nonzero_fraction'][0],2/3)
        self.assertEqual(r['mean_direction_agreement'][1],1.)

    def test_old_tie_tolerance_preserved(self):
        s=np.ones((2,1));g=s+1e-13
        r=direction_statistics(g-s,g,s)
        self.assertEqual(r['equal_samples'][0],2)
        self.assertEqual(r['strict_positive_samples'][0],2)
        self.assertEqual(r['mean_direction'][0],0)

    def test_head_grid_missing_duplicate_invalid(self):
        h=pd.DataFrame([(l,k) for l in range(32) for k in range(32)],columns=['layer','head'])
        complete_heads(h,'fixture')
        for wrong in (h.iloc[:-1],pd.concat([h.iloc[:-1],h.iloc[:1]]),h.assign(head=32)):
            with self.assertRaisesRegex(ValueError,'correspondence'):complete_heads(wrong,'fixture')

    def test_pair_alignment_independent_of_row_order(self):
        m,p=self.fixture()
        g,s,d=align_pairs(m.sample(frac=1,random_state=4),p.iloc[::-1],[70,68],'full')
        self.assertEqual(g.shape,(2,1024));np.testing.assert_array_equal(d[:,0],[0.,1.])
        np.testing.assert_array_equal(g[:,0],[1.,2.]);np.testing.assert_array_equal(s,1.)

    def test_reversed_delta_missing_and_duplicate_pair_fail(self):
        m,p=self.fixture()
        with self.assertRaises(AssertionError):align_pairs(m,p.assign(delta_R_UU_OO_full=-p.delta_R_UU_OO_full),[68,70],'full')
        for wrong in (p.iloc[:-1],pd.concat([p,p.iloc[:1]])):
            with self.assertRaises(ValueError):align_pairs(m,wrong,[68,70],'full')
        with self.assertRaises(ValueError):align_pairs(m.iloc[:-1],p,[68,70],'full')

    def test_nonfinite_or_wrong_shape_rejected(self):
        for d in (np.array([[np.nan]]),np.ones((2,1))):
            with self.assertRaises(ValueError):direction_statistics(d,np.ones((1,1)),np.ones((1,1)))

    def tables(self):
        m,p=self.fixture();g,s,d=align_pairs(m,p,[68,70],'full')
        values=np.stack([g,s],axis=1)[...,None]
        stats=paired_stats(values,[0,1]);h=pd.DataFrame(dict(layer=np.arange(1024)//32,head=np.arange(1024)%32,metric='R_UU_OO_full',sample_count=2))
        for k,v in stats.items():h[k]=v[:,0]
        maps=h[['layer','head']].assign(delta_R_UU_OO_full=h.delta_mean)
        ov=pd.DataFrame([dict(subset=2,sample_count=2,metric='R_UU_OO_full',**{k:float(v[:,0].mean()) for k,v in stats.items()})])
        return m,p,h,maps,ov

    def test_saved_ratio_mean_counts_and_threshold(self):
        m,p,h,maps,ov=self.tables()
        heads,o,l,s,e=analyze_heads(m,p,h,maps,ov,[68,70],'full')
        self.assertEqual(o['delta_mean'],.5)
        self.assertEqual(o['positive_heads'],1024)
        self.assertEqual(o['gcg_head_mean_gt1'],1024)
        self.assertEqual(o['positive_all_samples_heads'],0)
        self.assertEqual(o['mean_direction_agreement_median'],.5)
        self.assertEqual(len(l),32);self.assertEqual(len(s),2)
        self.assertTrue(all(v==0 for v in e.values()))
        self.assertEqual(heads.gcg_mean.iloc[0],1.5)

    def test_saved_summary_or_map_mismatch_detected(self):
        m,p,h,maps,ov=self.tables()
        with self.assertRaises(ValueError):analyze_heads(m,p,h.assign(delta_mean=2),maps,ov,[68,70],'full')
        with self.assertRaises(AssertionError):analyze_heads(m,p,h,maps.assign(delta_R_UU_OO_full=9.),ov,[68,70],'full')

    def test_fixed_group_membership_preserved_and_validated(self):
        m,p,h,maps,ov=self.tables();heads,*_=analyze_heads(m,p,h,maps,ov,[68,70],'full')
        groups=[];members=[]
        for i in range(12):
            group='fixed_'+str(i);row=h.iloc[0].to_dict();row.pop('layer');row.pop('head')
            groups.append(dict(**row,view_id=group,view_kind='selected_group',head_count=5))
            for j in range(5):members.append(dict(subset=2,selected_group=group,**h.iloc[i*5+j].to_dict()))
        views=pd.DataFrame(groups);detail=pd.DataFrame(members)
        g,d=fixed_group_tables(views,detail,heads,2,'full')
        self.assertEqual(g.view_id.tolist(),views.view_id.tolist());self.assertEqual(len(d),60)
        with self.assertRaises(ValueError):fixed_group_tables(views,detail.iloc[:-1],heads,2,'full')
        with self.assertRaises(ValueError):fixed_group_tables(views,detail.assign(selected_group='unknown'),heads,2,'full')


if __name__=='__main__':unittest.main()
