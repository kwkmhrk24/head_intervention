import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_gcg_shuffled_compatibility import compare_array,compare_trees
from utils.gcg_shuffled_common import head_id,head_entities,region_layout,reduce_views,paired_stats


class CompatibilityTests(unittest.TestCase):
    def test_head_id_order_and_range(self):
        entities=head_entities();self.assertEqual(len(entities),1024)
        self.assertEqual([head_id(**e) for e in entities],list(range(1024)))
        for args in [(-1,0),(0,32),(1.5,2)]:
            with self.assertRaises(ValueError):head_id(*args)

    def test_condition_reversal_and_sample_shift_detected(self):
        x=np.arange(24,dtype=float).reshape(3,2,4)
        with self.assertRaises(AssertionError):compare_array(x,x[:,::-1],'condition reversal')
        with self.assertRaises(AssertionError):compare_array(x,x[::-1],'sample order')

    def test_dtype_shape_nan_and_sign_detection(self):
        for b in [np.array([1],dtype=np.float32),np.array([[1.]]),np.array([-1.]),np.array([np.nan])]:
            with self.assertRaises(AssertionError):compare_array(np.array([1.]),b,'change')
        self.assertEqual(compare_array(np.array([np.nan]),np.array([np.nan]),'same')['max_absolute_error'],0)

    def test_csv_schema_order_and_ID_exactness(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';a.mkdir();b.mkdir()
            x=pd.DataFrame({'index':[68,70],'metric':['mu','mu'],'value':[1.,-2.]})
            x.to_csv(a/'metrics.csv',index=False);x.to_csv(b/'metrics.csv',index=False)
            self.assertEqual(compare_trees(a,b)['max_absolute_error'],0)
            x.iloc[::-1].to_csv(b/'metrics.csv',index=False)
            with self.assertRaises(AssertionError):compare_trees(a,b)
            x[['value','index','metric']].to_csv(b/'metrics.csv',index=False)
            with self.assertRaises(AssertionError):compare_trees(a,b)

    def test_npz_key_order_and_head_shift(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';a.mkdir();b.mkdir()
            x=np.arange(1024)
            np.savez(a/'v.npz',head=x,mask=np.tri(2,dtype=bool))
            np.savez(b/'v.npz',head=x,mask=np.tri(2,dtype=bool))
            self.assertEqual(compare_trees(a,b)['max_absolute_error'],0)
            np.savez(b/'v.npz',head=np.roll(x,1),mask=np.tri(2,dtype=bool))
            with self.assertRaises(AssertionError):compare_trees(a,b)

    def test_layout_bounds_causal_counts_and_sign_preserved(self):
        x=np.ones((1,25,25),dtype=np.float32)*-3
        blocks,masks,n=region_layout(x,dict(original=[1,3],suffix=[3,23]))
        self.assertEqual(n,dict(original_length=2,suffix_length=20));self.assertEqual(masks['UU'].sum(),210)
        self.assertEqual(masks['UO'].sum(),40);self.assertTrue((blocks['UU']==-3).all())
        with self.assertRaises(ValueError):region_layout(x,dict(original=[1,4],suffix=[3,23]))

    def test_fixed_view_weights_and_features_before_averaging(self):
        x=np.array([[1.],[-3.],[20.]])
        views=[dict(members=[1,0]),dict(members=[2])]
        np.testing.assert_array_equal(reduce_views(x,views),[[-1.],[20.]])
        pair=np.zeros((2,2,3,1));pair[:,0]=x
        a=paired_stats(pair,[0,1]);self.assertEqual(a['delta_mean'][1,0],-3)
        self.assertEqual(a['shuffled_larger_count'][1,0],2)

    def test_legacy_imports_export_same_implementations(self):
        from scripts.analyze_gcg_shuffled_four_metrics import inspect_inputs,write_csv
        from scripts.analyze_gcg_shuffled_regions import summary_frame
        from utils.gcg_shuffled_token_metrics import reduce_views as legacy
        from utils.gcg_shuffled_region_metrics import paired_stats as legacy_stats
        import utils.gcg_shuffled_common as c
        self.assertIs(inspect_inputs,c.inspect_inputs);self.assertIs(write_csv,c.write_csv)
        self.assertIs(summary_frame,c.summary_frame);self.assertIs(legacy,c.reduce_views);self.assertIs(legacy_stats,c.paired_stats)


if __name__=='__main__':unittest.main()

class CatalogTests(unittest.TestCase):
    def test_relative_links_keep_old_path_and_reject_retargeting(self):
        from scripts.index_gcg_shuffled_phase import safe_link
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'old';old.mkdir();(old/'x.csv').write_text('id,value\n1,-2\n')
            alias=root/'new'/'stage';safe_link(alias,old);safe_link(alias,old)
            self.assertTrue((old/'x.csv').exists());self.assertEqual((alias/'x.csv').read_bytes(),(old/'x.csv').read_bytes())
            other=root/'other';other.mkdir()
            with self.assertRaises(ValueError):safe_link(alias,other)
            with self.assertRaises(FileNotFoundError):safe_link(root/'bad',root/'missing')

    def test_catalog_never_replaces_directory(self):
        from scripts.index_gcg_shuffled_phase import safe_link
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            with self.assertRaises(FileExistsError):safe_link(a,b)
