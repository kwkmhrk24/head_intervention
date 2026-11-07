import unittest
import numpy as np
from utils.gcg_shuffled_additional_metrics import sign_metrics, query_diagnostics, QUERY_FIELDS


class AdditionalTests(unittest.TestCase):
    def fixture(self):
        ids=list(range(27));ids[-4:]=[518,29914,25580,29962]
        item=dict(input_ids=ids,token_ranges=dict(template_prefix=[0,1],original=[1,3],suffix=[3,23],template_closing=[23,27]))
        return np.zeros((32,32,27,27),dtype=np.float32),item

    def test_hand_sign_energy_and_zero_fraction(self):
        x,item=self.fixture();x[:,:,1:3,1:3]=[[-3,2],[0,-1]]
        v,n=sign_metrics(x,item['token_ranges'])
        for key,value in {'E_pos_OO_full':1,'E_neg_OO_full':2.5,'mu_pos_OO_full':.5,
                          'mu_neg_OO_full':-1,'positive_fraction_OO_full':.25,'negative_fraction_OO_full':.5,
                          'median_OO_full':-.5}.items():np.testing.assert_allclose(v[key],value)
        np.testing.assert_allclose(v['E_pos_OO_causal'],0)
        np.testing.assert_allclose(v['E_neg_OO_causal'],10/3)
        self.assertEqual(n['UU_causal'],210);self.assertEqual(n['UO_causal'],40)

    def test_uniform_causal_mass_all_keys_in_denominator(self):
        x,item=self.fixture();v,p=query_diagnostics(x,item)
        np.testing.assert_allclose(v[0,:20,QUERY_FIELDS.index('M_U')],np.arange(1,21)/np.arange(4,24))
        np.testing.assert_allclose(v[0,:20,QUERY_FIELDS.index('M_O')],2/np.arange(4,24))
        np.testing.assert_allclose(v[0,23,QUERY_FIELDS.index('M_U')],20/27)
        self.assertTrue((p['weight'][:,0,4:]==0).all())
        np.testing.assert_allclose(v[:,24,:-1],v[:,:20,:-1].mean(axis=1))
        self.assertTrue(np.isnan(v[:,:20,-1]).all())

    def test_future_values_never_enter_row_center_or_softmax(self):
        x,item=self.fixture();before,_=query_diagnostics(x,item)
        x[:,:,3,4:]=10000;after,_=query_diagnostics(x,item)
        np.testing.assert_allclose(before[:,0,:-1],after[:,0,:-1])

    def test_row_shift_center_and_D_invariance(self):
        x,item=self.fixture();x[:,:,3:23,1:3]=-3;x[:,:,3:23,3:23]=2
        a,_=query_diagnostics(x,item);b,_=query_diagnostics(x+7,item)
        for key in ['M_U','M_O','D','conditional_M_U','centered_O','centered_U','gap_U_minus_O']:
            j=QUERY_FIELDS.index(key);np.testing.assert_allclose(a[...,j],b[...,j],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(b[...,QUERY_FIELDS.index('row_mean')]-a[...,QUERY_FIELDS.index('row_mean')],7)

    def test_softmax_per_head_before_averaging(self):
        x,item=self.fixture();x[0,0,:,3:23]=10
        _,p=query_diagnostics(x,item)
        g=p['weight'][:,23,3:23].sum(axis=-1).mean()
        row=x[:,:,26,:].reshape(1024,27).mean(axis=0);w=np.exp(row-row.max());w/=w.sum()
        self.assertNotAlmostEqual(g,w[3:23].sum())

    def test_q4_matches_prior_implementation(self):
        from utils.gcg_shuffled_token_metrics import closing_queries
        x,item=self.fixture();x[:,:,24,3:23]=-1
        values,_=query_diagnostics(x,item);old=closing_queries(x,item)['metrics']
        for j,name in enumerate(['M_U','M_O','D']):
            np.testing.assert_allclose(values[:,20:24,QUERY_FIELDS.index(name)],old[:,:,j],rtol=1e-12,atol=1e-12)

    def test_bad_boundaries_and_nonfinite_rejected(self):
        x,item=self.fixture();x[0,0,4,4]=np.nan
        with self.assertRaises(ValueError):query_diagnostics(x,item)
        with self.assertRaises(ValueError):sign_metrics(x,item['token_ranges'])
        x,item=self.fixture();item['input_ids'][-1]=0
        with self.assertRaises(ValueError):query_diagnostics(x,item)


if __name__=='__main__':unittest.main()
