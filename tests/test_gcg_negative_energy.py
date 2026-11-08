import unittest
import numpy as np
from utils.gcg_negative_energy import factors,paired_factors,concentration,head_distribution,causal_query_factors,CONDITION_FIELDS,PAIR_FIELDS


class NegativeEnergyTests(unittest.TestCase):
    def test_hand_factors_include_all_elements_in_denominator(self):
        f,e=factors(np.array([[-3.,-1.,0.,2.]]))
        self.assertEqual(f['E_neg'][0],2.5);self.assertEqual(f['E_pos'][0],1.)
        self.assertEqual(f['p_neg'][0],.5);self.assertEqual(f['m2_neg'][0],5.)
        self.assertEqual(f['negative_count'][0],2)

    def test_zero_negative_is_undefined_with_explicit_identity_extension(self):
        f,_=factors(np.array([[0.,1.]]));self.assertTrue(np.isnan(f['m2_neg'][0]))
        self.assertTrue(f['zero_negative'][0]);self.assertEqual(f['m2_neg_zero_extension'][0],0.)
        g,s,p,_,_=paired_factors(np.array([[-2.,0.]]),np.array([[0.,1.]]))
        self.assertEqual(p['delta_E_neg'][0],2.)
        self.assertEqual(p['frequency_contribution'][0],1.)
        self.assertEqual(p['magnitude_contribution'][0],1.)
        self.assertFalse(p['both_have_negative'][0])

    def test_midpoint_decomposition_and_signed_element_changes(self):
        g=np.array([[-4.,-2.,1.,0.]]);s=np.array([[-3.,1.,-1.,0.]])
        _,_,p,e,errors=paired_factors(g,s)
        self.assertEqual(p['delta_E_neg'][0],2.5)
        self.assertEqual(p['frequency_contribution'][0],0.)
        self.assertEqual(p['magnitude_contribution'][0],2.5)
        self.assertEqual(p['positive_element_fraction'][0],.5)
        self.assertEqual(p['negative_element_fraction'][0],.25)
        self.assertEqual(p['equal_element_fraction'][0],.25)
        self.assertEqual(p['positive_contribution'][0],11/4)
        self.assertEqual(p['positive_element_count'][0],2)
        self.assertEqual(p['negative_element_count'][0],1)
        self.assertEqual(p['positive_contribution_total'][0],11)
        self.assertEqual(p['negative_contribution_total'][0],-1)
        self.assertEqual(p['negative_contribution'][0],-1/4)
        np.testing.assert_allclose(e['delta'].mean(axis=-1),p['delta_E_neg'])
        self.assertTrue(all(v==0 for v in errors.values()))

    def test_top_tail_denominators_rounding_and_zero_total(self):
        x=np.array([[9.,1.,0.,0.,0.,0.,0.,0.,0.,0.]])
        allshare,k,_=concentration(x,.2,'all');active,ka,_=concentration(x,.2,'active')
        self.assertEqual(k[0],2);self.assertEqual(ka[0],1)
        self.assertEqual(allshare[0],1.);self.assertEqual(active[0],.9)
        share,k,_=concentration(np.zeros((1,20)),.01,'active')
        self.assertTrue(np.isnan(share[0]));self.assertEqual(k[0],0)
        _,k,_=concentration(np.ones((1,210)),.01,'all');self.assertEqual(k[0],3)

    def test_head_distribution_and_signs_are_exact(self):
        s=head_distribution(np.array([-1.,0.,1.,4.]))
        self.assertEqual(s['mean'],1.);self.assertEqual(s['median'],.5)
        self.assertEqual(s['positive_count'],2);self.assertEqual(s['negative_count'],1)
        self.assertEqual(s['equal_count'],1)
        self.assertAlmostEqual(s['std'],np.std([-1,0,1,4],ddof=1))

    def test_causal_query_positions_weighting_not_future_values(self):
        g=np.zeros((1,20,20));s=g.copy()
        for q in range(20):g[:,q,:q+1]=-(q+1);g[:,q,q+1:]=-1000
        c,p,n=causal_query_factors(g,s,'UU')
        self.assertEqual(c.shape,(2,1,20,len(CONDITION_FIELDS)))
        self.assertEqual(p.shape,(1,20,len(PAIR_FIELDS)))
        np.testing.assert_array_equal(n,np.arange(1,21))
        energy=c[0,0,:,CONDITION_FIELDS.index('E_neg')]
        np.testing.assert_array_equal(energy,np.arange(1,21)**2)
        self.assertNotEqual(energy.mean(),np.average(energy,weights=n))
        f,_=factors(g[:,np.tril_indices(20)[0],np.tril_indices(20)[1]])
        self.assertAlmostEqual(f['E_neg'][0],np.average(energy,weights=n))

    def test_uo_query_equal_key_counts(self):
        g=np.full((2,20,3),-2.);s=np.full_like(g,-1.)
        c,p,n=causal_query_factors(g,s,'UO')
        np.testing.assert_array_equal(n,3)
        np.testing.assert_allclose(p[:,:,PAIR_FIELDS.index('delta_E_neg')],3.)

    def test_invalid_data_shapes_and_tail_policies_fail(self):
        for x in (np.array([]),np.array([np.nan]),np.array([np.inf])):
            with self.assertRaises(ValueError):factors(x)
        with self.assertRaises(ValueError):paired_factors(np.zeros((2,3)),np.zeros((2,4)))
        for x,p,basis in [(np.array([-1.]),.1,'all'),(np.ones(2),0.,'all'),(np.ones(2),.1,'bad')]:
            with self.assertRaises(ValueError):concentration(x,p,basis)
        with self.assertRaises(ValueError):causal_query_factors(np.zeros((2,19,20)),np.zeros((2,19,20)),'UU')


if __name__=='__main__':unittest.main()
