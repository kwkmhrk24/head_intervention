import unittest
import numpy as np
from utils.gcg_shuffled_signed_means import signed_means
from utils.gcg_shuffled_region_metrics import region_metrics, paired_stats


class SignedMeanTests(unittest.TestCase):
    def fixture(self):
        x = np.zeros((2, 25, 25), dtype=np.float32)
        x[:, 1:3, 1:3] = [[-3, 5], [-1, -1]]
        x[:, 3:23, 3:23] = -2
        x[:, 3:23, 1:3] = [-3, 1]
        return x, dict(original=[1, 3], suffix=[3, 23])

    def test_hand_signed_means_and_causal_counts(self):
        x, r = self.fixture(); v, n = signed_means(x, r)
        self.assertTrue((v['mu_OO_full'] == 0).all())
        np.testing.assert_allclose(v['mu_OO_causal'], -5/3)
        np.testing.assert_array_equal(v['mu_UU_full'], [-2, -2])
        np.testing.assert_array_equal(v['mu_UO_full'], [-1, -1])
        _, old_counts = region_metrics(x, r)
        self.assertEqual(n, old_counts)
        self.assertEqual(n['UU_causal'], 210)

    def test_future_values_excluded_without_inf_or_zero_padding(self):
        x, r = self.fixture(); x[:, 3:23, 3:23] = np.where(np.tri(20), -2, 10)
        v, _ = signed_means(x, r)
        np.testing.assert_allclose(v['mu_UU_causal'], -2)
        np.testing.assert_allclose(v['mu_UU_full'], (-2*210+10*190)/400)

    def test_no_absolute_value_or_clipping_and_equal_head_weight(self):
        x, r = self.fixture(); x[0, 3:23, 3:23] = -4; x[1, 3:23, 3:23] = 2
        v, _ = signed_means(x, r)
        self.assertEqual(v['mu_UU_full'].mean(), -1)
        pair = np.zeros((3, 2, 1, 1)); pair[:, 0, 0, 0] = [-1, 2, 3]
        result = paired_stats(pair, [0, 1])
        self.assertEqual(result['delta_mean'][0, 0], .5)
        self.assertEqual(result['gcg_larger_count'][0, 0], 1)
        self.assertEqual(result['shuffled_larger_count'][0, 0], 1)

    def test_invalid_and_nonfinite_fail(self):
        x, r = self.fixture()
        with self.assertRaises(ValueError): signed_means(x[..., :-1], r)
        with self.assertRaises(ValueError): signed_means(x, dict(original=[1, 4], suffix=[3, 23]))
        with self.assertRaises(ValueError): signed_means(x, dict(original=[1, 3], suffix=[3, 22]))
        for bad in (np.inf, np.nan):
            x[0, 4, 4] = bad
            with self.assertRaises(ValueError): signed_means(x, r)

    def test_unchanged_original_is_equal(self):
        x, r = self.fixture(); y = x.copy(); y[:, 3:23, 3:23] += 3
        a, _ = signed_means(x, r); b, _ = signed_means(y, r)
        for p in ('full', 'causal'):
            np.testing.assert_array_equal(a['mu_OO_'+p], b['mu_OO_'+p])
            np.testing.assert_allclose(a['mu_UU_'+p]-b['mu_UU_'+p], -3)


if __name__ == '__main__': unittest.main()
