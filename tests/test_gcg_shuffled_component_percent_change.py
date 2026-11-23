import unittest

import numpy as np

from scripts.analyze_gcg_shuffled_component_percent_change import changes


class PercentChangeTests(unittest.TestCase):
    def test_doubling_halving_and_zero_change(self):
        p, log = changes([4, 1, 2], [2, 2, 2])
        np.testing.assert_array_equal(p, [100, -50, 0])
        np.testing.assert_allclose(log, [np.log(2), -np.log(2), 0])

    def test_zero_and_nonfinite(self):
        p, log = changes([0, 2, np.inf, 1, 1e-15], [2, 0, 1, np.nan, 2e-15])
        np.testing.assert_allclose(p, [-100, np.nan, np.nan, np.nan, -50])
        np.testing.assert_allclose(log, [np.nan, np.nan, np.nan, np.nan, -np.log(2)])

    def test_percent_must_precede_head_mean(self):
        p, log = changes([2, 8], [1, 10])
        self.assertAlmostEqual(p.mean(), 40)
        self.assertAlmostEqual(np.abs(p).mean(), 60)
        self.assertNotAlmostEqual(p.mean(), 100 * (5 - 5.5) / 5.5)
        self.assertAlmostEqual(log.mean(), np.log(1.6) / 2)

    def test_log_sign_near_rounding_limit(self):
        baseline = 64.0
        p, log = changes([np.nextafter(baseline,np.inf), np.nextafter(baseline,0)], baseline)
        np.testing.assert_array_equal(np.sign(log), [1,-1])
        np.testing.assert_array_equal(np.sign(p), np.sign(log))
