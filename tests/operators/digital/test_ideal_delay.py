"""
-------------------------------------------------------------------------------

Unit tests for the .ideal_delay module of spqf.operators.digital.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.operators.digital import ideal_delay as f_sig


# noinspection SpellCheckingInspection,PyPep8Naming
class TestDigitalIdealDelay(unittest.TestCase):
    """
    Unit tests to verify the correct operation of the ideal-delay operator.
    """

    def test_delay_is_correct(self):
        """Tests that the design-based ideal delay is correct"""

        with self.subTest(msg="explicit mu argument"):

            mu_ans = 11
            mu_test = f_sig.delay(mu_ans)

            self.assertEqual(mu_test, mu_ans)

        with self.subTest(msg="default argument"):

            mu_test = f_sig.delay()

            self.assertTrue(np.isnan(mu_test))

    def test_ideal_delay_has_correct_array_length(self):
        """Tests that the sequence has the correct length"""

        mu_v = np.array([-4, 0, 12])
        ans_lengths = np.array([5, 1, 13])

        for i, mu in enumerate(mu_v):
            ds = f_sig.generate_impulse_response(mu)
            self.assertEqual(ds.n_axis.shape[0], ans_lengths[i])

    def test_ideal_delay_has_correct_indices_and_weights(self):
        """Tests that the sequence has the correct index and weight values"""

        mu_v = np.array([-4, 0, 12])

        for i, mu in enumerate(mu_v):
            ds = f_sig.generate_impulse_response(mu)

            # test n axis
            if mu >= 0:
                ans_n_axis = np.arange(0, mu + 1)
            else:
                ans_n_axis = np.arange(mu, 0 + 1)

            self.assertSequenceEqual(list(ds.n_axis), list(ans_n_axis))

            # test v axis
            index = ds.index_at(mu)
            self.assertEqual(ds.v_axis[index], 1)
            self.assertEqual(np.sum(ds.v_axis), 1)
