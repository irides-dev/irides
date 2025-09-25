"""
-------------------------------------------------------------------------------

Unit tests for the .unit_step module of spqf.operators.digital.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.operators.digital import unit_step as f_sig


# noinspection SpellCheckingInspection,PyPep8Naming
class TestDigitalUnitStep(unittest.TestCase):
    """
    Unit tests to verify the correct operation of the unit-step operator.
    """

    def test_delay_is_correct(self):
        """Tests that the unit-step delay is correct"""

        with self.subTest(msg="explicit n_end argument"):

            n_end = 11
            delay_ans = n_end / 2
            delay_test = f_sig.delay(n_end)

            self.assertEqual(delay_test, delay_ans)

        with self.subTest(msg="default n_end argument"):

            delay_test = f_sig.delay()

            self.assertTrue(np.isinf(delay_test))

    def test_unit_step_has_correct_array_length(self):
        """Tests that the sequence has the correct length"""

        ds = f_sig.generate_impulse_response(17)
        self.assertEqual(ds.n_axis.shape[0], 17)

    def test_unit_step_has_correct_indices_and_weights(self):
        """Tests that the sequence has the correct index and weight values"""

        n_end = 17
        ds = f_sig.generate_impulse_response(n_end)

        ans_n_axis = np.arange(0, n_end)
        ans_v_axis = np.ones_like(ans_n_axis)

        self.assertSequenceEqual(list(ds.n_axis), list(ans_n_axis))
        self.assertSequenceEqual(list(ds.v_axis), list(ans_v_axis))

    def test_alternating_unit_step_has_correct_values(self):
        """Tests that an alternating unit step goes as 1, -1, 1, -1, """

        n_end = 9
        ds = f_sig.generate_impulse_response(n_end, f_sig.FrequencyBand.HIGH)

        ans_v_axis = np.power(-1, ds.n_axis)
        self.assertSequenceEqual(list(ds.v_axis), list(ans_v_axis))

