"""
-------------------------------------------------------------------------------

Unit tests for the .first_difference module of spqf.operators.digital.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.operators.digital import first_difference as f_sig


# noinspection SpellCheckingInspection,PyPep8Naming
class TestDigitalFirstDifference(unittest.TestCase):
    """
    Unit tests to verify the correct operation of the 1st difference operator.
    """

    def test_delay_is_correct(self):
        """Tests that the 1st-difference delay = 1/2"""

        delay_ans = 0.5
        delay_test = f_sig.delay()
        self.assertEqual(delay_test, delay_ans)

    def test_first_difference_has_correct_array_length(self):
        """Tests that the sequence has the correct length"""

        ds = f_sig.generate_impulse_response()
        self.assertEqual(ds.n_axis.shape[0], 2)

    def test_first_difference_has_correct_indices_and_weights(self):
        """Tests that the sequence has the correct index and weight values"""

        ds = f_sig.generate_impulse_response()
        self.assertSequenceEqual(list(ds.n_axis), list(np.array([0, 1])))
        self.assertSequenceEqual(list(ds.v_axis), list(np.array([+1.0, -1.0])))
