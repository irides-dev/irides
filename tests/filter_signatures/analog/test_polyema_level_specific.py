"""
-------------------------------------------------------------------------------

Unit tests for the .polyema_level module of spqf.design.filter_signatures.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.design.analog import polyema_level as design
from irides.filter_signatures.analog import polyema_level as f_sig


# noinspection SpellCheckingInspection,PyPep8Naming
class TestPolyemaLevelSpecific(unittest.TestCase):
    """
    Unit tests for the polyema-level filter are predominantly covered
    by the parameterized tests in test-level-filters. Here, tests that
    are not common to all level filters are made available.
    """

    def test_generate_poles_and_zeros_gives_correct_locations(self):
        """
        Test that the f_sig poles are correctly located on the real axis and
        that there are no zeros.
        """

        # defs
        tau = 11.0  # prime and outside of the range of filter orders

        # fetch valid filter orders
        filter_orderes = design.get_valid_filter_orders()

        # iter over filter orders
        for order in filter_orderes:

            # fetch poles and zeros
            poles_and_zeros = f_sig.generate_poles_and_zeros(tau, order)

            # the theo pole location
            pole_theo = -1.0 / (tau / order) + 1j * 0.0

            with self.subTest(msg="order: {0}".format(order)):

                # expect no zeros
                self.assertTrue(poles_and_zeros["zeros"].shape == (0, 0))

                # test theo pole
                pole_test = poles_and_zeros["poles"]
                self.assertAlmostEqual(np.real(pole_test), np.real(pole_theo))
                self.assertAlmostEqual(np.imag(pole_test), np.imag(pole_theo))
