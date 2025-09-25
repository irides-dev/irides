"""
-------------------------------------------------------------------------------

Unit tests for the .bessel_level module of spqf.design.filter_signatures.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.design.analog import bessel_level as design
from irides.filter_signatures.analog import bessel_level as f_sig


# noinspection SpellCheckingInspection,PyPep8Naming
def clean_ndarray_for_precision_comparison(
    array: np.array, precision
) -> np.array:
    """
    Float comparisons are evil. Here I round to extra precision, rebase by
    multiplying by 10^precision, and then cast to int (which is not a
    rounding function).
    """

    # rebase value
    rebase = np.power(10, precision)

    # covert
    return rebase * np.round(array, precision + 1).astype(int)


# noinspection SpellCheckingInspection,PyPep8Naming
class TestBesselLevelSpecific(unittest.TestCase):
    """
    Unit tests for the bessel-level filter are predominantly covered
    by the parameterized tests in test-level-filters. Here, tests that
    are not common to all level filters made available.
    """

    def test_generate_poles_and_zeros_gives_correct_locations(self):
        """
        Simple test to confirm that there are no zeros and that the poles
        are in the right place.
        """

        # defs
        tau = 3.0
        filter_orders = design.get_valid_filter_orders()

        # test setup
        precision = 4

        # iter over orders
        clean = clean_ndarray_for_precision_comparison
        for filter_order in filter_orders:

            design_theo = design.designs(filter_order)
            poles_theo = design_theo["poles"] / tau

            poles_zeros = f_sig.generate_poles_and_zeros(tau, filter_order)
            poles_test = poles_zeros["poles"]
            zeros_test = poles_zeros["zeros"]

            # test no zeros
            self.assertTrue(zeros_test.shape == (0,))

            # test that poles match
            self.assertSequenceEqual(
                list(clean(np.real(poles_test), precision)),
                list(clean(np.real(poles_theo), precision)),
            )
            self.assertSequenceEqual(
                list(clean(np.imag(poles_test), precision)),
                list(clean(np.imag(poles_theo), precision)),
            )
