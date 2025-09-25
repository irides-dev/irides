"""
-------------------------------------------------------------------------------

Unit tests for the .polyema_composite_slope module of spqf.design.filter_signatures.

-------------------------------------------------------------------------------
"""

import unittest

from irides.design.analog import polyema_composite_slope as design
from irides.filter_signatures.analog import polyema_composite_slope as f_sig


# noinspection SpellCheckingInspection,PyPep8Naming
class TestPolyemaCompositeSlopeSpecific(unittest.TestCase):
    """
    Unit tests for the polyema-composite-slope filter are predominantly covered
    by the parameterized tests in test-slope-filters. Here, tests that
    are not common to all slope filters are made available.
    """

    def test_sacf_dc_value_is_approx_unity_for_nondefault_arm_ratio(self):
        """
        """

        # defs
        xi_start = 0.0
        xi_end = 4.0
        dxi = 0.005
        tau = 1.0
        arm_ratio = 1.0 / 3.0

        # fetch filter orders
        filter_orders = design.get_valid_filter_orders()

        # test defs
        precision = 2

        # iter on orders
        for order in filter_orders:

            sacf_norm = f_sig.generate_sacf_correlogram(
                xi_start,
                xi_end,
                dxi,
                tau,
                order,
                arm_ratio=arm_ratio,
                normalize=True,
            )

            kh0_test = sacf_norm[0, 1]

            with self.subTest(msg="order: {0}".format(order)):

                self.assertAlmostEqual(kh0_test, 1.0, precision)
