"""
-------------------------------------------------------------------------------

Unit tests for the .polyema_composite_slope module of spqf.design.analog.

-------------------------------------------------------------------------------
"""

import unittest

from irides.design.analog import polyema_composite_slope as design


# noinspection SpellCheckingInspection,PyPep8Naming
class TestDesignPolyemaCompositeSlopeSpecific(unittest.TestCase):
    """
    Unit tests for the src.spqf.design.analog.polyema_composite_slope module.
    """

    def test_get_minimum_valid_filter_order_is_simply_one(self):
        """Test that the minimum order is 1."""

        # note: unclear what strictness means for a comp-slope filter
        min_order_theo = {True: 1, False: 1}
        for strict in [True, False]:
            self.assertEqual(
                design.get_minimum_valid_filter_order(strict),
                min_order_theo[strict],
            )

    def test_convert_armratio_order_params_to_tp_tm_params(self):
        """Test the (alpha, m) -> (tp, tm) conversion."""

        # defs
        arm_ratio = 1.0 / 3.0
        tau = 2.0
        filter_order = 3

        # theo values
        tp_theo = 3.0 / 4.0
        tm_theo = 9.0 / 4.0

        # test values
        tp_test, tm_test = design.convert_armratio_order_params_to_tp_tm_params(
            arm_ratio, tau, filter_order
        )

        precision = 4
        self.assertAlmostEqual(tp_test, tp_theo, precision)
        self.assertAlmostEqual(tm_test, tm_theo, precision)

    def test_convert_tp_tm_params_to_armratio_order_params(self):
        """Test that (tp, tm) -> (alpha, m) conversion."""

        # defs
        tau = 2.0
        tp = 3.0 / 4.0
        tm = 9.0 / 4.0

        # theo values
        arm_ratio_theo = 1.0 / 3.0
        filter_order_theo = 3

        # test values
        # fmt: off
        arm_ratio_test, filter_order_test = design. \
            convert_tp_tm_params_to_armratio_order_params(
                tp, tm, tau
        )
        # fmt: on

        precision = 4
        self.assertAlmostEqual(arm_ratio_test, arm_ratio_theo, precision)
        self.assertAlmostEqual(filter_order_test, filter_order_theo, precision)
