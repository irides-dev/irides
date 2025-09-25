"""
-------------------------------------------------------------------------------

Unit tests for the .polyema_inline_slope module of spqf.design.analog.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.design.analog import polyema_inline_slope as design


# noinspection SpellCheckingInspection,PyPep8Naming
class TestDesignPolyemaInlineSlopeSpecific(unittest.TestCase):
    """
    Unit tests for the src.spqf.design.analog.polyema_inline_slope module.
    """

    def test_get_minimum_valid_filter_order_is_simply_one(self):
        """Test that the minimum order is 1."""

        # note: need to address the strictness issue
        min_order_theo = {True: 3, False: 2}
        for strict in [True, False]:
            self.assertEqual(
                design.get_minimum_valid_filter_order(strict),
                min_order_theo[strict],
            )

    def test_integer_sequence_a000169_matches_first_eight_values(self):
        """
        Test that the integer sequence gives expected results on [1, 8].
        """

        filter_orders = design.get_valid_filter_orders()

        # setup theo
        seq_theo = np.array([2, 9, 64, 625, 7776, 117649, 2097152], dtype=int)

        # setup test
        seq_test = np.zeros_like(seq_theo, dtype=int)
        for i, m in enumerate(filter_orders):
            seq_test[i] = design.integer_sequence_a000169(m)

        # test
        self.assertSequenceEqual(list(seq_test), list(seq_theo))

    def test_integer_sequence_a063170_matches_first_eight_values(self):
        """
        Test that the integer sequence gives expected results on [1, 8].
        """

        filter_orders = design.get_valid_filter_orders()

        # setup theo
        seq_theo = np.array([2, 10, 78, 824, 10970, 176112, 3309110], dtype=int)

        # setup test
        seq_test = np.zeros_like(seq_theo, dtype=int)
        for i, m in enumerate(filter_orders):
            seq_test[i] = design.integer_sequence_a063170(m)

        # test
        self.assertSequenceEqual(list(seq_test), list(seq_theo))

    def test_autocorrelation_peak_value_discretizaiton_corrections(self):
        """
        Test that the kh0 correction for temporal discretization is correct.
        """

        # defs
        filter_orders = design.get_valid_filter_orders()
        dt = 0.001
        tau = 2.0

        # theo
        kh0_adjust_theo = np.zeros(filter_orders.shape[0])
        kh0_adjust_theo[0] = 8.0 * dt / np.power(tau, 4)

        precision = 4

        # iter over filters
        kh0_adjust_test = np.zeros_like(kh0_adjust_theo)
        for i, order in enumerate(filter_orders):

            kh0_adjust_test[
                i
            ] = design.autocorrelation_peak_value_discretization_correction(
                tau, order, dt
            )

            with self.subTest(msg="order: {0}".format(order)):
                if order == 1:
                    self.assertTrue(np.isnan(kh0_adjust_test[i]))
                else:
                    self.assertAlmostEqual(
                        kh0_adjust_test[i], kh0_adjust_theo[i], precision
                    )

    def test_impulse_response_t0_value_for_order_equal_one(self):
        """Tests the special case for m=1, which is analytic for pemas"""

        # defs
        tau = 1.0
        order = 1
        dt = 0.001

        # theo val
        val_theo = 1.0 / (dt * tau) - 1.0 / np.power(tau, 2)

        # test val
        val_test = design.impulse_response_t0_value(tau, order, dt)

        # test
        precision = 4
        self.assertAlmostEqual(val_test, val_theo, precision)

    def test_moment_value_generator_for_order_equal_one(self):
        """Tests the special case for m=1, which is analytic for pemas"""

        # defs
        tau = 1.0
        order = 1
        moment = 0
        dt = 0.001

        # theo value
        m0_theo = -dt / (2.0 * np.power(tau, 2))

        # test value
        m0_gen = design.moment_value_generator(moment, tau, dt)
        m0_test = m0_gen(order)

        # test
        precision = 4
        self.assertAlmostEqual(m0_test, m0_theo, precision)

    def test_autocorrelation_peak_and_stride_values_for_order_equal_one(self):
        """Tests the special case for m=1, which produces np.nans."""

        # defs
        tau = 1.0
        order = 1
        dt = 0.001

        # test value
        vals_test = design.autocorrelation_peak_and_stride_values(
            tau, order, dt
        )

        # test
        for k, v in vals_test.items():

            self.assertTrue(np.isnan(v))
