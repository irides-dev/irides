"""
-------------------------------------------------------------------------------

Unit tests for the .impulse_response_tools module of spqf.tools.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.tools import impulse_response_tools
from irides.design.analog import mbox_level as mbox_design


# noinspection SpellCheckingInspection,PyPep8Naming
def make_box_impulse_response(dt: float, T=1.0):
    """
    Make a simple box impulse response.
    """

    time_series = np.zeros((np.arange(0.0, T, dt).shape[0], 2))
    time_series[:, 0] = np.arange(0.0, T, dt)
    time_series[:, 1] = 1.0

    time_series[:, 1] /= sum(time_series[:, 1]) * dt

    return time_series


# noinspection SpellCheckingInspection,PyPep8Naming
class TestImpulseResponseTools(unittest.TestCase):
    """
    Unit tests to verify the correct operation of the impulse-response tools
    under src.spqf.tools.impulse_response_tools.
    """

    def test_calculate_impulse_response_moment_is_correct_for_a_box(self):
        """
        Validate that 0th, 1st and 2nd moments are correct for the sample box
        impulse response in this test file.

        The discrete-time corrections are because the h_box weight at t = T
        is zero, as a discretization of u(t) - u(t - T) shows.
        """

        # setup array of moments
        moments = np.array([0, 1, 2])

        # make box impulse response
        dt = 0.001
        impulse_response = make_box_impulse_response(dt)

        # theoretical moments for a box with T = 1
        moment_ans = np.array(
            [
                1.0,
                1.0 / 2.0 - dt / 2,  # discrete-time correction
                1.0 / 3.0 - dt / 2,  # discrete-time correction
            ]
        )

        # test values
        for i, moment in enumerate(moments):
            moment_test = impulse_response_tools.calculate_impulse_response_moment(
                impulse_response, moment
            )

            # test
            self.assertAlmostEqual(moment_ans[i], moment_test, 4)

    def test_calculate_impulse_response_full_width_is_correct_for_a_box(self):
        """
        Test that the full width is correct, where

            FW = 2 T sqrt(M2 - M1^2).

        Given that M1 = 1/2 - dt/2 and M2 = 1/3 - dt/2, we have

            FW = sqrt(1/3 - dt^2).
        """

        # make box impulse response
        dt = 0.001
        impulse_response = make_box_impulse_response(dt)

        # theoretic value
        fw_ans = np.sqrt(1.0 / 3 - np.power(dt, 2))

        # test value
        fw_test = impulse_response_tools.calculate_impulse_response_full_width(
            impulse_response
        )

        # test
        self.assertAlmostEqual(fw_ans, fw_test, 4)

    def test_calculate_auto_correlation_function_is_correct_for_a_box(self):
        """
        Test that sacf for a box has the correct properties.
        """

        # make box impulse response
        dt = 0.001
        impulse_response = make_box_impulse_response(dt)

        # theoretic properties
        sacf_integral = 1.0  # integrated value of sacf spectrum
        kh0_theo = 1.0  # zero-lag sacf value is 1.0
        xi_5pct = 0.95  # expect 5% residual acf at this lag
        kh_5pct = 0.05  # the expect 5%

        # run sacf
        sacf = impulse_response_tools.calculate_auto_correlation_function(
            impulse_response
        )

        # tests
        # integrated value
        self.assertAlmostEqual(sum(sacf[:, 1]) * dt, sacf_integral, 4)

        # zero-lag value
        i_zero = np.where(sacf[:, 0] >= 0.0 - dt / 2.0)[0][0]
        self.assertAlmostEqual(sacf[i_zero, 1], kh0_theo, 4)

        # residual acf
        i_5pct = i_zero + np.where(sacf[i_zero:, 0] >= xi_5pct - dt / 2.0)[0][0]
        self.assertAlmostEqual(sacf[i_5pct, 1], kh_5pct, 4)

    def test_get_valid_moments_is_correct(self):
        """Validate that moments 0, 1 and 2 are captured as np array."""

        moments_theo = np.arange(0, 3)
        moments_test = impulse_response_tools.get_valid_moments()

        self.assertSequenceEqual(moments_test.tolist(), moments_theo.tolist())

    def test_validate_filter_order_or_die_returns_for_valid_filter_order(self):
        """
        Tests that validate_filter_order_or_die() returns when the filter
        order is valid for a given design.
        """

        with self.subTest(msg="design: mbox"):
            valid_filter_orders = mbox_design.get_valid_filter_orders()

            for valid_order in valid_filter_orders:
                impulse_response_tools.validate_filter_order_or_die(
                    mbox_design, valid_order
                )
                self.assertTrue(True)

    def test_validate_filter_order_or_die_throws_for_invalid_filter_order(self):
        """
        Tests that validate_filter_order_or_die() dies when the filter
        order is invalid for a given design.
        """

        with self.subTest(msg="design: mbox"):
            valid_filter_orders = mbox_design.get_valid_filter_orders()
            invalid_filter_order = max(valid_filter_orders) + 1

            self.assertRaises(
                IndexError,
                lambda: impulse_response_tools.validate_filter_order_or_die(
                    mbox_design, invalid_filter_order
                ),
            )
