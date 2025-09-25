"""
-------------------------------------------------------------------------------

Unit tests for the .design.digital.damped_oscillator design code.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.design.digital import damped_oscillator as dsgn_dig_dosc
from irides.design.analog import bessel_level as dsgn_alg_bssl
from irides.tools import analog_to_digital_conversion_tools as a2d_tools


# noinspection SpellCheckingInspection
class TestDigitalDesignDampedOscillator(unittest.TestCase):
    """
    Unit tests against the digital damped-oscillator design component.
    """

    def setUp(self) -> None:
        """Recurring setup for reference poles"""

        # setup
        self.filter_order = 5
        self.splane_poles = dsgn_alg_bssl.designs(self.filter_order)["poles"]
        self.tau_default = 13.0  # nice prime number
        self.zplane_poles = a2d_tools.convert_analog_poles_to_digital_poles(
            self.tau_default, self.splane_poles
        )

    def test_gain_adjustment(self):
        """Tests that the gain adjustment is correct (the same as h[0] test)"""

        # convert to polar coords
        r = np.abs(self.zplane_poles)
        phi = np.arctan2(np.imag(self.zplane_poles), np.real(self.zplane_poles))

        # compute and fetch gain adjustment
        gain_adj_answers = 1.0 - 2.0 * np.cos(phi) * r + r * r
        gain_adj_tests = dsgn_dig_dosc.gain_adjustment(self.zplane_poles)

        # test as a whole
        self.assertAlmostEqual(
            np.linalg.norm(gain_adj_tests - gain_adj_answers), 0, 4
        )

    def test_initial_hn_values(self):
        """Tests that h[0], h[1], and h[2] values are correct"""

        # convert to polar coords
        r = np.abs(self.zplane_poles)
        phi = np.arctan2(np.imag(self.zplane_poles), np.real(self.zplane_poles))

        # compute h[0..2]
        hn_answers = np.zeros((self.zplane_poles.shape[0], 3))
        hn_answers[:, 0] = 1.0 - 2.0 * np.cos(phi) * r + r * r
        hn_answers[:, 1] = 2.0 * np.cos(phi) * r * hn_answers[:, 0]
        hn_answers[:, 2] = (
            (1.0 + 2.0 * np.cos(2.0 * phi)) * r * r * hn_answers[:, 0]
        )

        # fetch test values
        hn_tests = dsgn_dig_dosc.initial_hn_values(self.zplane_poles)

        # iter over
        for k, hk_answers in enumerate(hn_answers.T):

            # test hk values across zplane-poles as a whole
            with self.subTest("h[{0}] test".format(k)):

                self.assertAlmostEqual(
                    np.linalg.norm(hn_tests[:, k] - hk_answers), 0, 4,
                )

    def test_moment_values(self):
        """Tests that the M0, M1, and M2 values match expectation"""

        # pre-calc'd answers, so this is really a regression test
        moments_answers = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [6.17635013, 5.14918192, 5.14918192, 2.35885378, 2.35885378],
                [63.39730154, 39.95176358, 39.95176358, -2.137015, -2.137015],
            ]
        ).T

        moments_test = dsgn_dig_dosc.moment_values(self.zplane_poles)

        self.assertAlmostEqual(
            np.linalg.norm(moments_test - moments_answers), 0.0, 4
        )

    def test_full_width_value(self):
        """Tests that FW values match expectations"""

        # calculate answer
        r = np.abs(self.zplane_poles)
        phi = np.arctan2(np.imag(self.zplane_poles), np.real(self.zplane_poles))

        with np.errstate(invalid="ignore"):

            arg = 2.0 * r * ((1.0 + r * r) * np.cos(phi) - 2.0 * r)
            gain_adj = 1.0 - 2.0 * np.cos(phi) * r + r * r
            fw_answers = 2.0 * np.sqrt(arg) / gain_adj

        # call test values
        fw_test = dsgn_dig_dosc.full_width_value(self.zplane_poles)

        # test one by one, given that nan can be an answer
        for i, fw_answer in enumerate(fw_answers):

            if np.isnan(fw_answer):
                self.assertTrue(np.isnan(fw_test[i]))
            else:
                self.assertAlmostEqual(fw_answer, fw_test[i], 4)

    def test_tau_minimum_for_analog_pole_has_pdr_value_of_one_half(self):
        """Tests that this function produces a correct tau-min value"""

        # setup
        filter_orders = dsgn_alg_bssl.get_valid_filter_orders(strict=True)
        pdr_ans = 0.5

        # iter over order, use the dosc with least damping
        for order in filter_orders:

            # extract a pole from the first dosc stage
            dsgn = dsgn_alg_bssl.designs(order)
            splane_poles = dsgn["poles"][dsgn["stages"][0]["indices"]]

            # calc tau-min
            tau_min = dsgn_dig_dosc.tau_minimum_for_analog_pole(splane_poles)[0]
            zplane_poles = a2d_tools.convert_analog_poles_to_digital_poles(
                tau_min, splane_poles
            )

            with self.subTest("order: {0}".format(order)):

                for zplane_pole in zplane_poles:
                    pdr_test = np.real(zplane_pole)
                    self.assertAlmostEqual(pdr_ans, pdr_test, 4)
