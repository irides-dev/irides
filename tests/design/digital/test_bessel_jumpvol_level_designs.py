"""
-------------------------------------------------------------------------------

Unit tests for the .design.digital.bessel_jumpvol_level design code.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.design.digital import bessel_jumpvol_level as dsgn_d_jv_bssl


# noinspection SpellCheckingInspection
class TestDigitalDesignBesselJumpVol(unittest.TestCase):
    """
    Unit tests against the digital kt-bessel jump-vol design component.
    """

    def test_valid_filter_orders(self):
        """Test that the filter orders are valid"""

        # fetch from design
        valid_orders_test = dsgn_d_jv_bssl.get_valid_filter_orders()

        # setup answer
        valid_orders_ans = np.arange(3, valid_orders_test[-1] + 1)

        # test
        self.assertSequenceEqual(
            list(valid_orders_test), list(valid_orders_ans)
        )

    def test_kij_zero_tau_min_values(self):
        """Tests that the k_I-J[0] values are correct for tau-min"""

        valid_orders = dsgn_d_jv_bssl.get_valid_filter_orders()
        precision = 4

        # answers
        kij0_answers = np.array(
            [
                1.1819,
                1.26417,
                1.40365,
                1.55348,
                1.69833,
                1.83489,
            ]
        )

        # test
        for i, order in enumerate(valid_orders):
            kij0_test = dsgn_d_jv_bssl.designs(order)["kij_zero_at_tau_min"]
            self.assertAlmostEqual(kij0_test, kij0_answers[i], precision)

    def test_kij_zero_tau_inf_values(self):
        """Tests that the k_I-J[0] values are correct for tau-inf"""

        valid_orders = dsgn_d_jv_bssl.get_valid_filter_orders()
        precision = 4

        # answers
        kij0_answers = np.array(
            [
                0.942596,
                1.10617,
                1.26084,
                1.40435,
                1.5373,
                1.66093,
            ]
        )

        # test
        for i, order in enumerate(valid_orders):
            kij0_test = dsgn_d_jv_bssl.designs(order)["kij_zero_at_tau_inf"]
            self.assertAlmostEqual(kij0_test, kij0_answers[i], precision)
