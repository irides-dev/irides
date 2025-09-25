"""
-------------------------------------------------------------------------------

Unit tests for the .design.digital.polyema_jumpvol_level design code.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.design.digital import polyema_jumpvol_level as dsgn_d_jv_pema


# noinspection SpellCheckingInspection
class TestDigitalDesignPemaJumpVol(unittest.TestCase):
    """
    Unit tests against the digital pema jump-vol design component.
    """

    def test_valid_filter_orders(self):
        """Test that the filter orders are valid"""

        # fetch from design
        valid_orders_test = dsgn_d_jv_pema.get_valid_filter_orders()

        # setup answer
        valid_orders_ans = np.arange(3, valid_orders_test[-1] + 1)

        # test
        self.assertSequenceEqual(
            list(valid_orders_test), list(valid_orders_ans)
        )

    def test_kij_zero_tau_min_values(self):
        """Tests that the k_I-J[0] values are correct for tau-min"""

        valid_orders = dsgn_d_jv_pema.get_valid_filter_orders()
        precision = 4

        # answers
        kij0_answers = np.array(
            [
                1.06966,
                1.06752,
                1.13212,
                1.2103,
                1.28814,
                1.36295,
            ]
        )

        # test
        for i, order in enumerate(valid_orders):
            kij0_test = dsgn_d_jv_pema.designs(order)["kij_zero_at_tau_min"]
            self.assertAlmostEqual(kij0_test, kij0_answers[i], precision)

    def test_kij_zero_tau_inf_values(self):
        """Tests that the k_I-J[0] values are correct for tau-inf"""

        valid_orders = dsgn_d_jv_pema.get_valid_filter_orders()
        precision = 4

        # answers
        kij0_answers = np.array(
            [
                0.883706,
                0.981931,
                1.07404,
                1.16004,
                1.24068,
                1.31675,
            ]
        )

        # test
        for i, order in enumerate(valid_orders):
            kij0_test = dsgn_d_jv_pema.designs(order)["kij_zero_at_tau_inf"]
            self.assertAlmostEqual(kij0_test, kij0_answers[i], precision)
