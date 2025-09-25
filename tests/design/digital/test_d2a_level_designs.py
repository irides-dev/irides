"""
-------------------------------------------------------------------------------

Unit tests for the .design.digital.{mbox|pema|bssL} design code that use
digital-to-analog conversions.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.design.digital import bessel_level as dsgn_d_l_bssl
from irides.design.digital import polyema_level as dsgn_d_l_pema
from irides.design.digital import mbox_level as dsgn_d_l_mbox

from irides.tools import analog_to_digital_conversion_tools as a2d_tools


# noinspection SpellCheckingInspection
class TestD2AConversionsForDigitalLevelFilters(unittest.TestCase):
    """
    Unit tests against the digital designs tied to d2a conversion.
    """

    def setUp(self):
        """Capture recurring defs"""

        # setup
        self.mu = 20
        self.precision = 4

    def test_mbox_tau_values_from_mu(self):
        """Tests that tau == mu (for the mbox)"""

        # setup
        filter_orders = dsgn_d_l_mbox.get_valid_filter_orders()
        tau_answers = self.mu * np.ones_like(filter_orders)

        for i, filter_order in enumerate(filter_orders):
            tau_test = dsgn_d_l_mbox.analog_tau_value(self.mu, filter_order)

            with self.subTest(f"order: {filter_order}"):
                self.assertAlmostEqual(tau_answers[i], tau_test, self.precision)

    def test_pema_tau_values_from_mu(self):
        """Tests tau values against a known mu value across filter orders"""

        # setup
        filter_orders = dsgn_d_l_pema.get_valid_filter_orders()

        # this is basically a regression test given these pre-computed answers
        tau_answers = np.array(
            [
                20.495934,
                20.984117,
                21.465071,
                21.939260,
                22.407101,
                22.868968,
                23.325201,
                23.776107,
            ]
        )

        # print("pema:")
        for i, filter_order in enumerate(filter_orders):
            tau_test = dsgn_d_l_pema.analog_tau_value(self.mu, filter_order)
            # print(f"i: {i}, tau: {tau_test:-.6f}")

            with self.subTest(f"order: {filter_order}"):
                self.assertAlmostEqual(tau_answers[i], tau_test, self.precision)

    def test_bessel_tau_values_from_mu(self):
        """Tests tau values against a known mu value across filter orders"""

        # setup
        filter_orders = dsgn_d_l_bssl.get_valid_filter_orders()

        # this is basically a regression test given these pre-computed answers
        tau_answers = np.array(
            [
                20.495934,
                20.988088,
                21.476718,
                21.962051,
                22.444296,
                22.923638,
                23.400248,
                23.874281,
            ]
        )

        # print("bssl:")
        for i, filter_order in enumerate(filter_orders):
            tau_test = dsgn_d_l_bssl.analog_tau_value(self.mu, filter_order)
            # print(f"i: {i}, tau: {tau_test:-.6f}")

            with self.subTest(f"order: {filter_order}"):
                self.assertAlmostEqual(tau_answers[i], tau_test, self.precision)
