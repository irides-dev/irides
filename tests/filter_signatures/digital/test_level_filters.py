"""
-------------------------------------------------------------------------------

Parameterized unit tests for level-filter signatures.

-------------------------------------------------------------------------------
"""

import parameterized

import unittest

from irides.design.digital import polyema_level as design_pema
from irides.design.digital import bessel_level as design_bssl
from irides.design.digital import mbox_level as design_mbox

from irides.filter_signatures.digital import polyema_level as f_sig_pema
from irides.filter_signatures.digital import bessel_level as f_sig_bssl
from irides.filter_signatures.digital import mbox_level as f_sig_mbox

from irides.tools import impulse_response_tools as ir_tools
from tests.filter_signatures.digital import common_tooling


# noinspection SpellCheckingInspection,PyPep8Naming
# setup to iter over filter types
@parameterized.parameterized_class(
    ("f_type", "design", "f_sig"),
    [
        ("pema", design_pema, f_sig_pema),
        ("bessel", design_bssl, f_sig_bssl),
        ("mbox", design_mbox, f_sig_mbox),
    ],
)
class TestLevelFilters(unittest.TestCase):
    """
    Parameterized, common unit tests for standard level filters.
    """

    def test_impulse_response_moments_for_all_valid_orders(self):
        """Tests that the moments M0, M1, M2 are correct"""

        precisions = {
            "bessel": {"M0": 4, "M1": 4, "M2": 4},
            "pema": {"M0": 4, "M1": 4, "M2": 4},
            "mbox": {"M0": 4, "M1": 4, "M2": 4},
        }[self.f_type]

        common_tooling.common_impulse_response_moment_tests(
            self, precisions, proportional=True
        )

    def test_impulse_response_initial_values_are_correct_for_all_orders(self):
        """Tests that h[0], h[1], and h[2] are correct"""

        precisions = {
            "bessel": {"hk": 4},
            "pema": {"hk": 4},
            "mbox": {"hk": 4},
        }[self.f_type]

        common_tooling.common_impulse_response_initial_values_tests(
            self, precisions
        )

    def test_impulse_response_initial_value_and_gain_adjustment_match(self):
        """Tests that h[0] = gain-adj for these level filters"""

        # filter orders and moments
        filter_orders = self.design.get_valid_filter_orders(strict=False)

        # iter over filter orders
        precision = 3
        for i, filter_order in enumerate(filter_orders):

            # make impulse response
            ds, mu_trg = common_tooling.make_impulse_response(
                self, filter_order
            )

            # gain adjustment
            gain_adj_design = self.design.gain_adjustment(mu_trg, filter_order)
            gain_adj_test = ds.v_axis[0]

            # tests
            with self.subTest(
                msg="order: {0}, gain adjustment".format(filter_order)
            ):

                self.assertAlmostEqual(
                    gain_adj_test / gain_adj_design, 1.0, precision
                )

    def test_impulse_response_full_widths_are_correct(self):
        """Tests that FW values are correct for these level filters"""

        # filter orders and moments
        filter_orders = self.design.get_valid_filter_orders(strict=False)

        # iter over filter orders
        precision = 3
        for i, filter_order in enumerate(filter_orders):

            # make impulse response
            ds, mu_trg = common_tooling.make_impulse_response(
                self, filter_order
            )

            # full width
            fw_test = ir_tools.calculate_impulse_response_full_width(
                ds.to_ndarray()
            )
            fw_design = self.design.full_width_value(mu_trg, filter_order)

            with self.subTest(msg="order: {0}, FW test".format(filter_order)):

                self.assertAlmostEqual(fw_test / fw_design, 1.0, precision)
