"""
-------------------------------------------------------------------------------

Parameterized unit tests for inline curvature-filter signatures.

-------------------------------------------------------------------------------
"""

import parameterized

import unittest

from irides.design.digital import polyema_inline_curvature as design_pema
from irides.design.digital import bessel_inline_curvature as design_bssl
from irides.design.digital import mbox_inline_curvature as design_mbox

from irides.filter_signatures.digital import \
    polyema_inline_curvature as f_sig_pema
from irides.filter_signatures.digital import bessel_inline_curvature as f_sig_bssl
from irides.filter_signatures.digital import mbox_inline_curvature as f_sig_mbox

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
class TestInlineCurvatureFilters(unittest.TestCase):
    """
    Parameterized, common unit tests for inline curvature filters.
    """

    def test_impulse_response_moments_for_all_valid_orders(self):
        """Tests that the moments M0, M1, M2 are correct"""

        precisions = {
            "bessel": {"M0": 4, "M1": 4, "M2": 4},
            "pema": {"M0": 4, "M1": 4, "M2": 4},
            "mbox": {"M0": 4, "M1": 4, "M2": 4},
        }[self.f_type]

        common_tooling.common_impulse_response_moment_tests(
            self, precisions, proportional=False
        )

    def test_impulse_response_initial_values_are_correct_for_all_orders(self):
        """Tests that h[0], h[1], and h[2] are correct"""

        precisions = {
            "bessel": {"hk": 3},
            "pema": {"hk": 3},
            "mbox": {"hk": 3},
        }[self.f_type]

        common_tooling.common_impulse_response_initial_values_tests(
            self, precisions
        )
