"""
-------------------------------------------------------------------------------

Parameterized unit tests for inline-slope filter signatures.

-------------------------------------------------------------------------------
"""

import parameterized

import unittest
import numpy as np

from irides.design.analog import bessel_inline_slope as design_bessel
from irides.design.analog import polyema_inline_slope as design_pema
from irides.design.analog import mbox_inline_slope as design_mbox
from irides.design.analog import polyema_composite_slope as design_pema_comp

from irides.filter_signatures.analog import polyema_inline_slope as f_sig_pema
from irides.filter_signatures.analog import bessel_inline_slope as f_sig_bessel
from irides.filter_signatures.analog import mbox_inline_slope as f_sig_mbox
from irides.filter_signatures.analog import \
    polyema_composite_slope as f_sig_pema_comp

from irides.tools import transfer_function_tools as xfer_tools

from tests.filter_signatures.analog import common_tooling


# noinspection SpellCheckingInspection,PyPep8Naming
# setup to iter over filter types
@parameterized.parameterized_class(
    ("f_type", "design", "f_sig"),
    [
        ("pema", design_pema, f_sig_pema),
        ("bessel", design_bessel, f_sig_bessel),
        ("mbox", design_mbox, f_sig_mbox),
        ("pema-comp", design_pema_comp, f_sig_pema_comp),
    ],
)
class TestInlineSlopeFilters(unittest.TestCase):
    """
    Parameterized, common unit tests for standard level filters.
    """

    def test_impulse_response_moments_for_all_valid_orders(self):
        """
        Validate that discretized continuous-time impulse responses
        have the expected 0th, 1st and 2nd moments.
        """

        precisions = {
            "pema": {"M0": 4, "M1": 4, "M2": 3},
            "bessel": {"M0": 4, "M1": 3, "M2": 2},
            "mbox": {"M0": 3, "M1": 4, "M2": 3},
            "pema-comp": {"M0": 4, "M1": 4, "M2": 3},
        }[self.f_type]

        common_tooling.common_impulse_response_moment_tests(self, precisions)

    def test_impulse_response_is_correct_at_t0(self):
        """
        Test that h_m(t=0) is correct across valid filter orders.
        """

        # params
        params = {
            "pema": {"t_start": -0.1, "t_end": 0.1},
            "bessel": {"t_start": -0.1, "t_end": 0.1},
            "mbox": {"t_start": -0.1, "t_end": 2.0},
            "pema-comp": {"t_start": -0.1, "t_end": 0.1},
        }

        # defs
        filter_orders = self.design.get_valid_filter_orders(strict=False)
        tau = 2.0
        t_start = params[self.f_type]["t_start"]
        t_end = params[self.f_type]["t_end"] * tau
        dt = 0.001 * tau

        # compute and test
        for order in filter_orders:

            # theo value
            h_t0_theo = self.design.impulse_response_t0_value(tau, order, dt)

            # test value
            ts: np.ndarray = self.f_sig.generate_impulse_response(
                t_start, t_end, dt, tau, order
            )
            i_zero = np.where(0.0 - dt / 2 < ts[:, 0])[0][0]
            h_t0_test = ts[i_zero, 1]

            with self.subTest(msg="order: {0}".format(order)):

                self.assertAlmostEqual(h_t0_test, h_t0_theo, 4)

                # if exists t < 0 points, then test that the preceding
                # point is zero
                if i_zero > 0:
                    h_t0_minus = ts[i_zero - 1, 1]
                    self.assertAlmostEqual(h_t0_minus, 0.0, 4)

    def test_impulse_response_has_correct_approx_zero_crossing_time(self):
        """
        Test that h_m(tx) = 0 for valid orders.
        """

        # defs
        filter_orders = self.design.get_valid_filter_orders(strict=False)
        tau = 0.5
        dt = 0.001 * tau
        t_start = 0.0
        t_end = 2.0 * tau + dt

        # compute and test
        for order in filter_orders:

            # theo value
            tx_theo = self.design.zero_crossing_time(tau, order)

            # test value
            ts: np.ndarray = self.f_sig.generate_impulse_response(
                t_start, t_end, dt, tau, order
            )
            t_axis = ts[:, 0]
            h_slope = ts[:, 1]

            # bound on interval
            # first, find t > 0
            i_t0 = np.min(np.argwhere(t_axis > 0.0))

            # find the last point before the first sign change
            # along the principal positive lobe.
            i_end = h_slope.shape[0]
            i_last_pos = i_t0
            while i_last_pos < i_end and h_slope[i_last_pos + 1] >= 0.0:
                i_last_pos += 1
            i_first_neg = i_last_pos + 1

            tx_leq = t_axis[i_last_pos]
            tx_geq = t_axis[i_first_neg]
            h_geq = h_slope[i_last_pos]
            h_lt = h_slope[i_first_neg]

            with self.subTest(msg="order: {0}".format(order)):

                # confirm that the crossing is bound
                self.assertTrue(h_geq >= 0.0)
                self.assertTrue(h_lt < 0.0)

                # confirm crossing time
                self.assertTrue(tx_leq <= tx_theo <= tx_geq)

    def test_wireframe_moments_for_all_valid_orders(self):
        """
        Validate that discretized continuous-time wireframes
        have the expected 0th, 1st and 2nd moments.
        """

        precisions = {
            "pema": {"M0": 4, "M1": 4, "M2": 3},
            "bessel": {"M0": 4, "M1": 4, "M2": 3},
            "mbox": {"M0": 4, "M1": 4, "M2": 4},
            "pema-comp": {"M0": 4, "M1": 4, "M2": 3},
        }[self.f_type]

        common_tooling.common_impulse_response_moment_tests(
            self, precisions, proportional=False, use_wireframe_signature=True
        )

    def test_sacf_has_correct_unnormalized_peak_values(self):
        """
        Tests that the kh(0) calculated from an impulse response is close
        to the theoretic design value.
        """

        # params
        params = {
            "pema": {"dt": 0.001, "precision": 4},
            "bessel": {"dt": 0.0005, "precision": 2},
            "mbox": {"dt": 0.0002, "precision": 4},
            "pema-comp": {"dt": 0.001, "precision": 4},
        }

        # defs
        filter_orders = self.design.get_valid_filter_orders(strict=False)
        t_end = 5.0
        tau = 1.0
        dt = params[self.f_type]["dt"]

        # test
        precision = params[self.f_type]["precision"]

        # prepare precision exception
        exception_type = self.f_type == "bessel"

        for i, order in enumerate(filter_orders):

            # prepare precision exception
            exception_order = order == 2

            # fetch design points with dt correction
            kh0_theo = self.design.autocorrelation_peak_and_stride_values(
                tau, order, dt
            )["kh0"]

            # fetch sacf, take peak value
            sacf = self.f_sig.generate_sacf_correlogram(
                -t_end, t_end, dt, tau, order
            )
            kh0_test = max(sacf[:, 1])

            # test
            with self.subTest(msg="order: {0}".format(order)):

                prec_adj = -1 if (exception_type and exception_order) else 0
                self.assertAlmostEqual(
                    kh0_test / kh0_theo, 1.0, precision + prec_adj
                )

    def test_sacf_has_approx_correct_5pct_lag_values(self):
        """
        Tests that the xi_5pct lag calculated from an impulse response is
        close to the theoretic design value.

        The strategy steps are
            *) isolate the positive-xi side of the sacf
            *) find the minimum sacf value
            *) find the first sacf value about -5% beyond the minimum

             ^ sacf[xi, 1]
             |.
             | .
            -|--.------------------------------------------> xi
                  .                      .       .      <---- -5% level
                     .           .       ^
                           .             |
                           ^           1st val above -5%
                           | min

             |--> pos-xi
        """

        # params
        params = {
            "pema": {"t_end_coef": 5.0, "dt_v_coef": 0.0005, "precision": 3},
            "bessel": {"t_end_coef": 5.0, "dt_v_coef": 0.0005, "precision": 2},
            "mbox": {"t_end_coef": 2.0, "dt_v_coef": 0.0002, "precision": 3},
            "pema-comp": {
                "t_end_coef": 5.0,
                "dt_v_coef": 0.0005,
                "precision": 3,
            },
        }

        # defs
        filter_orders = self.design.get_valid_filter_orders(strict=False)
        tau = 1.0

        t_end_coeff = params[self.f_type]["t_end_coef"]
        t_end_v = t_end_coeff * np.ones(filter_orders.shape[0])
        t_end_v[0] = 2.0 * t_end_coeff

        dt_v_coef = params[self.f_type]["dt_v_coef"]
        dt_v = dt_v_coef * np.ones_like(t_end_v)
        dt_v[0] = dt_v_coef / 2.0

        # test
        precision = params[self.f_type]["precision"]

        for i, order in enumerate(filter_orders):

            # fetch design points
            xi_5pct_theo = self.design.autocorrelation_peak_and_stride_values(
                tau, order
            )["xi_5pct"]

            # fetch sacf, take peak value
            t_end = t_end_v[i]
            dt = dt_v[i]
            sacf = self.f_sig.generate_sacf_correlogram(
                -t_end, t_end, dt, tau, order, normalize=True
            )

            # infer dxi from sacf
            dxi = min(np.diff(sacf[:, 0]))

            # isolate the positive-xi side
            i_zero = np.where(0.0 - dxi / 2 < sacf[:, 0])[0][0]
            sacf_posxi = sacf[i_zero:, :]

            # find minimum sacf on the positive side of xi
            i_posxi_min = np.argmin(sacf_posxi[:, 1])
            sacf_segment = sacf_posxi[i_posxi_min:, :]

            # find 1st xi where sacf_segment[:, 1] >= -5%
            i_5pct = np.where(sacf_segment[:, 1] >= -0.05)[0][0]

            # manage xi discretization
            xi_5pct_cands = sacf_segment[i_5pct - 1 : i_5pct + 1, 0]
            errs = np.abs(xi_5pct_cands - xi_5pct_theo)
            xi_5pct_test = xi_5pct_cands[np.argmin(errs)]

            with self.subTest(msg="order: {0}".format(order)):
                self.assertAlmostEqual(xi_5pct_test, xi_5pct_theo, precision)

    def test_sscf_has_approx_correct_5pct_stride_values(self):
        """
        Tests that the sti-5pct strides calculated from mathematica
        approximately match those calculated from the transfer functions in
        this library.

        Because of the wide range of scale-decorrelation strides, these
        tests fetch the theorectic strides from the design files (and assoc'd
        residuals), and then calculates sscf on a domain that surrounds the
        expected stride.
        """

        # params
        precision = {"pema": 3, "bessel": 2, "mbox": 4, "pema-comp": 3}[
            self.f_type
        ]

        common_tooling.common_sscf_decorrelation_stride_test(self, precision)

    def test_generate_transfer_function_has_correct_size_and_type(self):
        """
        The transfer-function panel is testable for gain, phase and group
        delay, but not very well by itself. Here, only shape and type checks
        are tested.
        """

        common_tooling.common_transfer_function_has_correct_size_and_type(self)

    def test_generate_spectra_for_correct_gain_at_dc_and_peak(self):
        """
        Test that the DC and cutoff gain is correct for all orders.
        """

        precision = {"pema": 4, "bessel": 2, "mbox": 2, "pema-comp": 4}[
            self.f_type
        ]

        # noinspection PyTypeChecker
        common_tooling.common_gain_phase_group_delay_at_dc_test(
            self,
            self.design.gain_at_dc,
            xfer_tools.SpectraColumns.GAIN.value,
            "gain",
        )

        # noinspection PyTypeChecker
        common_tooling.common_gain_phase_group_delay_at_feature_freq_test(
            self,
            self.design.frequency_of_peak_gain,
            self.design.gain_at_peak_frequency,
            xfer_tools.SpectraColumns.GAIN.value,
            precision,
            "gain",
        )

    def test_generate_spectra_for_correct_phase_at_dc_and_peak(self):
        """
        Test that the DC and cutoff phase is correct for all orders.
        """

        precision = {"pema": 4, "bessel": 2, "mbox": 4, "pema-comp": 4}[
            self.f_type
        ]

        # noinspection PyTypeChecker
        common_tooling.common_gain_phase_group_delay_at_dc_test(
            self,
            self.design.phase_at_dc,
            xfer_tools.SpectraColumns.PHASE.value,
            "phase",
        )

        # noinspection PyTypeChecker
        common_tooling.common_gain_phase_group_delay_at_feature_freq_test(
            self,
            self.design.frequency_of_peak_gain,
            self.design.phase_at_peak_frequency,
            xfer_tools.SpectraColumns.PHASE.value,
            precision,
            "phase",
        )

    def test_generate_spectra_for_correct_group_delay_at_dc_and_peak(self):
        """
        Test that the DC and cutoff group delay is correct for all orders.
        """

        precision = {"pema": 4, "bessel": 2, "mbox": 4, "pema-comp": 4}[
            self.f_type
        ]

        # noinspection PyTypeChecker
        common_tooling.common_gain_phase_group_delay_at_dc_test(
            self,
            self.design.group_delay_at_dc,
            xfer_tools.SpectraColumns.GROUPDELAY.value,
            "group-delay",
        )

        # noinspection PyTypeChecker
        common_tooling.common_gain_phase_group_delay_at_feature_freq_test(
            self,
            self.design.frequency_of_peak_gain,
            self.design.group_delay_at_peak_frequency,
            xfer_tools.SpectraColumns.GROUPDELAY.value,
            precision,
            "group-delay",
        )

    def test_callable_transfer_function_has_correct_kh0_value(self):
        """
        Tests that the kh(0) calculated from the integral

          kh(0) = int_-inf^inf H(j 2 pi f) H(-j 2 pi f) df

        matches the design value.
        """

        # params
        params = {
            "pema": {"precision": 4},
            "bessel": {"precision": 2},
            "mbox": {"precision": 3},
            "pema-comp": {"precision": 4},
        }

        precision = params[self.f_type]["precision"]
        common_tooling.common_callable_xfer_fcxn_has_correct_kh0_value(
            self, precision
        )
