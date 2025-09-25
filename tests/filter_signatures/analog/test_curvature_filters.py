"""
-------------------------------------------------------------------------------

Parameterized unit tests for inline-curvature filter signatures.

-------------------------------------------------------------------------------
"""

import parameterized

import unittest
import numpy as np

from irides.design.analog import bessel_inline_curvature as design_bessel
from irides.design.analog import polyema_inline_curvature as design_pema
from irides.design.analog import mbox_inline_curvature as design_mbox

from irides.filter_signatures.analog import polyema_inline_curvature as f_sig_pema
from irides.filter_signatures.analog import \
    bessel_inline_curvature as f_sig_bessel
from irides.filter_signatures.analog import mbox_inline_curvature as f_sig_mbox

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
    ],
)
class TestInlineCurvatureFilters(unittest.TestCase):
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

    def test_impulse_response_has_correct_crossing_times_and_imbalance(self):
        """
        Tests that the first and second principal crossing times are correct
        to within dt discretization and result precision, and that the lobe
        imbalance is correct to within precision.
        """

        # params
        params = {
            "pema": {"dt": 0.0005, "precision": 3},
            "bessel": {"dt": 0.0005, "precision": 2},
            "mbox": {"dt": 0.0005, "precision": 3},
        }

        # defs
        filter_orders = self.design.get_valid_filter_orders(strict=False)
        tau = 1.0
        dt = params[self.f_type]["dt"] * tau
        t_start = 0.0
        t_end = 10.0 * tau + dt

        # compute and test
        for order in filter_orders:

            # theo values
            res = self.design.impulse_response_lobe_details(tau, order)
            txl_theo = res["tx1"]
            txr_theo = res["tx2"]
            imb_theo = res["imbalance"]

            # fetch impulse response
            ts: np.ndarray = self.f_sig.generate_impulse_response(
                t_start, t_end, dt, tau, order
            )
            t_axis = ts[:, 0]
            h = ts[:, 1]

            # find crossing points
            i = 0
            i_cand = np.zeros(2, dtype=int)
            crossed_once = False
            crossed_twice = False
            while True:
                i += 1
                if not crossed_once and h[i - 1] >= 0 > h[i]:
                    i_cand[0] = i
                    crossed_once = True
                if not crossed_twice and crossed_once and h[i - 1] < 0 <= h[i]:
                    i_cand[1] = i
                    break

            # test values (a bracket to either side)
            txl_leq = t_axis[i_cand[0] - 1]
            txl_geq = t_axis[i_cand[0] + 1]
            txr_leq = t_axis[i_cand[1] - 1]
            txr_geq = t_axis[i_cand[1] + 1]

            with self.subTest(msg="order: {0}, txl-times".format(order)):

                self.assertTrue(txl_leq <= txl_theo <= txl_geq)

            with self.subTest(msg="order: {0}, txr-times".format(order)):

                self.assertTrue(txr_leq <= txr_theo <= txr_geq)

            with self.subTest(msg="order: {0}, lobe imbalance".format(order)):

                # imbalance of principal positive lobes
                A_pos_l = np.trapz(h[: i_cand[0]]) * dt
                A_pos_r = np.trapz(h[i_cand[1] :]) * dt
                imb_test = A_pos_l / A_pos_r

                precision = params[self.f_type]["precision"]

                # relative-precision test
                self.assertAlmostEqual(imb_test / imb_theo, 1.0, precision)

    def test_wireframe_moments_for_all_valid_orders(self):
        """
        Validate that discretized continuous-time wireframes
        have the expected 0th, 1st and 2nd moments.
        """

        precisions = {
            "pema": {"M0": 4, "M1": 2, "M2": 2},
            "bessel": {"M0": 4, "M1": 2, "M2": 2},
            "mbox": {"M0": 4, "M1": 4, "M2": 4},
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
            "pema": {"dt": 0.0001, "precision": 4},
            "bessel": {"dt": 0.0001, "precision": 2},
            "mbox": {"dt": 0.0002, "precision": 4},
        }

        # defs
        filter_orders = self.design.get_valid_filter_orders(strict=False)
        tau = 1.0
        dt = params[self.f_type]["dt"] * tau
        t_end = 10.0 * tau

        # test
        precision = params[self.f_type]["precision"]

        # prepare precision exception
        exception_type = self.f_type in ["pema", "mbox"]

        for i, order in enumerate(filter_orders):

            # prepare precision exception
            exception_order = order == 3

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

    def test_sacf_has_approx_correct_stride_and_residual_acf_values(self):
        """
        Test that the residual acf is as calculated at the decorrelation
        stride value.

        This test strategy differs from others and is rather simple. The
        challenges are that the bessel sacf oscillates in the tail, and the
        pema sacf does not reach 5% until 7th order.

        Thus, the strategy is to build a (xi, sacf) box along the realized
        sacf correlogram that contains the design point. This works well when
        there is a true 5% level to find, but in other cases (pema m < 7,
        bessel m = 3) then the ratio of actual to design residual is tested
        (forming a relative-precision test).
        """

        # params
        params = {
            "pema": {"dt": 0.0001, "precision": 3},
            "bessel": {"dt": 0.0001, "precision": 3},
            "mbox": {"dt": 0.0002, "precision": 3},
        }

        # defs
        filter_orders = self.design.get_valid_filter_orders(strict=False)
        tau = 1.0
        dt = params[self.f_type]["dt"] * tau
        t_end = 10.0 * tau

        # test
        precision = params[self.f_type]["precision"]

        # prepare precision exception
        exception_type = self.f_type in ["pema", "bessel"]

        for i, order in enumerate(filter_orders):

            # prepare precision exception
            exception_order = order == 3

            # design values
            sacf_design = self.design.autocorrelation_peak_and_stride_values(
                tau, order, dt
            )
            xi_5pct_theo = sacf_design["xi_5pct"]
            sacf_residual_acf_theo = sacf_design["residual_acf"]
            is_5pct = (
                int(np.round(1000.0 * np.round(sacf_residual_acf_theo, 3)))
                == 5000
            )

            # test series
            sacf_norm = self.f_sig.generate_sacf_correlogram(
                -t_end, t_end, dt, tau, order, normalize=True
            )
            xi_axis = sacf_norm[:, 0]
            sacf_axis = sacf_norm[:, 1]

            # construct box around (xi, sacf)_theo
            i_cand = np.min(np.argwhere(xi_axis >= xi_5pct_theo))
            i_sacf_xi_5pct_high = i_cand + 1
            i_sacf_xi_5pct_low = i_cand - 1

            xi_5pct_test_geq = xi_axis[i_sacf_xi_5pct_high]
            xi_5pct_test_leq = xi_axis[i_sacf_xi_5pct_low]
            sacf_residual_acf_test_geq = 100.0 * sacf_axis[i_sacf_xi_5pct_low]
            sacf_residual_acf_test_leq = 100.0 * sacf_axis[i_sacf_xi_5pct_high]

            with self.subTest(msg="order: {0}, test stride".format(order)):

                # test xi bound
                self.assertTrue(
                    xi_5pct_test_leq <= xi_5pct_theo <= xi_5pct_test_geq
                )

            with self.subTest(msg="order: {0}, test residual".format(order)):

                # test sacf bound
                if is_5pct:

                    self.assertTrue(
                        sacf_residual_acf_test_leq
                        <= sacf_residual_acf_theo
                        <= sacf_residual_acf_test_geq
                    )

                else:

                    prec_adj = -1 if (exception_type and exception_order) else 0
                    sacf_residual_acf_mean = np.mean(
                        [sacf_residual_acf_test_leq, sacf_residual_acf_test_geq]
                    )
                    self.assertAlmostEqual(
                        sacf_residual_acf_mean / sacf_residual_acf_theo,
                        1.0,
                        precision + prec_adj,
                    )

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
        precision = {"pema": 2, "bessel": 2, "mbox": 3}[self.f_type]

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

        precision = {"pema": 4, "bessel": 2, "mbox": 2}[self.f_type]

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

        precision = {"pema": 4, "bessel": 2, "mbox": 4}[self.f_type]

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

        precision = {"pema": 4, "bessel": 2, "mbox": 4}[self.f_type]

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
            "bessel": {"precision": 2},
            "pema": {"precision": 4},
            "mbox": {"precision": 4},
        }

        precision = params[self.f_type]["precision"]
        common_tooling.common_callable_xfer_fcxn_has_correct_kh0_value(
            self, precision
        )
