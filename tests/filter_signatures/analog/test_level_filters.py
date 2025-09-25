"""
-------------------------------------------------------------------------------

Parameterized unit tests for level-filter signatures.

-------------------------------------------------------------------------------
"""

import parameterized

import unittest
import numpy as np

from irides.design.analog import bessel_level as design_bessel
from irides.design.analog import polyema_level as design_pema
from irides.design.analog import mbox_level as design_mbox

from irides.filter_signatures.analog import bessel_level as f_sig_bessel
from irides.filter_signatures.analog import polyema_level as f_sig_pema
from irides.filter_signatures.analog import mbox_level as f_sig_mbox

from irides.tools import impulse_response_tools as ir_tools
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
class TestLevelFilters(unittest.TestCase):
    """
    Parameterized, common unit tests for standard level filters.
    """

    def test_impulse_response_moments_for_all_valid_orders(self):
        """
        Validate that discretized continuous-time impulse responses
        have the expected 0th, 1st and 2nd moments.
        """

        precisions = {
            "bessel": {"M0": 4, "M1": 4, "M2": 2},
            "pema": {"M0": 4, "M1": 4, "M2": 3},
            "mbox": {"M0": 4, "M1": 4, "M2": 3},
        }[self.f_type]

        common_tooling.common_impulse_response_moment_tests(
            self, precisions, proportional=True
        )

    def test_impulse_response_full_width_is_correct_for_all_orders(self):
        """
        Validate that the discretized continuous-time impulse-response
        full width is correct to within precision.
        """

        # filter orders and moments
        filter_orders = self.design.get_valid_filter_orders()

        # defs
        tau = 1.0
        dt = 0.0005 * tau
        t_start = 0.0
        t_end = 20.0 * tau

        # fetch fw-generator function
        fw_gen = self.design.full_width_generator(tau, dt)

        # iter over orders
        for order in filter_orders:
            # fetch theo value
            fw_theo = fw_gen(order)

            # build impulse response
            ir: np.ndarray = self.f_sig.generate_impulse_response(
                t_start, t_end, dt, tau, order
            )

            # calc fw
            fw_test = ir_tools.calculate_impulse_response_full_width(ir)

            # test
            precision = 4

            with self.subTest(msg="order: {0}".format(order)):
                self.assertAlmostEqual(fw_test, fw_theo, precision)

    def test_impulse_response_is_nonzero_at_t0_for_order_eq_1(self):
        """
        Test that h_m(t=0) = const for order m = 1
        """

        # params
        params = {
            "bessel": {"t_start": -0.1, "t_end": 0.1, "t0_value": 1.0},
            "pema": {"t_start": -0.1, "t_end": 0.1, "t0_value": 1.0},
            "mbox": {"t_start": -0.1, "t_end": 2.0, "t0_value": 0.5},
        }

        # defs
        filter_order = 1
        t_start = params[self.f_type]["t_start"]
        t_end = params[self.f_type]["t_end"]
        tau = 1.0
        dt = 0.001

        ts: np.ndarray = self.f_sig.generate_impulse_response(
            t_start, t_end, dt, tau, filter_order
        )

        # index to t = 0
        i_zero = np.where(0.0 - dt / 2 < ts[:, 0])[0][0]
        t0_value_theo = params[self.f_type]["t0_value"]
        self.assertAlmostEqual(ts[i_zero, 1], t0_value_theo, 4)

        # if exists t < 0 points, then test that the preceding
        # point is zero
        if i_zero > 0:
            self.assertAlmostEqual(ts[i_zero - 1, 1], 0.0, 4)

    def test_impulse_response_is_zero_at_t0_for_orders_ge_2(self):
        """
        Test that h_m(t=0) = 0 for orders m >= 2
        """

        # params
        params = {
            "bessel": {"t_start": -0.1, "t_end": 0.1},
            "pema": {"t_start": -0.1, "t_end": 0.1},
            "mbox": {"t_start": -0.1, "t_end": 2.0},
        }

        # defs
        filter_orders = self.design.get_valid_filter_orders()
        higher_orders = np.delete(
            filter_orders, np.argwhere(filter_orders == 1)
        )
        t_start = params[self.f_type]["t_start"]
        t_end = params[self.f_type]["t_end"]
        tau = 1.0
        dt = 0.001

        def zero_with_dt_correction(this_order):
            """
            A discretization correction must be applied for
            h_mbox(t=0, order=2) = dt / 2.
            """
            return (
                dt / 2.0 if self.f_type == "mbox" and this_order == 2 else 0.0
            )

        # compute and test
        for order in higher_orders:

            with self.subTest(msg="order: {0}".format(order)):

                ts: np.ndarray = self.f_sig.generate_impulse_response(
                    t_start, t_end, dt, tau, order
                )

                # index to t = 0
                i_zero = np.where(0.0 - dt / 2 < ts[:, 0])[0][0]
                self.assertAlmostEqual(
                    ts[i_zero, 1], zero_with_dt_correction(order), 4
                )

                # if exists t < 0 points, then test that the preceding
                # point is zero
                if i_zero > 0:
                    self.assertAlmostEqual(ts[i_zero - 1, 1], 0.0, 4)

    def test_impulse_response_has_correct_unit_peak_coordinates(self):
        """
        Test that t-peak and h(t-peak) are correct, to within precision, for
        valid orders.
        """

        # params
        params = {
            "bessel": {"precision": 3},
            "pema": {"precision": 4},
            "mbox": {"precision": 3},
        }

        # defs
        filter_orders = self.design.get_valid_filter_orders()
        t_start = 0.0
        t_end = 2.0
        tau = 1.0
        dt = 0.0001

        # test setup
        precision = params[self.f_type]["precision"]

        def pick_i_t_peak(this_order, this_peak_coord_theo, this_ts):
            """
            The box profile is flat so a different index picker is required.
            """
            if not (self.f_type == "mbox" and this_order == 1):
                return np.argmax(this_ts[:, 1])
            else:
                return np.where(
                    this_peak_coord_theo.time - dt / 2.0 < this_ts[:, 0]
                )[0][0]

        # iter on filter orders
        for order in filter_orders:
            # theo coord
            peak_coord_theo = self.design.peak_value_coordinate(tau, order, dt)

            # impulse response
            ts = self.f_sig.generate_impulse_response(
                t_start, t_end, dt, tau, order
            )

            # find t-peak and h(t-peak)
            i_t_peak = pick_i_t_peak(order, peak_coord_theo, ts)
            t_peak_test = ts[i_t_peak, 0]
            h_peak_test = ts[i_t_peak, 1]

            with self.subTest(msg="order: {0}".format(order)):
                self.assertAlmostEqual(
                    t_peak_test, peak_coord_theo.time, precision
                )
                self.assertAlmostEqual(
                    h_peak_test, peak_coord_theo.value, precision
                )

    def test_wireframe_moments_for_all_valid_orders(self):
        """
        Validate that discretized continuous-time wireframes
        have the expected 0th, 1st and 2nd moments.
        """

        precisions = {
            "pema": {"M0": 4, "M1": 4, "M2": 4},
            "bessel": {"M0": 4, "M1": 4, "M2": 4},
            "mbox": {"M0": 4, "M1": 4, "M2": 4},
        }[self.f_type]

        common_tooling.common_impulse_response_moment_tests(
            self, precisions, proportional=True, use_wireframe_signature=True
        )

    def test_sacf_has_correct_unnormalized_peak_values(self):
        """
        Tests that the kh(0) calculated from an impulse response is close
        to the theoretic design value.
        """

        # params
        params = {
            "bessel": {"dt": 0.0002, "precision": 2},
            "pema": {"dt": 0.0005, "precision": 4},
            "mbox": {"dt": 0.001, "precision": 4},
        }

        # defs
        filter_orders = self.design.get_valid_filter_orders(strict=False)
        t_end = 5.0
        tau = 1.0
        dt = params[self.f_type]["dt"]
        acf_arg = 1.0

        # test
        precision = params[self.f_type]["precision"]

        for i, order in enumerate(filter_orders):

            # fetch design points
            kh0_theo = self.design.autocorrelation_peak_and_stride_values(
                acf_arg, order
            )["kh0"]

            # discretization correction
            if self.f_type in ["bessel", "pema"] and order == 1:
                kh0_theo += dt / 2.0

            # fetch sacf, take peak value
            sacf = self.f_sig.generate_sacf_correlogram(
                -t_end, t_end, dt, tau, order
            )
            kh0_test = max(sacf[:, 1])

            # test
            with self.subTest(msg="order: {0}".format(order)):
                self.assertAlmostEqual(kh0_test, kh0_theo, precision)

    def test_sacf_has_approx_correct_5pct_lag_values(self):
        """
        Tests that the xi_5pct lag calculated from an impulse response is
        close to the theoretic design value.
        """

        # params
        params = {
            "bessel": {"t_end_coef": 5.0, "dt_v_coef": 0.0005, "precision": 2},
            "pema": {"t_end_coef": 5.0, "dt_v_coef": 0.0005, "precision": 3},
            "mbox": {"t_end_coef": 2.0, "dt_v_coef": 0.0002, "precision": 3},
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

            # find xi = 0 index
            i_zero = np.where(0.0 - dxi / 2 < sacf[:, 0])[0][0]

            # find xi where sacf[:, 1] is first < 5%
            i_5pct = i_zero + np.where(sacf[i_zero:, 1] <= 0.05)[0][0]

            xi_5pct_test = sacf[i_5pct, 0]

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
        precision = {"pema": 3, "bessel": 2, "mbox": 3}[self.f_type]

        common_tooling.common_sscf_decorrelation_stride_test(self, precision)

    def test_generate_transfer_function_has_correct_size_and_type(self):
        """
        The transfer-function panel is testable for gain, phase and group
        delay, but not very well by itself. Here, only shape and type checks
        are tested.
        """

        common_tooling.common_transfer_function_has_correct_size_and_type(self)

    def test_generate_spectra_for_correct_gain_at_dc_and_cutoff(self):
        """
        Test that the DC and cutoff gain is correct for all orders.
        """

        precision = {"bessel": 2, "pema": 4, "mbox": 2}[self.f_type]

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
            self.design.cutoff_frequency,
            self.design.gain_at_cutoff,
            xfer_tools.SpectraColumns.GAIN.value,
            precision,
            "gain",
        )

    def test_generate_spectra_for_correct_phase_at_dc_and_cutoff(self):
        """
        Test that the DC and cutoff phase is correct for all orders.
        """

        precision = {"bessel": 2, "pema": 4, "mbox": 4}[self.f_type]

        with self.subTest(msg="dc test"):
            # noinspection PyTypeChecker
            common_tooling.common_gain_phase_group_delay_at_dc_test(
                self,
                self.design.phase_at_dc,
                xfer_tools.SpectraColumns.PHASE.value,
                "phase",
            )

        with self.subTest(msg="cutoff test"):
            # noinspection PyTypeChecker
            common_tooling.common_gain_phase_group_delay_at_feature_freq_test(
                self,
                self.design.cutoff_frequency,
                self.design.phase_at_cutoff,
                xfer_tools.SpectraColumns.PHASE.value,
                precision,
                "phase",
            )

    def test_generate_spectra_for_correct_group_delay_at_dc_and_cutoff(self):
        """
        Test that the DC and cutoff group delay is correct for all orders.
        """

        precision = {"bessel": 2, "pema": 4, "mbox": 4}[self.f_type]

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
            self.design.cutoff_frequency,
            self.design.group_delay_at_cutoff,
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
