"""
-------------------------------------------------------------------------------

Tests against an analog damped oscillator filter signature

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.design.analog import damped_oscillator as design
from irides.filter_signatures.analog import damped_oscillator as f_sig

from irides.design.analog import bessel_level as bessel_design

from irides.tools import impulse_response_tools as ir_tools
from irides.tools import transfer_function_tools as xfer_tools


# noinspection SpellCheckingInspection,PyPep8Naming
def clean_ndarray_for_precision_comparison(
    array: np.array, precision
) -> np.array:
    """
    Float comparisons are evil. Here I round to extra precision, rebase by
    multiplying by 10^precision, and then cast to int (which is not a
    rounding function).
    """

    # rebase value
    rebase = np.power(10, precision)

    # covert
    return rebase * np.round(array, precision + 1).astype(int)


# noinspection SpellCheckingInspection,PyPep8Naming
class TestDampedOscillatorElement(unittest.TestCase):
    """
    Unit tests for the src.spqf.filter_signature.analog.damped_oscillator
    module.
    """

    def setUp(self) -> None:
        """
        Recurring setup for reference poles from the Bessel-7 filter design.
        """

        # fetch test poles
        self.b7_design = bessel_design.designs(7)
        self.stages = self.b7_design["stages"]
        self.n_stages = len(self.stages)

        # build array
        self.reference_poles = np.zeros(self.n_stages, dtype=complex)
        for i, stage in enumerate(self.stages):
            i_pole = stage["indices"][0]  # first pole from array
            self.reference_poles[i] = self.b7_design["poles"][i_pole]

    def test_impulse_response_moments_for_bessel7_poles(self):
        """
        Tests that the realized moments and full width are nearly the
        same as the design values for per-stage poles in the 7th-order
        bessel design.

        It is recognized that the last Bessel-7 stage is an ema. This test
        seeks only to use several relevant pole positions, and a purely real
        pole is an important case to test. There is no confusion between an
        ema and a damped oscillator with a coincident pole location.
        """

        # defs
        t_start = 0.0
        t_end = 20.0
        tau = 1.0
        dt = 0.0005
        moments = np.arange(3)

        # construct theo panel: rows ~ stage, cols ~ [M0, M1, M2, FW]
        moment_values_theo = np.zeros((self.n_stages, 4))
        m0_gen = design.moment_value_generator(0, dt)
        m1_gen = design.moment_value_generator(1, dt)
        m2_gen = design.moment_value_generator(2, dt)
        fw_gen = design.full_width_generator(dt)

        # construct test panel
        moment_values_test = np.zeros_like(moment_values_theo)

        # iter on stages
        for i, stage in enumerate(self.stages):

            reference_pole = self.reference_poles[i]

            # theo setup
            wns, thetas = design.extract_wn_and_theta(
                np.array([reference_pole])
            )
            wn = wns[0]
            theta = thetas[0]

            # theo values
            moment_values_theo[i, :] = np.array(
                [
                    m0_gen(wn, theta, tau),
                    m1_gen(wn, theta, tau),
                    m2_gen(wn, theta, tau),
                    fw_gen(wn, theta, tau),
                ]
            )

            # test setup
            ir: np.ndarray = f_sig.generate_impulse_response(
                t_start, t_end, dt, reference_pole, tau
            )

            # test values
            for j, moment in enumerate(moments):
                moment_values_test[
                    i, j
                ] = ir_tools.calculate_impulse_response_moment(ir, moment)
            moment_values_test[
                i, -1
            ] = ir_tools.calculate_impulse_response_full_width(ir)

        # test setup
        precision = 4

        # test by stage
        clean = clean_ndarray_for_precision_comparison
        for i, stage in enumerate(self.stages):

            with self.subTest(msg="stage: {0}".format(i)):
                self.assertSequenceEqual(
                    list(clean(moment_values_test[i, :], precision)),
                    list(clean(moment_values_theo[i, :], precision)),
                )

    def test_impulse_response_has_correct_t0_and_null_and_peak_coordinates(
        self,
    ):
        """
        Test that t-peak, h(t-peak), h(t=0), and t(1st null) generated from
        sample impulse responses match the theoretic design points.
        """

        # defs
        t_start = -0.1
        t_end = 2.0
        tau = 1.0
        dt = 0.0001

        # setup theo
        peak_coordinates_theo = design.peak_value_coordinates(
            self.reference_poles, tau
        )
        fth_null_times_theo = design.nth_null_times(
            self.reference_poles, 1, tau
        )

        # iter on poles, create impulse responses
        h_zeros_test = np.zeros(self.n_stages, dtype=float)
        t_peaks_test = np.zeros_like(h_zeros_test)
        h_peaks_test = np.zeros_like(h_zeros_test)
        fth_null_times_test = np.zeros_like(h_zeros_test)

        for i, reference_pole in enumerate(self.reference_poles):

            # build ir
            ir: np.ndarray = f_sig.generate_impulse_response(
                t_start, t_end, dt, reference_pole, tau
            )

            # index to t = 0
            i_zero = np.where(0.0 - dt / 2 < ir[:, 0])[0][0]
            h_zeros_test[i] = ir[i_zero, 1]

            # index to t-peak and h(t-peak)
            i_t_peak = np.argmax(ir[:, 1])
            t_peaks_test[i] = ir[i_t_peak, 0]
            h_peaks_test[i] = ir[i_t_peak, 1]

            # index to 1st-null time
            if np.imag(reference_pole) != 0.0:
                i_1st_null = np.where(ir[:, 1] < 0.0)[0][0]
                t_1st_null = ir[i_1st_null, 0]
            else:
                t_1st_null = np.inf
            fth_null_times_test[i] = t_1st_null

        # test setup
        precision = 4
        time_precision = 3

        # test by stage
        clean = clean_ndarray_for_precision_comparison

        with self.subTest(msg="h(t=0)"):
            self.assertSequenceEqual(
                list(clean(h_zeros_test, precision)),
                list(clean(np.zeros(self.n_stages), precision)),
            )
        with self.subTest(msg="t-peak"):
            self.assertSequenceEqual(
                list(clean(t_peaks_test, precision)),
                list(clean(peak_coordinates_theo.times(), precision)),
            )
        with self.subTest(msg="h(t-peak)"):
            self.assertSequenceEqual(
                list(clean(h_peaks_test, precision)),
                list(clean(peak_coordinates_theo.values(), precision)),
            )
        with self.subTest(msg="t(1st null)"):
            self.assertSequenceEqual(
                list(clean(fth_null_times_test, time_precision)),
                list(clean(fth_null_times_theo, time_precision)),
            )

    def test_sacf_has_correct_kh0_and_decorrelation_stride_values(self):
        """
        Test that sacfs calculated from an impulse response approximately
        matches the theoretical design points. For the decorrelation stride,
        the design point reports the exponential envelope and does not
        account for the underlying oscillation. As a consequence, the design
        point is pessimistic by quoting a larger stride than might be the
        case. For a purely real reference_pole, the design point and realized
        decorrelation strides are approximately the same.
        """

        # defs
        t_end = 2.0
        tau = 1.0
        dt = 0.0001

        # fetch theo values
        res = design.autocorrelation_peak_and_stride_values(
            self.reference_poles, tau
        )
        kh0_theos = res["kh0"]
        xi_5pct_theos = res["xi_5pct"]

        # test
        precision = 3

        # iter on poles
        for i, reference_pole in enumerate(self.reference_poles):

            # test sacf values
            sacf_unnorm = f_sig.generate_sacf_correlogram(
                -t_end, t_end, dt, reference_pole, tau
            )

            sacf_norm = f_sig.generate_sacf_correlogram(
                -t_end, t_end, dt, reference_pole, tau, normalize=True
            )

            kh0_test = max(sacf_unnorm[:, 1])

            # infer dxi from sacf
            dxi = min(np.diff(sacf_norm[:, 0]))

            # find xi = 0 index
            i_zero = np.where(0.0 - dxi / 2 < sacf_norm[:, 0])[0][0]

            # find xi where sacf[:, 1] is first < 5%
            i_5pct = i_zero + np.where(sacf_norm[i_zero:, 1] <= 0.05)[0][0]

            xi_5pct_test = sacf_norm[i_5pct, 0]

            with self.subTest(msg="stage: {0}, kh0 test".format(i)):
                self.assertAlmostEqual(kh0_test, kh0_theos[i], precision)

            with self.subTest(msg="stage: {0}, xi_5pct stride test".format(i)):
                self.assertLessEqual(xi_5pct_test, xi_5pct_theos[i])

    def test_generate_transfer_function_has_correct_size_and_type(self):
        """
        The transfer-function panel is testable for gain, phase and group
        delay, but not very well by itself. Here, only shape and type checks
        are tested.
        """

        # defs
        n_pts = 100
        f_start = 0.0
        f_end = 1.0
        df = f_end / n_pts
        tau = 1.0
        filter_order = 3

        # create H panel
        H_dosc: np.ndarray = f_sig.generate_transfer_function(
            f_start, f_end, df, tau, filter_order
        )

        # simple tests on the shape and type of the return
        self.assertEqual(H_dosc.shape[0], n_pts)
        self.assertEqual(H_dosc.shape[1], 2)
        self.assertTrue(H_dosc.dtype == np.dtype("complex128"))

    def test_generate_spectra_for_correct_features_at_dc_and_peak_freq(self):
        """
        Tests for gain, phase and group-delay levels at DC and, when not
        np.nan, at the underdamped peak frequency. Design points are compared
        with generated spectra.
        """

        # defs
        df = 0.0005
        f_start = -4.0 * df
        tau = 1.0

        # fetch theo values
        features_theo = design.key_spectral_features(self.reference_poles, tau)

        # test
        precision = 3
        cols = xfer_tools.SpectraColumns

        # encapsulate test
        def exec_test(i_pole: int, i_col: int, i_freq: int, damp_type: str):

            val_theo = features_theo[damp_type][i_pole, i_col]
            val_test = spectra_test[i_freq, i_col]

            self.assertAlmostEqual(val_test, val_theo, precision)

        # iter on poles
        for i, reference_pole in enumerate(self.reference_poles):

            # compute f_end based on theo wmax
            wmax_under = features_theo["under"][i, cols.FREQ.value]
            f_end = (
                2.0
                if np.isnan(wmax_under)
                else (2.0 * wmax_under / (2.0 * np.pi))
            )

            # generate a spectrum panel
            spectra_test = f_sig.generate_spectra(
                f_start, f_end, df, reference_pole, tau
            )

            # always test overdamped
            with self.subTest(msg="stage: {0}, overdamped"):

                # find DC freq index
                i_freq = np.where(
                    0.0 - df / 2 < spectra_test[:, cols.FREQ.value]
                )[0][0]

                exec_test(i, cols.GAIN.value, i_freq, "over")
                exec_test(i, cols.PHASE.value, i_freq, "over")
                exec_test(i, cols.GROUPDELAY.value, i_freq, "over")

            # conditionally test underdamped
            if np.isnan(wmax_under):
                continue

            with self.subTest(msg="stage: {0}, underdamped"):

                # find wmax freq index
                fmax = wmax_under / (2.0 * np.pi)
                i_freq = np.where(
                    fmax - df / 2 < spectra_test[:, cols.FREQ.value]
                )[0][0]

                exec_test(i, cols.GAIN.value, i_freq, "under")
                exec_test(i, cols.PHASE.value, i_freq, "under")
                exec_test(i, cols.GROUPDELAY.value, i_freq, "under")
