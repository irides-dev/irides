"""
-------------------------------------------------------------------------------

Common tooling for filter-signature unit tests.

-------------------------------------------------------------------------------
"""

import numpy as np
import scipy.integrate as integrate

from irides.tools import design_tools
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


# noinspection SpellCheckingInspection
def common_impulse_response_moment_tests(
    self, precisions: dict, proportional=False, use_wireframe_signature=False
):
    """
    Tests that the realized M0, M1 and M2 moments from an impulse response
    approximately match the design values.
    """

    # filter orders and moments
    filter_orders = self.design.get_valid_filter_orders(strict=False)
    moments = ir_tools.get_valid_moments()

    # defs
    tau = 0.5
    dt = 0.0002 * tau

    # construct theo panel, rows ~ order, cols ~ [M0, M1, M2]
    moment_cols = ["M0", "M1", "M2"]
    n_moments = len(moment_cols)
    moment_values_theo = np.zeros((filter_orders.shape[0], n_moments))

    if not use_wireframe_signature:

        t_start = 0.0
        t_end = 20.0 * tau

        f_sig_fcxn = self.f_sig.generate_impulse_response

        m0_gen = self.design.moment_value_generator(0, tau, dt)
        m1_gen = self.design.moment_value_generator(1, tau, dt)
        m2_gen = self.design.moment_value_generator(2, tau, dt)

    else:

        t_start = -0.5 * tau
        t_end = 2.0 * tau

        f_sig_fcxn = self.f_sig.generate_wireframe

        m0_gen = design_tools.wireframe_moment_value_generator(
            self.design, 0, tau, dt
        )
        m1_gen = design_tools.wireframe_moment_value_generator(
            self.design, 1, tau, dt
        )
        m2_gen = design_tools.wireframe_moment_value_generator(
            self.design, 2, tau, dt
        )

    for i, order in enumerate(filter_orders):
        moment_values_theo[i, :] = np.array(
            [m0_gen(order), m1_gen(order), m2_gen(order)]
        )

    # build test values
    moment_values_test = np.zeros_like(moment_values_theo)
    for i, order in enumerate(filter_orders):

        # build impulse response
        ir: np.ndarray = f_sig_fcxn(t_start, t_end, dt, tau, order)

        # calc moments
        for j, moment in enumerate(moments):
            moment_values_test[
                i, j
            ] = ir_tools.calculate_impulse_response_moment(ir, moment)

    # test by order
    for i, order in enumerate(filter_orders):
        for j, mc in enumerate(moment_cols):
            with self.subTest(msg="order: {0}, M: {1}".format(order, j)):

                precision = precisions[moment_cols[j]]

                if proportional:
                    self.assertAlmostEqual(
                        moment_values_test[i, j] / moment_values_theo[i, j],
                        1.0,
                        precision,
                    )
                else:
                    self.assertAlmostEqual(
                        moment_values_test[i, j],
                        moment_values_theo[i, j],
                        precision,
                    )


# noinspection SpellCheckingInspection,PyPep8Naming
def common_transfer_function_has_correct_size_and_type(self):
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
    order = 3

    # create H panel
    H: np.ndarray = self.f_sig.generate_transfer_function(
        f_start, f_end, df, tau, order
    )

    # simple tests on the shape and type of the return
    self.assertEqual(H.shape[0], n_pts)
    self.assertEqual(H.shape[1], 2)
    self.assertTrue(H.dtype == np.dtype("complex128"))


# noinspection SpellCheckingInspection
def common_gain_phase_group_delay_at_dc_test(
    this, design_fcxn, value_col, caller
):
    """
    Common test for phase and group delay values at DC frequency.
    """

    # defs
    df = 0.001
    tau = 1.0
    dc_precision = 4

    # fetch valid filter orders
    filter_orders = this.design.get_valid_filter_orders(strict=False)

    # feature at DC
    f_theo = 0.0
    feature_theo = design_fcxn(tau)

    for order in filter_orders:

        # generate a spectrum panel
        spectra_test = this.f_sig.generate_spectra(
            -5 * df, 5 * df, df, tau, order
        )

        # find the first freq at or above f_theo (0)
        f_axis = spectra_test[:, xfer_tools.SpectraColumns.FREQ.value]
        i_f_test = np.where(f_theo - df / 2.0 < f_axis)[0][0]
        f_test = f_axis[i_f_test]
        feature_test = spectra_test[i_f_test, value_col]

        # test
        with this.subTest(msg=(caller + " at DC -- order: {0}".format(order))):
            this.assertAlmostEqual(f_test, f_theo, dc_precision)
            this.assertAlmostEqual(feature_test, feature_theo, dc_precision)


# noinspection SpellCheckingInspection
def common_gain_phase_group_delay_at_feature_freq_test(
    this, design_freq_fcxn, design_feature_fcxn, value_col, precision, caller
):
    """
    Common tests for gain, phase and group delay at a feature frequency.
    For level filters, the feature frequency is at gain cutoff.
    For slope and curvature filters, the feature frequency is the frequency
        at peak gain.
    """

    # defs
    df = 0.001
    tau = 1.0

    # fetch valid filter orders
    filter_orders = this.design.get_valid_filter_orders(strict=False)

    # feature at cutoff
    for order in filter_orders:
        # compute [f_start, f_end] from the theo peak freq
        w_theo = design_freq_fcxn(tau, order)
        f_theo = w_theo / (2.0 * np.pi)
        f_start = f_theo - 0.1
        f_end = f_theo + 0.1

        # fetch the theoretic feature (gain, phase, group-delay)
        feature_theo = design_feature_fcxn(tau, order)

        # generate a spectrum panel
        spectra_test = this.f_sig.generate_spectra(
            f_start, f_end, df, tau, order
        )

        # find the first freq at or above f_theo
        f_axis = spectra_test[:, xfer_tools.SpectraColumns.FREQ.value]
        i_f_test = np.where(f_theo - df / 2.0 < f_axis)[0][0]
        f_test = f_axis[i_f_test]
        feature_test = spectra_test[i_f_test, value_col]

        # test
        with this.subTest(
            msg=(caller + " at feature freq -- order: {0}".format(order))
        ):
            this.assertAlmostEqual(f_test, f_theo, precision)
            this.assertAlmostEqual(feature_test, feature_theo, precision)


# noinspection SpellCheckingInspection,PyPep8Naming
def common_callable_xfer_fcxn_has_correct_kh0_value(this, precision):
    """
    Calculates the integral

      kh(0) = int_-inf^inf H(j 2 pi f) H(-j 2 pi f) df

    and compares it with the design value.
    """

    # defs
    tau = 1.0

    # fetch valid filter orders
    filter_orders = this.design.get_valid_filter_orders(strict=False)

    # prepare precision exception
    exception_type = this.f_type in ["mbox"]

    # feature at cutoff
    for order in filter_orders:

        # prepare precision exception
        exception_order = order == np.min(filter_orders)

        # get design value
        kh0_theo = this.design.autocorrelation_peak_and_stride_values(
            tau, order
        )["kh0"]

        # calculate test value
        H = this.f_sig.make_callable_transfer_function(order)
        kh0_test = 2.0 * np.real(
            integrate.quad(lambda f: H(f) * H(-f), 0.0, np.inf)[0]
        )

        # test
        with this.subTest(msg="order: {0}".format(order)):

            prec_adj = -1 if (exception_type and exception_order) else 0

            this.assertAlmostEqual(
                kh0_test / kh0_theo, 1.0, precision + prec_adj
            )


# noinspection SpellCheckingInspection,PyPep8Naming
def common_sscf_decorrelation_stride_test(this, precision):
    """
    Tests that the sti-5pct strides calculated from mathematica
    approximately match those calculated from the transfer functions in
    this library.

    Because of the wide range of scale-decorrelation strides, these
    tests fetch the theorectic strides from the design files (and assoc'd
    residuals), and then calculates sscf on a domain that surrounds the
    expected stride.
    """

    # defs
    filter_orders = this.design.get_valid_filter_orders(strict=False)
    tau = 1.0
    dsti = 0.002
    sti_band_scale = 20

    # iter on orders
    for i, order in enumerate(filter_orders):

        d = this.design.scale_correlation_stride_and_residual_values(tau, order)
        sti_stride_theo = d["sti_5pct"]
        sti_residual = d["residual"]

        infer_crossing = 0.995 <= np.abs(sti_residual / 0.05) <= 1.005

        sti_start = sti_stride_theo - sti_band_scale * dsti
        sti_end = sti_stride_theo + sti_band_scale * dsti

        sscf_norm = this.f_sig.generate_sscf_correlogram(
            sti_start, sti_end, dsti, order
        )

        res_min = np.min(sscf_norm[:, 1])
        res_max = np.max(sscf_norm[:, 1])

        with this.subTest(msg="order: {0}".format(order)):

            if infer_crossing:

                this.assertTrue(
                    np.min(sscf_norm[:, 1])
                    <= sti_residual
                    <= np.max(sscf_norm[:, 1])
                )

            else:

                this.assertAlmostEqual(sti_residual / res_max, 1.0, precision)
