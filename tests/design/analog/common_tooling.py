"""
-------------------------------------------------------------------------------

Common tooling for filter-design unit tests.

-------------------------------------------------------------------------------
"""

import numpy as np


# noinspection SpellCheckingInspection
def design_id_test(this, design, design_id_theo):
    """Tests that a correct design-id is coded."""

    this.assertEqual(design.design_id(), design_id_theo)


def design_type_test(this, design, design_type_theo):
    """Tests that the design-type enum is set correctly"""

    this.assertEqual(design.design_type(), design_type_theo)


def loose_valid_filter_orders_are_as_expected(this, design, seq_theo):
    """Tests for correct filter orders."""

    this.assertSequenceEqual(
        list(design.get_valid_filter_orders(strict=False)), list(seq_theo)
    )


def impulse_response_t0_value_is_correct_for_valid_orders(
    this, design, seq_theo, tau, dt, strict=False
):
    """Tests for correct h(t=0) values."""

    filter_orders = design.get_valid_filter_orders(strict)

    # calc test values
    seq_test = np.zeros_like(seq_theo)
    for i, order in enumerate(filter_orders):
        seq_test[i] = design.impulse_response_t0_value(tau, order, dt)

    # test
    precision = 4
    assert_sequence_equal_for_float_arrays(this, seq_test, seq_theo, precision)


def zero_crossing_time_is_correct_for_valid_orders(
    this, design, seq_theo, tau, strict=False
):
    """Test that the crossing times are correct."""

    filter_orders = design.get_valid_filter_orders(strict)

    # calc test values
    seq_test = np.zeros_like(seq_theo)
    for i, order in enumerate(filter_orders):
        seq_test[i] = design.zero_crossing_time(tau, order)

    # test
    precision = 4
    assert_sequence_equal_for_float_arrays(this, seq_test, seq_theo, precision)


def impulse_response_lobe_details_are_correct(
    this, design, seq_theo, feature, tau, strict=False
):
    """For curvature filters, test for tx{1,2} and lobe imbalance."""

    filter_orders = design.get_valid_filter_orders(strict)

    # calc test values
    seq_test = np.zeros_like(seq_theo)
    for i, order in enumerate(filter_orders):
        seq_test[i] = design.impulse_response_lobe_details(tau, order)[feature]

    # test
    precision = 4
    assert_sequence_equal_for_float_arrays(this, seq_test, seq_theo, precision)


def moment_value_generators_for_a_given_moment_are_correct(
    this, design, seq_theo, moment, tau, dt, strict=False
):
    """Test that the moments are correct across orders."""

    filter_orders = design.get_valid_filter_orders(strict)
    moment_gen = design.moment_value_generator(moment, tau, dt)

    # calc test values
    seq_test = np.zeros_like(seq_theo, dtype=float)
    for i, order in enumerate(filter_orders):
        seq_test[i] = moment_gen(order)

    # test
    precision = 4
    assert_sequence_equal_for_float_arrays(this, seq_test, seq_theo, precision)


# noinspection SpellCheckingInspection
def autocorrelation_peak_and_stride_values_are_correct(
    this, design, sacf_pairs_theo, tau, dt, strict=False
):
    """Test that the kh0 and xi_5pct values are correct."""

    filter_orders = design.get_valid_filter_orders(strict)

    # test
    precision = 4

    for order in filter_orders:

        # fetch design values
        sacf_pairs_test = design.autocorrelation_peak_and_stride_values(
            tau, order, dt
        )

        # test w/out discretization correction
        for k in ["kh0", "xi_5pct"]:

            with this.subTest(msg="order: {0}, feature: {1}".format(order, k)):

                this.assertAlmostEqual(
                    sacf_pairs_test[k], sacf_pairs_theo[order][k], precision
                )


# noinspection SpellCheckingInspection
def autocorrelation_peak_and_stride_values_scale_with_tau(
    this, design, kh0_power_law
):
    """Test that sacf values scale correctly with tau."""

    tau_1 = 1.0
    tau_2 = 3.0

    scale_kh0 = 1.0 / np.power(tau_2 / tau_1, kh0_power_law)
    scale_xi_5cpt = tau_2 / tau_1

    filter_orders = design.get_valid_filter_orders(strict=False)

    # test
    precision = 4

    for order in filter_orders:
        with this.subTest(msg="order: {0}".format(order)):
            ans_1 = design.autocorrelation_peak_and_stride_values(tau_1, order)
            ans_2 = design.autocorrelation_peak_and_stride_values(tau_2, order)

            ratio_kh0 = ans_2["kh0"] / ans_1["kh0"]
            ratio_xi_5pct = ans_2["xi_5pct"] / ans_1["xi_5pct"]

            this.assertAlmostEqual(ratio_kh0, scale_kh0, precision)
            this.assertAlmostEqual(ratio_xi_5pct, scale_xi_5cpt, precision)


# noinspection SpellCheckingInspection
def scale_correlation_strides(this, design, strides_theo: np.ndarray):
    """Tests that the scale-decorrelation strides are correct"""

    # defs
    tau = 1.0
    filter_orders = design.get_valid_filter_orders(strict=False)

    # defs for test
    precision = 4

    for i, order in enumerate(filter_orders):

        # fetch design values
        # fmt: off
        strides_test = design. \
            scale_correlation_stride_and_residual_values(tau, order)["sti_5pct"]
        # fmt: on

        # test
        with this.subTest(msg="order: {0}".format(order)):

            this.assertAlmostEqual(strides_test, strides_theo[i], precision)


# noinspection SpellCheckingInspection
def gain_phase_gd_at_dc_frequency(
    this, design, ans_theo_dict: dict, tau: float
):
    """Tests that the dc feature value is correct."""

    # defs
    precision = 4

    ans_test_dict = {
        "gain": design.gain_at_dc(tau),
        "phase": design.phase_at_dc(tau),
        "group-delay": design.group_delay_at_dc(tau),
    }

    for feature in ans_theo_dict.keys():

        feature_test = ans_test_dict[feature]
        feature_theo = ans_theo_dict[feature]

        with this.subTest(msg="{0}".format(feature)):
            this.assertAlmostEqual(feature_test, feature_theo, precision)


# noinspection SpellCheckingInspection
def peak_spectral_features(this, design, values_theo, design_fcxn, tau, caller):
    """Common test fixure for peak-frequency features."""

    # defs
    filter_orders = design.get_valid_filter_orders(strict=False)

    # defs for test
    precision = 4
    rebase = np.power(10.0, precision)

    values_test = np.zeros_like(values_theo)

    for i, order in enumerate(filter_orders):
        values_test[i] = design_fcxn(tau, order)

    with this.subTest(msg=(caller + " peak-frequency feature")):

        this.assertSequenceEqual(
            list(rebase * np.round(values_test, precision)),
            list(rebase * np.round(values_theo, precision)),
        )


# noinspection SpellCheckingInspection,PyPep8Naming
def wireframe_level_features(this, design, tau: float):
    """Tests that the wf timepoint and weight are correct."""

    # defs
    filter_orders = design.get_valid_filter_orders(strict=False)

    # defs for test
    precision = 4
    TP = 0

    # def fixed theo values
    wf_tp_theo = tau
    wf_wgt_theo = 1.0

    for order in filter_orders:

        # fetch design value
        wf_test = design.wireframe(tau, order)

        # test
        this.assertAlmostEqual(wf_test.timepoints[TP], wf_tp_theo, precision)

        this.assertAlmostEqual(wf_test.weights[TP], wf_wgt_theo, precision)


# noinspection SpellCheckingInspection,PyPep8Naming
def wireframe_slope_features(this, design, wfs_theo: dict, tau: float):
    """Tests that the wf timepoints and weights are correct"""

    # defs
    filter_orders = design.get_valid_filter_orders(strict=False)

    # defs for test
    precision = 4
    TPOS = 0
    TNEG = 1

    for order in filter_orders:

        # select theo values
        wf_theo = wfs_theo[order]

        # fetch design values
        wf_test = design.wireframe(tau, order)

        # test
        with this.subTest(msg="order: {0}, timepoints".format(order)):

            this.assertAlmostEqual(
                wf_test.timepoints[TPOS], wf_theo["tpos"] * tau, precision
            )
            this.assertAlmostEqual(
                wf_test.timepoints[TNEG], wf_theo["tneg"] * tau, precision
            )

        with this.subTest(msg="order: {0}, weights".format(order)):

            Twf_theo = (wf_theo["tneg"] - wf_theo["tpos"]) * tau
            weight_theo = 1.0 / Twf_theo

            this.assertAlmostEqual(
                wf_test.weights[TPOS], weight_theo, precision
            )
            this.assertAlmostEqual(
                wf_test.weights[TNEG], -1.0 * weight_theo, precision
            )


# noinspection SpellCheckingInspection,PyPep8Naming
def wireframe_curvature_features(this, design, wfs_theo: dict, tau: float):
    """Tests that the wf timepoints and weights are correct"""

    # defs
    filter_orders = design.get_valid_filter_orders(strict=False)

    # defs for test
    precision = 4
    TPOSL = 0
    TNEG = 1
    TPOSR = 2

    for order in filter_orders:

        # select theo values
        wf_theo = wfs_theo[order]

        # fetch design values
        wf_test = design.wireframe(tau, order)

        # test
        with this.subTest(msg="order: {0}, timepoints".format(order)):

            this.assertAlmostEqual(
                wf_test.timepoints[TPOSL], wf_theo["tposl"] * tau, precision
            )
            this.assertAlmostEqual(
                wf_test.timepoints[TNEG], wf_theo["tneg"] * tau, precision
            )
            this.assertAlmostEqual(
                wf_test.timepoints[TPOSR], wf_theo["tposr"] * tau, precision
            )

        with this.subTest(msg="order: {0}, weights".format(order)):

            Twf_theo = (wf_theo["tposr"] - wf_theo["tposl"]) * tau
            weight_theo = np.power(Twf_theo / 2.0, -2)

            this.assertAlmostEqual(
                wf_test.weights[TPOSL], weight_theo, precision
            )
            this.assertAlmostEqual(
                wf_test.weights[TNEG], -2.0 * weight_theo, precision
            )
            this.assertAlmostEqual(
                wf_test.weights[TPOSR], weight_theo, precision
            )


# noinspection SpellCheckingInspection,PyPep8Naming
def wavenumber_features(this, design, wns_theo: np.ndarray):
    """Tests that wavenumbers are correct"""

    # defs
    filter_orders = design.get_valid_filter_orders(strict=False)

    # defs for test
    precision = 4

    for i, order in enumerate(filter_orders):

        # fetch design values
        wn_test = design.wavenumber(order)

        # test
        with this.subTest(msg="order: {0}".format(order)):

            this.assertAlmostEqual(wn_test, wns_theo[i], precision)


# noinspection SpellCheckingInspection
def uncertainty_products(this, design, ucps_theo: np.ndarray):
    """Tests that uncertainty products are correct"""

    # defs
    filter_orders = design.get_valid_filter_orders(strict=False)

    # defs for test
    precision = 4

    for i, order in enumerate(filter_orders):

        # fetch design values
        ucp_test = design.uncertainty_product(order)["UCP"]

        # test
        with this.subTest(msg="order: {0}".format(order)):

            if ucps_theo[i] == np.inf:
                this.assertTrue(np.isinf(ucp_test))
            else:
                this.assertAlmostEqual(ucp_test, ucps_theo[i], precision)


# ------------------------------------------------------------------------------
def assert_sequence_equal_for_float_arrays(this, seq_test, seq_theo, precision):
    """Common test code for testing of sequences."""

    rebase = np.power(10.0, precision)

    this.assertSequenceEqual(
        list(rebase * np.round(seq_test, precision)),
        list(rebase * np.round(seq_theo, precision)),
    )
