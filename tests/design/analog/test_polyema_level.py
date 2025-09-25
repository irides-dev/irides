"""
-------------------------------------------------------------------------------

Unit tests for the .polyema_level module of spqf.design.analog.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np
import scipy.special

from irides.design.analog import polyema_level as design
from irides.tools import design_tools

from tests.design.analog import common_tooling


# noinspection SpellCheckingInspection,PyPep8Naming
class TestDesignPolyemaLevel(unittest.TestCase):
    """
    Unit tests for the src.spqf.design.analog.polyema_level module.
    """

    def test_design_id_is_correct(self):
        """
        Tests that the correct design-id is coded.
        """

        design_id_theo = "polyema-level"
        design_id_test = design.design_id()

        self.assertEqual(design_id_test, design_id_theo)

    def test_design_type_is_correct(self):
        """Sanity test to confirm we have a level filter type"""

        design_type_theo = design_tools.FilterDesignType.LEVEL
        common_tooling.design_type_test(self, design, design_type_theo)

    def test_get_valid_filter_orders_returns_basic_array(self):
        """Tests the return of a valid filter-order array."""

        ans = design.get_valid_filter_orders()
        self.assertEqual(ans[0], 1)
        self.assertTrue(ans.shape[0] > 1)

    def test_designs_produce_correct_pole_constellation(self):
        """Tests that the number and coordinate of the poles are correct"""

        filter_orders = design.get_valid_filter_orders()
        for filter_order in filter_orders:
            poles_test = design.designs(filter_order)["poles"]
            poles_theo = -filter_order * np.ones(filter_order, dtype=float)
            self.assertAlmostEqual(
                np.linalg.norm(np.real(poles_test) - np.real(poles_theo)),
                0.0,
                4,
            )
            self.assertAlmostEqual(sum(np.imag(poles_test)), 0, 4)

    def test_designs_produce_correct_polynomial_coefficients(self):
        """Tests that the polynomial coefficiets are binomial"""

        filter_orders = design.get_valid_filter_orders()
        for filter_order in filter_orders:
            coefs_test = design.designs(filter_order)["poly_coeffs"]
            coefs_theo = [
                scipy.special.binom(filter_order, k)
                for k in range(filter_order + 1)
            ]
            self.assertSequenceEqual(coefs_test, coefs_theo)

    def test_integer_sequence_a055897_matches_first_eight_values(self):
        """
        Test that the integer sequence gives expected results on [1, 8].
        """

        # setup theo
        seq_theo = np.array(
            [1, 2, 12, 108, 1280, 18750, 326592, 6588344], dtype=int
        )

        # setup test
        seq_test = np.zeros(8, dtype=int)
        for i, n in enumerate(np.arange(1, 9)):
            seq_test[i] = design.integer_sequence_a055897(n)

        # test
        self.assertSequenceEqual(list(seq_test), list(seq_theo))

    def test_convert_reference_poles_to_temporal_scales_is_correct(self):
        """
        Confirms that a reference pole is correctly converted to a temporal
        scale.
        """

        # these are the order 1, 2 pole locations from the Bessel filter.
        poles = np.array(
            [
                -1.0000000000000000000 + 0j,
                -1.5000000000000000000 - 0.8660254037844386468j,
                -1.5000000000000000000 + 0.8660254037844386468j,
            ]
        )

        # theo values
        tau_theo = 1.0 / np.abs(poles)

        # from the design code
        tau_test = design.convert_reference_poles_to_temporal_scales(poles)

        # test
        precision = 4
        rebase = np.power(10.0, precision)

        self.assertSequenceEqual(
            list(rebase * np.round(tau_test, precision)),
            list(rebase * np.round(tau_theo, precision)),
        )

    def test_tau_per_stage_scales_with_filter_order(self):
        """
        Test for per_stage_value = tau / filter_order.
        """

        # defs
        filter_orders = design.get_valid_filter_orders()
        tau = 1.0

        # setup theo
        moments_theo = tau / filter_orders

        # setup test
        moments_test = np.zeros(filter_orders.shape[0])
        for i, m in enumerate(filter_orders):
            moments_test[i] = design.tau_per_stage(tau, m)

        # test
        precision = 4
        rebase = np.power(10.0, precision)

        self.assertSequenceEqual(
            list(rebase * np.round(moments_test, precision)),
            list(rebase * np.round(moments_theo, precision)),
        )

    def test_unit_peak_time_is_correct_for_eight_orders(self):
        """
        Test that the unit peak time goes as (m - 1) / m.
        """

        # defs
        filter_orders = design.get_valid_filter_orders()

        # setup theo
        unit_peak_times_theo = np.array(
            [0.0000, 0.5000, 0.6667, 0.7500, 0.8000, 0.8333, 0.8571, 0.8750]
        )

        # setup test
        unit_peak_times_test = np.zeros(filter_orders.shape[0])
        for i, m in enumerate(filter_orders):
            unit_peak_times_test[i] = design.unit_peak_time(m)

        # test
        precision = 4
        rebase = np.power(10.0, precision)

        self.assertSequenceEqual(
            list(rebase * np.round(unit_peak_times_test, precision)),
            list(rebase * np.round(unit_peak_times_theo, precision)),
        )

    def test_unit_peak_value_is_correct_for_eight_orders(self):
        """
        Test that unit peak value follows the analytic expression.
        """

        # defs
        filter_orders = design.get_valid_filter_orders()

        # setup test and theo
        unit_peak_values_test = np.zeros(filter_orders.shape[0])

        # theo values calculate from the analytic expression
        unit_peak_values_theo = np.array(
            [1.0000, 0.7358, 0.8120, 0.8962, 0.9768, 1.0528, 1.1244, 1.1920]
        )

        # test value
        for i, m in enumerate(filter_orders):
            unit_peak_values_test[i] = design.unit_peak_value(m)

        # test
        precision = 4
        rebase = np.power(10.0, precision)

        self.assertSequenceEqual(
            list(rebase * np.round(unit_peak_values_test, precision)),
            list(rebase * np.round(unit_peak_values_theo, precision)),
        )

    def test_peak_value_coordinate_scales_with_tau_for_pema(self):
        """
        Tests that the peak time and value are scaled by tau and 1 / tau,
        respectively, for polyema's. For the ema (order=1), the peak time
        is always zero and the peak value scales as 1 / tau.
        """

        # defs
        tau_1 = 1.0
        tau_2 = 3.0

        filter_orders = design.get_valid_filter_orders()
        for m in filter_orders:
            ans_1 = design.peak_value_coordinate(tau_1, m)
            ans_2 = design.peak_value_coordinate(tau_2, m)

            if m == 1:  # ema case
                self.assertAlmostEqual(ans_1.time, 0.0)
                self.assertAlmostEqual(ans_2.time, 0.0)

            else:  # pema case
                ratio_time = ans_2.time / ans_1.time
                self.assertAlmostEqual(ratio_time, tau_2 / tau_1)

            ratio_value = ans_2.value / ans_1.value
            self.assertAlmostEqual(ratio_value, 1.0 / (tau_2 / tau_1))

    def test_moment_value_generators_are_correct(self):
        """
        Test that the anomymous functions returned by moment_value_generator()
        are correct.
        """

        # defs
        tau = 2.0
        dt = 0.005

        # fetch generators
        m0_gen = design.moment_value_generator(0, tau, dt)
        m1_gen = design.moment_value_generator(1, tau, dt)
        m2_gen = design.moment_value_generator(2, tau, dt)

        # test m = 1
        m = 1
        m0_theo = 1.0 + (dt / tau) / 2.0
        m1_theo = tau
        m2_theo = 2.0 * np.power(tau, 2)
        self.assertAlmostEqual(m0_gen(m), m0_theo, 4)
        self.assertAlmostEqual(m1_gen(m), m1_theo, 4)
        self.assertAlmostEqual(m2_gen(m), m2_theo, 4)

        # test m = 3 (!= 1)
        m = 3
        m0_theo = 1.0
        m1_theo = tau
        m2_theo = 1.33333 * np.power(tau, 2)
        self.assertAlmostEqual(m0_gen(m), m0_theo, 4)
        self.assertAlmostEqual(m1_gen(m), m1_theo, 4)
        self.assertAlmostEqual(m2_gen(m), m2_theo, 4)

    def test_moment_value_generator_throws_on_out_of_bounds_moment(self):
        """
        Test that the target function throws.
        """

        self.assertRaises(IndexError, lambda: design.moment_value_generator(10))

    def test_full_width_generator_is_correct(self):
        """
        Test that the anonymous function returned by full_width_generator()
        is correct.
        """

        # defs
        tau = 2.0
        dt = 0.005
        m = 3

        # fetch generator
        fw_gen = design.full_width_generator(tau, dt)

        # test
        fw_theo = 2.0 * np.sqrt(1.0 / m) * tau
        self.assertAlmostEqual(fw_gen(m), fw_theo)

    def test_gain_phase_and_gd_at_dc_frequency(self):
        """
        Test that the gain, phase and group-delay are correct at DC.
        """

        # defs
        tau = 1.0
        precision = 4

        # define setup and theo answers
        setup = {
            "gain": {"call": design.gain_at_dc, "theo": 1.0},
            "phase": {"call": design.phase_at_dc, "theo": 0.0},
            "gd": {"call": design.group_delay_at_dc, "theo": tau},
        }

        for feature, v in setup.items():

            feature_theo = v["theo"]
            feature_test = v["call"](tau)

            with self.subTest(msg="{0}".format(feature)):
                self.assertAlmostEqual(feature_test, feature_theo, precision)

    def test_cutoff_frequencies_are_correct(self):
        """
        Test that the cutoff frequencies follow the analytic expression.
        """

        # defs
        filter_orders = design.get_valid_filter_orders()
        tau = 1.0

        # setup theo
        cutoff_frequencies_theo = (
            filter_orders
            / tau
            * np.sqrt(np.power(2.0, 1.0 / filter_orders) - 1.0)
        )

        # setup test
        cutoff_frequencies_test = np.zeros(filter_orders.shape[0])
        for i, m in enumerate(filter_orders):
            cutoff_frequencies_test[i] = design.cutoff_frequency(tau, m)

        # test
        precision = 4
        rebase = np.power(10.0, precision)

        self.assertSequenceEqual(
            list(rebase * np.round(cutoff_frequencies_test, precision)),
            list(rebase * np.round(cutoff_frequencies_theo, precision)),
        )

    def test_cutoff_frequencies_scale_correctly_with_tau(self):
        """
        Test that wc scales as 1 / tau.
        """

        # defs
        filter_orders = design.get_valid_filter_orders()
        tau_1 = 1.0
        tau_2 = 3.0

        # setup test
        for i, m in enumerate(filter_orders):
            wc_1 = design.cutoff_frequency(tau_1, m)
            wc_2 = design.cutoff_frequency(tau_2, m)

            self.assertAlmostEqual(wc_2 / wc_1, 1.0 / (tau_2 / tau_1))

    def test_gain_at_cutoff_frequencies_is_correct(self):
        """
        Test that the gain at cutoff matches these design value.
        """

        # gain_theo
        tau = 1.0
        gain_theo = np.sqrt(1.0 / 2.0)

        # iter on orders
        filter_orders = design.get_valid_filter_orders()
        precision = 3

        for order in filter_orders:

            gain_test = design.gain_at_cutoff(tau, order)
            self.assertAlmostEqual(gain_test, gain_theo, precision)

    def test_phase_at_cutoff_frequencies_are_correct(self):
        """
        Test
        """

        # defs
        filter_orders = design.get_valid_filter_orders()
        tau = 1.0

        # setup theo
        phase_at_wc_theo = np.array(
            [
                -0.7854,
                -1.1437,
                -1.4144,
                -1.6412,
                -1.8402,
                -2.0198,
                -2.1846,
                -2.3379,
            ]
        )

        # setup test
        phase_at_wc_test = np.zeros(filter_orders.shape[0])
        for i, order in enumerate(filter_orders):
            phase_at_wc_test[i] = design.phase_at_cutoff(tau, order)

        # test
        precision = 4
        rebase = np.power(10.0, precision)

        self.assertSequenceEqual(
            list(rebase * np.round(phase_at_wc_test, precision)),
            list(rebase * np.round(phase_at_wc_theo, precision)),
        )

    def test_group_delay_at_cutoff_frequencies_are_correct(self):
        """
        Test that the group delay at cutoff frequency grp-dly(wc)
        matches analytic expression.
        """

        # defs
        filter_orders = design.get_valid_filter_orders()
        tau = 1.0

        # setup theo
        group_delay_at_wc_theo = np.power(2, -1.0 / filter_orders) * tau

        # setup test
        group_delay_at_wc_test = np.zeros(filter_orders.shape[0])
        for i, order in enumerate(filter_orders):
            group_delay_at_wc_test[i] = design.group_delay_at_cutoff(tau, order)

        # test
        precision = 4
        rebase = np.power(10.0, precision)

        self.assertSequenceEqual(
            list(rebase * np.round(group_delay_at_wc_test, precision)),
            list(rebase * np.round(group_delay_at_wc_theo, precision)),
        )

    def test_autocorrelation_peak_and_stride_values_are_correct(self):
        """
        Test that the sacf zero-lag and 5% stride values are correct.
        """

        # defs
        filter_orders = design.get_valid_filter_orders()
        tau = 1.0

        unit_sacf_values_theo = {
            1: {"kh0": 0.50000, "xi_5pct": 2.99573},
            2: {"kh0": 0.50000, "xi_5pct": 2.37193},
            3: {"kh0": 0.56250, "xi_5pct": 1.97288},
            4: {"kh0": 0.62500, "xi_5pct": 1.71918},
            5: {"kh0": 0.68359, "xi_5pct": 1.54189},
            6: {"kh0": 0.73828, "xi_5pct": 1.40953},
            7: {"kh0": 0.78955, "xi_5pct": 1.30603},
            8: {"kh0": 0.83789, "xi_5pct": 1.22228},
        }

        # test
        precision = 4

        for order in filter_orders:

            unit_sacf_value_test = design.autocorrelation_peak_and_stride_values(
                tau, order
            )

            for k in ["kh0", "xi_5pct"]:
                with self.subTest(
                    msg="order: {0}, feature: {1}".format(order, k)
                ):
                    self.assertAlmostEqual(
                        unit_sacf_value_test[k],
                        unit_sacf_values_theo[order][k],
                        precision,
                    )

        # test for invalid order
        with self.subTest(msg="for invalid order"):

            invalid_order = max(filter_orders) + 1
            unit_sacf_value_test = design.autocorrelation_peak_and_stride_values(
                tau, invalid_order
            )
            self.assertEqual(unit_sacf_value_test["kh0"], 0.0)
            self.assertEqual(unit_sacf_value_test["xi_5pct"], 0.0)

    def test_autocorrelation_peak_and_stride_values_scale_with_tau(self):
        """
        Test that kh(0) scales with 1 / tau and that xi_5pct scales with tau.
        """

        tau_1 = 1.0
        tau_2 = 3.0

        scale_kh0 = 1.0 / (tau_2 / tau_1)
        scale_xi_5cpt = tau_2 / tau_1

        filter_orders = design.get_valid_filter_orders()
        for order in filter_orders:

            ans_1 = design.autocorrelation_peak_and_stride_values(tau_1, order)
            ans_2 = design.autocorrelation_peak_and_stride_values(tau_2, order)

            ratio_kh0 = ans_2["kh0"] / ans_1["kh0"]
            ratio_xi_5pct = ans_2["xi_5pct"] / ans_1["xi_5pct"]

            self.assertAlmostEqual(ratio_kh0, scale_kh0)
            self.assertAlmostEqual(ratio_xi_5pct, scale_xi_5cpt)

    def test_scale_correlation_strides_are_correct(self):
        """Validates that the scale-decorrelation strides are correct."""

        # defs
        strides_theo = np.array(
            [
                1598.00,
                27.4358,
                11.1683,
                7.27680,
                5.60510,
                4.68270,
                4.09790,
                3.69310,
            ]
        )

        # send to test
        common_tooling.scale_correlation_strides(self, design, strides_theo)

    def test_wireframe_is_correct(self):
        """Validates the Fourier wireframe design."""

        # defs
        tau = 2.0

        # send to test
        common_tooling.wireframe_level_features(self, design, tau)

    def test_wavenumber_is_correct(self):
        """Validates the Fourier wireframe wavenumber."""

        # defs
        wns_theo = np.array(
            [
                0.64780859,
                0.72623682,
                0.74588861,
                0.75626296,
                0.76143075,
                0.76197976,
                0.75833512,
                0.75110948,
            ]
        )

        # send to test
        common_tooling.wavenumber_features(self, design, wns_theo)

    def test_uncertainty_product_is_correct(self):
        """Validates that the uncertainty products are correct."""

        # defs
        ucps_theo = np.array(
            [
                np.inf,
                0.866025,
                0.645497,
                0.591608,
                0.566947,
                0.552771,
                0.543557,
                0.537086,
            ]
        )

        # send to test
        common_tooling.uncertainty_products(self, design, ucps_theo)
