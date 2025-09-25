"""
-------------------------------------------------------------------------------

Unit tests for the .mbox_level module of spqf.design.analog.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.design.analog import mbox_level as design
from irides.tools import design_tools

from tests.design.analog import common_tooling


# noinspection SpellCheckingInspection,PyPep8Naming
class TestDesignMBoxLevel(unittest.TestCase):
    """
    Unit tests for the src.spqf.design.analog.mbox_level module.
    """

    def test_design_id_is_correct(self):
        """
        Tests that the correct design-id is coded.
        """

        design_id_theo = "mbox-level"
        design_id_test = design.design_id()

        self.assertEqual(design_id_test, design_id_theo)

    def test_design_type_is_correct(self):
        """Sanity test to confirm we have a level filter type"""

        design_type_theo = design_tools.FilterDesignType.LEVEL
        common_tooling.design_type_test(self, design, design_type_theo)

    def test_get_valid_filter_order_returns_basic_array(self):
        """Tests the return of a valid filter-order array."""

        ans = design.get_valid_filter_orders()
        self.assertEqual(ans[0], 1)
        self.assertTrue(ans.shape[0] > 1)

    def test_convert_tau_to_T(self):
        """Confirm that T = tau / 2."""

        tau = 3.0
        T_theo = 2.0 * tau
        T_test = design.convert_tau_to_T(tau)

        self.assertAlmostEqual(T_test, T_theo, 4)

    def test_convert_T_to_tau(self):
        """Confirm that tau = 2 T."""

        T = 3.0
        tau_theo = T / 2.0
        tau_test = design.convert_T_to_tau(T)

        self.assertAlmostEqual(tau_test, tau_theo, 4)

    def test_unit_peak_times_are_always_one(self):
        """
        Tests that the unit peak time is always 1, since M1 = 1.
        """

        unit_peak_time_theo = 1.0
        filter_orders = design.get_valid_filter_orders()
        for order in filter_orders:
            ans = design.unit_peak_time(order)
            self.assertEqual(ans, unit_peak_time_theo)

    def test_unit_peak_values_are_approx_correct_for_valid_orders(self):
        """
        Tests that the unit peak values are correct for valid orders,
        and 0 otherwise.
        """

        filter_orders = design.get_valid_filter_orders()
        dt = 0.05
        precision = 2
        peak_values_theo = np.array(
            [0.5, 1.0, 1.13, 1.33, 1.5, 1.65, 1.79, 1.92]
        )
        order_correction = 2
        peak_values_theo[order_correction - 1] -= dt

        for i, order in enumerate(filter_orders):
            peak_value_test = design.unit_peak_value(order, dt)
            self.assertAlmostEqual(
                peak_value_test, peak_values_theo[i], precision
            )

        invalid_orders = max(filter_orders) + filter_orders
        for order in invalid_orders:
            ans = design.unit_peak_value(order)
            self.assertEqual(ans, 0.0)

    def test_peak_value_coordinate_scale_with_T(self):
        """
        Tests that the peak time and value are scaled by T and 1 / T,
        respectively.
        """

        T_1 = 1.0
        T_2 = 3.0
        tau_1 = design.convert_T_to_tau(T_1)
        tau_2 = design.convert_T_to_tau(T_2)

        filter_orders = design.get_valid_filter_orders()
        for order in filter_orders:
            ans_1 = design.peak_value_coordinate(tau_1, order)
            ans_2 = design.peak_value_coordinate(tau_2, order)

            ratio_time = ans_2.time / ans_1.time
            ratio_value = ans_2.value / ans_1.value

            self.assertAlmostEqual(ratio_time, T_2 / T_1)
            self.assertAlmostEqual(ratio_value, 1.0 / (T_2 / T_1))

    def test_moment_value_generators_are_correct(self):
        """
        Test that the anomymous function returned by moment_value_generator()
        are correct.
        """

        # defs
        tau = 2.0
        dt = 0.005
        m = 3

        # fetch generators
        m0_gen = design.moment_value_generator(0, tau, dt)
        m1_gen = design.moment_value_generator(1, tau, dt)
        m2_gen = design.moment_value_generator(2, tau, dt)

        # theo vals
        m0_theo = 1.0
        m1_theo = tau - dt / 2.0
        m2_theo = (3.0 * m + 1.0) / (3.0 * m) * np.power(tau, 2) - dt / 2.0

        # test
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
        fw_theo = 2.0 * np.sqrt(
            1.0 / (3.0 * m) * np.power(tau, 2) - np.power(dt / tau, 2)
        )

        self.assertAlmostEqual(fw_gen(m), fw_theo, 4)

    def test_first_lobe_gain_and_frequency_generator_for_correct_values(self):
        """
        Tests that the 1st-lobe freqs and gain levels are correct
        """

        T = 2.0
        tau = design.convert_T_to_tau(T)

        filter_orders = design.get_valid_filter_orders()
        for order in filter_orders:
            w_test, g_test = design.first_lobe_frequency_and_gain_values(
                tau, order
            )
            w_theo = 3.0 * np.pi / (T / order)
            g_theo = np.power(2.0 / (3.0 * np.pi), order)

            self.assertAlmostEqual(w_test, w_theo)
            self.assertAlmostEqual(g_test, g_theo)

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

    def test_cutoff_frequnecies_are_non_zero_for_valid_orders(self):
        """
        Test that cutoff freqs are > 0 for valid orders, and zero otherwise.
        """

        T = 1.0
        tau = design.convert_T_to_tau(T)

        filter_orders = design.get_valid_filter_orders()
        for order in filter_orders:
            ans = design.cutoff_frequency(tau, order)
            self.assertTrue(ans > 1.0)

    def test_gain_at_cutoff_frequencies_is_correct(self):
        """
        Test that the gain at cutoff matches these design value.
        """

        # gain_theo
        gain_theo = np.sqrt(1.0 / 2.0)
        tau = 1.0

        # iter on orders
        filter_orders = design.get_valid_filter_orders()
        precision = 3

        for order in filter_orders:

            gain_test = design.gain_at_cutoff(tau, order)
            self.assertAlmostEqual(gain_test, gain_theo, precision)

    def test_phase_at_cutoff_frequencies_are_correct(self):
        """
        Test that the phase at cutoff matches these test values
        (not a strong test).
        """

        # phase_theo
        phases_theo = {
            1: -1.39,
            2: -2.00,
            3: -2.47,
            4: -2.86,
            5: -3.20,
            6: -3.51,
            7: -3.80,
            8: -4.06,
        }
        tau = 1.0

        # iter on orders
        filter_orders = design.get_valid_filter_orders()
        precision = 2

        for order in filter_orders:

            phase_test = design.phase_at_cutoff(tau, order)
            phase_theo = phases_theo[order]

            self.assertAlmostEqual(phase_test, phase_theo, precision)

    def test_group_delay_at_cutoff_is_always_one_half(self):
        """
        Test the group-delay values at cutoff are always T / 2.
        """

        T = 1
        tau = design.convert_T_to_tau(T)

        filter_orders = design.get_valid_filter_orders()
        for order in filter_orders:
            ans = design.group_delay_at_cutoff(tau, order)
            self.assertEqual(ans, 1.0 / 2.0)

    def test_sacf_values_have_suitable_values_for_valid_orders(self):
        """
        Test that the kh(0) and xi_5pct values are >= 1 and < 1, respectively.
        """

        Tcal = 2.0
        tau_cal = design.convert_T_to_tau(Tcal)
        precision = 4
        kh0_theo = np.array(
            [0.5000, 0.6667, 0.8250, 0.9587, 1.0760, 1.1818, 1.2788, 1.3690]
        )
        xi_5pct_theo = np.array(
            [
                1.9,
                1.4152,
                1.15309,
                0.998982,
                0.893614,
                0.815806,
                0.755319,
                0.706554,
            ]
        )

        filter_orders = design.get_valid_filter_orders()
        for i, order in enumerate(filter_orders):
            ans_test = design.autocorrelation_peak_and_stride_values(
                tau_cal, order
            )
            self.assertAlmostEqual(ans_test["kh0"], kh0_theo[i], precision)
            self.assertAlmostEqual(
                ans_test["xi_5pct"], xi_5pct_theo[i], precision
            )

        invalid_orders = max(filter_orders) + filter_orders
        for order in invalid_orders:
            ans_test = design.autocorrelation_peak_and_stride_values(
                tau_cal, order
            )
            self.assertEqual(ans_test["kh0"], 0.0)
            self.assertEqual(ans_test["xi_5pct"], 0.0)

    def test_sacf_values_scale_correctly_with_T(self):
        """
        Test that kh(0) scales with 1 / T and that xi_5pct scales with T.
        """

        T_1 = 1.0
        T_2 = 3.0

        scale_kh0 = 1.0 / (T_2 / T_1)
        scale_xi_5cpt = T_2 / T_1

        filter_orders = design.get_valid_filter_orders()
        for order in filter_orders:
            ans_1 = design.autocorrelation_peak_and_stride_values(T_1, order)
            ans_2 = design.autocorrelation_peak_and_stride_values(T_2, order)

            ratio_kh0 = ans_2["kh0"] / ans_1["kh0"]
            ratio_xi_5pct = ans_2["xi_5pct"] / ans_1["xi_5pct"]

            self.assertAlmostEqual(ratio_kh0, scale_kh0)
            self.assertAlmostEqual(ratio_xi_5pct, scale_xi_5cpt)

    def test_scale_correlation_strides_are_correct(self):
        """Validates that the scale-decorrelation strides are correct."""

        # defs
        strides_theo = np.array(
            [
                400.000,
                9.65489,
                4.60289,
                3.36003,
                2.81947,
                2.51131,
                2.30873,
                2.16416,
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
                0.76907485,
                0.79476859,
                0.76418367,
                0.72115753,
                0.68203083,
                0.65227649,
                0.63221627,
                0.62143716,
            ]
        )

        # send to test
        common_tooling.wavenumber_features(self, design, wns_theo)

    def test_uncertainty_product_is_correct(self):
        """Validates that the uncertainty products are correct."""

        # defs
        ucps_theo = np.array(
            [
                247.023,
                0.547723,
                0.503822,
                0.50123,
                0.500706,
                0.500474,
                0.500342,
                0.500258,
            ]
        )

        # send to test
        common_tooling.uncertainty_products(self, design, ucps_theo)
