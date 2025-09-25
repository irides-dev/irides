"""
-------------------------------------------------------------------------------

Unit tests for the .bessel_level module of spqf.design.analog.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np
import scipy.signal
import scipy.special

from irides.design.analog import bessel_level as design
from irides.tools import design_tools

from tests.design.analog import common_tooling


# noinspection SpellCheckingInspection,PyPep8Naming
class TestDesignBesselLevel(unittest.TestCase):
    """
    Unit tests for the src.spqf.design.analog.bessel_level module.
    """

    def test_design_id_is_correct(self):
        """
        Tests that the correct design-id is coded.
        """

        design_id_theo = "bessel-level"
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

    def test_designs_poles_and_residues_produce_correct_xfer_function(self):
        """
        Partial-fraction expansion (PFE) of the transfer function reads

                     r[0]       r[1]             r[-1]
            H(s) = -------- + -------- + ... + --------- + k(s)
                   (s-p[0])   (s-p[1])         (s-p[-1])

        where p[] are the poles, r[] are the associated residues, and k[] are
        direct terms. There are no direct terms for Bessel transfer functions.

        Multiplication of the PFE terms reconstructs H(s), and thereby
        constructs the original numerator and denominator polynomials.

        The numerator of the Bessel transfer functions is a constant that
        matches the 0th-order Bessel polynomial coefficient. So, for a 3rd
        order transfer function, the numerator polynomial reads

            n(s) = 0 s^3 + 0 s^2 + 0 s^1 + 15.

        The poles coded in the design.designs[<order>]['poles'] dictionary
        correspond to coefficients of the reverse Bessel polynomial theta_n(s)
        (https://en.wikipedia.org/wiki/Bessel_polynomials). For instance, a
        3rd order polynomial reads

            p3(s) = s^3 + 6 s^2 + 15 s + 15.

        The coefficients in the arrays below follow this pattern of descending
        order in polynomial power.

        When this test passes, the design poles, residues, and their
        association, are clearly correct. What is not tested is the precision
        with which the poles and residues are recorded. For that I defer to
        Mathematica (v12) for extended precision.
        """

        # original Bessel polynomial coefficients
        bessel_poly_coeffs_dict = {
            1: np.array([1, 1]),
            2: np.array([1, 3, 3]),
            3: np.array([1, 6, 15, 15]),
            4: np.array([1, 10, 45, 105, 105]),
            5: np.array([1, 15, 105, 420, 945, 945]),
            6: np.array([1, 21, 210, 1260, 4725, 10395, 10395]),
            7: np.array([1, 28, 378, 3150, 17325, 62370, 135135, 135135]),
            8: np.array(
                [1, 36, 630, 6930, 51975, 270270, 945945, 2027025, 2027025]
            ),
        }

        # fetch valid orders and designs
        orders = design.get_valid_filter_orders()

        # test prep
        precision = 4
        rebase = np.power(10.0, precision)

        # iter over order
        for order in orders:

            # fetch poles and residues for this order
            this_design = design.designs(order)
            poles = this_design["poles"]
            residues = this_design["residues"]
            k = np.array([])

            # multiply out the PFE coeffs
            num_denom_coeffs_test = scipy.signal.invres(residues, poles, k)

            with self.subTest(msg="order: {0}, test poles".format(order)):

                # extract the test answer
                bessel_poly_coeffs_test = num_denom_coeffs_test[-1]

                # fetch theo answer
                bessel_poly_coeffs_theo = bessel_poly_coeffs_dict[order]

                # test
                self.assertSequenceEqual(
                    list(rebase * np.round(bessel_poly_coeffs_test, precision)),
                    list(rebase * np.round(bessel_poly_coeffs_theo, precision)),
                )

            with self.subTest(msg="order: {0}, test residues".format(order)):

                # extract the test answer
                zeroth_order_num_coeff_test = num_denom_coeffs_test[0][-1]

                # fetch the theo answer
                zeroth_order_num_coeff_theo = bessel_poly_coeffs_dict[order][-1]

                # test
                self.assertAlmostEqual(
                    zeroth_order_num_coeff_test,
                    zeroth_order_num_coeff_theo,
                    precision,
                )

                # test that any higher-order poly coeffs are essentially zero
                for coeff_test in num_denom_coeffs_test[0][:-1]:
                    self.assertAlmostEqual(np.abs(coeff_test), 0.0, precision)

    def test_designs_poles_inverses_sum_to_minus_one_for_each_order(self):
        """
        Confirm a property of reverse Bessel polynomials that the sum of
        inverse roots is -1.
        """

        # fetch valid orders
        orders = design.get_valid_filter_orders()

        # test prep
        precision = 4
        sum_theo = -1.0

        # iter over order
        for order in orders:

            poles = design.designs(order)["poles"]
            sum_test = np.sum(1.0 / poles)

            # test
            self.assertAlmostEqual(sum_test, sum_theo, precision)

    def test_designs_residues_sum_to_zero_for_higher_orders(self):
        """
        For orders > 1, the Bessel impulse response at t = 0 is zero:

            h_m(t = 0) = 0,  m > 1.

        A consequence is that

            Sum_i A_i = 0

        where A_i is the ith residue of the partial-fraction expansion of Hm(s).
        This test confirms this trait.
        """

        # fetch valid orders
        orders = design.get_valid_filter_orders()

        # test prep
        precision = 4

        # iter over order
        for order in orders:

            # ignore order = 1 because this test does not apply
            if order == 1:
                continue

            # fetch residues for this order
            residues = design.designs(order)["residues"]

            # test
            self.assertAlmostEqual(np.sum(residues), 0.0, precision)

    def test_designs_pole_residue_products_sum_to_zero_for_m_gt_two(self):
        """
        For orders > 2, the Bessel impulse response has zero derivative
        at t = 0:

            h_m'(t=0) = 0, m > 2.

        A consequence is that

            Sum_i p_i A_i = 0.

        This test confirms this trait.
        """

        # fetch valid orders
        orders = design.get_valid_filter_orders()

        # test prep
        precision = 4

        # iter over order
        for order in orders:

            # ignore orders 1 and 2 because this test does not apply
            if 1 <= order < 3:
                continue

            # fetch poles and residues for this order
            this_design = design.designs(order)
            poles = this_design["poles"]
            residues = this_design["residues"]

            # test
            self.assertAlmostEqual(np.sum(poles * residues), 0.0, precision)

    def test_designs_per_stage_imaginary_parts_sum_to_zero(self):
        """
        There are no purely imaginary components in the purely real
        impulse response of this filter. Therefore, confirm that no stage
        has an imaginary component that is not zero.
        """

        # fetch valid orders
        orders = design.get_valid_filter_orders()

        # test prep
        precision = 4

        # iter over order
        for order in orders:

            # fetch stage indexing and poles
            this_design = design.designs(order)
            stages = this_design["stages"]
            poles = this_design["poles"]

            with self.subTest(msg="order: {0}".format(order)):

                # iter on stages
                for stage in stages:
                    # each stage is a dict:
                    #   {'type': <stage-type>, 'indices': <indices>}

                    # imag part of stage pole sum
                    imag_total_test = np.imag(np.sum(poles[stage["indices"]]))

                    # test
                    self.assertAlmostEqual(imag_total_test, 0.0, precision)

    def test_unit_peak_time_is_approx_correct_for_valid_orders(self):

        # defs
        filter_orders = design.get_valid_filter_orders()

        # setup theo
        unit_peak_times_theo = np.array(
            [0.0000, 0.6046, 0.8135, 0.9005, 0.9426, 0.9650, 0.9778, 0.9855]
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

    def test_unit_peak_value_is_approx_correct_for_valid_orders(self):

        # defs
        filter_orders = design.get_valid_filter_orders()

        # setup test and theo
        unit_peak_values_test = np.zeros(filter_orders.shape[0])

        # theo values calculate from the analytic expression
        unit_peak_values_theo = np.array(
            [1.0000, 0.6994, 0.8164, 0.9522, 1.0855, 1.2120, 1.3311, 1.4430]
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
        respectively, for Bessel filters. For the ema (order=1), the peak time
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
        are correct to within error bounds.
        """

        # defs
        tau = 1.0
        dt = 0.005

        # fetch generators
        m0_gen = design.moment_value_generator(0, tau, dt)
        m1_gen = design.moment_value_generator(1, tau, dt)
        m2_gen = design.moment_value_generator(2, tau, dt)

        # theo values
        m0_theo = 1.0
        m1_theo = tau

        def m2_theo_gen(_m):
            return (2.0 * _m) / (2.0 * _m - 1.0) * np.power(tau, 2)

        # test that even order have no dt-scale correction applied,
        #   which is to say, the test and theo values match.
        even_orders = np.arange(2, 10, 2)
        for m in even_orders:
            with self.subTest(msg="order: {0}".format(m)):
                m2_theo = m2_theo_gen(m)
                self.assertAlmostEqual(m0_gen(m), m0_theo, 4)
                self.assertAlmostEqual(m1_gen(m), m1_theo, 4)
                self.assertAlmostEqual(m2_gen(m), m2_theo, 4)

        # test that odd orders have a dt-scale correction applied.
        # In all cases, the correction is positive. Here, the correction
        # is only bounded.
        m0_dt_scale_correction_upper = 5.0 * (dt / 2.0)
        m1_dt_scale_correction_upper = 4.0 * (dt / 2.0)
        m2_dt_scale_correction_upper = 3.5 * (dt / 2.0)

        odd_orders = np.arange(1, 9, 2)
        for m in odd_orders:
            with self.subTest(msg="order: {0}".format(m)):
                m2_theo = m2_theo_gen(m)

                self.assertTrue(
                    m0_theo
                    <= m0_gen(m)
                    <= m0_theo + m0_dt_scale_correction_upper
                )
                self.assertTrue(
                    m1_theo
                    <= m1_gen(m)
                    <= m1_theo + m1_dt_scale_correction_upper
                )
                self.assertTrue(
                    m2_theo
                    <= m2_gen(m)
                    <= m2_theo + m2_dt_scale_correction_upper
                )

    def test_moment_value_generator_throws_on_out_of_bounds_moment(self):
        """
        Test that the target function throws.
        """

        self.assertRaises(IndexError, lambda: design.moment_value_generator(10))

    def test_full_width_generator_is_correct(self):
        """
        Test that the anonymous function returned by full_width_generator()
        is correct to within bounds.
        """

        # defs
        tau = 1.0
        dt = 0.005
        m = 3

        # fetch generator
        fw_gen = design.full_width_generator(tau, dt)

        # theo values
        fw_theo_gen = lambda _m: 2.0 * np.sqrt(1.0 / (2.0 * _m - 1.0))

        # even orders have no dt correction
        even_orders = np.arange(2, 10, 2)
        for m in even_orders:
            with self.subTest(msg="order: {0}".format(m)):
                fw_theo = fw_theo_gen(m)
                self.assertAlmostEqual(fw_gen(m), fw_theo, 4)

        # odd orders have a dt correction
        fw_dt_scale_correction_lower = -18.0 * (dt / 2.0)

        odd_orders = np.arange(1, 9, 2)
        for m in odd_orders:
            with self.subTest(msg="order: {0}".format(m)):
                fw_theo = fw_theo_gen(m)
                self.assertTrue(
                    fw_theo + fw_dt_scale_correction_lower
                    <= fw_gen(m)
                    <= fw_theo
                )

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
        Test that cutoff freqs match these test values (not a strong test).
        """

        # defs
        tau = 1.0

        # cutoff_theo
        cutoff_freqs_theo = {
            1: 1.00,
            2: 1.36,
            3: 1.76,
            4: 2.11,
            5: 2.43,
            6: 2.70,
            7: 2.95,
            8: 3.18,
        }

        # iter on orders
        filter_orders = design.get_valid_filter_orders()
        precision = 2

        for order in filter_orders:
            cutoff_freq_test = design.cutoff_frequency(tau, order)
            cutoff_freq_theo = cutoff_freqs_theo[order]

            self.assertAlmostEqual(
                cutoff_freq_test, cutoff_freq_theo, precision
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
            1: -0.785,
            2: -1.297,
            3: -1.736,
            4: -2.109,
            5: -2.426,
            6: -2.703,
            7: -2.952,
            8: -3.180,
        }
        tau = 1.0

        # iter on orders
        filter_orders = design.get_valid_filter_orders()
        precision = 3

        for order in filter_orders:
            phase_test = design.phase_at_cutoff(tau, order)
            phase_theo = phases_theo[order]

            self.assertAlmostEqual(phase_test, phase_theo, precision)

    def test_group_delay_at_cutoff_frequencies_are_correct(self):
        """
        Test that the group delay at cutoff matches these test values
        (not a strong test).
        """

        # defs
        tau = 3

        # gds_theo
        gds_theo = {
            1: 0.500,
            2: 0.809,
            3: 0.935,
            4: 0.982,
            5: 0.996,
            6: 0.999,
            7: 1.000,
            8: 1.000,
        }

        # iter on orders
        filter_orders = design.get_valid_filter_orders()
        precision = 3

        for order in filter_orders:
            gd_test = design.group_delay_at_cutoff(tau, order)
            gd_theo = gds_theo[order] * tau

            self.assertAlmostEqual(gd_test, gd_theo, precision)

    def test_autocorrelation_peak_and_stride_values_are_correct(self):
        """
        Test that the sacf zero-lag and 5% stride values are correct.
        Scaling by the first moment is embedded into this test in that
        a non-unit-scale value is used.
        """

        # defs
        filter_orders = design.get_valid_filter_orders()
        first_moment = 3.0
        apply_first_moment = {
            "kh0": lambda tau: 1.0 / tau,
            "xi_5pct": lambda tau: tau,
        }

        unit_sacf_values_theo = {
            1: {"kh0": 0.5000, "xi_5pct": 2.9957},
            2: {"kh0": 0.4994, "xi_5pct": 2.1882},
            3: {"kh0": 0.5999, "xi_5pct": 1.6556},
            4: {"kh0": 0.7034, "xi_5pct": 1.3604},
            5: {"kh0": 0.8027, "xi_5pct": 1.1733},
            6: {"kh0": 0.8949, "xi_5pct": 1.0473},
            7: {"kh0": 0.9779, "xi_5pct": 0.9594},
            8: {"kh0": 1.0563, "xi_5pct": 0.8909},
        }

        # test
        precision = 4

        for order in filter_orders:

            unit_sacf_value_test = design.autocorrelation_peak_and_stride_values(
                first_moment, order
            )

            for k in ["kh0", "xi_5pct"]:
                with self.subTest(
                    msg="order: {0}, feature: {1}".format(order, k)
                ):
                    self.assertAlmostEqual(
                        unit_sacf_value_test[k],
                        unit_sacf_values_theo[order][k]
                        * apply_first_moment[k](first_moment),
                        precision,
                    )

    def test_scale_correlation_strides_are_correct(self):
        """Validates that the scale-decorrelation strides are correct."""

        # defs
        strides_theo = np.array(
            [
                1598.00,
                22.9564,
                8.64193,
                5.46444,
                4.17151,
                3.48665,
                3.06588,
                2.78165,
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
                0.64778095,
                0.75316630,
                0.77811737,
                0.78250289,
                0.76528238,
                0.74164804,
                0.71391692,
                0.68803663,
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
                0.763763,
                0.569868,
                0.531105,
                0.517241,
                0.510926,
                0.507532,
                0.505463,
            ]
        )

        # send to test
        common_tooling.uncertainty_products(self, design, ucps_theo)
