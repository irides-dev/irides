"""
-------------------------------------------------------------------------------

Unit tests for the .damped_oscillator module of spqf.design.analog.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.design.analog import damped_oscillator as design
from irides.tools import transfer_function_tools


def generate_cartesian_coord_sequence() -> np.ndarray:
    """
    Generates a test sequence of pole locations in the left-half s-plane.
    """

    # pi/2 < angles < 3pi/2 with respect to positive real axis
    phi_v = (5.0 / 6.0) * (np.pi / 2.0) * np.linspace(-1, 1, 7) + np.pi
    radii = 3.0 * np.ones_like(phi_v)

    return radii * np.cos(phi_v) + 1j * radii * np.sin(phi_v)


# noinspection SpellCheckingInspection,PyPep8Naming
class TestDesignDampedOscillatorElement(unittest.TestCase):
    """
    Unit tests for the src.spqf.design.analog.damped_oscillator module.
    """

    def test_cast_to_cartesian_form_and_cast_to_polar_form(self):
        """
        Tests the cartesian -> polar and polar -> cartesian conversions
        work as expected.
        """

        # build test candidates
        phi_v = 2 * np.pi * np.linspace(-2, 2, 41)
        radii = 5.0 * np.ones_like(phi_v)

        cartesian_cands = radii * np.cos(phi_v) + 1j * radii * np.sin(phi_v)
        polar_cands = np.array([radii, phi_v]).T

        # defs
        precision = 4
        rebase = np.power(10, precision)

        # test convertion to polar
        with self.subTest(msg="cartesian -> polar"):
            polar_test = design.cast_to_polar_form(cartesian_cands)

            # both angles are passed through this wrap function because
            # of the seemingly inevitable +/- pi quirks.
            angle_test = transfer_function_tools.wrap_phase(polar_test[:, 1])
            angle_theo = transfer_function_tools.wrap_phase(polar_cands[:, 1])

            self.assertSequenceEqual(
                list(rebase * np.round(polar_test[:, 0], precision)),
                list(rebase * np.round(polar_cands[:, 0], precision)),
            )

            self.assertSequenceEqual(
                list(rebase * np.round(angle_test, precision)),
                list(rebase * np.round(angle_theo, precision)),
            )

        # test convertion to cartesian
        with self.subTest(msg="polar -> cartesian"):
            cartesian_test = design.cast_to_cartesian_form(polar_cands)

            self.assertSequenceEqual(
                list(rebase * np.round(cartesian_test, precision)),
                list(rebase * np.round(cartesian_cands, precision)),
            )

    def test_unit_peak_times_and_values(self):
        """
        Tests that unit peak times and values calculations are correctly
        implemented.
        """

        # fetch cart coords to test
        cartesian_coords = generate_cartesian_coord_sequence()

        # call test functions
        unit_peak_times_test = design.unit_peak_times(cartesian_coords)
        unit_peak_values_test = design.unit_peak_values(cartesian_coords)

        # record theo values
        unit_peak_times_theo = (1.0 / 3.0) * np.array(
            [1.355173, 1.139183, 1.032451, 1.0, 1.032451, 1.139183, 1.355173]
        )
        unit_peak_values_theo = 3.0 * np.array(
            [
                0.704164,
                0.480825,
                0.392304,
                0.367879,
                0.392304,
                0.480825,
                0.704164,
            ]
        )

        # defs
        precision = 4
        rebase = np.power(10, precision)

        with self.subTest(msg="unit peak times"):
            self.assertSequenceEqual(
                list(rebase * np.round(unit_peak_times_test, precision)),
                list(rebase * np.round(unit_peak_times_theo, precision)),
            )

        with self.subTest(msg="unit peak values"):
            self.assertSequenceEqual(
                list(rebase * np.round(unit_peak_values_test, precision)),
                list(rebase * np.round(unit_peak_values_theo, precision)),
            )

    def test_peak_value_coordinate_scales_with_tau(self):
        """
        Tests that the peak time and value are scaled by tau and 1 / tau,
        respectively, for the damped oscillator.
        """

        # defs
        tau_1 = 1.0
        tau_2 = 3.0

        # fetch cart coords to test
        cartesian_poles = generate_cartesian_coord_sequence()

        # calc coords for taus
        ans_1 = design.peak_value_coordinates(cartesian_poles, tau_1)
        ans_2 = design.peak_value_coordinates(cartesian_poles, tau_2)

        ratio_times = ans_2.times() / ans_1.times()
        ratio_values = ans_2.values() / ans_1.values()

        # defs
        precision = 4
        rebase = np.power(10, precision)

        with self.subTest(msg="scaled times"):
            scaled_times_theo = (tau_2 / tau_1) * np.ones_like(ratio_times)

            self.assertSequenceEqual(
                list(rebase * np.round(ratio_times, precision)),
                list(rebase * np.round(scaled_times_theo, precision)),
            )

        with self.subTest(msg="scaled values"):
            scaled_values_theo = (tau_1 / tau_2) * np.ones_like(ratio_times)

            self.assertSequenceEqual(
                list(rebase * np.round(ratio_values, precision)),
                list(rebase * np.round(scaled_values_theo, precision)),
            )

    def test_nth_null_times_is_correct_for_1st_null(self):
        """
        Test that the 1st null timepoints are correct.
        """

        # fetch cart coords to test
        cartesian_poles = generate_cartesian_coord_sequence()

        # calc 1st null times
        first_null_times_test = design.nth_null_times(cartesian_poles)

        # recorded times
        first_null_times_theo = np.array(
            [1.084139, 1.367019, 2.477881, np.inf, 2.477881, 1.367019, 1.084139]
        )

        # defs
        precision = 4
        rebase = np.power(10, precision)

        self.assertSequenceEqual(
            list(rebase * np.round(first_null_times_test, precision)),
            list(rebase * np.round(first_null_times_theo, precision)),
        )

    def test_moment_value_generators_are_correct(self):
        """
        Test that the anomymous functions returned by moment_value_generator()
        are correct.
        """

        # fetch generators
        m0_gen = design.moment_value_generator(0)
        m1_gen = design.moment_value_generator(1)
        m2_gen = design.moment_value_generator(2)

        # fetch cart coords to test
        cartesian_poles = generate_cartesian_coord_sequence()

        # convert coords to wn, theta
        polar_poles = design.convert_reference_angle_to_negative_real_axis(
            design.cast_to_polar_form(cartesian_poles)
        )
        wn = polar_poles[:, 0]
        theta = polar_poles[:, 1]
        tau = 1.0

        m0_test = m0_gen(wn, theta, tau)
        m1_test = m1_gen(wn, theta, tau)
        m2_test = m2_gen(wn, theta, tau)

        # record theo values
        m0_theo = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        m1_theo = np.array(
            [
                0.172546,
                0.428525,
                0.604205,
                0.666667,
                0.604205,
                0.428525,
                0.172546,
            ]
        )
        m2_theo = np.array(
            [
                -0.16267796,
                0.14504525,
                0.5079056,
                0.66666667,
                0.5079056,
                0.14504525,
                -0.16267796,
            ]
        )

        # defs
        precision = 4
        rebase = np.power(10, precision)

        with self.subTest(msg="M0 test"):
            self.assertSequenceEqual(
                list(rebase * np.round(m0_test, precision)),
                list(rebase * np.round(m0_theo, precision)),
            )
        with self.subTest(msg="M1 test"):
            self.assertSequenceEqual(
                list(rebase * np.round(m1_test, precision)),
                list(rebase * np.round(m1_theo, precision)),
            )
        with self.subTest(msg="M2 test"):
            self.assertSequenceEqual(
                list(rebase * np.round(m2_test, precision)),
                list(rebase * np.round(m2_theo, precision)),
            )

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

        # fetch generators
        fw_gen = design.full_width_generator()

        # fetch cart coords to test
        cartesian_poles = generate_cartesian_coord_sequence()

        # convert coords to wn, theta
        polar_poles = design.convert_reference_angle_to_negative_real_axis(
            design.cast_to_polar_form(cartesian_poles)
        )
        wn = polar_poles[:, 0]
        theta = polar_poles[:, 1]
        tau = 1.0

        fw_test = fw_gen(wn, theta, tau)

        # record theo values
        fw_theo = np.array(
            [np.nan, np.nan, 0.75588806, 0.94280904, 0.75588806, np.nan, np.nan]
        )

        # defs
        precision = 4

        with self.subTest(msg="FW test"):

            # break down this test b/c of the nan values
            for i, fw in enumerate(fw_test):

                if np.isnan(fw_theo[i]):
                    self.assertTrue(np.isnan(fw))
                else:
                    self.assertAlmostEqual(fw, fw_theo[i], precision)

    def test_key_spectral_features_are_correct_for_over_and_underdamped(self):
        """
        Tests key spectral features, which included the damping type (under or
        over), wmax, g(wmax), phase(wmax) and group-delay(wmax) for each type.
        """

        # fetch cart coords to test
        cartesian_coords = generate_cartesian_coord_sequence()

        # defs
        tau = 1.0
        n_pts = cartesian_coords.shape[0]

        precision = 4
        rebase = np.power(10, precision)

        # test features
        features_test = design.key_spectral_features(cartesian_coords, tau)

        # theo features
        features_theo = {
            "damp-type": np.array(
                ["under", "under", "over", "over", "over", "under", "under"]
            ),
            "over": {
                "wmax": np.zeros(n_pts),
                "gain": np.ones(n_pts),
                "phase": np.zeros(n_pts),
                "gd": np.array(
                    [
                        0.17254603,
                        0.42852507,
                        0.60420519,
                        0.66666667,
                        0.60420519,
                        0.42852507,
                        0.17254603,
                    ]
                ),
            },
            "under": {
                "wmax": np.array(
                    [
                        2.79181458,
                        1.25013343,
                        np.nan,
                        np.nan,
                        np.nan,
                        1.25013343,
                        2.79181458,
                    ]
                ),
                "gain": np.array(
                    [2.0, 1.01542661, np.nan, np.nan, np.nan, 1.01542661, 2.0]
                ),
                "phase": np.array(
                    [
                        -1.29953257,
                        -0.57517038,
                        np.nan,
                        np.nan,
                        np.nan,
                        -0.57517038,
                        -1.29953257,
                    ]
                ),
                "gd": np.array(
                    [
                        1.2879011,
                        0.51857461,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.51857461,
                        1.2879011,
                    ]
                ),
            },
        }

        with self.subTest(msg="damp-type"):

            for i, dt in enumerate(features_theo["damp-type"]):

                self.assertEqual(features_test["damp-type"][i], dt)

        cols = {0: "wmax", 1: "gain", 2: "phase", 3: "gd"}
        cols_i = np.arange(4)

        for col_i in cols_i:

            col_name = cols[col_i]

            with self.subTest(msg="over {0}".format(col_name)):

                self.assertSequenceEqual(
                    list(
                        rebase
                        * np.round(features_test["over"][:, col_i], precision)
                    ),
                    list(
                        rebase
                        * np.round(features_theo["over"][col_name], precision)
                    ),
                )

        for col_i in cols_i:

            col_name = cols[col_i]

            with self.subTest(msg="under {0}".format(col_name)):

                for i in range(cartesian_coords.shape[0]):

                    v_test = features_test["under"][i, col_i]
                    v_theo = features_theo["under"][col_name][i]

                    if not np.isnan(v_theo):

                        self.assertAlmostEqual(v_test, v_theo, precision)

                    else:

                        self.assertTrue(np.isnan(v_test))

    # noinspection SpellCheckingInspection,PyPep8Naming
    def test_autocorrelation_peak_and_stride_values_are_correct(self):

        # fetch cart coords to test
        cartesian_coords = generate_cartesian_coord_sequence()

        # set scale
        tau = 7.0

        # fetch sacf test values
        sacf_test = design.autocorrelation_peak_and_stride_values(
            cartesian_coords, tau
        )

        # record theo values
        sacf_theo = {
            "kh0": (1.0 / tau)
            * np.array(
                [
                    2.897777,
                    1.166793,
                    0.827533,
                    0.75,
                    0.827533,
                    1.166793,
                    2.897777,
                ]
            ),
            "xi_5pct": tau
            * np.array(
                [
                    3.902856,
                    1.691719,
                    1.418583,
                    1.581288,
                    1.418583,
                    1.691719,
                    3.902856,
                ]
            ),
        }

        # defs
        precision = 4
        rebase = np.power(10, precision)

        for k in sacf_theo:
            with self.subTest(msg="test {0}".format(k)):
                self.assertSequenceEqual(
                    list(rebase * np.round(sacf_test[k], precision)),
                    list(rebase * np.round(sacf_theo[k], precision)),
                )
