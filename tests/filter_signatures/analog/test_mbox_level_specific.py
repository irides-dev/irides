"""
-------------------------------------------------------------------------------

Unit tests for the .mbox_level module of spqf.design.filter_signatures.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.design.analog import mbox_level as design
from irides.filter_signatures.analog import mbox_level as f_sig

from irides.tools import transfer_function_tools as xfer_tools


# noinspection SpellCheckingInspection,PyPep8Naming
class TestMBoxLevelSpecific(unittest.TestCase):
    """
    Unit tests for the mbox-level filter are predominantly covered
    by the parameterized tests in test-level-filters. Here, tests that
    are not common to all level filters made available.
    """

    def test_impulse_response_has_zero_value_at_T(self):
        """
        The t = T position shall have a zero value.
        """

        # defs
        filter_orders = design.get_valid_filter_orders()
        t_start = -1.0
        t_end = 2.0
        T = 1.0
        tau = design.convert_T_to_tau(T)
        dt = 0.001

        # compute and test
        for order in filter_orders:
            ts: np.array = f_sig.generate_impulse_response(
                t_start, t_end, dt, tau, order
            )

            # index to t = T
            i_T = np.where(T - dt / 2 < ts[:, 0])[0][0]
            self.assertEqual(ts[i_T, 1], 0.0)

            # if exists t < T points, then test that the preceding pt is nonzero
            if i_T > 0:
                self.assertGreater(ts[i_T - 1, 1], 0.0)

    def test_generate_spectra_for_1st_lobe_freq_and_gain_over_all_orders(self):
        """
        Tests that the test frequency and gain at the peak of the first
        sidelobe match theory.
        """

        # defs
        df = 0.001
        T = 2.0
        tau = design.convert_T_to_tau(T)

        # fetch valid filter orders
        filter_orders = design.get_valid_filter_orders()

        # iter over filter orders
        for order in filter_orders:

            # compute [f_start, f_end] from the theo peak lobe freq
            w_theo, g_theo = design.first_lobe_frequency_and_gain_values(
                tau, order
            )
            f_theo = w_theo / (2.0 * np.pi)
            f_start = f_theo - 0.25
            f_end = f_theo + 0.25

            # generate a spectrum panel
            spectra_test = f_sig.generate_spectra(
                f_start, f_end, df, tau, order
            )

            # find the peak freq on the f_axis, find assoc gain
            f_axis = spectra_test[:, xfer_tools.SpectraColumns.FREQ.value]
            i_f_test = np.where(f_theo - df / 2.0 < f_axis)[0][0]
            f_test = f_axis[i_f_test]
            g_test = spectra_test[
                i_f_test, xfer_tools.SpectraColumns.GAIN.value
            ]

            # tests
            with self.subTest(msg="order: {0}".format(order)):
                self.assertAlmostEqual(f_test, f_theo, 4)
                self.assertAlmostEqual(g_test, g_theo, 4)

    def test_generate_spectra_for_correct_linear_phase(self):
        """
        Test that the phase spectrum is linear in frequency.
        """

        # defs
        f_start = 0.0
        f_end = 2.0
        df = 0.01
        T = 1.0
        tau = design.convert_T_to_tau(T)

        # fetch valid filter orders
        filter_orders = design.get_valid_filter_orders()

        # iter over filter orders
        for order in filter_orders:

            # generate a spectrum panel
            spectra_test = f_sig.generate_spectra(
                f_start, f_end, df, tau, order
            )

            # compute phase_theo
            f_axis = spectra_test[:, xfer_tools.SpectraColumns.FREQ.value]
            phase_theo = -(2.0 * np.pi * f_axis) * T / 2.0

            # take the phase_test
            phase_test = spectra_test[:, xfer_tools.SpectraColumns.PHASE.value]

            # test
            precision = 4
            rebase = np.power(10.0, precision)

            with self.subTest(msg="order: {0}".format(order)):
                self.assertSequenceEqual(
                    list(rebase * np.round(phase_test, precision)),
                    list(rebase * np.round(phase_theo, precision)),
                )

    def test_generate_spectra_for_correct_constant_group_delay(self):

        # defs
        f_start = 0.0
        f_end = 2.0
        df = 0.01
        T = 1.0
        tau = design.convert_T_to_tau(T)

        # fetch valid filter orders
        filter_orders = design.get_valid_filter_orders()

        # iter over filter orders
        for order in filter_orders:

            # generate a spectrum panel
            spectra_test = f_sig.generate_spectra(
                f_start, f_end, df, tau, order
            )

            # compute gd_theo
            f_axis = spectra_test[:, xfer_tools.SpectraColumns.FREQ.value]
            gd_theo = T / 2.0 * np.ones_like(f_axis)

            # take the phase_test
            gd_test = spectra_test[
                :, xfer_tools.SpectraColumns.GROUPDELAY.value
            ]

            # test
            precision = 4
            rebase = np.power(10.0, precision)

            with self.subTest(msg="order: {0}".format(order)):
                self.assertSequenceEqual(
                    list(rebase * np.round(gd_test[1:-2], precision)),
                    list(rebase * np.round(gd_theo[1:-2], precision)),
                )

    def test_generate_poles_and_zeros_gives_correct_locations(self):
        """
        Test that the f_sig poles are zeros match the theoretic locations.
        There are no poles for this filter, and zeros lie along the jw
        axis but with no zero at the origin.
        """

        # defs
        f_start = -20.0
        f_end = 20.0
        T = 1.0
        tau = design.convert_T_to_tau(T)

        # fetch valid filter orders
        filter_orders = design.get_valid_filter_orders()

        # iter over filter orders
        for order in filter_orders:

            # fetch poles and zeros
            poles_and_zeros = f_sig.generate_poles_and_zeros(
                tau, order, f_start, f_end
            )

            # theo zero locations
            n_end = np.floor(np.floor(f_end) * (T / order))
            n_axis = np.arange(-n_end, n_end + 1, dtype=int)
            n_axis = np.delete(n_axis, np.where(n_axis == 0)[0])
            zeros_theo = np.zeros(n_axis.shape[0], dtype=complex)
            zeros_theo[:] = np.zeros_like(n_axis) + 1j * n_axis / (T / order)

            # test
            with self.subTest(msg="order: {0}".format(order)):

                # expect no poles
                self.assertTrue(poles_and_zeros["poles"].shape == (0, 0))

                # test zeros
                zeros_test = poles_and_zeros["zeros"]

                # expect no zero at origin
                self.assertFalse((0.0 + 1j * 0.0) in zeros_test)

                # test and theo nearly match
                self.assertAlmostEqual(
                    np.linalg.norm(zeros_test - zeros_theo), 0.0
                )
