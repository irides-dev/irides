"""
-------------------------------------------------------------------------------

Unit tests for the .tramsfer_function_tools module of spqf.tools.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.tools import transfer_function_tools as xfer_tools


# noinspection SpellCheckingInspection,PyPep8Naming
class TestTransferFunctionTools(unittest.TestCase):
    """
    Unit tests to verify the correct operation of the transfer-function tools
    under src.spqf.tools.transfer_function_tools.
    """

    def test_calculate_gain_spectrum_computes_abs_val_of_complex_input(self):
        """
        Tests that gain is calculated by taking the absolute value, or
        radial value, of a complex input.
        """

        # create xfer function and calculate gain
        n_pts = 100
        H_test = make_complex_circle(n_pts, 1.0)
        gain_test = xfer_tools.calculate_gain_spectrum(H_test)

        # test
        precision = 4
        rebase = np.power(10.0, precision)
        self.assertSequenceEqual(
            list(rebase * np.round(gain_test[:, 1], precision)),
            list(rebase * np.ones(n_pts)),
        )

    def test_calculate_group_delay_spectrum_is_const_for_linear_phase(self):
        """
        Tests that the group-delay spectrum is constant for linear phase.
        """

        # create xfer function and calculate phase
        n_pts = 100
        phase_test = make_linear_phase(n_pts, 2.0)
        gd_test = xfer_tools.calculate_group_delay_spectrum(phase_test)

        # test
        precision = 4
        rebase = np.power(10.0, precision)

        with self.subTest(msg="constant group delay"):
            self.assertSequenceEqual(
                list(rebase * np.round(gd_test[1:-2, 1], precision)),
                list(rebase * np.ones(n_pts)[1:-2]),
            )
        with self.subTest(msg="array ends are np.nan"):
            self.assertTrue(np.isnan(gd_test[0, 1]))
            self.assertTrue(np.isnan(gd_test[-1, 1]))

    def test_calculate_group_delay_spectrum_is_linear_for_quadratic_phase(self):
        """
        Tests that the group-delay spectrum is linear for quadratic phase.
        """

        # create xfer function and calculate phase
        n_pts = 100
        phase_test = make_quadratic_phase(n_pts, 2.0)
        gd_test = xfer_tools.calculate_group_delay_spectrum(phase_test)

        # theo result
        gd_theo = 2.0 * np.pi * gd_test[:, 0]

        # test, exclude the end points
        precision = 4
        rebase = np.power(10.0, precision)

        with self.subTest(msg="linear group delay"):
            self.assertSequenceEqual(
                list(rebase * np.round(gd_test[1:-2, 1], precision)),
                list(rebase * np.round(gd_theo[1:-2], precision)),
            )
        with self.subTest(msg="array ends are np.nan"):
            self.assertTrue(np.isnan(gd_test[0, 1]))
            self.assertTrue(np.isnan(gd_test[-1, 1]))

    def test_perturb_dc_frequency_alters_zero_frequency_and_not_others(self):
        """
        Tests that the frequency column (col 0) of a panel has all zero-value
        frequency entries altered on the panel.
        """

        # setup freq axis and transfer-function panel
        n_pts = 100
        f = np.arange(0.0, 1.0, 1.0 / n_pts)
        H_test = np.ndarray([n_pts, 2], dtype=complex)
        H_test[:, 0] = f

        # send to perturb
        H_pert = xfer_tools.perturb_dc_frequency(H_test)

        with self.subTest(msg="perturb for f = 0.0"):

            self.assertEqual(f[0], 0.0)
            self.assertNotEqual(H_pert[0, 0], 0.0)
            self.assertGreater(H_pert[0, 0], 0.0)

        # perturb again, expecting no change
        f_pert_0 = H_pert[0, 0]
        H_pert_pert = xfer_tools.perturb_dc_frequency(H_pert)

        with self.subTest(msg="not perturb for f != 0.0."):

            self.assertAlmostEqual(f_pert_0, H_pert_pert[0, 0], 8)

    def test_make_conformant_float_panel_and_copy_axis_0_casts_correctly(self):
        """
        The input transfer-function panel has dtype=complex. The return
        panel shall have dtype=float.
        """

        # setup input panel
        H_test = np.ndarray([1, 2], dtype=complex)

        # fetch return panel
        panel = xfer_tools.make_conformant_float_panel_and_copy_axis_0(H_test)

        # test
        self.assertTrue(H_test.dtype, np.dtype("complex128"))
        self.assertTrue(panel.dtype, np.dtype("float"))

    def test_make_conformant_float_panel_and_copy_axis_0_copies_first_column(
        self,
    ):
        """
        Tests that the first column, which holds frequency, is copied to
        the output panel.
        """

        # setup for test
        H_test = np.ndarray([10, 2], dtype=complex)
        H_test[:, 0] = np.arange(H_test.shape[0])
        panel = xfer_tools.make_conformant_float_panel_and_copy_axis_0(H_test)

        # test
        precision = 4
        rebase = np.power(10.0, precision)
        self.assertSequenceEqual(
            list(rebase * np.round(H_test[:, 0], precision)),
            list(rebase * np.round(panel[:, 0], precision)),
        )

    def test_spectra_column_enums_have_expected_int_values(self):
        """
        Test to confirm that the spectra-column enum calls out the expected
        column values.
        """

        with self.subTest(msg="freq:"):
            self.assertEqual(xfer_tools.SpectraColumns.FREQ.value, 0)
        with self.subTest(msg="gain:"):
            self.assertEqual(xfer_tools.SpectraColumns.GAIN.value, 1)
        with self.subTest(msg="phase:"):
            self.assertEqual(xfer_tools.SpectraColumns.PHASE.value, 2)
        with self.subTest(msg="group-delay:"):
            self.assertEqual(xfer_tools.SpectraColumns.GROUPDELAY.value, 3)


# noinspection SpellCheckingInspection,PyPep8Naming
def make_complex_circle(n_pts: int, n_wraps: float) -> np.ndarray:
    """
    Creates a [n_pts, 2] panel with cols (freq, complex value xfer fcxn).
    The values in the 2nd column trace a clockwise circle in the complex
    plane:

        H(f) = exp(-j 2 pi f).

    Parameters
    ----------
    n_pts: int
        Number of points in the panel.
    n_wraps: float
        Number of times to wrap around 2 pi.

    Returns
    -------
    H_test: np.ndarray
        Panel, dtype=complex, [n_pts, 2] in size.
    """

    # setup the freq axis
    f = np.arange(0.0, n_wraps, n_wraps / n_pts)

    # create H, trace unit-radius circle in the complex plane
    # note the clockwise rotation (to better align with the negative phase
    #   of all causal filters).
    H_test = np.ndarray([n_pts, 2], dtype=complex)
    H_test[:, 0] = f
    H_test[:, 1] = np.exp(-1j * 2.0 * np.pi * f)

    return H_test


# noinspection SpellCheckingInspection,PyPep8Naming
def make_linear_phase(n_pts: int, n_wraps: float) -> np.ndarray:
    """
    Creates a [n_pts, 2] panel with cols (freq, phase).
    The linear phase is

        phase(f) = - 2 pi f .

    Parameters
    ----------
    n_pts: int
        Number of points in the panel.
    n_wraps: float
        Number of times to wrap around 2 pi.

    Returns
    -------
    panel: np.ndarray
        Panel, dtype=float, [n_pts, 2] in size.
    """

    # setup the freq axis
    f_max = np.sqrt(2.0 * n_wraps)
    f = np.arange(0.0, f_max, f_max / n_pts)

    # quadratic phase
    phase = np.ndarray([n_pts, 2], dtype=float)
    phase[:, 0] = f
    phase[:, 1] = -2.0 * np.pi * f

    return phase


# noinspection SpellCheckingInspection,PyPep8Naming
def make_quadratic_phase(n_pts: int, n_wraps: float) -> np.ndarray:
    """
    Creates a [n_pts, 2] panel with cols (freq, phase).
    The quadratic phase is

        phase(f) = - (2 pi f)^2 / 2 .

    Parameters
    ----------
    n_pts: int
        Number of points in the panel.
    n_wraps: float
        Number of times to wrap around 2 pi.

    Returns
    -------
    panel: np.ndarray
        Panel, dtype=float, [n_pts, 2] in size.
    """

    # setup the freq axis
    f_max = np.sqrt(2.0 * n_wraps)
    f = np.arange(0.0, f_max, f_max / n_pts)

    # quadratic phase
    phase = np.ndarray([n_pts, 2], dtype=float)
    phase[:, 0] = f
    phase[:, 1] = -0.5 * np.power(2.0 * np.pi * f, 2)

    return phase
