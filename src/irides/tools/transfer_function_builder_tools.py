"""
-------------------------------------------------------------------------------

Tools to build transfer functions and spectra.

-------------------------------------------------------------------------------
"""

from typing import Callable
import numpy as np


# noinspection SpellCheckingInspection,PyPep8Naming
def truncate_and_normalize_sacf_correlogram(
    sacf: np.ndarray, kh0: float, normalize: bool, xi_start: float
) -> np.ndarray:
    """Truncates sacf to xi >= xi_start, normalizes by kh0 if requested.

    Parameters
    ----------
    sacf: np.ndarray
        The system autocorrelation correlogram [n, 2] format.
    kh0: float
        The sacf(0) value.
    normalize: bool
        The normalization flag.
    xi_start: float
        Starting point along the xi axis

    Returns
    -------
    np.ndarray
        Modified correlogram.
    """

    # infer
    dxi = np.min(np.diff(sacf[:, 0]))

    # truncate
    sacf = sacf[xi_start - dxi / 2.0 <= sacf[:, 0], :]

    # normalize is requested
    if normalize:
        sacf[:, 1] /= kh0

    # return
    return sacf


# noinspection SpellCheckingInspection,PyPep8Naming
def apply_leading_jw_coefficient_to_transfer_function(
    H: np.ndarray, power: int = 1
) -> np.ndarray:
    """Multiplies the transfer function by a power of jw.

    Parameters
    ----------
    H: np.ndarray
        Current transfer function, [n, 2] layout
    power: int
        Applies (jw)^power to the transfer function. Default = 1.

    Returns
    -------
    np.ndarray
        Updated transfer function
    """

    f_axis = H[:, 0]
    coef = (1j * 2.0 * np.pi) * f_axis
    H[:, 1] *= np.power(coef, power)

    return H


# noinspection SpellCheckingInspection,PyPep8Naming
def apply_leading_jw_coefficient_to_phase_spectrum(
    H: np.ndarray, power: int = 1
) -> np.ndarray:
    """Adds multiple of np.pi / 2.0 to the phase.

    Parameters
    ----------
    H: np.ndarray
        Current phase spectrum, [n, 2] layout
    power: int
        Applies (np.pi/2) * power to the phase. Default = 1.

    Returns
    -------
    np.ndarray
        Updated phase spectrum
    """

    phase_shift = (np.pi / 2.0) * power
    H[:, 1] += phase_shift

    return H


# noinspection SpellCheckingInspection,PyPep8Naming
def make_callable_transfer_function_with_leading_jw_power_term(
    f_sig_ref, filter_order: int, power: int
) -> Callable[[float, float], complex]:
    """Creates new callable with same signature with leading (jw)^power.

    Parameters
    ----------
    f_sig_ref
        Name of reference filter-signature module.
    filter_order: int
        The filter order, which is captured by the definition herein.
    power: int
        Power to raise (jw) term

    Returns
    -------
    Callable[[float, float], complex]
    """
    H_ref = f_sig_ref.make_callable_transfer_function(filter_order)

    # noinspection PyPep8Naming
    def H_amended(f: float, tau: float = 1.0) -> complex:

        return np.power(1j * 2.0 * np.pi * f, power) * H_ref(f, tau)

    return H_amended
