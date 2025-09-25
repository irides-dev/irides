"""
-------------------------------------------------------------------------------

Tools to compute spectra from a transfer function, and various helpers, too.

-------------------------------------------------------------------------------
"""

import numpy as np
from enum import Enum
import scipy.integrate as integrate


# noinspection SpellCheckingInspection,PyPep8Naming
def calculate_gain_spectrum(transfer_function_panel: np.ndarray) -> np.ndarray:
    """
    The gain spectrum g(w) is

        g(w) = |H(jw)|,

    where w = 2 pi f. The input and returned np arrays read

        [f, H(j 2 pi f)]
    and
        [f,  g(f)].

    Parameters
    ----------
    transfer_function_panel: np.ndarray
        Transfer function panel with cols [f, H(j 2 pi f)].

    Returns
    -------
    panel: np.ndarray
        Gain spectrum panel with cols [f, g(2 pi f)].
    """

    # take the absolute value of the complex values
    panel = make_conformant_float_panel_and_copy_axis_0(transfer_function_panel)
    panel[:, 1] = abs(transfer_function_panel[:, 1])

    # return
    return panel


# noinspection SpellCheckingInspection,PyPep8Naming
def calculate_group_delay_spectrum(
    phase_spectrum_panel: np.ndarray,
) -> np.ndarray:
    """
    Returns the group-delay spectrum from an input phase spectrum.
    The group-delay spectrum is defined with respect to the phase
    spectrum phase(w) as

        grp_delay(w) = - d phase(w) / dw .

    The derivative is taken numerically as a first difference over
    two grid steps.

    Parameters
    ----------
    phase_spectrum_panel: np.ndarray
        Transfer function panel with cols [f, angle_H(j 2 pi f)].

    Returns
    -------
    panel: np.ndarray
        Group-delay spectrum panel with cols [f, group-delay(2 pi f)].
    """

    # make conformant group-delay panel
    panel = phase_spectrum_panel.copy()

    # infer df, calc dw
    df = min(np.diff(phase_spectrum_panel[:, 0]))

    # take 1st difference two steps apart (for O(2) error)
    # mark first and last entries as np.nan since those values are
    panel[1:-1, 1] = -(panel[2:, 1] - panel[:-2, 1]) / (2.0 * 2.0 * np.pi * df)
    panel[0, 1] = np.nan
    panel[-1, 1] = np.nan

    # return
    return panel


# noinspection SpellCheckingInspection,PyPep8Naming
def calculate_spectra(H: np.ndarray, phase_H: np.ndarray) -> np.ndarray:
    """Calculates gain, phase and group delay from H and angle-H.

    Parameters
    ----------
    H: np.ndarray
        Transfer function values in frequency.
    phase_H
        Phase of transfer function values in frequency.

    Returns
    -------
    spectra: np.ndarray
        Spectra: (n, 4) with cols [f, gain(f), phase(f), group-delay(f)]
    """

    # alloc ndarray [n, 4] with cols [f, gain(f), phase(f), grp-delay(f)]
    spectra = np.zeros((H.shape[0], 4))

    # call up the column enum
    cols = SpectraColumns

    # frequency axis
    spectra[:, cols.FREQ.value] = np.real(H[:, 0].copy())

    # spectra
    spectra[:, cols.GAIN.value] = calculate_gain_spectrum(H)[:, 1]

    spectra[:, cols.PHASE.value] = phase_H[:, 1].copy()

    spectra[:, cols.GROUPDELAY.value] = calculate_group_delay_spectrum(phase_H)[
        :, 1
    ]

    # return
    return spectra


# noinspection SpellCheckingInspection,PyPep8Naming
def perturb_dc_frequency(
    transfer_function_panel: np.ndarray, scale=1.0e-6
) -> np.ndarray:
    """
    In the case that there is a zero value for time or frequency in
    the panel, this function replaces that value with a perturbated value.
    This function is useful when computing transfer functions that have
    poor behavior around f = 0 (eg an mbox level), or for phase
    spectra where the magnitude of the transfer function is zero at DC.

    Parameters
    ----------
    transfer_function_panel: np.ndarray
        Transfer function panel with cols [f, H(j 2 pi f)].
    scale: float
        Scale factor applied to the perturbation step.

    Returns
    -------
    transfer_function_panel: np.ndarray
        Transfer function panel with any 0 values in col 0 perturbed.
    """

    # pull the freq axis
    f = np.real(transfer_function_panel[:, 0])
    df = min(np.diff(f))

    # see if there is a replacement to make
    replace_indices = np.where(f == 0.0)[0]

    # replace as necessary
    if len(replace_indices) > 0:

        transfer_function_panel[replace_indices, 0] = scale * df

    # return
    return transfer_function_panel


# noinspection SpellCheckingInspection,PyPep8Naming
def make_conformant_float_panel_and_copy_axis_0(transfer_function_panel):
    """."""

    # alloc a zeros panel, copy over the freq axis
    panel = np.zeros_like(transfer_function_panel, dtype=float)
    panel[:, 0] = np.real(transfer_function_panel[:, 0])

    return panel


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wraps input phase to the domain [-pi, pi].

    This is an imperfect function. The map +pi -> +pi works, but the
    map for -pi is -pi -> +pi. The meaning of this function wrap()
    is itself ambiguous when applied to the entire real number line,
    rather than non-negative numbers.

    Parameters
    ----------
    phase: np.ndarray
        Input phase array.

    Returns
    -------
    np.ndarray
        Output phase array where phase is bound to [-pi, pi].
    """

    # calc and return
    shift = np.pi

    return (phase + shift) % (2 * np.pi) - shift


# noinspection SpellCheckingInspection,PyPep8Naming
def calculate_scale_correlation_function(
    sti_start: float,
    sti_end: float,
    dsti: float,
    filter_order: int,
    this,
    sti_norm_correction_fcxn,
):
    """Calculates the scale-correlation correlogram.

    `sti` is the ratio of temporal scales between a first and second filter.

    The S-SCF correlogram goes as

        sscf(sti) = 1/(2pi) int_reals H(w) H(w sti) dw

    where `H` is the Fourier transfer function. A normalization correction
    is required for inline-slope and -curvature filters because the
    leading `s` or `s^2` is attached before the integral is taken, which
    in turn produces an errant `sti` or `sti^2` scale factor.

    Parameters
    ----------
    sti_start: float
        Starting scale ratio, typically 1.0
    sti_end: float
        Ending scale ratio
    dsti: float
        Step size for scale-ratio increments
    filter_order: int
        The order of the filter
    this
        Handle to the calling filter-signature object
    sti_norm_correction_fcxn
        Function of `sti`: f(sti) -> float.

    Returns
    -------
    np.ndarray
        Array [n, 2] with cols (stigma_axis, sscf-norm-value).
    """

    # defs
    tau_base = 1.0
    stigma_axis = np.arange(sti_start, sti_end + dsti, dsti)

    # setup sscf correlogram array [n, 2]
    sscf_norm = np.zeros((stigma_axis.shape[0], 2))
    sscf_norm[:, 0] = stigma_axis

    # fetch callable H function and base-line scale (kh0)
    H = this.make_callable_transfer_function(filter_order)
    kh0 = this.design.autocorrelation_peak_and_stride_values(
        tau_base, filter_order
    )["kh0"]

    # iter on stigma, evaluating the integral over the xfer fcxn each step
    for i, sti in enumerate(stigma_axis):

        # integral eval
        sscf_norm[i, 1] = (
            (np.sqrt(sti) / kh0)
            * 2.0
            * np.real(
                integrate.quad(lambda f: H(f) * H(-f * sti), 0.0, np.inf)
            )[0]
            * sti_norm_correction_fcxn(sti)
        )

    return sscf_norm


# noinspection SpellCheckingInspection,PyPep8Naming
class SpectraColumns(Enum):
    """
    Names for the columns in the panel returned by *.generate_spectra(..)
    in the figure signatures.
    """

    FREQ = 0
    GAIN = 1
    PHASE = 2
    GROUPDELAY = 3
