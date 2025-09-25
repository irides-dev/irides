"""
-------------------------------------------------------------------------------

Common tools for digital designs.

-------------------------------------------------------------------------------
"""

import numpy as np
from enum import Enum
from types import ModuleType

from numpy.polynomial import Polynomial

from irides.tools import impulse_response_tools as ir_tools
from irides.tools import analog_to_digital_conversion_tools as a2d_tools
from irides.resources.core_enumerations import FilterDesignType
from irides.tools.design_tools import StageTypes
from irides.resources.containers.discrete_sequence import DiscreteSequence
from irides.resources.containers.transfer_functions import (
    IIRTransferFunctionDiscreteTime,
)


class FrequencyBand(Enum):
    """Defines the set feasible frequency bands, whether low or high"""

    LOW = 0
    HIGH = 1


# noinspection PyPep8Naming
def construct_transfer_function(
    mu: float,
    filter_order: int,
    design_module: ModuleType,
    filter_class: FilterDesignType = FilterDesignType.LEVEL,
    frequency_band: FrequencyBand = FrequencyBand.LOW,
    T: float = 1,
) -> IIRTransferFunctionDiscreteTime:
    """Constructs H(z) from H(s) using impulse-invariant design.

    The transfer-function representation is multistage. Accordingly, the
    stages are built but not combined; instead the per-stage polynomials
    are passed to the constructor of the TransferFunction object for storage
    and evaluation.

    Parameters
    ----------
    mu: float
        Temporal scale
    filter_order: int
        The order of the filter
    design_module: ModuleType
        A design module that implements .designs(filter_order)
    filter_class: FilterDesignType
        Class of filter {level|slope|curve}
    frequency_band: FrequencyBand
        Specifies LOW or HIGH frequency (default = LOW)
    T: float
        Sampling rate (default = 1)

    Returns
    -------
    IIRTransferFunctionDiscreteTime
        Object that contains a transfer-function representation
    """

    # validate order
    ir_tools.validate_filter_order_or_die(design_module, filter_order)

    # fetch analog design
    filter_design = design_module.designs(filter_order)

    # prepare zplane poles
    zplane_poles_lf = a2d_tools.zplane_poles_from_analog_design(
        design_module, mu, filter_order, T
    )
    zplane_poles_signed = convert_frequency_band(
        zplane_poles_lf, frequency_band
    )

    # build the polynomial fragments stage by stage
    num_polys = np.array([])
    den_polys = np.array([])

    for stage in filter_design["stages"]:
        if stage["type"] == StageTypes.EMA.value:
            # poly: 1 - p z^-1
            zpole = np.real(zplane_poles_signed[stage["indices"]])[0]
            p = Polynomial([1.0, -zpole])

            zpole_lf = np.real(zplane_poles_lf[stage["indices"]])[0]
            p_gain_adj = Polynomial([1.0, -zpole_lf])

        else:
            # poly: 1 - (p + p*) z^-1 + pp* z^-2
            zpoles = zplane_poles_signed[stage["indices"]]
            a0 = 1.0
            a1 = -np.real(np.sum(zpoles))
            a2 = np.real(np.prod(zpoles))
            p = Polynomial([a0, a1, a2])

            zpoles_lf = zplane_poles_lf[stage["indices"]]
            a0 = 1.0
            a1 = -np.real(np.sum(zpoles_lf))
            a2 = np.real(np.prod(zpoles_lf))
            p_gain_adj = Polynomial([a0, a1, a2])

        # append polynomial fragment to frag array
        num_polys = np.append(num_polys, Polynomial(p_gain_adj(1.0)))
        den_polys = np.append(den_polys, p)

    # construct container and return
    return IIRTransferFunctionDiscreteTime(
        filter_order, num_polys, den_polys, filter_class
    )


def compute_sample_dtft_spectra(
    ds_in: DiscreteSequence,
    freq_max: float,
    n_freqs: int,
    v_col: int = 0,
) -> DiscreteSequence:
    """Implements the DTFT analysis equation for an input sample time series.

    X(e^jW) is returned by this function, using the equation

        X(e^jW) = sum_{i=0}^m x[n] e^jWn.

    Parameters
    ----------
    ds_in: DiscreteSequence
        Input sample time sequence
    freq_max: float
        Maximum frequency (eg 0.5)
    n_freqs: int
        Number of frequencies in the [0, pi) range
    v_col: int
        The column in ds_in.v_axes[:, `v_col`] to use (default = 0)

    Returns
    -------
    DiscreteSequence
        DTFT analysis output, n_axis -> frequencies, v_axis -> complex spectra
    """

    freqs = np.linspace(0, freq_max, n_freqs)
    coef = 1j * 2 * np.pi
    ds_out = DiscreteSequence(0, n_freqs, v_type="complex")

    for i, freq in enumerate(freqs):
        ejWn = np.exp(coef * freq * ds_in.n_axis)
        ds_out.v_axis[i] = np.sum(ds_in.v_axes[:, v_col] * ejWn)

    return ds_out


def identify_frequency_bands_of_zplane_roots(zplane_roots: np.ndarray) -> list:
    """Returns list of {LOW|HIGH} frequency types by inspecting zplane roots

    The frequency band is LOW if Re(root) >= 0, HIGH otherwise.

    Parameters
    ----------
    zplane_roots: np.ndarray
        Coordinates of zplane roots (float or complex)

    Returns
    -------
    list
        List of FrequencyBand values
    """

    selector = lambda b: FrequencyBand.LOW if b else FrequencyBand.HIGH
    is_nonnegative_real = np.real(zplane_roots) >= 0.0

    return list(map(selector, is_nonnegative_real))


def convert_frequency_band(
    zplane_roots: np.ndarray, target_frequency_band: FrequencyBand
) -> np.ndarray:
    """Converts `zplane_roots` to low- or high-freq roots given target type

    Parameters
    ----------
    zplane_roots: np.ndarray
        Coordinates of zplane roots (float or complex)
    target_frequency_band: FrequencyBand
        Target band, whether LOW or HIGH

    Returns
    -------
    np.ndarray
        Updated zplane_roots such that all roots have the specified freq band
    """

    # list of actual frequency bands
    freq_bands = identify_frequency_bands_of_zplane_roots(zplane_roots)

    # make list of signs so that (sign * zplane_pole) is in the spec'd freq band
    sign_converter = (
        lambda freq_type: 1.0 if freq_type == target_frequency_band else -1.0
    )

    signs_to_apply = np.array(list(map(sign_converter, freq_bands)))

    # apply and return
    return signs_to_apply * zplane_roots


# noinspection SpellCheckingInspection
def qz_coefs_from_zplane_poles(zplane_poles: np.ndarray) -> np.ndarray:
    """Returns array of Q(z) coefficients given an input array of z-poles

    Expressed in terms of zeta ( = z^-1 ), the zplane poles associate
    with simple polynomials according to

        (1 - p_1 zeta) (1 - p_2 zeta) ....

    Once multiplied through, the polynomial reads

        1 + a_1 zeta + a_2 zeta^2 + ....

    This routine returns the array [1, a_1, a_2, ...].

    Parameters
    ----------
    zplane_poles: np.ndarray
        Array of zplane poles (eg from zplane_poles_from_analog_design)

    Returns
    -------
    np.ndarray
        Coefficients to Q(z) polynomial
    """

    # Note that .fromroots() treats the input array as associated with
    #   (x - r[0]) (x - r[1]) ... (x - r[n-1])
    # rather than the simple polynomials expressed above. So, the
    # result from the routine needs to be reversed in order to correspond.

    return np.real(Polynomial.fromroots(zplane_poles).coef[::-1])


def build_a_coefs_matrix_from_qz_coefs(
    design_analog_level,
    mu: float,
    order: int,
) -> np.ndarray:
    """Builds a matrix of Q(z) coefs for subsequent use.

    The (order + 1) coefs from Q(z) are [a_0, a_1, a_2, ...]. ZIR analysis
    requires a matrix of the form

            | a_1  a_2  a_3 |
        A = | a_2  a_3  0   |
            | a_3  0    0   |

    which this function constructs.

    Parameters
    ----------
    design_analog_level:
        an analog design module for level filters
    mu: float:
        timescale of the digital filter
    order: int:
        order of the filter

    Returns
    -------
    np.ndarray:
        An A matrix
    """

    # fetch the Q(z) coefs
    qz_coefs = qz_coefs_from_zplane_poles(
        a2d_tools.zplane_poles_from_analog_design(
            design_analog_level, mu, order
        )
    )

    # remove qz_coefs[0] term, rename to a_coefs
    a_coefs = qz_coefs[1:]
    a_coefs_padded = np.pad(a_coefs, (0, order + 1))

    # build matrix
    A = np.zeros((order, order))
    for i in np.arange(0, order):
        A[i, :] = np.roll(a_coefs_padded, -i)[:order]

    # return
    return A


def zir_coefs_at_fixed_horizon(
    design_analog_level,
    design_digital_level,
    filter_signature_digital_level,
    mu: float,
    order: int,
) -> np.ndarray:
    """Coefficients for a fixed-horizon ZIR filter.

    The ZIR filter in this case appears in the

        Y_zir(z) = H_zir(z) Y(z)

    system equation whose inverse transform is

        y_zir[n; mu] = h_zir[n; mu] .conv. y[n].

    Use the H_zir(z) filter when propagating forward in `n` while
    maintaining a fixed forward horizon of `n + mu`.

    The form of H(z) in the context of Y(z) = H(z) X(z), the
    original filter, the pema and bessel H(z) structure is

                              gain_adj
        H(z) = ------------------------------------------  .
                1 + a_1 z^-1 + a_2 z^-2 + ... + a_m z^-m

    Using this form and simplifying to a 3rd-order system, the H_zir(z)
    transfer function reads

        H_zir(z) = gain_adj^-1 x (
            - (a_1 + a_2 + a_3) z^-1 - (a_2 + a_3) z^-2 - a_3 z^-3
        ).

    For Y_zir(z) = H_zir(z) H(z) X(z), the fixed-horizon ZIR impulse response is

        ~~ validate -- I think there's delta[n - k] in each item
        h_zir[n; mu] = - gain_adj^-1 x (
            (a_1 + a_2 + a_3) h[mu-1] + (a_2 + a_3) h[mu-2] + a_3 h[mu-3]
        )
        ~~

    Parameters
    ----------
    design_analog_level:
        an analog design module for level filters
    design_digital_level:
        a digital design module for level filters
    filter_signature_digital_level:
        a filter-signature module for level filters
    mu: float
        timescale of the digital filter
    order: int
        order of the filter

    Returns
    -------
    np.ndarray:
        array of zir coefs
    """

    # coefs of Q(z)
    qz_coefs = qz_coefs_from_zplane_poles(
        a2d_tools.zplane_poles_from_analog_design(
            design_analog_level, mu, order
        )
    )

    # gain_adj
    gain_adj = design_digital_level.gain_adjustment(mu, order)

    # hl[n] values at n = mu and its lags
    hl_weights = filter_signature_digital_level.generate_impulse_response(
        0, 4 * mu, mu, order
    ).v_axis[mu - order : mu + 1]

    # calculate zir coefs
    zir_coef_vals = np.zeros_like(qz_coefs)

    # these equations need to be checked (probably ok but I don't understand
    # how the math aligns with the code here)
    for i in np.arange(1, zir_coef_vals.shape[0]):
        alpha_ic_coefs = -qz_coefs[i:]
        hn_coefs = hl_weights[-(order + 1) + i :][::-1]
        zir_coef_vals[i] = (alpha_ic_coefs @ hn_coefs) / gain_adj

    return zir_coef_vals


def hzir_for_variable_horizon(
    yn_stream: np.ndarray,
    design_analog_level,
    design_digital_level,
    mu: float,
    order: int,
) -> DiscreteSequence:
    """Impulse response for the ZIR continuation of a yn stream.

    The returned impulse response h_zir[ell] is the IIR component of a
    continuation of a filtered stream y[n] = (h * x)[x], where h[n] may be
    a level or slope filter. That is, at sample `n`, further input x[n] is
    frozen to _x_[n], after which the ZIR response of the filter h[n] continues
    due to the energy stored in the filter.

    The real-world sample axis is denoted `n` while the continuation axis
    on which the ZIR response propagates is denoted by `ell` (after the
    Cyrillic character).

    The complete ZIR continuation for a level filter is

        y_zir[ell] = (h_zir * h)[ell] + _x_[n] (h * u)[ell].

    The impulse response of h_zir[ell] is `order` long in length, and the
    term-by-term entries go as

        h_zir[k] = - 1 / gain_adj alpha_ic[k],  k = 0, 1, 2

    where

                   | a_1  a_2  a_3 |   | y[n-1] |
        alpha_ic = | a_2  a_3  0   |   | y[n-2] |
                   | a_3  0    0   |   | y[n-3] |

    and y[n-k] are the input initial conditions.


    Parameters
    ----------
    yn_stream: np.ndarray
        Stream of output values from a preceding filter. The stream is to be
        order + 1 in length and ordered in ascending time.
    design_analog_level:
        an analog design module for level filters
    design_digital_level:
        a digital design module for level filters
    mu: float
        timescale of the digital filter
    order: int
        order of the filter

    Returns
    -------
    DiscreteSequence:
        The sequence of h_zir `order` samples long.
    """

    # build A matrix from coefs of Q(z)
    A = build_a_coefs_matrix_from_qz_coefs(design_analog_level, mu, order)

    # convert yn_stream to yn initial conditions
    yn_ic = yn_stream[1 : order + 1][::-1]

    # build alpha coefs based on A @ yn_ic
    alpha_ic_coefs = A @ yn_ic

    # gain_adj
    gain_adj = design_digital_level.gain_adjustment(mu, order)

    # prepare an impulse response
    ds_hzir = DiscreteSequence(0, order)
    ds_hzir.v_axis = -alpha_ic_coefs / gain_adj

    return ds_hzir


def hzir_for_fixed_horizon(
    hn_level: DiscreteSequence,
    design_analog_level,
    design_digital_level,
    mu: float,
    order: int,
    ell_select: int = None,
) -> DiscreteSequence:
    """Impulse response for the fixed-horizon.

    The impulse response goes as

        h_zir[n] = - 1 / gain_adj x
            (
                beta_1 delta[n-1] + beta_2 delta[n-2] + beta_3 delta[n-3] + ...
            )

    where

               | a_1  a_2  a_3 |   | hl[ell]   |
        beta = | a_2  a_3  0   |   | hl[ell-1] | .
               | a_3  0    0   |   | hl[ell-2] |

    The z-transform then reads

        H_zir(z) = - 1 / gain_adj x
            (
                beta_1 z^{-1} + beta_2 z^{-2} + beta_3 z^{-3} + ...
            ).

    Parameters
    ----------
    design_analog_level
    design_digital_level
    mu
    order

    Returns
    -------
    DiscreteSequence:
        The sequence of hzir `order` samples long.

    """

    # build A matrix from coefs of Q(z)
    A = build_a_coefs_matrix_from_qz_coefs(design_analog_level, mu, order)

    # pick hl values from hn_level (note reverse order)
    ell = mu if ell_select is None else ell_select
    hl = hn_level.v_axis[ell - order + 1 : ell + 1][::-1]

    # compute beta coefs
    beta = A @ hl

    # gain_adj
    gain_adj = design_digital_level.gain_adjustment(mu, order)

    # prepare the hl_zir impulse response
    # observe that beta_0 = 0
    ds_hzir = DiscreteSequence(0, order + 1)
    ds_hzir.v_axis[1:] = -beta / gain_adj

    # return
    return ds_hzir


# noinspection SpellCheckingInspection
def upsample_sequence(
    original_sequence: DiscreteSequence, upsample_stride: int
) -> DiscreteSequence:
    """Returns an upsampled sequence based on the input sequence.

    Parameters
    ----------
    original_sequence: DiscreteSequence
        Sequence to be upsampled
    upsample_stride: int
        Stride of adjacent upsampled values (lefthand point plus intermediate zeros)

    Returns
    -------
    DiscreteSequence
        Upsampled sequence
    """

    # set up new discrete sequence
    len_orig = original_sequence.len
    len_up = upsample_stride * len_orig
    n_start = original_sequence.n_axis[0]
    n_cols = original_sequence.v_axes.shape[1]
    ds_up = DiscreteSequence(n_start, n_start + len_up, n_cols)

    # for each v_axes column, expand
    for i in range(n_cols):
        # original values
        v_axis = original_sequence.v_axes[:, i][:, np.newaxis]

        # block of zeros where column count = (stride - 1)
        zeros_block = np.zeros((len_orig, upsample_stride - 1))

        # augment and collapse into a single vector
        ds_up.v_axes[:, i] = np.ravel(np.hstack((v_axis, zeros_block)))

    # return
    return ds_up
