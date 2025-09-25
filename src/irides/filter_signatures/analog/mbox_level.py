"""
-------------------------------------------------------------------------------

Analog multistage box:

Implements temporal, spectral, autocorrelation and pole / zero features
for this continuous-time level filter.

-------------------------------------------------------------------------------
"""

from typing import Callable
import numpy as np
import sys

from irides.design.analog import mbox_level as design
from irides.tools import impulse_response_builder_tools
from irides.tools import impulse_response_tools
from irides.tools import transfer_function_tools


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_impulse_response(
    t_start: float, t_end: float, dt: float, tau: float, filter_order: int
) -> np.ndarray:
    """
    Creates a continuous-time-like impulse response for the multistage box
    level filter. The multistage box is constructed from a cascade of
    elementary boxes.

    Parameters
    ----------
    t_start: float
        Start time.
    t_end: float
        End time.
    dt: float
        Time increment.
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Filter order (number of stages).

    Returns
    -------
    np.ndarray
        A time series array (n, 2) having columns [t_axis, h_mbox].
    """

    # validate
    impulse_response_tools.validate_filter_order_or_die(design, filter_order)

    # convert
    T = design.convert_tau_to_T(tau)

    # alias
    m = filter_order

    # make time series template
    (
        time_series,
        i_t_0plus,
    ) = impulse_response_builder_tools.make_impulse_response_template(
        t_start, t_end, dt
    )

    # alias
    h = time_series[:, 1]

    # prepare for and exec m-fold convolution
    ref_box = np.ones(int(np.floor(T / dt)))
    conv_box = np.array([1])
    for i in range(m):
        conv_box = np.convolve(ref_box, conv_box)

    # downsample candidate box and normalize gain to unity
    cand_box = conv_box[::m]
    cand_box /= sum(cand_box) * dt

    # make h_mbox by inserting the cand_box into an array of zeros
    i_start = i_t_0plus
    i_end = i_start + cand_box.shape[0]
    h[i_start:i_end] = cand_box

    # return
    return time_series


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_sacf_correlogram(
    xi_start: float,
    xi_end: float,
    dxi: float,
    tau: float,
    filter_order: int,
    normalize=False,
) -> np.ndarray:
    """
    Computes the system autocorrelation response from the correlation
    of the impulse response with itself.

    Parameters
    ----------
    xi_start: float
        Start lag, |xi_start| <= |xi_end|
    xi_end: float
        End lag
    dxi: float
        Lag increment
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Filter order (number of stages).
    normalize: bool
        T/F whether to normalize the sacf spectrum so sacf(0) = 1.

    Returns
    -------
    np.ndarray
        System autocorrelation (n, 2) np.array w/ cols [xi, sacf-value].
    """

    # fetch impulse response
    ir = generate_impulse_response(0.0, xi_end, dxi, tau, filter_order)

    # calc sacf
    sacf = impulse_response_tools.calculate_auto_correlation_function(ir)

    # truncate at xi_start
    sacf = sacf[xi_start - dxi / 2.0 <= sacf[:, 0], :]

    # normalize by theoretic peak value if requested
    if normalize:
        kh0 = design.autocorrelation_peak_and_stride_values(tau, filter_order)[
            "kh0"
        ]
        sacf[:, 1] /= kh0

    return sacf


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_transfer_function(
    f_start: float, f_end: float, df: float, tau: float, filter_order: int
) -> np.ndarray:
    r"""
    Computes the transfer function H_mbox(f) of the mutistage box. To avoid a
    numeric overflow for f = 0, the transfer function is expanded in the
    region f ~ 0. We have

                 /
                 |    / 1 - exp(-s T / m) \ m
                 |   |---------------------|             |s| >= 2 sqrt(m) TOL
                 |    \      s T / m      /
        H(s) =  <
                 |
                 |   (1 - s T / 2 + (1+3m)/24m (s T)^2   |s| < 2 sqrt(m) TOL
                 |
                 \

    where s = jw, w = 2pi f, and TOL << 1.

    Parameters
    ----------
    f_start: float
        Start cyclic frequency.
    f_end: float
        End cyclic frequency.
    df: float
        Cyclic frequency increment.
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Filter order (number of stages).

    Returns
    -------
    np.ndarray
        The transfer function with cols [f_axis, H_mbox(f)].
    """

    # validate
    impulse_response_tools.validate_filter_order_or_die(design, filter_order)

    # convert
    T = design.convert_tau_to_T(tau)

    # alias
    m = filter_order

    # make transfer-function panel
    npts = np.arange(f_start, f_end, df).shape[0]
    H_mbox = np.ndarray([npts, 2], dtype=complex)

    # setup f_axis
    H_mbox[:, 0] = np.arange(f_start, f_end, df)
    f_axis = H_mbox[:, 0]

    # make f_axis and sTm vector
    sTm = (1j * 2.0 * np.pi * T / m) * f_axis
    sT = m * sTm

    # index into f ~ 0 region and outside that region
    tol = 0.001  # hardcode choice
    thresh = 2.0 * np.sqrt(m) / T * tol
    i_in = np.where(np.abs(f_axis) < thresh)[0]
    i_out = np.where(np.abs(f_axis) >= thresh)[0]

    # calculate transfer function
    H_mbox[i_in, 1] = (
        1.0
        - 0.5 * sT[i_in]
        + (1.0 + 3.0 * m) / (24.0 * m) * np.power(sT[i_in], 2)
    )

    H_mbox[i_out, 1] = np.power((1.0 - np.exp(-sTm[i_out])) / sTm[i_out], m)

    # return
    return H_mbox


# noinspection SpellCheckingInspection,PyPep8Naming
def make_callable_transfer_function(
    filter_order: int,
) -> Callable[[float, float], complex]:
    """Returns a callable function with signature (f, tau) -> complex.

    This transfer function is restricted to the jw axis:

        H(s) -> H(j w).

    Parameters
    ----------
    filter_order: int
        The filter order, which is captured by the definition herein.

    Returns
    -------
    Callable[[float, float], complex]
        A callable function (f: float, tau: float) -> complex
    """

    # noinspection PyPep8Naming
    def H(f: float, tau: float = 1.0) -> complex:
        """Compute complex xfer fcxn H at freq `f` with optional scale tau.

        Note that `filter_order` is captured, leaving only (f, tau) as free
        variables.

        Parameters
        ----------
        f: float
            Start cyclic frequency.
        tau: float
            Temporal scaling, which ='s the first moment of the final filter.

        Returns
        -------
        complex
            H(f, tau) value.
        """

        jw = 1j * 2.0 * np.pi * f
        T = 2.0 * tau
        sTm = jw * T / filter_order

        return np.power((1.0 - np.exp(-sTm)) / sTm, filter_order)

    return H


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_sscf_correlogram(
    sti_start: float, sti_end: float, dsti: float, filter_order: int
) -> np.ndarray:
    """
    Computes the normalized system scale-correlation correlogram from the
    filter's transfer function.

    No sti-normalization correction is necessary for level filters.

    Parameters
    ----------
    sti_start: float
        Start scale, |sti_start| <= |sti_end|
    sti_end: float
        End scale
    dsti: float
        Scale increment
    filter_order: int
        Filter order.

    Returns
    -------
    np.ndarray
        System scale-correlation normalized (n, 2)
        np.array w/ cols [sti, sscf-value].
    """

    # correction function
    sti_norm_correction_fcxn = lambda sti: 1.0

    # return sscf-norm np-array
    return transfer_function_tools.calculate_scale_correlation_function(
        sti_start,
        sti_end,
        dsti,
        filter_order,
        sys.modules[__name__],
        sti_norm_correction_fcxn,
    )


# noinspection SpellCheckingInspection,PyPep8Naming,PyUnusedLocal
def generate_phase_spectrum(
    f_start: float, f_end: float, df: float, tau: float, filter_order: int
) -> np.ndarray:
    r"""
    Computes the phase of H_mbox(f) from its analytic expression. The phase
    spectrum is independent of filter_order (in this case) and is simply

        phase-H_mbox(f) = - (2 pi f) T / 2.

    Parameters
    ----------
    f_start: float
        Start cyclic frequency.
    f_end: float
        End cyclic frequency.
    df: float
        Cyclic frequency increment.
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Filter order (number of stages).  (unused)

    Returns
    -------
    np.ndarray
        The phase spectrum with cols [f_axis, AngleH_mbox(f)].
    """

    # convert
    T = design.convert_tau_to_T(tau)

    # make phase-spectrum panel
    npts = np.arange(f_start, f_end, df).shape[0]
    angle_H_mbox = np.ndarray([npts, 2], dtype=float)

    # setup f_axis
    angle_H_mbox[:, 0] = np.arange(f_start, f_end, df)
    f_axis = angle_H_mbox[:, 0]

    # compute the phase
    angle_H_mbox[:, 1] = -np.pi * T * f_axis

    return angle_H_mbox


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_spectra(
    f_start: float, f_end: float, df: float, tau: float, filter_order: int
) -> np.ndarray:
    """
    Generate gain, phase and group-delay spectra, based on the transfer
    function.

    Parameters
    ----------
    f_start: float
        Start cyclic frequency.
    f_end: float
        End cyclic frequency.
    df: float
        Cyclic frequency increment.
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Filter order (number of stages).

    Returns
    -------
    np.ndarray
        Spectra panel [n_pts, 4] with cols [f, g(f), phase(f), grpdly(f)].
    """

    # fetch the transfer function for gain calculation
    H: np.ndarray = generate_transfer_function(
        f_start, f_end, df, tau, filter_order
    )

    # fetch the phase spectrum for group-delay calculation
    phase_H: np.ndarray = generate_phase_spectrum(
        f_start, f_end, df, tau, filter_order
    )

    # calculate and return spectra
    return transfer_function_tools.calculate_spectra(H, phase_H)


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_wireframe(
    t_start: float, t_end: float, dt: float, tau: float, filter_order: int
) -> np.ndarray:
    """Returns a wireframe equivalent for this filter.

    Parameters
    ----------
    t_start: float
        Start time.
    t_end: float
        End time.
    dt: float
        Time increment.
    tau: float
        First moment of the filter.
    filter_order: int
        Filter order.

    Returns
    -------
    WireframeContinuousTime
        A wireframe container.
    """

    # fetch the wireframe, create a timeseries template
    wf = design.wireframe(tau, filter_order)
    ts = impulse_response_builder_tools.discretize_wireframe_onto_time_grid(
        t_start, t_end, dt, wf
    )

    return ts


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_poles_and_zeros(
    tau: float, filter_order: int, f_start=0.0, f_end=0.0
) -> dict:
    """
    Calculates poles and zeros for the multistage box in the s-plane.
    There are no poles, and zeros are periodic along the jw axis, with
    the exception of the origin (w, sigma) = 0, where there is no zero.

    The zero positions are

        f[n] = n / (T / m),  n = +/- 1, +/- 2, ...  (n != 0).

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Filter order (number of stages).
    f_start: float
        Start cyclic frequency.
    f_end: float
        End cyclic frequency.

    Returns
    -------
    dict
        Dictionary with keys ('poles', 'zeros') and values as np.ndarray
        with dtype=complex.
    """

    # convert
    T = design.convert_tau_to_T(tau)

    # the request is for domain [f_min, f_max], and this needs to be
    # converted to the domain [n_min, n_max]. The trick is to pull toward
    # zero frequency for either frequency sign. That is,
    # np.floor(2.2) = 2.0 while np.floor(-2.2) = -3.0.
    n_start = np.sign(f_start) * np.floor(
        np.floor(np.abs(f_start)) * (T / filter_order)
    )
    n_end = np.sign(f_end) * np.floor(
        np.floor(np.abs(f_end)) * (T / filter_order)
    )

    n_axis = np.arange(n_start, n_end + 1, dtype=int)
    n_axis = np.delete(n_axis, np.where(n_axis == 0)[0])  # remove any n=0 entry

    # zero locations
    zeros = np.zeros(n_axis.shape[0], dtype=complex)
    zeros[:] = np.zeros_like(n_axis) + 1j * n_axis / (T / filter_order)

    # fill in poles_zeros dict
    poles_zeros = {"poles": np.ndarray((0, 0), dtype=complex), "zeros": zeros}

    # return
    return poles_zeros
