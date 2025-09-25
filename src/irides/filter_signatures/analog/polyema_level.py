"""
-------------------------------------------------------------------------------

Analog polynomial ema:

Implements the temporal, spectral, autocorrelation and pole / zero features
for this continuous-time level filter.

-------------------------------------------------------------------------------
"""

from typing import Callable
import numpy as np
import scipy.special
import sys

from irides.design.analog import polyema_level as design
from irides.tools import impulse_response_builder_tools
from irides.tools import impulse_response_tools
from irides.tools import transfer_function_tools


# noinspection SpellCheckingInspection
def generate_impulse_response(
    t_start: float, t_end: float, dt: float, tau: float, filter_order: int
) -> np.ndarray:
    r"""
    Creates a continuous-time-like impulse response for the poly-ema
    level filter.

    The impulse response of an mth order filter reads

                        1          /    t     \ (m-1)
        h(t) = ------------------ | ---------- |      e^(-t / (tau / m)) u(t),
               (tau / m) Gamma(m)  \  tau / m /

    where tau is the first-moment of the h(t).

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
        Filter order .

    Returns
    -------
    np.ndarray
        A time series array (n, 2) having columns [t_axis, h_pema].
    """

    # validate
    impulse_response_tools.validate_filter_order_or_die(design, filter_order)

    # make time series template
    # fmt: off
    time_series, i_t_0plus = impulse_response_builder_tools. \
        make_impulse_response_template(
            t_start, t_end, dt
        )
    # fmt: on

    # alias
    t_axis = time_series[:, 0]
    t_0plus = t_axis[i_t_0plus:]
    h = time_series[:, 1]

    # compute impulse response t >= 0
    m = filter_order
    tau_m = design.tau_per_stage(tau, m)

    h[i_t_0plus:] = (
        (1.0 / (tau_m * scipy.special.gamma(m)))
        * np.power(t_0plus / tau_m, m - 1)
        * np.exp(-t_0plus / tau_m)
    )

    # return
    return time_series


# noinspection SpellCheckingInspection
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
        Filter order.
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
    Computes the transfer function H_pema(f) of the polynomial ema.
    The defining transfer function is

                             1
        H(tau s) = --------------------- .
                    (1 + (tau / m) s)^m

    In general, s = omega + j w, a complex value. Here, however, omega = 0,
    so /s/ is replaced by /j w/. Note that w = 2 pi f.

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
        Filter order.

    Returns
    -------
    np.ndarray
        (n, 2), dtype=complex
        The transfer function with cols [f_axis, H_pema(f)].
    """

    # validate
    impulse_response_tools.validate_filter_order_or_die(design, filter_order)

    # alias
    m = filter_order

    # make transfer-function panel
    npts = np.arange(f_start, f_end, df).shape[0]
    H_pema = np.ndarray([npts, 2], dtype=complex)

    # setup f_axis
    H_pema[:, 0] = np.arange(f_start, f_end, df)
    f_axis = H_pema[:, 0]

    # make f_axis and sTm vector
    jwtaum = (1j * 2.0 * np.pi * tau / m) * f_axis

    # calculate transfer function
    H_pema[:, 1] = 1.0 / np.power(1.0 + jwtaum, m)

    # return
    return H_pema


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

        return np.power(1.0 + jw * tau / filter_order, -filter_order)

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


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_phase_spectrum(
    f_start: float, f_end: float, df: float, tau: float, filter_order: int
) -> np.ndarray:
    r"""
    Computes the phase of H_pema(f) from the analytic form

        angle_H_pema(f) = -m arctan( j 2 pi f tau / m ).

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
        (n, 2), dtype=complex
        The phase spectrum with cols [f_axis, angle_H_pema(f)].
    """

    # make phase-spectrum panel
    npts = np.arange(f_start, f_end, df).shape[0]
    angle_H_pema = np.ndarray([npts, 2], dtype=float)

    # setup f_axis
    angle_H_pema[:, 0] = np.arange(f_start, f_end, df)
    f_axis = angle_H_pema[:, 0]

    # compute the phase
    m = filter_order
    angle_H_pema[:, 1] = -m * np.arctan(2.0 * np.pi * (tau / m) * f_axis)

    return angle_H_pema


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_spectra(
    f_start: float, f_end: float, df: float, tau: float, filter_order: int
) -> np.ndarray:
    """Generate gain, phase and group-delay spectra.

    Parameters
    ----------
    f_start: float
        Start cyclic frequency.
    f_end: float
        End cyclic frequency.
    df: float
        Cyclic frequency increment.
    tau: float
        First moment of the filter.
    filter_order: int
        Filter order.

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

    # return spectra
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


# noinspection SpellCheckingInspection,PyPep8Naming,PyUnusedLocal
def generate_poles_and_zeros(
    tau: float, filter_order: int, f_start=0.0, f_end=0.0
) -> dict:
    """
    Calculates poles and zeros for the polyema filter in the s-plane.
    There are no zeros, and the only pole(s) is an mth-order degenerate
    pole at

        s = sigma + jw = -(tau / m)^-1 .

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Filter order.
    f_start: float
        Start cyclic frequency (not used in this signature).
    f_end: float
        End cyclic frequency (not used in this signature).

    Returns
    -------
    dict
        Dictionary with keys ('poles', 'zeros') and values as np.ndarray
        with dtype=complex.
    """

    pole_location = -1.0 / (tau / filter_order) + 1j * 0.0

    # fill in poles_zeros dict
    poles_zeros = {
        "poles": np.array([pole_location]),
        "zeros": np.ndarray((0, 0), dtype=complex),
    }

    # return
    return poles_zeros
