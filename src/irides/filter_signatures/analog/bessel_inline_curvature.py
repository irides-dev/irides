"""
-------------------------------------------------------------------------------

Analog Bessel inline curvature filter:

Implements the temporal, spectral, autocorrelation and pole/zero features
for this continuous-time inline-slope filter.

-------------------------------------------------------------------------------
"""

from typing import Callable
import numpy as np
import sys

from irides.design.analog import bessel_inline_curvature as design

from irides.filter_signatures.analog import bessel_level as f_sig_ref

from irides.tools import impulse_response_builder_tools
from irides.tools import transfer_function_builder_tools
from irides.tools import impulse_response_tools
from irides.tools import transfer_function_tools


# noinspection SpellCheckingInspection
def generate_impulse_response(
    t_start: float, t_end: float, dt: float, tau: float, filter_order: int
) -> np.ndarray:
    r"""
    Creates a continuous-time-like impulse response for this filter.
    The implementation starts with a temporally scaled level impulse
    response and then takes the causal second difference.

    Only orders [3..8] are implemented.

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
    np.ndarray
        A time series array (n, 2) having columns [t_axis, h].
    """

    # start with a level-filter impulse response
    time_series = f_sig_ref.generate_impulse_response(
        t_start, t_end, dt, tau, filter_order
    )

    # fmt: off
    i_t_0plus = impulse_response_builder_tools. \
        compute_noncausal_symmetric_second_difference(
            time_series
    )
    # fmt: on

    # h(0) correction
    ht0_value = design.impulse_response_t0_value(tau, filter_order, dt)
    if i_t_0plus is not None:
        time_series[i_t_0plus, 1] = ht0_value

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

    # fetch kh0
    kh0 = design.autocorrelation_peak_and_stride_values(tau, filter_order, dxi)[
        "kh0"
    ]

    # post process
    sacf = transfer_function_builder_tools.truncate_and_normalize_sacf_correlogram(
        sacf, kh0, normalize, xi_start
    )

    return sacf


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_transfer_function(
    f_start: float, f_end: float, df: float, tau: float, filter_order: int
) -> np.ndarray:
    r"""
    Computes the transfer function H_curv(f) based on the
    underlying reference filter H_ref(f):

        H_curv(tau s) = s^2 H_ref(tau s).

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
        The transfer function with cols [f_axis, H_curv(f)].
    """

    # compute reference transfer function
    H = f_sig_ref.generate_transfer_function(
        f_start, f_end, df, tau, filter_order
    )

    # convert to inline curvature
    # fmt: off
    transfer_function_builder_tools. \
        apply_leading_jw_coefficient_to_transfer_function(H, 2)
    # fmt: on

    # return
    return H


# noinspection SpellCheckingInspection,PyPep8Naming
def make_callable_transfer_function(
    filter_order: int,
) -> Callable[[float, float], complex]:
    """Returns a callable function with signature (f, tau) -> complex.

    Constructs H(jw) = (jw)^2 H_ref(tau jw).

    Parameters
    ----------
    filter_order: int
        The filter order, which is captured by the definition herein.

    Returns
    -------
    Callable[[float, float], complex]
        A callable function (f: float, tau: float) -> complex
    """

    # fmt: off
    power = 2
    return transfer_function_builder_tools. \
        make_callable_transfer_function_with_leading_jw_power_term(
            f_sig_ref, filter_order, power
        )
    # fmt: on


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_sscf_correlogram(
    sti_start: float, sti_end: float, dsti: float, filter_order: int
) -> np.ndarray:
    """
    Computes the normalized system scale-correlation correlogram from the
    filter's transfer function.

    Note: inline-curvature filters have the transfer function

        s^2 H(tau s)

    and the sscf integral covers

        int_reals s^4 H(s) H(-s sti) ds.

    The way the code is structured, `s` is attached to `H(s)` before this
    integral is evaluated, creating

        int_reals s^2 (s sti)^2 H(s) H(-s sti) ds.

    Therefore, the integral needs to be corrected by 1 / sti^2.

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
    sti_norm_correction_fcxn = lambda sti: 1.0 / np.power(sti, 2)

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
    Computes the phase of H_curv(f) based on the underlying reference
    filter H_ref(f):

        angle_H_curv(f) = pi + angle_H_ref(f).

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
        The phase spectrum with cols [f_axis, angle_H_curv(f)].
    """

    # compute reference transfer function
    H = f_sig_ref.generate_phase_spectrum(f_start, f_end, df, tau, filter_order)

    # convert to inline slope
    # fmt: off
    transfer_function_builder_tools. \
        apply_leading_jw_coefficient_to_phase_spectrum(H, 2)
    # fmt: on

    # return
    return H


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

    H = generate_transfer_function(f_start, f_end, df, tau, filter_order)

    phase_H = generate_phase_spectrum(f_start, f_end, df, tau, filter_order)

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
