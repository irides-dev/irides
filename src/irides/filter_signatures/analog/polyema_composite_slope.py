"""
-------------------------------------------------------------------------------

Analog polyema composite-slope filter:

Implements the temporal, spectral, autocorrelation and pole/zero features
for this continuous-time composite-slope filter.

-------------------------------------------------------------------------------
"""

from typing import Callable
import numpy as np
import sys

from irides.design.analog import polyema_composite_slope as design

from irides.filter_signatures.analog import polyema_level as f_sig_ref

from irides.tools import impulse_response_builder_tools
from irides.tools import transfer_function_builder_tools
from irides.tools import impulse_response_tools
from irides.tools import transfer_function_tools


# noinspection SpellCheckingInspection
def generate_impulse_response(
    t_start: float,
    t_end: float,
    dt: float,
    tau: float,
    filter_order: int,
    arm_ratio=0.5,
) -> np.ndarray:
    """

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
    arm_ratio: float
        Ratio between tp / tm (default = 1/2)

    Returns
    -------
    np.ndarray
        A time series array (n, 2) having columns [t_axis, h].
    """

    # convert parameters to (tau+, tau-) literals
    tp, tm = design.convert_armratio_order_params_to_tp_tm_params(
        arm_ratio, tau, filter_order
    )

    # calculate gain adjustment
    gain_adj = 1.0 / (tm - tp)

    # calculate impulse response for each arm
    ts_pos_arm = f_sig_ref.generate_impulse_response(
        t_start, t_end, dt, tp, filter_order
    )
    ts_neg_arm = f_sig_ref.generate_impulse_response(
        t_start, t_end, dt, tm, filter_order
    )

    # construct composite-slope impulse response
    time_series = ts_pos_arm
    time_series[:, 1] = gain_adj * (ts_pos_arm[:, 1] - ts_neg_arm[:, 1])

    return time_series


# noinspection SpellCheckingInspection
def generate_sacf_correlogram(
    xi_start: float,
    xi_end: float,
    dxi: float,
    tau: float,
    filter_order: int,
    arm_ratio=0.5,
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
    arm_ratio: float
        Ratio between tp / tm (default = 1/2)
    normalize: bool
        T/F whether to normalize the sacf spectrum so sacf(0) = 1.

    Returns
    -------
    np.ndarray
        System autocorrelation (n, 2) np.array w/ cols [xi, sacf-value].
    """

    # fetch impulse response
    ir = generate_impulse_response(
        0.0, xi_end, dxi, tau, filter_order, arm_ratio
    )

    # calc sacf
    sacf = impulse_response_tools.calculate_auto_correlation_function(ir)

    # fetch kh0
    is_default_arm_ratio = np.round(arm_ratio / 0.5, 4) == 1.0
    if is_default_arm_ratio:
        kh0 = design.autocorrelation_peak_and_stride_values(
            tau, filter_order, dxi
        )["kh0"]
    else:
        i_xi0 = np.where(0.0 - dxi / 2.0 < sacf[:, 0])[0][0]
        kh0 = sacf[i_xi0, 1]  # empirical value

    # post process
    # fmt: off
    sacf = transfer_function_builder_tools. \
        truncate_and_normalize_sacf_correlogram(
            sacf, kh0, normalize, xi_start
        )
    # fmt: on

    return sacf


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_transfer_function(
    f_start: float,
    f_end: float,
    df: float,
    tau: float,
    filter_order: int,
    arm_ratio=0.5,
) -> np.ndarray:
    r"""
    Computes the transfer function H_slope(f) based on the
    underlying level filter H_lvl(f):

        H_slope(tau s) = gain_adj (H_lvl(tau_+ s) - H_lvl(tau_- s).

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
    arm_ratio: float
        Ratio between tp / tm (default = 1/2)

    Returns
    -------
    np.ndarray
        (n, 2), dtype=complex
        The transfer function with cols [f_axis, H_slope(f)].
    """

    # convert parameters to (tau+, tau-) literals
    tp, tm = design.convert_armratio_order_params_to_tp_tm_params(
        arm_ratio, tau, filter_order
    )

    # calculate gain adjustment
    gain_adj = 1.0 / (tm - tp)

    # fetch transfer function for each arm
    H_pos_arm = f_sig_ref.generate_transfer_function(
        f_start, f_end, df, tp, filter_order
    )
    H_neg_arm = f_sig_ref.generate_transfer_function(
        f_start, f_end, df, tm, filter_order
    )

    # construct the comp-slope xfer function
    H = H_pos_arm
    H[:, 1] = gain_adj * (H_pos_arm[:, 1] - H_neg_arm[:, 1])

    # return
    return H


# noinspection SpellCheckingInspection,PyPep8Naming
def make_callable_transfer_function(
    filter_order: int, arm_ratio=0.5
) -> Callable[[float, float], complex]:
    """Returns a callable function with signature (f, tau) -> complex.

    This transfer function is restricted to the jw axis:

        H(s) -> H(j w).

    Parameters
    ----------
    filter_order: int
        The filter order, which is captured by the definition herein.
    arm_ratio: float
        Ratio between tp / tm (default = 1/2)

    Returns
    -------
    Callable[[float, float], complex]
        A callable function (f: float, tau: float) -> complex
    """

    # convert parameters to (tau+, tau-) literals (unit tau)
    tp, tm = design.convert_armratio_order_params_to_tp_tm_params(
        arm_ratio, 1.0, filter_order
    )

    # calc gain adjustment
    gain_adj = 1.0 / (tm - tp)

    H_ref = f_sig_ref.make_callable_transfer_function(filter_order)

    # noinspection PyPep8Naming
    def H(f: float, tau: float = 1.0) -> complex:
        """Compute complex xfer fcxn H at freq `f` with optional scale tau.

        Note that `filter_order` and `arm_ratio` are captured,
        leaving only (f, tau) as free variables.

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

        return gain_adj * (H_ref(f, tp * tau) - H_ref(f, tm * tau))

    return H


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_sscf_correlogram(
    sti_start: float, sti_end: float, dsti: float, filter_order: int
) -> np.ndarray:
    """
    Computes the normalized system scale-correlation correlogram from the
    filter's transfer function.

    Note: a composite-slope filter takes the difference between two
    level filters,

        H_slope(s) = gain_adj (H_ref(tau_+ s) - H_ref(tau_- s))

    and the sscf integral covers

        int_reals H_slope(s) H_slope(-s sti) ds.

    Unlike inline-slope filters, no `sti` correction is necessary.

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
    f_start: float,
    f_end: float,
    df: float,
    tau: float,
    filter_order: int,
    arm_ratio=0.5,
) -> np.ndarray:
    """The phase of the transfer function.

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
    arm_ratio: float
        Ratio between tp / tm (default = 1/2)

    Returns
    -------
    np.ndarray
        The transfer function with cols [f_axis, angle_H(f)].
    """

    # convert parameters to (tau+, tau-) literals
    tp, tm = design.convert_armratio_order_params_to_tp_tm_params(
        arm_ratio, tau, filter_order
    )

    # fetch transfer function for each arm
    H_pos_arm = f_sig_ref.generate_transfer_function(
        f_start, f_end, df, tp, filter_order
    )
    H_neg_arm = f_sig_ref.generate_transfer_function(
        f_start, f_end, df, tm, filter_order
    )

    # alias
    f_axis = np.real(H_pos_arm[:, 0])
    H_pos = H_pos_arm[:, 1]
    H_neg = H_neg_arm[:, 1]

    # compute per-arm gain and phase
    g_pos = np.abs(H_pos)
    phi_pos = np.arctan2(np.imag(H_pos), np.real(H_pos))

    g_neg = np.abs(H_neg)
    phi_neg = np.arctan2(np.imag(H_neg), np.real(H_neg))

    # compute angle of composite
    im = g_pos * np.sin(phi_pos) - g_neg * np.sin(phi_neg)
    re = g_pos * np.cos(phi_pos) - g_neg * np.cos(phi_neg)
    angle = np.arctan2(im, re)

    # correct for f=0 value (given that the value comes thru a limit)
    i_f0 = np.where(0.0 - df / 2.0 < f_axis)[0][0]
    angle[i_f0] = design.phase_at_dc(tau)

    # unwrap
    angle = np.unwrap(angle)

    # force continuity across w=0 (if applicable)
    if i_f0 > 0:
        err = np.pi / 100
        if angle[i_f0] - angle[i_f0 - 1] > np.pi - err:
            angle[:i_f0] += np.pi

    # pack the return array, apply .unwrap to the angle
    angle_H = np.zeros_like(H_pos_arm, dtype=float)
    angle_H[:, 0] = np.real(H_pos_arm[:, 0]).copy()
    angle_H[:, 1] = angle

    # return
    return angle_H


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_spectra(
    f_start: float,
    f_end: float,
    df: float,
    tau: float,
    filter_order: int,
    arm_ratio=0.5,
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
    arm_ratio: float
        Ratio between tp / tm (default = 1/2)

    Returns
    -------
    np.ndarray
        Spectra panel [n_pts, 4] with cols [f, g(f), phase(f), grpdly(f)].
    """

    # fetch the transfer function for gain calculation
    H: np.ndarray = generate_transfer_function(
        f_start, f_end, df, tau, filter_order, arm_ratio
    )

    # fetch the phase spectrum for group-delay calculation
    phase_spectrum = generate_phase_spectrum(
        f_start, f_end, df, tau, filter_order, arm_ratio
    )

    # return spectra
    return transfer_function_tools.calculate_spectra(H, phase_spectrum)


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
