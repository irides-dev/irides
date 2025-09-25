"""
-------------------------------------------------------------------------------

Analog damped oscillator:

Implements the temporal and spectral signatures of a continuous-time
damped-oscillator filter.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.design.analog import damped_oscillator as design
from irides.tools import impulse_response_builder_tools
from irides.tools import impulse_response_tools
from irides.tools import transfer_function_tools


# noinspection SpellCheckingInspection
def generate_impulse_response(
    t_start: float, t_end: float, dt: float, reference_pole: complex, tau: float
) -> np.ndarray:
    r"""Creates continuous-time-like impulse response of the damped oscillator.

    The angle of inclination of the reference pole with respect to the negative
    real axis governs which expression is used to construct the impulse
    response.

    In the general case, 0 < |angle(reference_pole)|, so

        h(t) = wn csc(theta) sin( sin(theta) wn t ) exp( -cos(theta) wn t ) u(t)

    but for angle(reference_pole) == 0,

        h(t) = wn^2 t exp( -wn t ) u(t).

    Here,

        wn = |reference_pole|
        theta = angle(reference_pole)

    where theta is respect to the negative real axis. Note that /reference_pole/
    is expected to be cartesian, and accordingly its angle will be wrt to
    positive real axis, so an adjustment is made.

    Scaling by tau is done through the law

        h(t) ->  tau^{-1} h(t/tau) .

    Parameters
    ----------
    t_start: float
        Start time.
    t_end: float
        End time.
    dt: float
        Time increment.
    reference_pole: complex
        Cartesian coordinates of pole.
    tau: float
        First moment of the filter.

    Returns
    -------
    np.ndarray
        A time series array (n, 2) having columns [t_axis, h_dosc].
    """

    # make time series, an (n, 2) matrix
    time_series = np.zeros((np.arange(t_start, t_end + dt, dt).shape[0], 2))

    # make time series template
    (
        time_series,
        i_t_0plus,
    ) = impulse_response_builder_tools.make_impulse_response_template(
        t_start, t_end, dt
    )

    # alias
    t_axis = time_series[:, 0]
    t_0plus = t_axis[i_t_0plus:]
    h = time_series[:, 1]

    # from /reference_pole/ extract wn and theta
    wns, thetas = design.extract_wn_and_theta(np.array([reference_pole]))
    wn = wns[0]
    theta = thetas[0]

    # branch on theta
    if theta == 0.0:

        h[i_t_0plus:] = (
            np.power(wn, 2) * (t_0plus / tau) * np.exp(-wn * t_0plus / tau)
        )

    else:

        h[i_t_0plus:] = (
            wn
            / (tau * np.sin(theta))
            * np.sin(np.sin(theta) * wn * t_0plus / tau)
            * np.exp(-np.cos(theta) * wn * t_0plus / tau)
        )

    # return
    return time_series


# noinspection SpellCheckingInspection
def generate_sacf_correlogram(
    xi_start: float,
    xi_end: float,
    dxi: float,
    reference_pole: complex,
    tau: float,
    normalize=False,
) -> np.ndarray:
    """Creates S-ACF response by impulse-response correlation with itself.

    Parameters
    ----------
    xi_start: float
        Start lag, |xi_start| <= |xi_end|
    xi_end: float
        End lag
    dxi: float
        Lag increment
    reference_pole: complex
        Cartesian coordinates of pole.
    tau: float
        First moment of the filter.
    normalize: bool
        T/F whether to normalize the sacf spectrum so sacf(0) = 1.

    Returns
    -------
    np.ndarray
        System autocorrelation (n, 2) np.array w/ cols [xi, sacf-value].
    """

    # fetch impulse response
    ir = generate_impulse_response(0.0, xi_end, dxi, reference_pole, tau)

    # calc sacf
    sacf = impulse_response_tools.calculate_auto_correlation_function(ir)

    # truncate at xi_start
    sacf = sacf[sacf[:, 0] >= xi_start - dxi / 2.0, :]

    # normalize by theoretic peak value if requested
    if normalize:
        kh0 = design.autocorrelation_peak_and_stride_values(
            np.array([reference_pole]), tau
        )["kh0"]
        sacf[:, 1] /= kh0

    return sacf


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_transfer_function(
    f_start: float, f_end: float, df: float, reference_pole: complex, tau: float
) -> np.ndarray:
    """
    Returns an np array, first column is the (cyclic) frequency axis,
    second column is the complex-valued transfer function.

    The defining transfer function is

                                 p p*
        H_dosc(tau s) = -------------------------
                        (tau s + p)(tau s + p*)

    In general, s = omega + j w, a complex value. Here, however, omega = 0,
    so /s/ is replaced by /j w/.

    Moreover, the api is for cyclic frequency /f/, measured in cycles per sec,
    but radial frequency /w/, measured in radians per sec, is used internally.

    Parameters
    ----------
    f_start: float
        Start cyclic frequency.
    f_end: float
        End cyclic frequency.
    df: float
        Cyclic frequency increment.
    reference_pole: complex
        Cartesian coordinates of pole.
    tau: float
        First moment of the final filter.

    Returns
    -------
    np.ndarray
        The transfer function with cols [f_axis, H_dosc(f)],
        (n, 2), dtype=complex
    """

    # aliases
    p = reference_pole
    pc = p.conjugate()

    # make transfer-function panel
    npts = np.arange(f_start, f_end, df).shape[0]
    H_dosc = np.ndarray([npts, 2], dtype=complex)

    # setup f_axis
    H_dosc[:, 0] = np.arange(f_start, f_end, df)
    f_axis = H_dosc[:, 0]

    # compute the complex transfer function along the /jw/ axis
    jw = 1j * 2.0 * np.pi * f_axis

    H_dosc[:, 1] = p * pc / ((tau * jw + p) * (tau * jw + pc))

    # return
    return H_dosc


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_phase_spectrum(
    f_start: float, f_end: float, df: float, reference_pole: complex, tau: float
) -> np.ndarray:
    r"""
    Computes the phase of H_dosc(f) from the analytic form

                                   / w + wn sin(th) \
        angle_H_dosc(w) = - arctan|------------------|
                                   \   wn cos(th)   /

                                     / w - wn sin(th) \
                            - arctan|------------------|
                                     \  wn cos(th)    /

    Temporal scaling is applied with the substitution w -> tau w.

    Parameters
    ----------
    f_start: float
        Start cyclic frequency.
    f_end: float
        End cyclic frequency.
    df: float
        Cyclic frequency increment.
    reference_pole: complex
        Cartesian coordinates of pole.
    tau: float
        First moment of the final filter.

    Returns
    -------
    np.ndarray
        The phase spectrum with cols [f_axis, angle_H_dosc(f)],
        (n, 2), dtype=complex
    """

    # make phase-spectrum panel
    npts = np.arange(f_start, f_end, df).shape[0]
    angle_H_dosc = np.ndarray([npts, 2], dtype=float)

    # setup f_axis
    angle_H_dosc[:, 0] = np.arange(f_start, f_end, df)
    tau_f = tau * angle_H_dosc[:, 0]

    # from /reference_pole/ extract wn and theta
    wns, thetas = design.extract_wn_and_theta(np.array([reference_pole]))
    fn = wns[0] / (2.0 * np.pi)
    theta = thetas[0]

    # compute the phase
    angle_H_dosc[:, 1] = -np.arctan2(
        tau_f + fn * np.sin(theta), fn * np.cos(theta)
    ) - np.arctan2(tau_f - fn * np.sin(theta), fn * np.cos(theta))

    return angle_H_dosc


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_spectra(
    f_start: float, f_end: float, df: float, reference_pole: complex, tau: float
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
    reference_pole: complex
        Cartesian coordinates of pole.
    tau: float
        First moment of the final filter.

    Returns
    -------
    np.ndarray
        Spectra panel [n_pts, 4] with cols [f, g(f), phase(f), grpdly(f)].
    """

    # fetch the transfer function for gain calculation
    H: np.ndarray = generate_transfer_function(
        f_start, f_end, df, reference_pole, tau
    )

    # fetch the phase spectrum for group-delay calculation
    phase_H: np.ndarray = generate_phase_spectrum(
        f_start, f_end, df, reference_pole, tau
    )

    # calculate and return spectra
    return transfer_function_tools.calculate_spectra(H, phase_H)
