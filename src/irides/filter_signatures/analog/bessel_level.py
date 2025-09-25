"""
-------------------------------------------------------------------------------

Analog Bessel filter signature:

Implements the temporal and spectral signatures of a continuous-time
Bessel filter.

-------------------------------------------------------------------------------
"""

from typing import Callable
import numpy as np
from numpy.polynomial import polynomial
import sys

from irides.design.analog import bessel_level as design
from irides.design.analog import polyema_level as pema_design
from irides.filter_signatures.analog import polyema_level as pema_f_sig
from irides.filter_signatures.analog import damped_oscillator as dosc_f_sig
from irides.tools.design_tools import StageTypes
from irides.tools import impulse_response_builder_tools
from irides.tools import impulse_response_tools
from irides.tools import transfer_function_tools


# noinspection SpellCheckingInspection
def generate_impulse_response(
    t_start: float, t_end: float, dt: float, tau: float, filter_order: int
) -> np.ndarray:
    """
    There are at least three ways to compute the Bessel-filter impulse response:

        1) simple partial-fraction expansion,
        2) multi-stage partial-fraction expansion, and
        3) multi-stage cascade.

    This function implements the multi-stage cascade form because it is the
    most numerically stable of the methods, and it is revealing to see how the
    final impulse-response is built from each additional stage.

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
        Filter order .

    Returns
    -------
    np.ndarray
        A time series array (n, 2 + n_stgs) having columns
        [t_axis, h_bessel, h_stg1, h_stg1 * h_stg2, ...].
    """

    # validate
    impulse_response_tools.validate_filter_order_or_die(design, filter_order)

    # fetch the design config for this filter order
    design_config = design.designs(filter_order)

    # aliases
    stages = design_config["stages"]
    poles = design_config["poles"]

    # calc the number of stages
    n_stgs = len(design_config["stages"])

    # make time series array, an (n, 2 + n_stgs)) matrix
    # the cols are [t, h(t), h_stg1(t), h_stg1(t) * h_stg2(t), ...]
    # where the cascaded convolution
    # make time series template
    (
        time_series,
        i_t_0plus,
    ) = impulse_response_builder_tools.make_impulse_response_template(
        t_start, t_end, dt, 2 + n_stgs
    )

    # aliases
    t_axis = time_series[:, 0]
    t_0plus = t_axis[i_t_0plus:]
    h = time_series[:, 1]

    # set effective t-start and -end points
    t_start_eff = min(t_0plus)
    t_end_eff = max(t_0plus)

    # create ht-generating fcxns

    # noinspection SpellCheckingInspection
    def ht_ema_gen(reference_poles: np.ndarray) -> np.ndarray:
        """Generator of h(t) for the ema. Captures scope-fixed arg values."""

        scaled_reference_tau = (
            tau
            * pema_design.convert_reference_poles_to_temporal_scales(
                reference_poles
            )[0]
        )
        return pema_f_sig.generate_impulse_response(
            t_start_eff, t_end_eff, dt, scaled_reference_tau, 1
        )[:, 1]

    # noinspection SpellCheckingInspection
    def ht_dosc_gen(reference_poles: np.ndarray) -> np.ndarray:
        """Generator of h(t) for the dosc. Captures scope-fixed arg values."""

        return dosc_f_sig.generate_impulse_response(
            t_start_eff, t_end_eff, dt, reference_poles[1], tau
        )[:, 1]

    # pack into a dict
    ht_gen_dict = {
        StageTypes.EMA.value: ht_ema_gen,
        StageTypes.DOSC.value: ht_dosc_gen,
    }

    # iter across stages
    ht_cand = np.array([1.0 / dt])  # starting point for cascade
    for i, stage in enumerate(stages):

        # gen h(t) for this stage (dosc or ema)
        ht_stage = ht_gen_dict[stage["type"]](poles[stage["indices"]])

        # convolve in new h(t) with running candidate
        ht_cand = np.convolve(ht_stage, ht_cand)[: t_0plus.shape[0]] * dt

        # persist running conv in an extended column
        time_series[i_t_0plus:, 2 + i] = ht_cand

    # persist the final cascade h(t) result
    h[i_t_0plus:] = ht_cand

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

    # fetch impulse response
    ir = generate_impulse_response(0.0, xi_end, dxi, tau, filter_order)

    # calc sacf
    sacf = impulse_response_tools.calculate_auto_correlation_function(ir)

    # truncate at xi_start
    sacf = sacf[xi_start - dxi / 2.0 <= sacf[:, 0], :]

    # normalize by numeric peak value if requested
    # note that the numeric value will diverge from the theo value,
    # and the diveragence is a function of dxi, so it's better in this case
    # to normalize by the numerically calculated kh0 value.
    # note also that a xi = 0 value is expected, else the results will be
    # in error.
    if normalize:
        i_zero = np.where(0.0 - dxi / 2 < sacf[:, 0])[0][0]
        sacf[:, 1] /= sacf[i_zero, 1]

    return sacf


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_transfer_function(
    f_start: float, f_end: float, df: float, tau: float, filter_order: int
) -> np.ndarray:
    """
    Computes the transfer function H_bessel(f) from an array of frequencies f.
    The Bessel-filter transfer function is determined by the Bessel polynomial
    values, and these values depend on the filter order.

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
        The transfer function with cols [f_axis, H_bessel(f)].
    """

    # validate
    impulse_response_tools.validate_filter_order_or_die(design, filter_order)

    # make transfer-function panel
    npts = np.arange(f_start, f_end, df).shape[0]
    H_bessel = np.ndarray([npts, 2], dtype=complex)

    # setup f_axis
    H_bessel[:, 0] = np.arange(f_start, f_end, df)
    f_axis = H_bessel[:, 0]

    # make f_axis and jwtau vector
    jwtau = (1j * 2.0 * np.pi * tau) * f_axis

    # fetch Bessel denominator coeffs, set numerator coeff
    denom_coeffs = design.designs(filter_order)["poly_coeffs"]
    num_coeff = denom_coeffs[-1]

    # calculate the transfer function
    H_bessel[:, 1] = num_coeff / polynomial.polyval(jwtau, denom_coeffs[::-1])

    # return
    return H_bessel


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

    # validate
    impulse_response_tools.validate_filter_order_or_die(design, filter_order)

    # create capture parameters
    poly_coefs = design.designs(filter_order)["poly_coeffs"]
    num_coef = poly_coefs[-1]
    denom_coefs = poly_coefs[::-1]

    # noinspection SpellCheckingInspection,PyPep8Naming
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

        return num_coef / polynomial.polyval(jw * tau, denom_coefs)

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
    """
    The phase of the Bessel-filter transfer function is the sum of phases from
    each stage.

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
        The transfer function with cols [f_axis, angle_H_bessel(f)].
    """

    # validate
    impulse_response_tools.validate_filter_order_or_die(design, filter_order)

    # make transfer-function panel
    npts = np.arange(f_start, f_end, df).shape[0]
    angle_H_bessel = np.zeros((npts, 2))

    # setup f_axis
    angle_H_bessel[:, 0] = np.arange(f_start, f_end, df)

    # fetch the design
    this_design = design.designs(filter_order)

    stages = this_design["stages"]
    poles = this_design["poles"]

    # iter over stages
    for stage in stages:

        stage_poles = poles[stage["indices"]]

        if stage["type"] == StageTypes.EMA.value:

            stage_order = 1
            stage_pole_taus = pema_design.convert_reference_poles_to_temporal_scales(
                stage_poles
            )
            stage_tau = tau * stage_pole_taus[0]

            angle_H_bessel[:, 1] += pema_f_sig.generate_phase_spectrum(
                f_start, f_end, df, stage_tau, stage_order
            )[:, 1]

        if stage["type"] == StageTypes.DOSC.value:

            reference_pole = stage_poles[0]

            angle_H_bessel[:, 1] += dosc_f_sig.generate_phase_spectrum(
                f_start, f_end, df, reference_pole, tau
            )[:, 1]

    # return
    return angle_H_bessel


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
    phase_spectrum: np.ndarray = generate_phase_spectrum(
        f_start, f_end, df, tau, filter_order
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


# noinspection SpellCheckingInspection,PyUnusedLocal
def generate_poles_and_zeros(
    tau: float, filter_order: int, f_start=0.0, f_end=0.0
) -> dict:
    """
    The Bessel-filter unit-scale poles are available in the design module, yet
    for consistency of the api, this function packs the poles for a particular
    order in a container. Continuous-time Bessel filters have no zeros.

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

    # validate
    impulse_response_tools.validate_filter_order_or_die(design, filter_order)

    # fetch the design config for this filter order
    design_config = design.designs(filter_order)

    # scale poles by tau
    poles = design_config["poles"] / tau

    # fill in poles_zeros dict
    poles_zeros = {"poles": poles, "zeros": np.array([], dtype=complex)}

    # return
    return poles_zeros
