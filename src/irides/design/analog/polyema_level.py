"""
-------------------------------------------------------------------------------

Analog poly-ema filter design:

Captures the design details necessary to design a polyema level filter,
and also captures details of its characteristic behavior.

The level filter is a low-pass filter that can be used itself or as
a component for a slope, curvature or mirror filter.

Wavenumbers are calculated numerically in Python in `calculate_wavenumber.py`.

-------------------------------------------------------------------------------
"""

import numpy as np
import scipy.special
import sys

from irides.resources.containers import points
from irides.tools import design_tools
from irides.tools.design_tools import StageTypes


# noinspection SpellCheckingInspection
def design_id() -> str:
    """
    Since `design` modules are passed around as objects in this framework,
    a `design_id` gives a unique name.

    Returns
    -------
    str
        The id of this design.
    """

    return "polyema-level"


def design_type() -> design_tools.FilterDesignType:
    """This is a level filter"""

    return design_tools.FilterDesignType.LEVEL


# noinspection SpellCheckingInspection
def get_valid_filter_orders(strict: bool = False) -> np.ndarray:
    """Returns valid filter orders for this filter design.

    Parameters
    ----------
    strict: bool
        When true, only orders where h(0)=0 are included.

    Returns
    -------
    np.ndarray
        Array of valid filter orders
    """

    return design_tools.get_valid_filter_orders(
        design_tools.FilterDesignType.LEVEL, strict
    )


"""
-------------------------------------------------------------------------------

Filter-specific definitions

-------------------------------------------------------------------------------
"""


# noinspection SpellCheckingInspection
def designs(filter_order: int) -> dict:
    """Design parameters as a function of filter order.

    Parameters
    ----------
    filter_order: int
        The order of the filter.

    Returns
    -------
    dict
        Design dictionary.
    """

    # design given the order of the filter
    m = filter_order
    design = {
        "stages": [
            {"type": StageTypes.EMA.value, "indices": np.array([i])}
            for i in range(m)
        ],
        "poles": -m * np.ones(m, dtype=complex),
        "poly_coeffs": [scipy.special.binom(m, k) for k in range(m + 1)],
    }

    # return the design
    return design


def integer_sequence_a055897(n: int) -> int:
    """
    Integer sequence related peak impulse-response value.

    See https://oeis.org/A055897

    Parameters
    ----------
    n: int
        Sequence index.

    Returns
    -------
    int
        Sequence value.
    """

    return n * np.power((n - 1), n - 1)


# noinspection SpellCheckingInspection
def convert_reference_poles_to_temporal_scales(
    reference_poles: np.ndarray,
) -> np.ndarray:
    """Convert a single pole to the equivalent temporal scale.

    The relation between a reference pole and temporal scale is

        tau = 1 / abs(reference_pole)

    It is expected that imag(reference_pole) = 0 (since the scope of this
    function is in the polyema design, rather than damped oscillator design).
    Nonetheless, it is canonical to calculate the radius of the pole in the
    s-plane.

    Parameters
    ----------
    reference_poles: np.ndarray (dtype = complex)
        Reference poles to convert to temporal scales.

    Returns
    -------
    np.ndarray
        tau-values associated with reference poles.

    """

    return 1.0 / np.abs(reference_poles)


def tau_per_stage(tau: float, filter_order: int) -> float:
    """
    Calculates the temporal scale factor /tau_m/ for any of the identical
    ema stages along the polyema cascade:

        tau_per_stage_value = tau / filter_order .

    The cascade of /filter_order/ identical ema stages generates a combined
    first moment of /tau/.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        The order of the filter.

    Returns
    -------
    float
        The first moment for each (identical) stage: `tau_per_stage_value`.
    """

    # this is a simple calculation
    tau_per_stage_value = tau / filter_order

    # return
    return tau_per_stage_value


"""
-------------------------------------------------------------------------------

Standard features

-------------------------------------------------------------------------------
"""


def unit_peak_time(filter_order: int) -> float:
    """
    As per doc string for peak_value_coordinate(..),

        t* = (m - 1) / m  tau .

    This function returns the filter-order-based coefficient

        (m - 1) / m

    Parameters
    ----------
    filter_order: int
        The order of the filter.

    Returns
    -------
    float
        The timepoint where h(t) peaks, for unit delay.
    """

    # alias
    m = filter_order

    # return
    return (m - 1.0) / m


def unit_peak_value(filter_order: int) -> float:
    """
    As per doc string for peak_value_coordinate(..),

                        m (m-1)^(m-1)
        h(t*) * tau = -----------------
                       Gamma(m) e^(m-1)

    Function returns the right-hand side.

    Parameters
    ----------
    filter_order: int
        The order of the filter.

    Returns
    -------
    float
        The value of h(t_peak), for unit delay.

    """

    # alias
    m = filter_order

    # construct fraction
    numerator = integer_sequence_a055897(m)
    denominator = scipy.special.gamma(m) * np.exp(m - 1)

    # return
    return numerator / denominator


# noinspection PyUnusedLocal
def peak_value_coordinate(tau: float, filter_order: int, dt=0.0):
    """
    Low-pass polyema filters have the characteristic behavior of
    having a single maximum value along its impulse response h(t).

                  ....x...  <---- h(t*)
              ....        ....
           ...                ..
         ..                     ...
        .                          ....
    ----|-------------|-------------------------> t
        0
                     t*

    From Mathematica script polyema_inline_slope.nb, the
    time t* where h(t) attains is maximum and the value of h(t*) are

              m-1                     m (m-1)^(m-1)        1
        t* = ----- tau,  and h(t*) = --------------- -------------,
               m                        Gamma(m)      e^(m-1) tau

    where m = /filter_order/. Note that
    the sequence m (m-1)^(m-1) is identified as https://oeis.org/A055897.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        The order of the filter.
    dt: float
        Unused.

    Returns
    -------
    ContinuousTimePoint
        Object `peak_coord` with .time and .value fields.
    """

    # alias
    m = filter_order

    # put together into a point coordinate
    peak_coord = points.ContinuousTimePoint(
        unit_peak_time(m) * tau, unit_peak_value(m) / tau
    )

    # return
    return peak_coord


# noinspection SpellCheckingInspection,PyUnusedLocal
def moment_value_generator(moment: int, tau: float = 1.0, dt: float = 0.0):
    """
    Returns an anonymous function with the single argument (order: int),
    where order is a `filter_order`. The returned function captures the `tau`
    and `dt` arguments in this calling function.

    Note that the generated functions are not purely continuous time.
    Instead, they bridge continuous and discrete time by applying order-dt
    corrections to the moments. This is required because the CT series are
    in fact implemented in DT.

    Parameters
    ----------
    moment: int
        Moment to calculate: [0, 1, 2]
    tau: float
        Temporal scale of the filter.
    dt: float
        Time interval for discretization.

    Returns
    -------
    Anonymous function with argument (order: int).
    """

    # generator defs with dt capture
    # noinspection PyUnusedLocal
    def m0_gen(order: int) -> float:
        if order == 1:
            return 1.0 + (dt / tau) / 2.0
        else:
            return 1.0

    # noinspection PyUnusedLocal
    def m1_gen(order: int) -> float:
        return tau

    # noinspection PyUnusedLocal
    def m2_gen(order: int) -> float:
        return ((order + 1.0) / order) * np.power(tau, 2)

    # return
    if moment == 0:
        return m0_gen
    elif moment == 1:
        return m1_gen
    elif moment == 2:
        return m2_gen
    else:
        msg = "Moments must be within (0, 1, 2). Calling value: {0}".format(
            moment
        )
        raise IndexError(msg)


# noinspection SpellCheckingInspection,PyPep8Naming
def autocorrelation_peak_and_stride_values(
    tau: float, filter_order: int
) -> dict:
    """
    The zero-lag S-ACF parameter, kh(0), is analytic:

                       Gamma(m-1/2)
        kh(0) = ----------------------------- ,
                 2 sqrt(pi) (tau/m) Gamma(m)

    where m = filter_order.

    The stride xi_5pct where sacf(lag=xi_5pct) = 0.05 is reported numerically
    (based on results from Mathematica).

    The zero-lag kh(0) scales as 1 / tau, and the stride xi_5pct scales as tau.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Order of the filter.

    Returns
    -------
    dict
        Peak value (scaled by 1/tau) and 5% pct stride (scaled by tau)
    """

    def kh0_gen(order: int) -> float:
        """Implements equation for kh(0) value, tau=1"""

        gamma = scipy.special.gamma
        return gamma(order - 0.5) / (
            2.0 * np.sqrt(np.pi) * (1.0 / order) * gamma(order)
        )

    # see Mathematica polyema_level.nb
    unit_sacf_values = {
        1: {"kh0": kh0_gen(1), "xi_5pct": 2.99573},
        2: {"kh0": kh0_gen(2), "xi_5pct": 2.37193},
        3: {"kh0": kh0_gen(3), "xi_5pct": 1.97288},
        4: {"kh0": kh0_gen(4), "xi_5pct": 1.71918},
        5: {"kh0": kh0_gen(5), "xi_5pct": 1.54189},
        6: {"kh0": kh0_gen(6), "xi_5pct": 1.40953},
        7: {"kh0": kh0_gen(7), "xi_5pct": 1.30603},
        8: {"kh0": kh0_gen(8), "xi_5pct": 1.22228},
    }
    zero_values = {"kh0": 0.0, "xi_5pct": 0.0}

    # check and return
    if filter_order in unit_sacf_values.keys():
        ans = unit_sacf_values[filter_order]
        ans["kh0"] /= tau
        ans["xi_5pct"] *= tau
    else:
        ans = zero_values

    return ans


# noinspection SpellCheckingInspection,PyPep8Naming
def scale_correlation_stride_and_residual_values(
    tau: float, filter_order: int
) -> dict:
    """Scale decorrelation strides and residual correlations.

    Parameters
    ----------
    tau: float
        The first moment of the filter.
    filter_order: int
        Order of the filter.

    Returns
    -------
    dict
        Scale decorrelation (scales as `tau`) and residual correlation
        (independent of `tau`).
    """

    # values from Mathematica
    unit_sscf_values = {
        1: {"sti_5pct": 1598.00, "residual": 0.05},
        2: {"sti_5pct": 27.4358, "residual": 0.05},
        3: {"sti_5pct": 11.1683, "residual": 0.05},
        4: {"sti_5pct": 7.27680, "residual": 0.05},
        5: {"sti_5pct": 5.60510, "residual": 0.05},
        6: {"sti_5pct": 4.68270, "residual": 0.05},
        7: {"sti_5pct": 4.09790, "residual": 0.05},
        8: {"sti_5pct": 3.69310, "residual": 0.05},
    }

    # fetch values for order and scale
    sscf_values = unit_sscf_values[filter_order]
    sscf_values["sti_5pct"] *= tau

    # return
    return sscf_values


# noinspection SpellCheckingInspection,PyPep8Naming,PyUnusedLocal
def full_width_generator(tau: float = 1.0, dt: float = 0.0):
    """
    Returns anonymous function for the full-width. `tau` and `dt` are captured
    here, and the anonymous function has one argument, (order: int), the filter
    order.

    Parameters
    ----------
    tau: float
        Temporal scale of the filter.
    dt: float
        Time interval for discretization.

    Returns
    -------
    Anonymous function with argument (order: int).
    """

    def fw_gen(order: int):
        return 2.0 * np.sqrt(1.0 / order) * tau

    return fw_gen


# noinspection PyUnusedLocal
def gain_at_dc(tau: float) -> float:
    """Filter gain at DC frequency. This is independent of filter order.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.

    Returns
    -------
    float
        The gain at DC frequency.
    """

    return 1.0


# noinspection PyUnusedLocal
def phase_at_dc(tau: float) -> float:
    """Filter phase at DC frequency. This is independent of filter order.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.

    Returns
    -------
    float
        The gain at DC frequency.
    """

    return 0.0


def group_delay_at_dc(tau: float) -> float:
    """Filter group delay at DC frequency. This is independent of filter order.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.

    Returns
    -------
    float
        The gain at DC frequency.
    """

    return tau


def cutoff_frequency(tau: float, filter_order: int) -> float:
    """
    The cutoff frequency /wc/ is the radial frequency wc where

        |H(wc)|^2 = 1/2 .

    That frequency is

               m
        wc = ----- sqrt( 2^(1/m) - 1 ) .
              tau

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        The order of the filter.

    Returns
    -------
    float
        The cutoff frequency.
    """

    # alias
    m = float(filter_order)

    # calculate the cutoff frequency
    wc = m / tau * np.sqrt(np.power(2.0, 1.0 / m) - 1.0)

    # return
    return wc


# noinspection PyUnusedLocal
def gain_at_cutoff(tau: float, filter_order: int) -> float:
    """
    The cutoff gain is always sqrt(1/2). The purpose of this function is
    to adhere to the phase- and group-delay-cutoff api so that any of the
    gain/phase/group-delay values @ cutoff can be determined uniformly.
    """

    return np.sqrt(1.0 / 2.0)


# noinspection PyUnusedLocal
def phase_at_cutoff(tau: float, filter_order: int) -> float:
    """
    The phase at cutoff frequency `wc`. The phase is independent of
    temporal scale `tau`, and the analytic expression is

        phase(wc) = -m arctan( sqrt( 2^(1/m) - 1 ) ) .

    Parameters
    ----------
    tau: float
        Unused but present for a consistent api
    filter_order: int
        The order of the filter.

    Returns
    -------
    float
        The phase at the cutoff frequency.
    """

    # calculate from analytic expression
    m = filter_order
    phase_wc = -m * np.arctan(np.sqrt(np.power(2.0, 1.0 / m) - 1.0))

    # return
    return phase_wc


def group_delay_at_cutoff(tau: float, filter_order: int) -> float:
    """
    The group delay at cutoff frequency /wc/. The group delay here is

        grp-delay(wc) = 2^(-1/m) tau .

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        The order of the filter.

    Returns
    -------
    float
        The group delay at cutoff.
    """

    # calculate the cutoff frequency
    grp_delay = np.power(2, -1.0 / filter_order) * tau

    # return
    return grp_delay


# noinspection SpellCheckingInspection
def wireframe(tau: float, filter_order: int):
    """Wireframe timepoint for this level filter.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        The order of the filter (unused)

    Returns
    -------
    WireframeContinuousTime
        Wireframe object
    """

    return design_tools.generate_level_wireframe_from_spectra(
        sys.modules[__name__], tau, filter_order
    )


# noinspection SpellCheckingInspection
def wavenumber(filter_order: int) -> float:
    """Report the wavenumber for this filter as a function of order.

    Parameters
    ----------
    filter_order: int
        The order of the filter.

    Returns
    -------
    float
        The wavenumber.
    """

    wavenumbers = {
        1: 0.64780859,
        2: 0.72623682,
        3: 0.74588861,
        4: 0.75626296,
        5: 0.76143075,
        6: 0.76197976,
        7: 0.75833512,
        8: 0.75110948,
    }

    return wavenumbers[filter_order]


# noinspection SpellCheckingInspection
def uncertainty_product(filter_order: int) -> dict:
    """Dict of TW, BW, and UCP for this filter order.

    Parameters
    ----------
    filter_order: int
        The order of the filter

    Returns
    -------
    dict
        TW is temporal width, BW is spectral width, and UCP is the
        uncertainty product.
    """

    uncertainty_products = {
        1: {"TW": 0.5, "BW": np.inf, "UCP": np.inf},
        2: {"TW": 0.433013, "BW": 2.0, "UCP": 0.866025},
        3: {"TW": 0.372678, "BW": 1.73205, "UCP": 0.645497},
        4: {"TW": 0.330719, "BW": 1.78885, "UCP": 0.591608},
        5: {"TW": 0.3, "BW": 1.88982, "UCP": 0.566947},
        6: {"TW": 0.276385, "BW": 2.0, "UCP": 0.552771},
        7: {"TW": 0.257539, "BW": 2.11058, "UCP": 0.543557},
        8: {"TW": 0.242061, "BW": 2.2188, "UCP": 0.537086},
    }

    return uncertainty_products[filter_order]
