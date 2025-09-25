"""
-------------------------------------------------------------------------------

Analog multistage-box level filter design:

Captures temporal, spectral and autocorrelation traits of this filter.

Wavenumbers are calculated numerically in Python in `calculate_wavenumber.py`.

-------------------------------------------------------------------------------
"""

import numpy as np
import sys

from irides.resources.containers import points
from irides.tools import design_tools


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

    return "mbox-level"


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


# noinspection SpellCheckingInspection,PyPep8Naming
def convert_tau_to_T(tau: float) -> float:
    """T = 2 tau.

    `tau` is a temporal scale, `T` is a parameter of the m-box filter.
    The first moment of an m-box level filter is M1 = tau = T / 2.
    Thus: T = 2 x tau.
    """

    return 2.0 * tau


# noinspection SpellCheckingInspection,PyPep8Naming
def convert_T_to_tau(T: float) -> float:
    """tau = T / 2.

    For an m-box level filter, M1 = tau = T / 2.
    Thus: tau = T / 2.
    """

    return T / 2.0


"""
-------------------------------------------------------------------------------

Standard features

-------------------------------------------------------------------------------
"""


# noinspection PyUnusedLocal
def unit_peak_time(filter_order: int, dt=0.0) -> float:
    """
    Unit in this case refers to the first moment = 1, so T = 2.
    Thus, the unit-peak time is 1.

    Parameters
    ----------
    filter_order: int
        Order of the filter.
    dt: float
        Grid step of continuous-time discretization (default=0.0).

    Returns
    -------
    float
        The unit peak time of a multistage box level filter.
    """

    # invariant of the design
    t_peak = 1.0 - dt

    return t_peak


# noinspection SpellCheckingInspection
def unit_peak_value(filter_order: int, dt=0.0) -> float:
    """
    The value of impulse response h(t_peak; T = 2). While perhaps there is an
    analytic expression for this peak value, here the results from Mathematica
    are tabulated.

    Only order 1-8 are available.

    Parameters
    ----------
    filter_order: int
        Order of the filter.
    dt: float
        Grid step of continuous-time discretization (default=0.0).

    Returns
    -------
    float
        The unit peak value of the impulse response.
    """

    # unit peak values (from Mathematica mbox_level.nb,
    # with discrete corrections from python)
    unit_peak_values = {
        1: 1.0 / 2.0,
        2: 1.0 * (1.0 - dt),
        3: 9.0 / 8.0,
        4: 4.0 / 3.0,
        5: 575.0 / 384.0,
        6: 33.0 / 20.0,
        7: 41209.0 / 23040.0,
        8: 604.0 / 315.0,
    }

    # check and return
    valid_filter_orders = get_valid_filter_orders()
    if filter_order in valid_filter_orders:
        return unit_peak_values[filter_order]
    else:
        return 0.0


# noinspection SpellCheckingInspection,PyPep8Naming
def peak_value_coordinate(tau: float, filter_order: int, dt=0.0):
    """
    Bundles the peak time and value, scaled by T, into a point container.

    Parameters
    ----------
    tau: float
        First moment of the filter.
    filter_order: int
        Order of the filter.
    dt: float
        Grid step of continuous-time discretization (default=0.0).

    Returns
    -------
    ContinuousTimePoint
        Object with .time and .value fields.
    """

    # construct
    T = convert_tau_to_T(tau)
    T_cal = 2.0
    peak_coord = points.ContinuousTimePoint(
        unit_peak_time(filter_order, dt) * (T / T_cal),
        unit_peak_value(filter_order, dt) / (T / T_cal),
    )

    # return
    return peak_coord


# noinspection SpellCheckingInspection,PyPep8Naming,PyUnusedLocal
def moment_value_generator(moment: int, tau: float = 1.0, dt: float = 0.0):
    """
    The unit moments here are for T/2 = 1, so that M1 = 1.

    Returns an anonymous function with the single argument (order: int),
    where order is a `filter_order`. The returned function captures the `tau`
    and `dt` arguments in this calling function.

    Note that the generated functions are not necy purely continuous time.
    Instead, they bridge continuous and discrete time by applying order-dt
    corrections to the moments. This is required because the CT series are
    in fact implemented in DT. Use dt=0.0 to recover pure CT moments.

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
    def m0_gen(order: int):
        return 1.0

    # noinspection PyUnusedLocal
    def m1_gen(order: int):
        return tau - dt / 2.0

    def m2_gen(order: int):
        return ((3.0 * order + 1.0) / (3.0 * order)) * np.power(
            tau, 2
        ) - dt / 2.0

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
    The S-ACF parameters are all numerically determined. Here, the sacf(lag=0)
    and sacf(lag=xi_5pct) = 0.05 values are available for T = 2.

    The peak value of the S-ACF, sacf(lag=0), scales as 1 / T. The
    5% stride value xi_5pct, scales as T.

    Parameters
    ----------
    tau: float
        The first moment of the filter.
    filter_order: int
        Order of the filter.

    Returns
    -------
    dict
        Peak value (scaled by 1/T) and 5% pct stride (scaled by T)
    """

    # convert 1st moment to T
    T_input = convert_tau_to_T(tau)

    # values for T = 2
    T_cal = 2.0
    unit_sacf_values = {
        1: {"kh0": 1.0 / 2.0, "xi_5pct": 1.90},
        2: {"kh0": 2.0 / 3.0, "xi_5pct": 1.4152},
        3: {"kh0": 33.0 / 40.0, "xi_5pct": 1.15309},
        4: {"kh0": 302.0 / 315.0, "xi_5pct": 0.998982},
        5: {"kh0": 78095.0 / 72576.0, "xi_5pct": 0.893614},
        6: {"kh0": 655177.0 / 554400.0, "xi_5pct": 0.815806},
        7: {"kh0": 189597667.0 / 148262400.0, "xi_5pct": 0.755319},
        8: {"kh0": 2330931341.0 / 1702701000.0, "xi_5pct": 0.706554},
    }
    zero_values = {"kh0": 0.0, "xi_5pct": 0.0}

    # check and return
    if filter_order in unit_sacf_values.keys():
        ans = unit_sacf_values[filter_order]
        ans["kh0"] *= T_cal / T_input
        ans["xi_5pct"] *= T_input / T_cal
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
        1: {"sti_5pct": 400.000, "residual": 0.05},
        2: {"sti_5pct": 9.65489, "residual": 0.05},
        3: {"sti_5pct": 4.60289, "residual": 0.05},
        4: {"sti_5pct": 3.36003, "residual": 0.05},
        5: {"sti_5pct": 2.81947, "residual": 0.05},
        6: {"sti_5pct": 2.51131, "residual": 0.05},
        7: {"sti_5pct": 2.30873, "residual": 0.05},
        8: {"sti_5pct": 2.16416, "residual": 0.05},
    }

    # fetch values for order and scale
    sscf_values = unit_sscf_values[filter_order]
    sscf_values["sti_5pct"] *= tau

    # return
    return sscf_values


# noinspection SpellCheckingInspection,PyPep8Naming
def full_width_generator(tau: float = 1.0, dt: float = 0.0):
    """
    Returns anonymous function for the full-width, given that T/2 = 1.
    `tau` and `dt` are captured here, and the anonymous function has one
    argument, (order: int), the filter order.

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
        return 2.0 * np.sqrt(1.0 / (3.0 * order) - np.power(dt / tau, 2)) * tau

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


# noinspection SpellCheckingInspection,PyPep8Naming
def cutoff_frequency(tau: float, filter_order: int) -> float:
    """
    At w_cutoff,

        |Hmbox(w_cutoff; T, m)|^2 = 1/2.

    The solution is not analytic so the values for order 1-8 are reported here
    for T_cal = 2.

    The scaling is clear when written as this:

        g^2(T_cal wc_cal) = g^2(T wc') = 1/2,

    so,

        wc' = wc_cal Tcal / T.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Order of the filter  (note: user is required to validate that
        the order is supported).

    Returns
    -------
    float
        Frequency at cutoff (radians).
    """

    # convert
    T = convert_tau_to_T(tau)

    # the calibration temporal scale
    T_cal = 2.0

    # cutoff freqs (from Mathematica mbox_level.nb, T=2)
    cutoff_frequencies = {
        1: 1.39,
        2: 2.00,
        3: 2.47,
        4: 2.86,
        5: 3.20,
        6: 3.51,
        7: 3.80,
        8: 4.06,
    }

    # check and return
    return cutoff_frequencies[filter_order] * T_cal / T


# noinspection PyUnusedLocal
def gain_at_cutoff(tau: float, filter_order: int) -> float:
    """
    The cutoff gain is always sqrt(1/2). The purpose of this function is
    to adhere to the phase- and group-delay-cutoff api so that any of the
    gain/phase/group-delay values @ cutoff can be determined uniformly.
    """

    return np.sqrt(1.0 / 2.0)


# noinspection SpellCheckingInspection,PyPep8Naming,PyUnusedLocal
def phase_at_cutoff(tau: float, filter_order: int) -> float:
    """
    The phase at cutoff frequency `wc`. Phase at cutoff is independent
    of temporal scale `tau`, and this being a multistage box filter,
    the phase is also independent of filter order.

    Parameters
    ----------
    tau: float
        Unused but present for a consistent api
    filter_order: int
        The order of the filter (unused).

    Returns
    -------
    float
        The phase at the cutoff frequency.
    """

    T_input = convert_tau_to_T(tau)
    T_cal = 2.0
    tau_cal = convert_T_to_tau(T_cal)
    w_cutoff = cutoff_frequency(tau_cal, filter_order)

    return -w_cutoff * T_input / T_cal


# noinspection SpellCheckingInspection,PyPep8Naming,PyUnusedLocal
def group_delay_at_cutoff(tau: float, filter_order: int) -> float:
    """
    The group delay for a multistage box filter is T/2 independent of the
    filter order.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Order of the filter.

    Returns
    -------
    float
    """

    # convert
    T = convert_tau_to_T(tau)

    return T / 2.0


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
        1: 0.76907485,
        2: 0.79476859,
        3: 0.76418367,
        4: 0.72115753,
        5: 0.68203083,
        6: 0.65227649,
        7: 0.63221627,
        8: 0.62143716,
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
        1: {"TW": 0.57735, "BW": 427.856, "UCP": 247.023},
        2: {"TW": 0.316228, "BW": 1.73205, "UCP": 0.547723},
        3: {"TW": 0.249096, "BW": 2.0226, "UCP": 0.503822},
        4: {"TW": 0.212513, "BW": 2.35858, "UCP": 0.50123},
        5: {"TW": 0.18846, "BW": 2.65682, "UCP": 0.500706},
        6: {"TW": 0.171088, "BW": 2.92524, "UCP": 0.500474},
        7: {"TW": 0.157781, "BW": 3.17111, "UCP": 0.500342},
        8: {"TW": 0.147166, "BW": 3.39929, "UCP": 0.500258},
    }

    return uncertainty_products[filter_order]


"""
-------------------------------------------------------------------------------

Filter-specific features

-------------------------------------------------------------------------------
"""


# noinspection SpellCheckingInspection,PyPep8Naming
def first_lobe_frequency_and_gain_values(
    tau: float, filter_order: int
) -> tuple:
    """
    Returns the radial frequency w for the gain peak of the first
    gain lobe, along with the gain value.

    Parameters
    ----------
    tau: float
        The first moment of the filter.
    filter_order: int
        Order of the filter.

    Returns
    -------
    tuple
        (w_1st_lobe, gain_1st_lobe)
        w_1st_lobe: Radial frequency for gain peak of first gain lobe
        gain_1st_lobe: Gain at w_1st_lobe
    """

    T = convert_tau_to_T(tau)
    w_1st_lobe = 3.0 * np.pi / (T / filter_order)
    gain_1st_lobe = np.power(2.0 / (3.0 * np.pi), filter_order)

    return w_1st_lobe, gain_1st_lobe
