"""
-------------------------------------------------------------------------------

Analog poly-ema composite-slope filter design:

Captures the design details for this filter. Design details are available
in Mathematica notebook polyema_composite_slope.nb. Wavenumbers are
calculated numerically in Python in `calculate_wavenumber.py`.

-------------------------------------------------------------------------------
"""

import numpy as np
import scipy.misc
import scipy.special
import sys

from irides.tools import design_tools


# noinspection SpellCheckingInspection
def design_id() -> str:
    """
    Since `design` modules are passed around as variables in this framework,
    a `design_id` gives a unique name.

    Returns
    -------
    str
        The id of this design.
    """

    return "polyema-composite-slope"


def design_type() -> design_tools.FilterDesignType:
    """This is a slope filter"""

    return design_tools.FilterDesignType.SLOPE


def get_minimum_valid_filter_order(strict: bool = False) -> int:
    """Returns minimum valid filter order."""

    return 1
    # return design_tools.get_minimum_valid_filter_order(design_type(), strict)


def get_valid_filter_orders(strict: bool = False) -> np.ndarray:
    """Returns array of valid filter orders based on strictness."""

    return np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # return design_tools.get_valid_filter_orders(design_type(), strict)


# noinspection SpellCheckingInspection
def convert_armratio_order_params_to_tp_tm_params(
    arm_ratio: float, tau: float, filter_order: int
) -> tuple:
    """Converts the (alpha, m) tuple to a (tau+, tau-) tuple, scaled by tau."""

    # alias
    a = arm_ratio
    m = filter_order

    # calculate
    coef = m / (m + 1.0)
    tp = 2.0 * a / (1.0 + a) * coef * tau
    tm = 2.0 / (1.0 + a) * coef * tau

    return tp, tm


# noinspection SpellCheckingInspection
def convert_tp_tm_params_to_armratio_order_params(
    tp: float, tm: float, tau: float
) -> tuple:
    """Functionally inverts convert_armratio_order_params_to_tp_tm_params()."""

    # calculate
    arm_ratio = tp / tm
    tau_avg = (tp + tm) / (2.0 * tau)
    filter_order = tau_avg / (1.0 - tau_avg)

    return arm_ratio, filter_order


# noinspection SpellCheckingInspection
def impulse_response_t0_value(
    tau: float, filter_order: int, dt: float, arm_ratio=0.5
) -> float:
    """Returns the h_m(t=0) value.

    The h(t=0) value goes as

                  (1+ar)^2
        h(t=0) =  -------- tau^{-2},   m = 1
                     ar

        h(t=0) = 0, m > 1

    where ar = arm ratio, which is nominally 1/2.

    Parameters
    ----------
    tau: float
        Temporal scale of the filter.
    filter_order: int
        The order of the filter.
    dt: float
        Temporal discretization
    arm_ratio: float
        Ratio of tau_+ / tau_- (default is 1/2).

    Returns
    -------
    float
        Theoretic value of h_m(t=0).
    """

    if filter_order == 1:
        ht0 = np.power(1.0 + arm_ratio, 2) / arm_ratio * np.power(tau, -2)
    else:
        ht0 = 0.0

    return ht0


def zero_crossing_time(tau: float, filter_order: int, arm_ratio=0.5) -> float:
    """The crossing time `tx` such that h(tx) = 0, tx > 0.

    Parameters
    ----------
    tau: float
        Temporal scale of the filter.
    filter_order: int
        The order of the filter.
    arm_ratio: float
        Ratio of tau_+ / tau_- (default is 1/2).

    Returns
    -------
    float
        Time tx where h_slope(tx) = 0.
    """

    # alias
    tx = (
        2.0
        * arm_ratio
        * np.log(1.0 / arm_ratio)
        / (1.0 - np.power(arm_ratio, 2))
        * filter_order
        / (filter_order + 1.0)
        * tau
    )

    return tx


# noinspection SpellCheckingInspection,PyUnusedLocal
def moment_value_generator(moment: int, tau: float = 1.0, dt: float = 0.0):
    """
    Returns an anonymous function with the single argument (order: int),
    where order is a `filter_order`. The returned function captures the `tau`
    and `dt` arguments in this calling function.

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

    # generator defs with `tau` and `dt` capture
    # noinspection PyUnusedLocal
    def m0_gen(order: int) -> float:
        if order == 1:
            return 0.0 + 9.0 / 4.0 * dt / np.power(tau, 2)
        else:
            return 0.0

    # noinspection PyUnusedLocal
    def m1_gen(order: int) -> float:
        return -1.0

    # noinspection PyUnusedLocal
    def m2_gen(order: int) -> float:
        return -2.0 * tau

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


# noinspection SpellCheckingInspection
def autocorrelation_peak_and_stride_values(
    tau: float, filter_order: int, dt=0.0
) -> dict:
    """
    System autocorrelation peak and stride values. Peak value is kh(0),
    and the stride comes from the trailing kh(xi_5pct) / kh(0) = -5%.

    Note: `arm_ratio` = 1/2 for these encoded values.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Order of the filter (note: user is required to validate that
        the order is supported).
    dt: float
        [Optional] Temporal increment dt in h(t).

    Returns
    -------
    dict
        Peak value (scaled by 1/tau^3) and 5% pct stride (scaled by tau)
    """

    # build unit_kh0 expr
    m = filter_order
    coef = (
        np.power(9.0, 2 - m)
        * (np.power(9.0, m) - np.power(8.0, m))
        / (32.0 * np.sqrt(np.pi))
    )
    unit_kh0 = (
        coef
        * m
        * scipy.special.gamma(m - 0.5)
        / scipy.special.gamma(m)
        * np.power((m + 1.0) / m, 3)
    )

    # values from Mathematica
    unit_sacf_values = {
        1: {"xi_5pct": 1.91744},
        2: {"xi_5pct": 2.13314},
        3: {"xi_5pct": 2.04505},
        4: {"xi_5pct": 1.94042},
        5: {"xi_5pct": 1.84766},
        6: {"xi_5pct": 1.76889},
        7: {"xi_5pct": 1.70218},
        8: {"xi_5pct": 1.64526},
    }

    # fetch values for order and scale
    sacf_values = unit_sacf_values[filter_order]
    sacf_values["kh0"] = unit_kh0 / np.power(tau, 3)
    sacf_values["kh0"] += autocorrelation_peak_value_discretization_correction(
        tau, filter_order, dt
    )
    sacf_values["xi_5pct"] *= tau

    # return
    return sacf_values


# noinspection SpellCheckingInspection
def autocorrelation_peak_value_discretization_correction(
    tau: float, filter_order: int, dt: float
) -> float:
    """
    Although this is a continuous-time context, the discretization of the
    system autocorrelation function requires a `dt`-order correction. This
    correction can be computed directly by taking the Maclaurin expansion
    of the summed correlogram in terms of dt.

    Parameters
    ----------
    tau: float
        Temporal scale of the filter.
    filter_order: int
        Order of the filter.
    dt: float
        Time-discretization step.

    Returns
    -------
    float
        Discretization correction.
    """

    if filter_order == 1:
        return 81.0 / 8.0 * dt / np.power(tau, 4)
    else:
        return 0.0


# noinspection SpellCheckingInspection,PyPep8Naming
def scale_correlation_stride_and_residual_values(
    tau: float, filter_order: int
) -> dict:
    """Scale decorrelation strides and residual correlations, `arm_ratio` = 1/2.

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
        1: {"sti_5pct": 29.5134, "residual": +0.05},
        2: {"sti_5pct": 44.9399, "residual": -0.05},
        3: {"sti_5pct": 21.3885, "residual": -0.05},
        4: {"sti_5pct": 14.0121, "residual": -0.05},
        5: {"sti_5pct": 10.6693, "residual": -0.05},
        6: {"sti_5pct": 8.80714, "residual": -0.05},
        7: {"sti_5pct": 7.63184, "residual": -0.05},
        8: {"sti_5pct": 6.82619, "residual": -0.05},
    }

    # fetch values for order and scale
    sscf_values = unit_sscf_values[filter_order]
    sscf_values["sti_5pct"] *= tau

    # return
    return sscf_values


# noinspection PyUnusedLocal
def gain_at_dc(tau: float) -> float:
    """Filter gain at DC frequency. This is independent of filter order.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.

    Returns
    -------
    float:
        The gain at DC frequency.
    """

    return 0.0


# noinspection PyUnusedLocal
def phase_at_dc(tau: float) -> float:
    """Filter phase at DC frequency. This is independent of filter order.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.

    Returns
    -------
    float:
        The gain at DC frequency.
    """

    return np.pi / 2


def group_delay_at_dc(tau: float) -> float:
    """Filter group delay at DC frequency. This is independent of filter order.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.

    Returns
    -------
    float:
        The gain at DC frequency.
    """

    return tau


# noinspection SpellCheckingInspection
def frequency_of_peak_gain(tau: float, filter_order: int) -> float:
    """Returns the radial frequency at which the gain peaks (arm-ratio=1/2)."""

    # values calculated from Mathematica
    unit_frequency_of_peak_gain = {
        1: 2.12132,
        2: 2.17439,
        3: 2.30366,
        4: 2.43475,
        5: 2.55634,
        6: 2.66689,
        7: 2.76701,
        8: 2.85781,
    }

    return unit_frequency_of_peak_gain[filter_order] / tau


# noinspection SpellCheckingInspection
def gain_at_peak_frequency(tau: float, filter_order: int) -> float:
    """Returns gain(w_peak), the peak gain (arm-ratio=1/2)."""

    # values calculated from Mathematica
    unit_gain_at_peak_frequency = {
        1: 1.00000,
        2: 1.12577,
        3: 1.23961,
        4: 1.33919,
        5: 1.42674,
        6: 1.50441,
        7: 1.57393,
        8: 1.63663,
    }

    return unit_gain_at_peak_frequency[filter_order] / tau


# noinspection SpellCheckingInspection
def phase_at_peak_frequency(tau: float, filter_order: int) -> float:
    """Returns phase(w_peak) (arm-ratio=1/2)."""

    # values calculated from Mathematica
    phase_at_peak_frequency_values = {
        1: 0.00000,
        2: -0.23888,
        3: -0.44046,
        4: -0.61179,
        5: -0.75981,
        6: -0.88945,
        7: -1.00429,
        8: -1.10693,
    }

    return phase_at_peak_frequency_values[filter_order]


# noinspection SpellCheckingInspection
def group_delay_at_peak_frequency(tau: float, filter_order: int) -> float:
    """Returns group-delay(w_peak) (arm-ratio=1/2)."""

    # values calculated from Mathematica
    unit_group_delay_at_peak_frequency = {
        1: 0.44444,
        2: 0.60142,
        3: 0.68240,
        4: 0.73221,
        5: 0.76620,
        6: 0.79104,
        7: 0.81009,
        8: 0.82526,
    }

    return unit_group_delay_at_peak_frequency[filter_order] * tau


# noinspection SpellCheckingInspection,PyPep8Naming
def wireframe(tau: float, filter_order: int):
    """Wireframe-timepoint-pair for this slope filter. (arm-ratio = 1/2)

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        The order of the filter.

    Returns
    -------
    WireframeContinuousTime
        Wireframe object
    """

    return design_tools.generate_slope_wireframe_from_spectra(
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
        1: 0.55424844,
        2: 0.68040373,
        3: 0.70865678,
        4: 0.71848684,
        5: 0.72331669,
        6: 0.72664995,
        7: 0.72956863,
        8: 0.73242194,
    }

    return wavenumbers[filter_order]


# noinspection SpellCheckingInspection
def uncertainty_product(filter_order: int) -> dict:
    """Dict of TW, BW, and UCP for this filter order. (arm-ratio = 1/2)

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
        1: {"TW": 0.40062, "BW": np.inf, "UCP": np.inf},
        2: {"TW": 0.47192, "BW": 5.85204, "UCP": 2.76168},
        3: {"TW": 0.46611, "BW": 4.42917, "UCP": 2.06448},
        4: {"TW": 0.45002, "BW": 4.21715, "UCP": 1.89782},
        5: {"TW": 0.43368, "BW": 4.20664, "UCP": 1.82433},
        6: {"TW": 0.41906, "BW": 4.25805, "UCP": 1.78436},
        7: {"TW": 0.40641, "BW": 4.33153, "UCP": 1.76036},
        8: {"TW": 0.39555, "BW": 4.41225, "UCP": 1.74528},
    }

    return uncertainty_products[filter_order]
