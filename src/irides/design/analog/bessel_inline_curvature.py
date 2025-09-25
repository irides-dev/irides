"""
-------------------------------------------------------------------------------

Analog bessel inline curvature filter design:

Captures the design details for this filter. Design details are available
in Mathematica notebook bessel_inline_curvature.nb. Wavenumbers are
calculated numerically in Python in `calculate_wavenumber.py`.

-------------------------------------------------------------------------------
"""

import numpy as np
import sys

from irides.design.analog import bessel_level as design_reference
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

    return "bessel-inline-curvature"


def design_type() -> design_tools.FilterDesignType:
    """This is a curvature filter"""

    return design_tools.FilterDesignType.CURVE


def get_minimum_valid_filter_order(strict: bool = False) -> int:
    """Returns minimum valid filter order."""

    return design_tools.get_minimum_valid_filter_order(design_type(), strict)


def get_valid_filter_orders(strict: bool = False) -> np.ndarray:
    """Returns array of valid filter orders based on strictness."""

    return design_tools.get_valid_filter_orders(design_type(), strict)


# noinspection SpellCheckingInspection,PyUnusedLocal
def impulse_response_t0_value(
    tau: float, filter_order: int, dt: float
) -> float:
    """
    Returns the h_m(t=0) value for inline slope filters of this type. Only
    orders [2..8] are provided.

    Parameters
    ----------
    dt: float
        Temporal discretization
    tau: float
        Temporal scale of the filter.
    filter_order: int
        The order of the filter.

    Returns
    -------
    float
        Theoretic value of h_m(t=0).
    """

    # values calculated from Mathematica
    unit_t0_values = np.array(
        [
            15.0,  # m = 3
            0.0,  # m = 4
            0.0,  # m = 5
            0.0,  # m = 6
            0.0,  # m = 7
            0.0,  # m = 8
        ]
    )

    # return
    index = filter_order - get_minimum_valid_filter_order()
    return unit_t0_values[index] / np.power(tau, 3)


def impulse_response_lobe_details(tau: float, filter_order: int) -> dict:
    """
    Curvature filters have two principal crossing times, and the two positive
    lobes may capture equal or unequal area beneath.

      h_curv(t)
        ^     .                  ..
        |   .. ..             ...  ....
           .     .           .         .....
        ---|--|---0---------0-----------------------> t
           0       .       .
                    .     .
                     .. ..
                  ^    .    ^
                  |         |
                 tx-1      tx-2

    Parameters
    ----------
    tau: float
        Temporal scale of the filter.
    filter_order: int
        The order of the filter.

    Returns
    -------
    dict
    """

    lobe_details = {
        3: {"tx1": 0.267707, "tx2": 1.36620, "imbalance": 2.17617},
        4: {"tx1": 0.428704, "tx2": 1.38202, "imbalance": 1.62999},
        5: {"tx1": 0.529771, "tx2": 1.36538, "imbalance": 1.39585},
        6: {"tx1": 0.597575, "tx2": 1.34178, "imbalance": 1.26717},
        7: {"tx1": 0.645709, "tx2": 1.31829, "imbalance": 1.18758},
        8: {"tx1": 0.681470, "tx2": 1.29685, "imbalance": 1.13492},
    }

    ans_dict = lobe_details[filter_order]
    ans_dict["tx1"] *= tau
    ans_dict["tx2"] *= tau

    return ans_dict


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
        Temporal scale of the filter (unused)
    dt: float
        Time interval for discretization.

    Returns
    -------
    Anonymous function with argument (order: int).
    """

    # generator defs with `tau` and `dt` capture
    # noinspection PyUnusedLocal
    def m0_gen(order: int) -> float:
        """Covers order [3..8] only."""
        return 0.0

    # noinspection PyUnusedLocal
    def m1_gen(order: int) -> float:
        return 0.0

    # noinspection PyUnusedLocal
    def m2_gen(order: int) -> float:
        return 2.0

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


# noinspection SpellCheckingInspection, PyUnusedLocal
def autocorrelation_peak_and_stride_values(
    tau: float, filter_order: int, dt=0.0
) -> dict:
    """
    System autocorrelation peak and stride values. Peak value is kh(0),
    and the stride comes from the trailing kh(xi_5pct) / kh(0) = 5%.


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
        Peak value (scaled by 1/tau^5) and 5% pct stride (scaled by tau)
    """

    # values from Mathematica
    unit_sacf_values = {
        3: {"kh0": 22.5000, "xi_5pct": 2.14021, "residual_acf": 2.27118},
        4: {"kh0": 22.5000, "xi_5pct": 2.16504, "residual_acf": 5.0},
        5: {"kh0": 34.9999, "xi_5pct": 1.99730, "residual_acf": 5.0},
        6: {"kh0": 55.9711, "xi_5pct": 1.78948, "residual_acf": 5.0},
        7: {"kh0": 86.4564, "xi_5pct": 1.61198, "residual_acf": 5.0},
        8: {"kh0": 127.745, "xi_5pct": 1.46765, "residual_acf": 5.0},
    }

    # fetch values for order and scale
    sacf_values = unit_sacf_values[filter_order]
    sacf_values["kh0"] /= np.power(tau, 5)
    sacf_values["xi_5pct"] *= tau

    # return
    return sacf_values


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
        3: {"sti_5pct": 2.85810, "residual": +0.05000},
        4: {"sti_5pct": 2.89022, "residual": -0.02701},
        5: {"sti_5pct": 2.73510, "residual": -0.05000},
        6: {"sti_5pct": 2.65474, "residual": -0.05000},
        7: {"sti_5pct": 2.49171, "residual": -0.05000},
        8: {"sti_5pct": 2.34451, "residual": -0.05000},
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

    return np.pi


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

    return design_reference.group_delay_at_dc(tau)


# noinspection SpellCheckingInspection
def frequency_of_peak_gain(tau: float, filter_order: int) -> float:
    """Returns the radial frequency at which the gain peaks."""

    # values calculated from Mathematica
    unit_frequency_of_peak_gain = np.array(
        [
            3.096475897,  # m = 3
            3.276767292,  # m = 4
            3.675631324,  # m = 5
            4.121800308,  # m = 6
            4.568617084,  # m = 7
            4.996291251,  # m = 8
        ]
    )

    # return
    index = filter_order - get_minimum_valid_filter_order()
    return unit_frequency_of_peak_gain[index] / tau


# noinspection SpellCheckingInspection
def gain_at_peak_frequency(tau: float, filter_order: int) -> float:
    """Returns gain(w_peak), the peak gain."""

    # values calculated from Mathematica
    unit_gain_at_peak_frequency = np.array(
        [
            3.146314744,  # m = 3
            4.286691821,  # m = 4
            5.647797444,  # m = 5
            7.109325059,  # m = 6
            8.610656060,  # m = 7
            10.12018502,  # m = 8
        ]
    )

    # return
    index = filter_order - get_minimum_valid_filter_order()
    return unit_gain_at_peak_frequency[index] / np.power(tau, 2)


# noinspection SpellCheckingInspection,PyUnusedLocal
def phase_at_peak_frequency(tau: float, filter_order: int) -> float:
    """Returns angle H_delta(s) at the peak frequency."""

    # values calculated from Mathematica
    phase_at_peak_frequency_values = np.array(
        [
            0.3753478245,  # m = 3
            -0.0295578201,  # m = 4
            -0.4929680867,  # m = 5
            -0.9637886427,  # m = 6
            -1.4207345620,  # m = 7
            -1.8524816260,  # m = 8
        ]
    )

    # return
    index = filter_order - get_minimum_valid_filter_order()
    return phase_at_peak_frequency_values[index]


# noinspection SpellCheckingInspection
def group_delay_at_peak_frequency(tau: float, filter_order: int) -> float:
    """Returns the group delay at the peak frequency."""

    # values calculated from Mathematica
    unit_group_delay_at_peak_frequency = np.array(
        [
            0.5781508094,  # m = 3
            0.8078464881,  # m = 4
            0.9119181817,  # m = 5
            0.9610324290,  # m = 6
            0.9839163584,  # m = 7
            0.9939684121,  # m = 8
        ]
    )

    # return
    index = filter_order - get_minimum_valid_filter_order()
    return unit_group_delay_at_peak_frequency[index] * tau


# noinspection SpellCheckingInspection,PyPep8Naming
def wireframe(tau: float, filter_order: int):
    """Wireframe-timepoint-pair for this slope filter.

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

    return design_tools.generate_curvature_wireframe_from_spectra(
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
        3: 0.56484587,
        4: 0.75055984,
        5: 0.79969197,
        6: 0.82082189,
        7: 0.83757011,
        8: 0.84829495,
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
        3: {"TW": 0.386092, "BW": np.inf, "UCP": np.inf},
        4: {"TW": 0.462026, "BW": 5.91608, "UCP": 2.73338},
        5: {"TW": 0.44201, "BW": 4.74342, "UCP": 2.09664},
        6: {"TW": 0.407745, "BW": 4.79826, "UCP": 1.95647},
        7: {"TW": 0.374184, "BW": 5.0704, "UCP": 1.89726},
        8: {"TW": 0.344446, "BW": 5.41243, "UCP": 1.86429},
    }

    return uncertainty_products[filter_order]
