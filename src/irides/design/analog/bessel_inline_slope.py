"""
-------------------------------------------------------------------------------

Analog Bessel inline slope filter design:

Captures the design details for this filter. Design details are available
in Mathematica notebook bessel_inline_slope.nb. Wavenumbers are
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

    return "bessel-inline-slope"


def design_type() -> design_tools.FilterDesignType:
    """This is a slope filter"""

    return design_tools.FilterDesignType.SLOPE


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
    tau: float
        Temporal scale of the filter.
    filter_order: int
        The order of the filter.
    dt: float
        Temporal discretization

    Returns
    -------
    float
        Theoretic value of h_m(t=0).
    """

    # values calculated from Mathematica
    unit_t0_values = np.array(
        [
            3.0,  # m = 2
            0.0,  # m = 3
            0.0,  # m = 4
            0.0,  # m = 5
            0.0,  # m = 6
            0.0,  # m = 7
            0.0,  # m = 8
        ]
    )

    # return
    index = filter_order - get_minimum_valid_filter_order()
    return unit_t0_values[index] / np.power(tau, 2)


def zero_crossing_time(tau: float, filter_order: int) -> float:
    r"""
    The zero-crossing time `tx` is when h_slope(tx) = 0. This is
    the same time when h_ref(tx) reaches its maximum.

      h_slope(t)
          ^       .
          |    ... ...
             ..       .
          ---|----|----0----|----------------------------> t
             0          .       ....
                         ... ...
                       ^    .
                       |
                      tx

    Only orders [2..8] are provided.

    Parameters
    ----------
    tau: float
        Temporal scale of the filter.
    filter_order: int
        The order of the filter.

    Returns
    -------
    float
        Time tx where h_slope(tx) = 0.
    """

    # values calculated from Mathematica
    unit_crossing_times = np.array(
        [
            0.6045997881,  # m = 2
            0.8134998949,  # m = 3
            0.9005481030,  # m = 4
            0.9425714677,  # m = 5
            0.9649848592,  # m = 6
            0.9778024596,  # m = 7
            0.9855139594,  # m = 8
        ]
    )

    # return
    index = filter_order - get_minimum_valid_filter_order()
    return unit_crossing_times[index] * tau


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
    # noinspection SpellCheckingInspection,PyPep8Naming,PyUnusedLocal
    def m0_gen(order: int) -> float:
        """Covers order [2..8] only."""
        if order == 2:
            """see Mathematica bessel_inline_slope.nb"""
            design_config = design_reference.designs(order)
            p = design_config["poles"][0]
            A = design_config["residues"][0]
            coef = np.real(A.real * p.real - A.imag * p.imag)
            return 0.0 + coef * dt / np.power(tau, 2)
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


# noinspection SpellCheckingInspection, PyUnusedLocal
def autocorrelation_peak_and_stride_values(
    tau: float, filter_order: int, dt=0.0
) -> dict:
    """
    System autocorrelation peak and stride values. Peak value is kh(0),
    and the stride comes from the trailing kh(xi_5pct) / kh(0) = -5%.

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

    # values from Mathematica

    unit_sacf_values = {
        2: {"kh0": 1.49468, "xi_5pct": 2.46176},
        3: {"kh0": 1.49843, "xi_5pct": 2.29374},
        4: {"kh0": 2.13784, "xi_5pct": 1.93200},
        5: {"kh0": 3.05729, "xi_5pct": 1.65981},
        6: {"kh0": 4.19589, "xi_5pct": 1.46236},
        7: {"kh0": 5.48266, "xi_5pct": 1.32120},
        8: {"kh0": 6.94603, "xi_5pct": 1.21256},
    }

    # fetch values for order and scale
    sacf_values = unit_sacf_values[filter_order]
    sacf_values["kh0"] /= np.power(tau, 3)
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
        2: {"sti_5pct": 5.93470, "residual": +0.05000},
        3: {"sti_5pct": 4.80664, "residual": -0.03718},
        4: {"sti_5pct": 4.84400, "residual": -0.05000},
        5: {"sti_5pct": 4.28553, "residual": -0.05000},
        6: {"sti_5pct": 3.79060, "residual": -0.05000},
        7: {"sti_5pct": 3.41999, "residual": -0.05000},
        8: {"sti_5pct": 3.14215, "residual": -0.05000},
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

    return design_reference.group_delay_at_dc(tau)


# noinspection SpellCheckingInspection
def frequency_of_peak_gain(tau: float, filter_order: int) -> float:
    """Returns the radial frequency at which the gain peaks."""

    # values calculated from Mathematica
    unit_frequency_of_peak_gain = np.array(
        [
            1.732050808,  # m = 2
            2.001732855,  # m = 3
            2.378885838,  # m = 4
            2.757342866,  # m = 5
            3.108535773,  # m = 6
            3.426226098,  # m = 7
            3.714335374,  # m = 8
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
            1.000000000,  # m = 2
            1.262110453,  # m = 3
            1.517338302,  # m = 4
            1.746638378,  # m = 5
            1.950576314,  # m = 6
            2.134070905,  # m = 7
            2.301949193,  # m = 8
        ]
    )

    # return
    index = filter_order - get_minimum_valid_filter_order()
    return unit_gain_at_peak_frequency[index] / tau


# noinspection SpellCheckingInspection,PyUnusedLocal
def phase_at_peak_frequency(tau: float, filter_order: int) -> float:
    """Returns angle H_delta(s) at the peak frequency."""

    # values calculated from Mathematica
    phase_at_peak_frequency_values = np.array(
        [
            0.00000000000,  # m = 2
            -0.3898549443,  # m = 3
            -0.7960272586,  # m = 4
            -1.1831948640,  # m = 5
            -1.5369255350,  # m = 6
            -1.8552614240,  # m = 7
            -2.1435091590,  # m = 8
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
            0.6666666667,  # m = 2
            0.8863324030,  # m = 3
            0.9621533147,  # m = 4
            0.9885851933,  # m = 5
            0.9970335598,  # m = 6
            0.9993473592,  # m = 7
            0.9998773101,  # m = 8
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
        2: 0.60892898,
        3: 0.77474505,
        4: 0.81632936,
        5: 0.83570741,
        6: 0.84774214,
        7: 0.85137154,
        8: 0.84539188,
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
        2: {"TW": 0.527046, "BW": np.inf, "UCP": np.inf},
        3: {"TW": 0.58515, "BW": 3.87298, "UCP": 2.26627},
        4: {"TW": 0.532289, "BW": 3.24037, "UCP": 1.72481},
        5: {"TW": 0.475521, "BW": 3.38446, "UCP": 1.60938},
        6: {"TW": 0.427613, "BW": 3.6588, "UCP": 1.56455},
        7: {"TW": 0.388683, "BW": 3.96906, "UCP": 1.5427},
        8: {"TW": 0.357044, "BW": 4.28644, "UCP": 1.53045},
    }

    return uncertainty_products[filter_order]
