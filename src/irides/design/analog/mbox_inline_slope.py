"""
-------------------------------------------------------------------------------

Analog multistage-box inline slope filter design:

Captures the design details for this filter. Design details are available
in Mathematica notebook mbox_inline_slope.nb. Wavenumbers are
calculated numerically in Python in `calculate_wavenumber.py`.

-------------------------------------------------------------------------------
"""

import numpy as np
import sys

from irides.design.analog import mbox_level as design_reference
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

    return "mbox-inline-slope"


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
            1.0,  # m = 2
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
            1.0,  # m = 2
            1.0,  # m = 3
            1.0,  # m = 4
            1.0,  # m = 5
            1.0,  # m = 6
            1.0,  # m = 7
            1.0,  # m = 8
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

    # generator defs with `tau` and `dt` capture
    # noinspection PyUnusedLocal
    def m0_gen(order: int) -> float:
        """Covers order [2..8] only."""
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
        2: {"kh0": 2.0, "xi_5pct": 1.9000000000},
        3: {"kh0": 27.0 / 8.0, "xi_5pct": 1.5537113670},
        4: {"kh0": 16.0 / 3.0, "xi_5pct": 1.3393447220},
        5: {"kh0": 4375.0 / 576.0, "xi_5pct": 1.1953057840},
        6: {"kh0": 809.0 / 80.0, "xi_5pct": 1.0894011890},
        7: {"kh0": 1481417.0 / 115200.0, "xi_5pct": 1.0073948210},
        8: {"kh0": 224228.0 / 14175.0, "xi_5pct": 0.9414763172},
    }

    # fetch values for order and scale
    sacf_values = unit_sacf_values[filter_order]
    if filter_order == 2:
        sacf_values["kh0"] -= 2.0 * dt
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
        2: {"sti_5pct": 1.87194, "residual": +0.05000},
        3: {"sti_5pct": 3.31446, "residual": -0.05000},
        4: {"sti_5pct": 3.21686, "residual": -0.05000},
        5: {"sti_5pct": 2.95635, "residual": -0.05000},
        6: {"sti_5pct": 2.73096, "residual": -0.05000},
        7: {"sti_5pct": 2.55439, "residual": -0.05000},
        8: {"sti_5pct": 2.41530, "residual": -0.05000},
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
            2.331122356,  # m = 2
            2.902207933,  # m = 3
            3.378923335,  # m = 4
            3.796538379,  # m = 5
            4.172699669,  # m = 6
            4.517719693,  # m = 7
            4.838240217,  # m = 8
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
            1.449222708,  # m = 2
            1.789626825,  # m = 3
            2.075053083,  # m = 4
            2.325758038,  # m = 5
            2.551973194,  # m = 6
            2.759718734,  # m = 7
            2.952891372,  # m = 8
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
            -0.760326029,  # m = 2
            -1.331411606,  # m = 3
            -1.808127008,  # m = 4
            -2.225742052,  # m = 5
            -2.601903342,  # m = 6
            -2.946923366,  # m = 7
            -3.267443890,  # m = 8
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
            1.0,  # m = 2
            1.0,  # m = 3
            1.0,  # m = 4
            1.0,  # m = 5
            1.0,  # m = 6
            1.0,  # m = 7
            1.0,  # m = 8
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
        2: 0.80573771,
        3: 0.83797329,
        4: 0.83308778,
        5: 0.82091467,
        6: 0.79784648,
        7: 0.76948133,
        8: 0.74256534,
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
        2: {"TW": 0.57735, "BW": 741.036, "UCP": 427.838},
        3: {"TW": 0.447214, "BW": 3.67423, "UCP": 1.64317},
        4: {"TW": 0.377964, "BW": 4.0, "UCP": 1.51186},
        5: {"TW": 0.333333, "BW": 4.51189, "UCP": 1.50396},
        6: {"TW": 0.301511, "BW": 4.98304, "UCP": 1.50244},
        7: {"TW": 0.27735, "BW": 5.4146, "UCP": 1.50174},
        8: {"TW": 0.258199, "BW": 5.81455, "UCP": 1.50131},
    }

    return uncertainty_products[filter_order]
