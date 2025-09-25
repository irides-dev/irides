"""
-------------------------------------------------------------------------------

Analog polyema inline curvature filter design:

Captures the design details for this filter. Design details are available
in Mathematica notebook polyema_inline_curvature.nb. Wavenumbers are
calculated numerically in Python in `calculate_wavenumber.py`.

-------------------------------------------------------------------------------
"""

import numpy as np
import sys

from irides.design.analog import polyema_level as design_reference
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

    return "pema-inline-curvature"


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
    Returns the h_m(t=0) value for inline slope filters of this type.

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
            27.0,  # m = 3
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
        3: {"tx1": 0.195262, "tx2": 1.13807, "imbalance": 2.90281},
        4: {"tx1": 0.316987, "tx2": 1.18301, "imbalance": 2.29374},
        5: {"tx1": 0.400000, "tx2": 1.20000, "imbalance": 2.02215},
        6: {"tx1": 0.460655, "tx2": 1.20601, "imbalance": 1.86347},
        7: {"tx1": 0.507216, "tx2": 1.20707, "imbalance": 1.75752},
        8: {"tx1": 0.544281, "tx2": 1.20572, "imbalance": 1.68087},
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
        if order == 3:
            return 0.0 + (27.0 / 2.0) * dt / np.power(tau, 3)
        else:
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
        3: {"kh0": 45.5625, "xi_5pct": 1.85385, "residual_acf": 0.78433},
        4: {"kh0": 32.0000, "xi_5pct": 1.69782, "residual_acf": 2.24215},
        5: {"kh0": 36.6211, "xi_5pct": 1.56878, "residual_acf": 3.41986},
        6: {"kh0": 45.5625, "xi_5pct": 1.46340, "residual_acf": 4.35727},
        7: {"kh0": 57.4458, "xi_5pct": 1.37610, "residual_acf": 5.11265},
        8: {"kh0": 72.0000, "xi_5pct": 1.30253, "residual_acf": 5.73130},
    }

    # fetch values for order and scale
    sacf_values = unit_sacf_values[filter_order]
    # adjust for temporal discretization
    if filter_order == 3:
        sacf_values["kh0"] += 729.0 / 2.0 * dt / tau
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
        3: {"sti_5pct": 3.07384, "residual": +0.05000},
        4: {"sti_5pct": 3.54070, "residual": -0.01764},
        5: {"sti_5pct": 2.69320, "residual": -0.04482},
        6: {"sti_5pct": 2.95804, "residual": -0.05000},
        7: {"sti_5pct": 2.89426, "residual": -0.05000},
        8: {"sti_5pct": 2.78014, "residual": -0.05000},
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

    return generic_peak_frequency_value("frequency", tau, filter_order)


# noinspection SpellCheckingInspection
def gain_at_peak_frequency(tau: float, filter_order: int) -> float:
    """Returns gain(w_peak), the peak gain."""

    return generic_peak_frequency_value("gain", tau, filter_order)


# noinspection SpellCheckingInspection,PyUnusedLocal
def phase_at_peak_frequency(tau: float, filter_order: int) -> float:
    """Returns angle H_delta(s) at the peak frequency."""

    return generic_peak_frequency_value("phase", tau, filter_order)


# noinspection SpellCheckingInspection
def group_delay_at_peak_frequency(tau: float, filter_order: int) -> float:
    """Returns the group delay at the peak frequency."""

    return generic_peak_frequency_value("group-delay", tau, filter_order)


# noinspection SpellCheckingInspection
def generic_peak_frequency_value(
    feature: str, tau: float, filter_order: int
) -> float:
    """Returns the value of a peak-frequency feature.

    Parameters
    ----------
    feature: str
        The caller: {frequency|gain|phase|group-delay}
    tau: float
        Temporal scale of the filter.
    filter_order: int
        Order of the filter.

    Returns
    -------
    float
        The value associated with the caller.
    """

    fcxns = {
        "frequency": lambda _tau, order: order
        / np.sqrt((order - 2.0) / 2.0)
        / _tau,
        "gain": lambda _tau, order: 2.0
        * np.sqrt(
            np.power(order, 2) * np.power((order - 2.0) / order, order - 2)
        )
        / np.power(_tau, 2),
        "phase": lambda _tau, order: np.pi
        - order * np.arctan(1.0 / np.sqrt((order - 2.0) / 2.0)),
        "group-delay": lambda _tau, order: (order - 2.0) / order * tau,
    }

    fcxn = fcxns[feature]
    value = fcxn(tau, filter_order) if filter_order > 1 else np.nan

    return value


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
        3: 0.50644018,
        4: 0.65761864,
        5: 0.70048551,
        6: 0.71784811,
        7: 0.72791012,
        8: 0.73608002,
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
        3: {"TW": 0.254588, "BW": np.inf, "UCP": np.inf},
        4: {"TW": 0.330719, "BW": 8.94427, "UCP": 2.95804},
        5: {"TW": 0.341565, "BW": 6.45497, "UCP": 2.20479},
        6: {"TW": 0.336788, "BW": 6.0, "UCP": 2.02073},
        7: {"TW": 0.327327, "BW": 5.91608, "UCP": 1.93649},
        8: {"TW": 0.316639, "BW": 5.96285, "UCP": 1.88807},
    }

    return uncertainty_products[filter_order]
