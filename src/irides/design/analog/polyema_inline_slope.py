"""
-------------------------------------------------------------------------------

Analog poly-ema inline slope filter design:

Captures the design details for this filter. Design details are available
in Mathematica notebook polyema_inline_slope.nb. Wavenumbers are
calculated numerically in Python in `calculate_wavenumber.py`.

-------------------------------------------------------------------------------
"""

import numpy as np
import scipy.misc
import scipy.special
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

    return "polyema-inline-slope"


def design_type() -> design_tools.FilterDesignType:
    """This is a slope filter"""

    return design_tools.FilterDesignType.SLOPE


def get_minimum_valid_filter_order(strict: bool = False) -> int:
    """Returns minimum valid filter order."""

    return design_tools.get_minimum_valid_filter_order(design_type(), strict)


def get_valid_filter_orders(strict: bool = False) -> np.ndarray:
    """Returns array of valid filter orders based on strictness."""

    return design_tools.get_valid_filter_orders(design_type(), strict)


def integer_sequence_a000169(n: int) -> int:
    """
    Integer sequence that is used in the peak-gain calculation.

    See https://oeis.org/A000169.

    Parameters
    ----------
    n: int
        Sequence index.

    Returns
    -------
    int
        Sequence value.
    """

    return np.power(n, n - 1)


def integer_sequence_a063170(n: int) -> int:
    """
    Integer sequence related to inline polyema slope-filter
    wireframe timepoints.

    See https://oeis.org/A063170. One note: The scipy.special
    function of gammaincc(a,b) is normalized by gamma(a). To align
    with the published sequence, the gammaincc(..) function is
    multiplied by gamma(a).

    Parameters
    ----------
    n: int
        Sequence index.

    Returns
    -------
    int
        Sequence value.
    """

    return int(
        np.round(
            scipy.special.gamma(n)
            * scipy.special.gammaincc(n, n - 1)
            * np.exp(n - 1)
        )
    )


# noinspection SpellCheckingInspection
def impulse_response_t0_value(
    tau: float, filter_order: int, dt: float
) -> float:
    """
    Returns the h_m(t=0) value for inline slope filters of this type.
    For order = 1, there is, technically, a leading Dirac delta function.
    While this module falls within the continuous-time context, here the
    time discretization is accounted for.

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

    if filter_order == 1:
        val = 1.0 / (dt * tau) - 1.0 / np.power(tau, 2)
    elif filter_order == 2:
        val = 4.0 / np.power(tau, 2)
    else:
        val = 0.0

    return val


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

    return (filter_order - 1.0) / filter_order * tau


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
        if order == 1:
            return 0.0 - dt / (2.0 * np.power(tau, 2))
        elif order == 2:
            return 0.0 + 2.0 * dt / np.power(tau, 2)
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
    The zero-lag system autocorrelation value, k_hslope(0), is analytic
    and can be expressed in terms of its reference-filter parent. We have

                                 1
        k_hslope(0) = ----------------------- k_href(0)
                       (2 m - 3) (tau / m)^2

    where m = filter_order.

    The stride xi_5pct where sacf(lag=xi_5pct) = 0.05 is reported numerically
    (based on results from Mathematica).

    If `dt` is not None, then the discretization correction is applied to kh0.

    k_hslope(0) scales as 1 / tau^3.
    xi_5pct scales as tau.

    Parameters
    ----------
    tau: float
        Temporal scale of the filter.
    filter_order: int
        Order of the filter.
    dt: float
        [Optional] Temporal increment dt in h(t).

    Returns
    -------
    dict
        Peak value and 5% pct stride, appropriately scaled by tau
    """

    # compute kh0 value
    #   note: returns {'kh0': <val>, 'xi_5pct': <val>}
    ref_sacf_values = design_reference.autocorrelation_peak_and_stride_values(
        tau, filter_order
    )
    slope_sacf_values = ref_sacf_values

    # noinspection SpellCheckingInspection
    def coef_gen(order: int) -> float:
        """Analytic coefficient to reference-filter kh0"""

        return 1.0 / ((2.0 * order - 3.0) * np.power(tau / order, 2))

    # adjust kh0 value
    if filter_order > 1:
        slope_sacf_values["kh0"] *= coef_gen(filter_order)
    else:
        slope_sacf_values["kh0"] = np.nan

    # adjust for discretization
    slope_sacf_values[
        "kh0"
    ] += autocorrelation_peak_value_discretization_correction(
        tau, filter_order, dt
    )

    # set xi_5pct value, see Mathematica polyema_inline_slope.nb
    unit_sacf_xi_5pct_values = {
        1: np.nan,
        2: 2.06997,
        3: 2.18896,
        4: 2.03310,
        5: 1.87929,
        6: 1.74845,
        7: 1.63877,
        8: 1.54614,
    }
    slope_sacf_values["xi_5pct"] = unit_sacf_xi_5pct_values[filter_order] * tau

    # return
    return slope_sacf_values


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
        return np.nan
    elif filter_order == 2:
        return 8.0 * dt / np.power(tau, 4)
    else:
        return 0.0


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
        2: {"sti_5pct": 6.39659, "residual": +0.05000},
        3: {"sti_5pct": 5.89306, "residual": -0.03034},
        4: {"sti_5pct": 5.47517, "residual": -0.05000},
        5: {"sti_5pct": 5.18021, "residual": -0.05000},
        6: {"sti_5pct": 4.72284, "residual": -0.05000},
        7: {"sti_5pct": 4.33074, "residual": -0.05000},
        8: {"sti_5pct": 4.01296, "residual": -0.05000},
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

    return generic_peak_frequency_value("frequency", tau, filter_order)


# noinspection SpellCheckingInspection
def gain_at_peak_frequency(tau: float, filter_order: int) -> float:
    """Returns gain(w_peak), the peak gain."""

    return generic_peak_frequency_value("gain", tau, filter_order)


# noinspection SpellCheckingInspection
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
        "frequency": lambda _tau, order: order / np.sqrt(order - 1) / _tau,
        "gain": lambda _tau, order: np.sqrt(
            order * np.power((order - 1.0) / order, order - 1)
        )
        / _tau,
        "phase": lambda _tau, order: np.pi / 2.0
        - order * np.arctan(1.0 / np.sqrt(order - 1)),
        "group-delay": lambda _tau, order: (order - 1.0) / order * tau,
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
        2: 0.56986212,
        3: 0.70527775,
        4: 0.73673970,
        5: 0.74872477,
        6: 0.75584547,
        7: 0.76189466,
        8: 0.76761364,
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
        2: {"TW": 0.433013, "BW": np.inf, "UCP": np.inf},
        3: {"TW": 0.5, "BW": 5.19615, "UCP": 2.59808},
        4: {"TW": 0.484123, "BW": 4.0, "UCP": 1.93649},
        5: {"TW": 0.458258, "BW": 3.87298, "UCP": 1.77482},
        6: {"TW": 0.433013, "BW": 3.92792, "UCP": 1.70084},
        7: {"TW": 0.410326, "BW": 4.04145, "UCP": 1.65831},
        8: {"TW": 0.390312, "BW": 4.17786, "UCP": 1.63067},
    }

    return uncertainty_products[filter_order]
