"""
-------------------------------------------------------------------------------

Analog bessel inline curvature filter design:

Captures the design details for this filter. Design details are available
in Mathematica notebook bessel_inline_curvature.nb.

-------------------------------------------------------------------------------
"""

import numpy as np
from enum import Enum

from irides.resources.core_enumerations import FilterDesignType
from irides.resources.containers.wireframes import WireframeContinuousTime


class StageTypes(Enum):
    """Defines the set of possible stage types"""

    EMA = 0
    DOSC = 1


def get_valid_filter_orders_for_level_filters() -> np.ndarray:
    """Returns array of valid filter orders across all level filters"""

    return np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)


# noinspection SpellCheckingInspection,PyUnusedLocal
def get_minimum_valid_filter_order(
    filter_type: FilterDesignType, strict: bool = False
) -> int:
    """Returns minimum valid filter order.

    Strict orders have the trait that h(0) = 0, regardless of filter design
    type.

    Parameters
    ----------
    filter_type: FilterDesignType
        Indicates level, slope, or curve filter type.
    strict: bool
        Select strict or not-strict (loose) lower bound

    Returns
    -------
    int
        Minimum valid filter order.
    """

    soln_dict = {
        FilterDesignType.LEVEL.value: 1,
        FilterDesignType.SLOPE.value: 2,
        FilterDesignType.CURVE.value: 3,
    }

    minimum_valid_order = soln_dict[filter_type.value]

    if strict:
        minimum_valid_order += 1

    return minimum_valid_order


# noinspection SpellCheckingInspection
def get_valid_filter_orders(
    filter_type: FilterDesignType, strict: bool = False
) -> np.ndarray:
    """Returns array of valid filter orders based on f-type and strictness

    Parameters
    ----------
    filter_type: FilterDesignType
        Indicates level, slope, or curve filter type.
    strict: bool
        If true, return only higher orders, removing lower orders that
        do not qualify as suitable filters.

    Returns
    --------
    np.ndarray:
        Array of valid filter orders
    """

    filter_orders = get_valid_filter_orders_for_level_filters()
    minimum_valid_order = get_minimum_valid_filter_order(filter_type, strict)

    filter_orders = np.delete(
        filter_orders, np.where(filter_orders < minimum_valid_order)
    )

    return filter_orders


# noinspection SpellCheckingInspection,PyUnusedLocal
def wireframe_moment_value_generator(
    design, moment: int, tau: float, dt: float
):
    """Returns M0, M1, or M2 moment generators given a filter-operation type."""
    moment_generating_functions = {
        FilterDesignType.LEVEL: {
            0: lambda order: 1.0,
            1: lambda order: tau,
            2: lambda order: np.power(tau, 2),
        },
        FilterDesignType.SLOPE: {
            0: lambda order: 0.0,
            1: lambda order: -1.0,
            2: lambda order: -(
                design.wireframe(tau, order).timepoints[0]
                + design.wireframe(tau, order).timepoints[1]
            ),
        },
        FilterDesignType.CURVE: {
            0: lambda order: 0.0,
            1: lambda order: 0.0,
            2: lambda order: 2.0,
        },
    }

    return moment_generating_functions[design.design_type()][moment]


# noinspection SpellCheckingInspection,PyPep8Naming,PyUnusedLocal
def generate_level_wireframe_from_spectra(
    design, tau: float, filter_order: int
) -> WireframeContinuousTime:
    """Calculates and constructs a level-type wireframe object."""

    return WireframeContinuousTime(FilterDesignType.LEVEL, np.array([tau]))


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_slope_wireframe_from_spectra(
    design, tau: float, filter_order: int
) -> WireframeContinuousTime:
    """Calculates and constructs a slope-type wireframe object.

    Parameters
    ----------
    design:
        Reference to a design module.
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        The order of the filter.

    Returns
    -------
    WireframeContinuousTime
        Wireframe object
    """

    # frequency coordinates
    w_peak = design.frequency_of_peak_gain(tau, filter_order)
    phase_peak = design.phase_at_peak_frequency(tau, filter_order)

    # convert frequency of peak gain to a (co)sinusoidal period
    T_peak = 2.0 * np.pi / w_peak

    # the slope wireframe is one-half of T_peak
    T_wf = T_peak / 2.0

    # the positive timepoint is calculated thru the argument of the cosine
    #   cos(w_peak t_pos + phase_peak) --> cos(0) = 1
    t_pos = -phase_peak / w_peak

    # offset by T_wf to find t_neg
    t_neg = t_pos + T_wf

    # construct and return a wireframe container
    return WireframeContinuousTime(
        FilterDesignType.SLOPE, np.array([t_pos, t_neg])
    )


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_curvature_wireframe_from_spectra(
    design, tau: float, filter_order: int
) -> WireframeContinuousTime:
    """Calculates and constructs a curvature-type wireframe object.

    Parameters
    ----------
    design:
        Reference to a design module.
    filter_order: int
        The order of the filter.
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.

    Returns
    -------
    WireframeContinuousTime
        Wireframe object
    """

    # frequency coordinates
    w_peak = design.frequency_of_peak_gain(tau, filter_order)
    phase_peak = design.phase_at_peak_frequency(tau, filter_order)

    # convert frequency of peak gain to a (co)sinusoidal period
    T_peak = 2.0 * np.pi / w_peak

    # the curvature wireframe is the same as T_peak
    T_wf = T_peak

    # the negative (center) timepoint is calculated thru
    # the argument of the cosine
    t_pos_l = -phase_peak / w_peak

    # the left and right positive timepoints are symmetrically offset from
    # the center timepoint
    t_neg = t_pos_l + T_wf / 2.0
    t_pos_r = t_pos_l + T_wf

    # construct and return a wireframe container
    return WireframeContinuousTime(
        FilterDesignType.CURVE, np.array([t_pos_l, t_neg, t_pos_r])
    )
