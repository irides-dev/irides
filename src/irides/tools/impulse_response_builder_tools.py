"""
-------------------------------------------------------------------------------

Tools to build impulse responses and autocorrelation functions.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.resources.core_enumerations import FilterDesignType
from irides.resources.containers.wireframes import WireframeContinuousTime


# noinspection SpellCheckingInspection
def make_impulse_response_template(
    t_start: float, t_end: float, dt: float, n_cols: int = 2
) -> tuple:
    """Make a common impulse-response template as an np.ndarray [n, 2].

    Parameters
    ----------
    t_start: float
        Start time.
    t_end: float
        End time.
    dt: float
        Time increment.
    n_cols: int
        Number of columns (default = 2)

    Returns
    -------
    tuple
        0: np.ndarray time series template [n, 2] panel
        1:  index into time_axis from which all times are >= 0.0
    """

    # make time series, an (n, 2) matrix
    time_series = np.zeros(
        (np.arange(t_start, t_end + dt, dt).shape[0], n_cols)
    )

    # make t_axis
    time_series[:, 0] = np.arange(t_start, t_end + dt, dt)

    # alias
    t_axis = time_series[:, 0]

    # index and alias t_axis for t_axis >= 0
    i_t_0plus = protected_index_search(t_axis, dt, 0.0)

    # return tuple
    return time_series, i_t_0plus


# noinspection SpellCheckingInspection
def compute_noncausal_symmetric_first_difference(
    time_series: np.ndarray, enforce_causality=True
) -> int:
    """Note that time_series is changed in place, the original being clobbered.

    Calculates the noncausal, symmetric first difference for the time series.

       h(t)
        ^                   .  .  .
        |             . x .
        |           x |
                  .   |
        - . . . |---------------------------------> t
                0     |
                      v
        dh(t) = (h(t+dt) - h(t-dt)) / (2 dt)

    Note that h(t=0) is not reliable because t may not be indexed below zero,
    so a subsequent call to design.impulse_response_to_value(..) may be
    necessary.

    Parameters
    ----------
    time_series: np.ndarray
        Input time series, [n, n_cols >= 2].
    enforce_causality: bool
        Sets h[t<0] = 0 when true.

    Returns
    -------
    int:
        i_t_0plus as int or None
    """

    def compute_difference(ts: np.ndarray, dt: float) -> np.ndarray:
        """Implementation of central calculation"""

        # copy required so that resultant values are purely based on original
        # data, not a mixture of original and resultant, which is the case
        # since, otherwise, `h` is a reference to `ts[:, 1]`.
        h = ts[:, 1].copy()

        # difference
        scale_adj: float = 2.0 * dt
        ts[1:-1, 1] = (h[2:] - h[:-2]) / scale_adj
        ts[0, 1] = h[1] / scale_adj

        # last point cannot be calculated from a partial difference
        # b/c the level of the partial difference and a complete difference
        # may be large. Here, the last point is a copy of the preceding point.
        # An alternative is to return a shortened array. Returning an np.nan
        # disrupts downstream summation calculations.
        ts[-1, 1] = ts[-2, 1].copy()

        return ts

    return compute_noncausal_symmetric_generic_difference(
        compute_difference, time_series, enforce_causality
    )


# noinspection SpellCheckingInspection
def compute_noncausal_symmetric_second_difference(
    time_series_orig: np.ndarray, enforce_causality=True
) -> int:
    """Note that time_series is changed in place, the original being clobbered.

    Calculates the noncausal, symmetric second difference for the time series.

       h(t)
        ^                   .  .  .
        |             x x .
        |           x |
                  .   |
        - . . . |---------------------------------> t
                0     |
                      v
        dh(t) = (h(t+dt) - 2 h(t) + h(t-dt)) / dt^2

    Parameters
    ----------
    time_series_orig: np.ndarray
        Input time series, [n, n_cols >= 2].
    enforce_causality: bool
        Sets h[t<0] = 0 when true.

    Returns
    -------
    int:
        i_t_0plus as int or None
    """

    def compute_difference(ts: np.ndarray, dt: float) -> np.ndarray:
        """Implementation of central calculation"""

        # copy required so that resultant values are purely based on original
        # data, not a mixture of original and resultant, which is the case
        # since, otherwise, `h` is a reference to `ts[:, 1]`.
        h = ts[:, 1].copy()

        # difference
        scale_adj = np.power(dt, 2)
        ts[1:-1, 1] = (h[2:] - 2.0 * h[1:-1] + h[:-2]) / scale_adj
        ts[0, 1] = (h[1] - 2.0 * h[0]) / scale_adj

        # last point cannot be calculated from a partial difference
        # b/c the level of the partial difference and a complete difference
        # may be large. Here, the last point is a copy of the preceding point.
        # An alternative is to return a shortened array. Returning an np.nan
        # disrupts downstream summation calculations.
        ts[-1, 1] = ts[-2, 1].copy()

        return ts

    return compute_noncausal_symmetric_generic_difference(
        compute_difference, time_series_orig, enforce_causality
    )


# noinspection SpellCheckingInspection
def compute_noncausal_symmetric_generic_difference(
    compute_difference, time_series: np.ndarray, enforce_causality=True
) -> int:
    """Computes nth difference, preserves causality (if req'd)"""

    # features along the time axis
    t_axis = time_series[:, 0]
    dt = np.min(np.diff(t_axis))
    i_t0_plus = protected_index_search(t_axis, dt, 0.0)

    # compute difference
    time_series = compute_difference(time_series, dt)

    # force set h(t<0) = 0
    if enforce_causality and i_t0_plus is not None:
        time_series[:i_t0_plus, 1] = 0.0

    # return
    return i_t0_plus


# noinspection SpellCheckingInspection
def protected_index_search(
    axis: np.ndarray, step: float, search_value: float
) -> int:
    """Returns index into axis at `search_value` or None.

    Parameters
    ----------
    axis: np.ndarray
        Axis to search.
    step: float
        Step size along axis.
    search_value

    Returns
    -------
    int:
        index value or None
    """

    # search
    candidate = np.where(search_value - step / 2.0 <= axis)[0]

    # protected return
    return candidate[0] if candidate.shape[0] > 0 else None


# noinspection SpellCheckingInspection
def discretize_wireframe_onto_time_grid(
    t_start: float, t_end: float, dt: float, wf: WireframeContinuousTime
) -> np.ndarray:
    """Embeds a continuous-time wireframe onto a discretized time grid.

    Parameters
    ----------
    Parameters
    ----------
    t_start: float
        Start time.
    t_end: float
        End time.
    dt: float
        Time increment.
    wf: WireframeContinuousTime
        A populated wireframe object

    Returns
    -------
    np.ndarray
        [n, 2] time-series array
    """

    # make ts template, take alias of the time axis
    ts, _ = make_impulse_response_template(t_start, t_end, dt)
    t_axis = ts[:, 0]

    # for each delta function, place it on the time grid
    tp_grid_indices = np.array([], dtype=int)
    for i, tp in enumerate(wf.timepoints):

        # check that this timepoint is in the [t_start, t_end] range
        if tp < t_start or tp > t_end:
            print(
                "warning: wireframe timepoint not in [t_start, t_end] window."
            )
            return ts

        # compute L1 norm between all t_axis points and tp, take shortest norm
        i_tp = np.abs(tp - t_axis).argmin()
        tp_grid_indices = np.append(tp_grid_indices, i_tp)

        # set delta (pre gain adjust)
        ts[i_tp, 1] = wf.weights[i]

    # correct for gain given the discretization
    if wf.type == FilterDesignType.LEVEL:
        gain_adj = 1.0
        ts[tp_grid_indices[0], 1] *= gain_adj

    if wf.type == FilterDesignType.SLOPE:
        wf_interval = t_axis[tp_grid_indices[1]] - t_axis[tp_grid_indices[0]]
        gain_adj = 1.0 / wf_interval
        ts[tp_grid_indices[0], 1] = +1.0 * gain_adj
        ts[tp_grid_indices[1], 1] = -1.0 * gain_adj

    if wf.type == FilterDesignType.CURVE:
        wf_interval = t_axis[tp_grid_indices[2]] - t_axis[tp_grid_indices[0]]
        gain_adj = np.power(wf_interval / 2.0, -2)

        ts[tp_grid_indices[0], 1] = +1.0 * gain_adj
        ts[tp_grid_indices[1], 1] = -2.0 * gain_adj
        ts[tp_grid_indices[2], 1] = +1.0 * gain_adj

    # universal dt adjustment
    ts[:, 1] /= dt

    return ts
