"""
-------------------------------------------------------------------------------

Implements an ideal delay in a discretized continuous time.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.tools import impulse_response_builder_tools


# noinspection SpellCheckingInspection
def generate_impulse_response(tau: float, dt: float) -> np.ndarray:
    """Creates an ideal delay time series that can be used for convolution.

    The discretization naturally replaces a Dirac delta with a Kronecker delta
    with a weight of 1 / dt.

    Parameters
    ----------
    tau: float
        Timepoint of the Dirac delta (tau >= 0).
    dt: float
        Time increment.

    Returns
    -------
    np.ndarray
        A time series array (n, 2) having columns [t_axis, h_delay].
    """

    # determine (t_start, t_end)
    t_start = 0.0
    t_end = np.ceil(tau / dt) * dt

    # init time series
    (
        time_series,
        i_t0_plus,
    ) = impulse_response_builder_tools.make_impulse_response_template(
        t_start, t_end, dt
    )

    # locate the `tau` timepoint
    tau_sample = int(np.round(tau / dt))

    # set the delta
    time_series[i_t0_plus + tau_sample, 1] = 1.0 / dt

    # return
    return time_series
