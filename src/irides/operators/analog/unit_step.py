"""
-------------------------------------------------------------------------------

Implements a unit step in a discretized continuous time.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.tools import impulse_response_builder_tools


# noinspection SpellCheckingInspection
def generate_impulse_response(t_end: float, dt: float) -> np.ndarray:
    """Creates a unit-step time series that can be used for convolution.

    Parameters
    ----------
    t_end: float
        End timepoint of the series.
    dt: float
        Time increment.

    Returns
    -------
    np.ndarray
        A time series array (n, 2) having columns [t_axis, h_ustep].
    """

    # init time series
    t_start = 0.0
    (
        time_series,
        i_t0_plus,
    ) = impulse_response_builder_tools.make_impulse_response_template(
        t_start, t_end, dt
    )

    # set the unit step
    time_series[i_t0_plus:, 1] = 1.0

    # return
    return time_series
