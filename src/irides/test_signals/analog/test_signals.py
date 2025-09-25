"""
-------------------------------------------------------------------------------

Continuous-time test signals.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.resources.core_enumerations import TestSignalType
from irides.tools import impulse_response_builder_tools


# noinspection SpellCheckingInspection
def generate_test_signal(
    t_start: float,
    t_end: float,
    dt: float,
    signal_type: TestSignalType,
    alternating_frequency: float = 0.0,
) -> tuple:
    """Approximation of continuous-time test signal on a discretized time grid.

    Parameters
    ----------
    t_start: float
        Start time.
    t_end: float
        End time.
    dt: float
        Time increment.
    signal_type: TestSignalType
        Type of test signal to construct.
    alternating_frequency: float
        Radial frequency of alternating signal, w: cos(w t) [optional]

    Returns
    -------
    tuple
        (np.ndarray, int)
        np.ndarray: A time series array (n, 2) having columns [t_axis, h_test].
        int: index to t=0 value (if it exists)
    """

    # create a time-series template
    # fmt: off
    ts, i_t0_plus = impulse_response_builder_tools.make_impulse_response_template(
        t_start, t_end, dt, n_cols=2
    )
    # fmt: on
    t_axis = ts[:, 0]

    # quick return if type is unkonwn
    if signal_type == TestSignalType.UNKNOWN:
        return ts, i_t0_plus

    # build lambda dict
    time_functions = {
        TestSignalType.IMPULSE: lambda t: 0.0,  # note: this requires a fixup
        TestSignalType.STEP: lambda t: 1.0,
        TestSignalType.RAMP: lambda t: t,
        TestSignalType.PARABOLA: lambda t: 0.5 * np.power(t, 2),
        TestSignalType.ALTERNATING: lambda t: np.cos(alternating_frequency * t),
        TestSignalType.WHITE_NOISE: lambda t: np.random.normal(0, 1, 1)[0],
    }

    # apply
    ts[:, 1] = np.array(list(map(time_functions[signal_type], t_axis)))

    # impulse fixup
    if signal_type == TestSignalType.IMPULSE:
        ts[i_t0_plus, 1] = 1.0 / dt

    # force causality
    ts[:i_t0_plus, 1] = 0.0

    # return
    return ts, i_t0_plus
