"""
-------------------------------------------------------------------------------

Discrete-time test signals.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.resources.core_enumerations import TestSignalType
from irides.resources.containers.discrete_sequence import DiscreteSequence


# noinspection SpellCheckingInspection
def generate_test_signal(
    n_start: int, n_end: int, signal_type: TestSignalType,
) -> DiscreteSequence:
    """Discrete-time test signals on an integer axis.

    Parameters
    ----------
    n_start: int
        Start index.
    n_end: int
        End index.
    signal_type: TestSignalType
        Type of test signal to construct.

    Returns
    -------
    DiscreteSequence
        Contains n-axis and v-axis
    """

    # create a template
    ds = DiscreteSequence(n_start, n_end)

    # quick return if type is unknown
    if signal_type == TestSignalType.UNKNOWN:
        return ds

    # build lambda dict
    sequence_functions = {
        TestSignalType.IMPULSE: lambda n: 1.0 if n == 0 else 0.0,
        TestSignalType.STEP: lambda n: 1.0,
        TestSignalType.RAMP: lambda n: n + 1.0,
        TestSignalType.PARABOLA: lambda n: 0.5 * (n + 1.0) * (n + 2.0),
        TestSignalType.ALTERNATING: lambda n: np.power(-1.0, n),
        TestSignalType.WHITE_NOISE: lambda n: np.random.normal(0, 1, 1)[0],
    }

    # apply
    ds.v_axis[ds.i_n_zero :] = np.array(
        list(map(sequence_functions[signal_type], ds.n_axis[ds.i_n_zero :]))
    )

    # force causality

    if ds.i_n_zero is not None:
        ds.v_axis[: ds.i_n_zero] = 0.0

    # return
    return ds
