"""
-------------------------------------------------------------------------------

Discrete-time unit-step operator.

-------------------------------------------------------------------------------
"""

import numpy as np
from irides.resources.containers.discrete_sequence import DiscreteSequence
from irides.tools.digital_design_tools import FrequencyBand


# -----------------------------------------------------------------------------
# design


def delay(n_end: int = np.inf) -> float:
    """Returns the average delay of the operator (applies only to LOW band)"""

    return n_end / 2


# -----------------------------------------------------------------------------
# filter signature


def generate_impulse_response(
    n_end: int, frequency_band: FrequencyBand = FrequencyBand.LOW
) -> DiscreteSequence:
    """Returns a discrete sequence of (+/-1)^n between [0, n_end).

    Parameters
    ----------
    n_end: int
        The value of the highest index (exclusive of this point).
    frequency_band: FrequencyBand
        LOW frequency -> 1^n, HIGH frequency -> (-1)^n

    Returns
    -------
    DiscreteSequence
        A preloaded discrete sequence.
    """

    # setup a discrete sequence
    ds = DiscreteSequence(0, n_end)

    # set weights
    signed_one = 1 if frequency_band == FrequencyBand.LOW else -1
    ds.v_axis = np.power(signed_one, ds.n_axis)

    # return
    return ds
