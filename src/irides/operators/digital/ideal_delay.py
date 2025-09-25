"""
-------------------------------------------------------------------------------

Discrete-time ideal-delay operator.

-------------------------------------------------------------------------------
"""

import numpy as np
from irides.resources.containers.discrete_sequence import DiscreteSequence


# -----------------------------------------------------------------------------
# design


def delay(mu: float = np.nan) -> float:
    """Returns the average delay of the operator, or nan absent an argument"""

    return mu


# -----------------------------------------------------------------------------
# filter signature


def generate_impulse_response(mu: float) -> DiscreteSequence:
    """Returns a discrete sequence with a single +1 value located at n = mu.

    Parameters
    ----------
    mu: float
        The point of ideal delay. (Note, will be rounded to nearest integer.)

    Returns
    -------
    DiscreteSequence
        A preloaded discrete sequence.
    """

    # round mu (note: mu can take either sign)
    mu_int = np.round(mu).astype(int)

    # determine the length of the n-axis
    n_start = 0
    n_end = 0
    if mu_int >= 0:
        n_end = mu_int
    else:
        n_start = mu_int

    n_end_offset = 1  # cannot exclude a sample for ideal delay

    # setup a discrete sequence
    ds = DiscreteSequence(n_start, n_end + n_end_offset)

    # set the +1 weight
    ds.v_axis = np.zeros_like(ds.n_axis)
    ds.v_axis[ds.index_at(mu_int)] = +1.0

    # return
    return ds
