"""
-------------------------------------------------------------------------------

Discrete-time second-difference operator.

-------------------------------------------------------------------------------
"""

import numpy as np
from irides.resources.containers.discrete_sequence import DiscreteSequence


# -----------------------------------------------------------------------------
# design


def delay() -> float:
    """Returns the average delay of the operator"""

    return 1.0


# -----------------------------------------------------------------------------
# filter signature


def generate_impulse_response() -> DiscreteSequence:
    """Returns a discrete sequence three samples long with weights [+1, -2, +1].

    Returns
    -------
    DiscreteSequence
        A preloaded discrete sequence.
    """

    # setup a discrete sequence
    ds = DiscreteSequence(0, 3)

    # set weights for the zeroth and first samples
    ds.v_axis = np.array([+1.0, -2.0, +1.0])

    # return
    return ds
