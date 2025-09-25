"""
-------------------------------------------------------------------------------

Digital KT-Bessel inline slope-filter signature.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.filter_signatures.digital import bessel_level as fsig_dig_l_bssl
from irides.operators.digital import first_difference as fsig_1st_diff
from irides.resources.containers.discrete_sequence import DiscreteSequence


# noinspection SpellCheckingInspection
def generate_impulse_response(
    n_start: int, n_end: int, mu: float, filter_order: int
) -> DiscreteSequence:
    """Creates a DT impulse response for the Bessel inline slope filter.

    Parameters
    ----------
    n_start: int
        Start sample
    n_end: int
        End sample (exclusive)
    mu: float
        First moment of the impulse response
    filter_order: int
        Filter order

    Returns
    -------
    DiscreteSequence
        Contains n-axis and v-axis
    """

    # create the level-filter and first-diff components
    ds = fsig_dig_l_bssl.generate_impulse_response(
        n_start, n_end, mu, filter_order
    )
    ds_diff = fsig_1st_diff.generate_impulse_response()

    # convolve
    cand = np.convolve(ds_diff.v_axis, ds.v_axis)

    # assign result back to original, clipping last value
    ds.v_axis = cand[: ds.n_axis.shape[0]]

    # return
    return ds
