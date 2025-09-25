"""
-------------------------------------------------------------------------------

Digital multistage-box filter signature.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.design.digital import mbox_level as dsgn_d_l_mbox

from irides.tools import impulse_response_tools as ir_tools
from irides.resources.containers.discrete_sequence import DiscreteSequence


# noinspection SpellCheckingInspection,PyPep8Naming
def generate_impulse_response(
    n_start: int, n_end: int, mu: float, filter_order: int
) -> DiscreteSequence:
    """Creates a discrete-time impulse response for the multistage-box filter.

    Note: n_end must be at least (2 mu + filter_order) in length in order to
          not clip the mbox impulse response.

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
        Contains n-axis and h-axis
    """

    # validate
    ir_tools.validate_filter_order_or_die(dsgn_d_l_mbox, filter_order)

    # alias
    m = filter_order

    # compute N_stage, which uses an adjusted `mu` value
    N_stage = dsgn_d_l_mbox.convert_adjusted_mu_to_N_stage(mu, filter_order)
    N_int = filter_order * N_stage

    # setup a discrete sequence
    ds = DiscreteSequence(n_start, n_end)

    # prepare for and exec m-fold box cascade
    h_box = np.ones(N_stage) / N_stage
    conv_mbox = np.array([1])
    for i in range(m):
        conv_mbox = np.convolve(h_box, conv_mbox)

    # trim box to length N
    conv_mbox = conv_mbox[:N_int]
    conv_mbox_neff_axis = np.arange(0, conv_mbox.shape[0])

    # identify overlapping indices of ds.n_axis and the implicit n-axis
    # of the conv_box array
    i_insertions = np.array(list(set(ds.n_axis) & set(conv_mbox_neff_axis)))

    # insert
    ds.v_axis[ds.i_n_zero : ds.i_n_zero + i_insertions.shape[0]] = conv_mbox[
        i_insertions
    ]

    # return
    return ds
