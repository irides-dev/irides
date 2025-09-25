"""
-------------------------------------------------------------------------------

Digital poly-ema level-filter signature.

-------------------------------------------------------------------------------
"""

import numpy as np
import scipy.special

from irides.design.digital import polyema_level as dsgn_d_l_pema

from irides.tools import digital_design_tools as dd_tools
from irides.resources.containers.discrete_sequence import DiscreteSequence
from irides.tools.digital_design_tools import FrequencyBand


# noinspection SpellCheckingInspection
def generate_impulse_response(
    n_start: int,
    n_end: int,
    mu: float,
    filter_order: int,
    frequency_band: FrequencyBand = FrequencyBand.LOW,
) -> DiscreteSequence:
    """Creates a discrete-time impulse response for the poly-ema filter.

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
    frequency_band: FrequencyBand
        Specifies LOW or HIGH frequency (default = LOW)

    Returns
    -------
    DiscreteSequence
        Contains n-axis and h-axis
    """

    # compute pole value
    p = dsgn_d_l_pema.convert_mu_to_pole(mu, filter_order)

    # establish the location of `p` based on the requested frequency band
    # fmt: off
    p_signed = dd_tools.\
        convert_frequency_band(np.array([p]), frequency_band)[0]
    # fmt: on

    # setup a sequence container
    ds = DiscreteSequence(n_start, n_end)

    # pull n-axis and setup hn
    i_n_zero = ds.i_n_zero
    n_0plus = ds.n_axis[i_n_zero:]

    hn = ds.v_axis

    # analytically compute response
    gain_adj = dsgn_d_l_pema.gain_adjustment(mu, filter_order)
    hn[i_n_zero:] = (
        gain_adj
        * scipy.special.binom(n_0plus + filter_order - 1, filter_order - 1)
        * np.power(p_signed, n_0plus)
    )

    # return
    return ds
