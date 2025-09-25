"""
-------------------------------------------------------------------------------

Digital damped oscillator:

Implements the temporal signature of a damped-oscillator filter.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.design.digital import damped_oscillator as dsgn_dig_dosc
from irides.resources.containers.discrete_sequence import DiscreteSequence


# noinspection SpellCheckingInspection
def generate_impulse_response(
    n_start: int, n_end: int, zplane_poles: np.ndarray
) -> DiscreteSequence:
    """Creates a discrete-time impulse response of the damped oscillator.

    In the case that Im(zplane_poles) == 0, the 2nd-order pole-ema solution
    is returned.

    Parameters
    ----------
    n_start: int
        Start sample.
    n_end: int
        End sample (exclusive).
    zplane_poles: np.ndarray
        Coordinates of complex-conjugate pole pair.

    Returns
    -------
    DiscreteSequence
        Contains n-axis and h-axis
    """

    # setup a sequence template
    ds = DiscreteSequence(n_start, n_end)

    # pull n-axis, n == 0 index, and alias v-axis
    i_n_zero = ds.i_n_zero
    n_0plus = ds.n_axis[i_n_zero:]
    hn = ds.v_axis

    # convert pole to polar form
    i_select = (
        0
        if zplane_poles[0] == zplane_poles[1]
        else np.argwhere(np.imag(zplane_poles) > 0)[0][0]
    )
    gain_adj_v, r_v, phi_v = dsgn_dig_dosc.gain_adjustment(
        zplane_poles, return_polar_coords=True
    )
    gain_adj = gain_adj_v[i_select]
    r = r_v[i_select]
    phi = phi_v[i_select]

    # compute h[n]
    if np.abs(np.imag(zplane_poles[0])) > 0.0:
        hn[i_n_zero:] = (
            gain_adj
            / np.sin(phi)
            * np.sin((n_0plus + 1) * phi)
            * np.power(r, n_0plus)
        )
    else:
        hn[i_n_zero:] = (
            np.cos(phi * n_0plus)
            * np.power(1.0 - r, 2)
            * (n_0plus + 1)
            * np.power(r, n_0plus)
        )

    # return
    return ds
