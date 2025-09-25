"""
-------------------------------------------------------------------------------

Digital multistage-box level-filter design.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.design.analog import mbox_level as dsgn_alg_l_mbox
from irides.tools import design_tools


def design_id() -> str:
    """Design id of this filter."""

    return "digital-{0}".format(dsgn_alg_l_mbox.design_id())


def design_type() -> design_tools.FilterDesignType:
    """This is a level filter"""

    return dsgn_alg_l_mbox.design_type()


def get_valid_filter_orders(strict: bool = False) -> np.ndarray:
    """Returns valid filter orders for this filter design."""

    return dsgn_alg_l_mbox.get_valid_filter_orders(strict)


# noinspection PyPep8Naming
def convert_mu_to_N(mu: float, filter_order: int) -> float:
    """Converts mu value to Mbox length N.

    No adjustments are made to ensure that N / m is an integer. The first
    moment of a mbox filter is
        mu = (N - m) / 2,
    thus
        N = 2 mu + m.
    """

    return 2 * mu + filter_order


def adjust_mu_to_ensure_integer_length_box(
    mu: float, filter_order: int
) -> float:
    """Adjusts `mu` to that box-length `N/m` is an integer.

    At issue is that `mu` is given, but mu = (N - m) / 2, so the
    resulting `N` may not be an integer. The strategy is to convert
    N -> N_int and then return mu* = (N_int - m) / 2.
    """

    N_cand = 2.0 * mu + filter_order
    N_int = filter_order * np.round(N_cand / filter_order).astype(int)

    return (N_int - filter_order) / 2


# noinspection PyPep8Naming
def convert_adjusted_mu_to_N_stage(mu: float, filter_order: int) -> int:
    """Returns an integral N_stage value after, in effect, adjusting `mu`."""

    N_cand = 2.0 * mu + filter_order

    return np.round(N_cand / filter_order).astype(int)


def minimum_mu_value(filter_order: int) -> float:
    """Returns the minimum mu value for a given filter order"""

    return filter_order / 2


# noinspection PyPep8Naming
def gain_adjustment(mu: float, filter_order: int) -> float:
    """Returns the gain adjustment so that the filter's level gain is unity."""

    # compute N
    N = convert_mu_to_N(mu, filter_order)

    # compute and return
    return np.power(filter_order / N, filter_order)


def initial_hn_values(mu: float, filter_order: int) -> np.ndarray:
    """Returns analytically calculated values of h[0], h[1], and h[2]."""

    # compute N, fetch gain adj
    gain_adj = gain_adjustment(mu, filter_order)

    # compute
    h0 = gain_adj
    h1 = filter_order * gain_adj
    h2 = 0.5 * (filter_order + 1) * filter_order * gain_adj

    # pack and return
    return np.array([h0, h1, h2])


# noinspection PyPep8Naming
def moment_values(mu: float, filter_order: int) -> np.ndarray:
    """Returns analytically calculated moment values (M0, M1, M2)"""

    # compute integer N
    N = convert_mu_to_N(mu, filter_order)
    m = filter_order

    M0 = 1.0
    M1 = 0.5 * (N - m)
    M2 = ((3 * m + 1) * N - (3 * m - 1) * m) / (6 * m) * M1

    return np.array([M0, M1, M2])


# noinspection PyPep8Naming
def full_width_value(mu: float, filter_order: int) -> np.ndarray:
    """Returns the analytically calculated FW value"""

    Mk = moment_values(mu, filter_order)

    return 2.0 * np.sqrt(Mk[2] - np.power(Mk[1], 2))


def analog_tau_value(mu: float, filter_order: int) -> float:
    """Returns `tau` associated with (`mu`, `filter_order`)"""

    # for FIR filters like mbox, tau = mu
    return mu
