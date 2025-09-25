"""
-------------------------------------------------------------------------------

Digital mbox inline-curvature filter design.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.design.analog import mbox_inline_curvature as dsgn_alg_c_mbox
from irides.design.digital import mbox_level as dsgn_dig_l_mbox
from irides.tools import design_tools


def design_id() -> str:
    """Design id of this filter."""

    return "digital-{0}".format(dsgn_alg_c_mbox.design_id())


def design_type() -> design_tools.FilterDesignType:
    """This is a curvature filter"""

    return dsgn_alg_c_mbox.design_type()


def get_valid_filter_orders(strict: bool = False) -> np.ndarray:
    """Returns valid filter orders for this filter design."""

    return dsgn_alg_c_mbox.get_valid_filter_orders(strict)


def adjust_mu_to_ensure_integer_length_box(
    mu: float, filter_order: int
) -> float:
    """Adjusts `mu` upwards so that box-length `N/m` is an integer"""

    return dsgn_dig_l_mbox.adjust_mu_to_ensure_integer_length_box(
        mu, filter_order
    )


def minimum_mu_value(filter_order: int) -> float:
    """Returns the minimum mu value for a given filter order"""

    return dsgn_dig_l_mbox.minimum_mu_value(filter_order)


def initial_hn_values(mu: float, filter_order: int) -> np.ndarray:
    """Returns analytically calculated values of h[0], h[1], and h[2]

    Since this is an inline-curvature filter, h[k] values can be derived from
    the associated level-filter h[k] values.
    """

    # retrieve level-filter hk values
    hk_level = dsgn_dig_l_mbox.initial_hn_values(mu, filter_order)

    # compute hk_slope values
    h0 = hk_level[0]
    h1 = hk_level[1] - 2 * hk_level[0]
    h2 = hk_level[2] - 2 * hk_level[1] + hk_level[0]

    # pack and return
    return np.array([h0, h1, h2])


# noinspection PyPep8Naming
def moment_values(mu: float, filter_order: int) -> np.ndarray:
    """Returns analytically calculated moment values (M0, M1, M2)

    In general, for an inline-slope filter,
        M0 = 0
        M1 = 0
        M2 = 2H(1).
    Rather than calculate H^k(1) terms again, they can be backed out from
    the level-filter calculations.
    """

    # retrieve level-filter Mk values
    Mk_level = dsgn_dig_l_mbox.moment_values(mu, filter_order)

    # convert Mk_level values to Mk_slope values
    M0 = 0
    M1 = 0
    M2 = 2.0 * Mk_level[0]

    # pack and return
    return np.array([M0, M1, M2])
