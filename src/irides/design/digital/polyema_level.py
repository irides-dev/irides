"""
-------------------------------------------------------------------------------

Digital poly-ema level-filter design.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.design.analog import polyema_level as dsgn_alg_pema
from irides.tools import analog_to_digital_conversion_tools as a2d_tools
from irides.tools import design_tools


def design_id() -> str:
    """Design id of this filter."""

    return "digital-{0}".format(dsgn_alg_pema.design_id())


def design_type() -> design_tools.FilterDesignType:
    """This is a level filter"""

    return dsgn_alg_pema.design_type()


def get_valid_filter_orders(strict: bool = False) -> np.ndarray:
    """Returns valid filter orders for this filter design."""

    return dsgn_alg_pema.get_valid_filter_orders(strict)


def convert_mu_to_pole(mu: float, filter_order: int) -> float:
    """Computes `p` value from parameters"""

    return mu / (filter_order + mu)


def minimum_mu_value(filter_order: int) -> float:
    """Returns the minimum mu value for a given filter order"""

    return filter_order


def gain_adjustment(mu: float, filter_order: int) -> float:
    """Returns the gain adjustment so that the filter's level gain is unity."""

    # compute the associated pole
    p = convert_mu_to_pole(mu, filter_order)

    # compute and return the gain adjustment
    return np.exp(filter_order * np.log(1 - p))


def initial_hn_values(mu: float, filter_order: int) -> np.ndarray:
    """Returns analytically calculated values of h[0], h[1], and h[2].

    From a general single-pole product expansion of H(z), we find that

        h[0] = gain_adj
        h[1] = gain_adj * sum( digital_poles )
        h[2] = gain_adj * (sum( digital_poles )^2 - residual)

    where, as a 4th-order example to show the pattern,

        residual = p1 (p2 + p3 + p4) + p2 (p3 + p4) + p3 * p4.
    """

    # prerequisites
    zplane_poles = a2d_tools.zplane_poles_from_analog_design(
        dsgn_alg_pema, mu, filter_order
    )

    # calculate h[0]
    h0 = gain_adjustment(mu, filter_order)

    # calculate h[1]
    zplane_poles_sum = np.sum(zplane_poles)
    h1 = h0 * np.real(zplane_poles_sum)

    # calculate h[2]
    residual = np.real(
        np.sum(
            [
                zplane_poles[i] * np.sum(zplane_poles[i + 1 :])
                for i in range(filter_order - 1)
            ]
        )
    )
    h2 = h0 * np.real(np.power(zplane_poles_sum, 2) - residual)

    # pack and return as an array
    return np.array([h0, h1, h2])


# noinspection PyPep8Naming
def moment_values(mu: float, filter_order: int) -> np.ndarray:
    """Returns analytically calculated moment values (M0, M1, M2)"""

    M0 = 1.0
    M1 = mu
    M1_s = M1 / filter_order
    M2 = (1.0 + M1 + M1_s) * M1

    return np.array([M0, M1, M2])


def full_width_value(mu: float, filter_order: int) -> float:
    """Returns the analytically calculated FW value"""

    mu = mu
    m = filter_order
    return 2.0 * np.sqrt(mu * (m + mu) / m)


def analog_tau_value(mu: float, filter_order: int) -> float:
    """Returns `tau` associated with (`mu`, `filter_order`)"""

    # compute and return
    splane_poles = dsgn_alg_pema.designs(filter_order)["poles"]

    return a2d_tools.solve_for_tau_from_mu_and_splane_poles(mu, splane_poles)
