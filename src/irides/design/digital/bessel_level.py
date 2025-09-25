"""
-------------------------------------------------------------------------------

Digital Bessel filter design:

The fundamental file is analog.bessel_level.py because KT-Bessel filters are
designed in continuous time. Yet, for design-level parameters necessary
for digital implementation, this file captures these additional values.
Bessel filters are designed in continuous time

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.design.analog import bessel_level as dsgn_alg_bssl
from irides.tools import design_tools
from irides.tools import digital_design_tools as dd_tools
from irides.tools import analog_to_digital_conversion_tools as a2d_tools

"""
-------------------------------------------------------------------------------

Filter-specific definitions

-------------------------------------------------------------------------------
"""


# noinspection SpellCheckingInspection
def designs(filter_order: int) -> dict:
    """Design points related to digital Bessel filter realizations."""

    # designs
    design = {}

    design[1] = {
        "tau_minimum": 1.4426950408889634,
        "mu_minimum": 1.0,
        "scaled_bandwidth_pct": 0.0,
    }

    design[2] = {
        "tau_minimum": 2.3950324087595174,
        "mu_minimum": 1.4994039481201957,
        "scaled_bandwidth_pct": 0.23019684134769067,
    }

    design[3] = {
        "tau_minimum": 3.3487313481270853,
        "mu_minimum": 1.9983665156513852,
        "scaled_bandwidth_pct": 0.33352141188877066,
    }

    design[4] = {
        "tau_minimum": 4.302913367060106,
        "mu_minimum": 2.4971877143015915,
        "scaled_bandwidth_pct": 0.39316730888500784,
    }

    design[5] = {
        "tau_minimum": 5.25732894539093,
        "mu_minimum": 2.9959541441532784,
        "scaled_bandwidth_pct": 0.4324218290845279,
    }

    design[6] = {
        "tau_minimum": 6.2118775638638635,
        "mu_minimum": 3.494696994129047,
        "scaled_bandwidth_pct": 0.46042833324907484,
    }

    design[7] = {
        "tau_minimum": 7.1665102003888395,
        "mu_minimum": 3.9934293398661236,
        "scaled_bandwidth_pct": 0.48153438243678787,
    }

    design[8] = {
        "tau_minimum": 8.121199813729849,
        "mu_minimum": 4.492157233932513,
        "scaled_bandwidth_pct": 0.49808226091458135,
    }

    # check and return
    return design[filter_order]


"""
-------------------------------------------------------------------------------

Standard features

-------------------------------------------------------------------------------
"""


def design_id() -> str:
    """Design id of this filter."""

    return "digital-{0}".format(dsgn_alg_bssl.design_id())


def design_type() -> design_tools.FilterDesignType:
    """This is a level filter"""

    return dsgn_alg_bssl.design_type()


def get_valid_filter_orders(strict: bool = False) -> np.ndarray:
    """Returns valid filter orders for this filter design."""

    return dsgn_alg_bssl.get_valid_filter_orders(strict)


def minimum_mu_value(filter_order: int) -> float:
    """Returns the minimum mu value for a given filter order"""

    return designs(filter_order)["mu_minimum"]


# noinspection PyPep8Naming
def gain_adjustment(mu: float, filter_order: int, T: float = 1) -> float:
    """Returns the gain adjustment so that the level-filter's gain is unity.

    The gain adjustment for digital poles pd_k is

        gain_adj = (1 - pd_1) (1 - pd_2) ... (1 - pd_m),

    and the real part is taken to annihilate the zero-valued imaginary part.
    """

    # fetch zplane poles
    zplane_poles = a2d_tools.zplane_poles_from_analog_design(
        dsgn_alg_bssl, mu, filter_order, T
    )

    # compute and return
    return np.real(np.exp(np.sum(np.log(1.0 - zplane_poles))))


# noinspection PyPep8Naming
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
        dsgn_alg_bssl, mu, filter_order
    )
    # Hz = construct_Hz(mu, filter_order)
    Hz = dd_tools.construct_transfer_function(mu, filter_order, dsgn_alg_bssl)

    # calculate h[0]
    h0 = Hz.initial_value()

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
    """Returns moment values (M0, M1, M2) computed from H(z).

    Since the analytic derivatives of H(z) are complicated, it is simpler
    to numerically compute its approximate derivatives. To improve accuracy,
    higher-order finite differences are taken.
    """

    # build an H(z) object
    # Hz = construct_Hz(mu, filter_order)
    Hz = dd_tools.construct_transfer_function(mu, filter_order, dsgn_alg_bssl)

    # setup to H'(1) and H''(1)
    scale = 1e-4
    eps = scale / mu
    zvals = 1.0 + eps * np.arange(-3, 3 + 1)
    Hz_pts = Hz.value(zvals)

    # approximate from differences

    # these coefs from
    #   https://en.wikipedia.org/wiki/Finite_difference_coefficient
    zero_diff_coefs = np.array([0, 0, 0, 1, 0, 0, 0])
    first_diff_coefs = (
        np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]) / eps
    )
    second_diff_coefs = np.array(
        [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]
    ) / np.power(eps, 2)

    H = np.sum(zero_diff_coefs * Hz_pts)
    Hp = np.sum(first_diff_coefs * Hz_pts)
    Hpp = np.sum(second_diff_coefs * Hz_pts)

    # M0 and M1 and M2
    M0 = H
    M1 = -Hp
    M2 = Hp + Hpp

    # pack and return
    return np.array([M0, M1, M2])


def full_width_value(mu: float, filter_order: int) -> float:
    """Returns the numerically approximated FW value"""

    moments = moment_values(mu, filter_order)

    return 2.0 * np.sqrt(moments[2] - np.power(moments[1], 2))


def analog_tau_value(mu: float, filter_order: int) -> float:
    """Returns `tau` associated with (`mu`, `filter_order`)"""

    # compute and return
    splane_poles = dsgn_alg_bssl.designs(filter_order)["poles"]

    return a2d_tools.solve_for_tau_from_mu_and_splane_poles(mu, splane_poles)
