"""
-------------------------------------------------------------------------------

Design details of the high-frequency poly-ema level filter as it relates
to the jump-volatility filter.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.design.analog import polyema_level as dsgn_a_l_pema


def get_valid_filter_orders() -> np.ndarray:
    """Returns valid filter orders for this filter design"""

    # start with the analog design space
    analog_candidate = dsgn_a_l_pema.get_valid_filter_orders(strict=True)

    # remove orders <= 2
    valid_orders = analog_candidate[np.where(analog_candidate >= 3)[0]]

    # return
    return valid_orders


def designs(filter_order: int) -> dict:
    """Design details for the jump-vol filter"""

    # designs
    design = {}

    design[3] = {
        "kij_zero_at_tau_min": 1.06966,
        "kij_zero_at_tau_inf": 0.883706,
    }

    design[4] = {
        "kij_zero_at_tau_min": 1.06752,
        "kij_zero_at_tau_inf": 0.981931,
    }

    design[5] = {
        "kij_zero_at_tau_min": 1.13212,
        "kij_zero_at_tau_inf": 1.07404,
    }

    design[6] = {
        "kij_zero_at_tau_min": 1.2103,
        "kij_zero_at_tau_inf": 1.16004,
    }

    design[7] = {
        "kij_zero_at_tau_min": 1.28814,
        "kij_zero_at_tau_inf": 1.24068,
    }

    design[8] = {
        "kij_zero_at_tau_min": 1.36295,
        "kij_zero_at_tau_inf": 1.31675,
    }

    # return
    return design[filter_order]
