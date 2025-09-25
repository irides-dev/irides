"""
-------------------------------------------------------------------------------

Design details of the high-frequency kt-bessel level filter as it relates
to the jump-volatility filter.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.design.analog import bessel_level as dsgn_a_l_bssl


def get_valid_filter_orders() -> np.ndarray:
    """Returns valid filter orders for this filter design"""

    # start with the analog design space
    analog_candidate = dsgn_a_l_bssl.get_valid_filter_orders(strict=True)

    # remove orders <= 2
    valid_orders = analog_candidate[np.where(analog_candidate >= 3)[0]]

    # return
    return valid_orders


def designs(filter_order: int) -> dict:
    """Design details for the jump-vol filter"""

    # designs
    design = {}

    design[3] = {
        "kij_zero_at_tau_min": 1.1819,
        "kij_zero_at_tau_inf": 0.942596,
    }

    design[4] = {
        "kij_zero_at_tau_min": 1.26417,
        "kij_zero_at_tau_inf": 1.10617,
    }

    design[5] = {
        "kij_zero_at_tau_min": 1.40365,
        "kij_zero_at_tau_inf": 1.26084,
    }

    design[6] = {
        "kij_zero_at_tau_min": 1.55348,
        "kij_zero_at_tau_inf": 1.40435,
    }

    design[7] = {
        "kij_zero_at_tau_min": 1.69833,
        "kij_zero_at_tau_inf": 1.5373,
    }

    design[8] = {
        "kij_zero_at_tau_min": 1.83489,
        "kij_zero_at_tau_inf": 1.66093,
    }

    # return
    return design[filter_order]
