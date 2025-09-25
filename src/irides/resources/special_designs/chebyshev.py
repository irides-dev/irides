"""
-------------------------------------------------------------------------------

Analog + Digital Chebyshev Type I filter design:

Captures the reference pole locations, pole groupings for multistage
implementation, and tau-minimum values for analog-to-digital conversion.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.design.analog import bessel_level as dsgn_alg_bssl
from irides.tools.design_tools import StageTypes


def get_valid_filter_orders(strict: bool = False) -> np.ndarray:
    """Returns valid filter orders for this filter design"""

    return dsgn_alg_bssl.get_valid_filter_orders(strict)


# noinspection SpellCheckingInspection
def designs(filter_order: int) -> dict:
    """Like .design.analog.bessel_level, returns basic cheby parameters.

    Pole locations from `classical_multipole_designs.nb` notebook.

    Parameters
    ----------
    filter_order: int
        A valid filter order.

    Returns
    -------
    dict
        Design parameters for input `filter_order`
    """

    # designs
    design = {}

    # enumerate designs
    #   stages  : applies to both s- and z-planes
    #   poles   : applies to s-plane
    #   tau_min : scaling of s-plane poles for map to z-plane

    design[1] = {
        "stages": [{"type": StageTypes.EMA.value, "indices": np.array([0])}],
        "poles": np.array([-1.0 + 0j]),
        "tau_minimum": 1.4426950408889634,
        "mu_minimum": 1.0,
    }

    design[2] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([1, 0])}
        ],
        "poles": np.array(
            [-0.29289322 + 0.70710678j, -0.29289322 - 0.70710678j]
        ),
        "tau_minimum": 0.8835885773211489,
        "mu_minimum": -0.05946658,
    }

    design[3] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([2, 1])},
            {"type": StageTypes.EMA.value, "indices": np.array([0])},
        ],
        "poles": np.array(
            [-1.1058925, -0.55294627 - 3.3531595j, -0.55294627 + 3.3531595j,]
        ),
        "tau_minimum": 3.5424207507888927,
        "mu_minimum": 2.09560491,
    }

    design[4] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([3, 2])},
            {"type": StageTypes.DOSC.value, "indices": np.array([1, 0])},
        ],
        "poles": np.array(
            [
                -0.46886226 - 0.89561117j,
                -0.46886226 + 0.89561117j,
                -0.19420911 - 2.1621966j,
                -0.19420911 + 2.1621966j,
            ]
        ),
        "tau_minimum": 2.1785910587712913,
        "mu_minimum": 0.23035264,
    }

    design[5] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([4, 3])},
            {"type": StageTypes.DOSC.value, "indices": np.array([2, 1])},
            {"type": StageTypes.EMA.value, "indices": np.array([0])},
        ],
        "poles": np.array(
            [
                -1.1555201,
                -0.93483542 - 3.8928896j,
                -0.93483542 + 3.8928896j,
                -0.35707536 - 6.2988277j,
                -0.35707536 + 6.2988277j,
            ]
        ),
        "tau_minimum": 6.219641205006642,
        "mu_minimum": 3.77073252,
    }

    design[6] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([5, 4])},
            {"type": StageTypes.DOSC.value, "indices": np.array([3, 2])},
            {"type": StageTypes.DOSC.value, "indices": np.array([1, 0])},
        ],
        "poles": np.array(
            [
                -0.52540545 - 0.96526464j,
                -0.52540545 + 0.96526464j,
                -0.38462349 - 2.637152j,
                -0.38462349 + 2.637152j,
                -0.14078197 - 3.6024167j,
                -0.14078197 + 3.6024167j,
            ]
        ),
        "tau_minimum": 3.5197984637944115,
        "mu_minimum": 0.57052267,
    }

    design[7] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([6, 5])},
            {"type": StageTypes.DOSC.value, "indices": np.array([4, 3])},
            {"type": StageTypes.DOSC.value, "indices": np.array([2, 1])},
            {"type": StageTypes.EMA.value, "indices": np.array([0])},
        ],
        "poles": np.array(
            [
                -1.1796685,
                -1.0628446 - 4.0865606j,
                -1.0628446 + 4.0865606j,
                -0.73551129 - 7.3637277j,
                -0.73551129 + 7.3637277j,
                -0.26250094 - 9.1824183j,
                -0.26250094 + 9.1824183j,
            ]
        ),
        "tau_minimum": 8.91618563425041,
        "mu_minimum": 5.46668783,
    }

    design[8] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([7, 6])},
            {"type": StageTypes.DOSC.value, "indices": np.array([5, 4])},
            {"type": StageTypes.DOSC.value, "indices": np.array([3, 2])},
            {"type": StageTypes.DOSC.value, "indices": np.array([1, 0])},
        ],
        "poles": np.array(
            [
                -0.55223598 - 1.0010796j,
                -0.55223598 + 1.0010796j,
                -0.46816306 - 2.8508335j,
                -0.46816306 + 2.8508335j,
                -0.31281655 - 4.2665738j,
                -0.31281655 + 4.2665738j,
                -0.10984657 - 5.0327669j,
                -0.10984657 + 5.0327669j,
            ]
        ),
        "tau_minimum": 4.86742509466394,
        "mu_minimum": 0.91778267,
    }

    # return
    return design[filter_order]
