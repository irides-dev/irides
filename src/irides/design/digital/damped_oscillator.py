"""
-------------------------------------------------------------------------------

Digital damped-oscillator filter-element design

-------------------------------------------------------------------------------
"""

import numpy as np
from typing import Union

from irides.design.analog import damped_oscillator as dsgn_alg_dosc
from irides.tools import analog_to_digital_conversion_tools as a2d_tools
from irides.tools import digital_design_tools as dd_tools


# -----------------------------------------------------------------------------
# Purely digital
# -----------------------------------------------------------------------------

# noinspection SpellCheckingInspection
def gain_adjustment(
    zplane_poles: np.ndarray, return_polar_coords=False
) -> Union[np.ndarray, tuple]:
    """Returns the gain adjustment for an array of zplane poles.

    Parameters
    ----------
    zplane_poles: np.ndarray
        The digital poles
    return_polar_coords: bool
        Returns (gain_adj, r, phi) as tuple of np.ndarrays if T (default F)

    Returns
    -------
    Union[np.ndarray, tuple]
        gain adjustment as an array, or tuple that includes polar coords
        of zplane_poles
    """

    # convert to polar form
    polar_poles = dsgn_alg_dosc.cast_to_polar_form(zplane_poles)
    r = polar_poles[:, 0]
    phi = polar_poles[:, 1]

    # generate polar form of low-frequency poles
    zplane_lf_poles = dd_tools.convert_frequency_band(
        zplane_poles, dd_tools.FrequencyBand.LOW
    )
    polar_lf_poles = dsgn_alg_dosc.cast_to_polar_form(zplane_lf_poles)
    phi_lf = polar_lf_poles[:, 1]

    # compute gain adjustment -- always at low frequency
    gain_adj = 1.0 - 2.0 * np.cos(phi_lf) * r + r * r

    if return_polar_coords:
        return gain_adj, r, phi
    else:
        return gain_adj


def initial_hn_values(zplane_poles: np.ndarray) -> np.ndarray:
    """Returns analytically calculated h[0], h[1], h[2] values.

    Parameters
    ----------
    zplane_poles: np.ndarray
        Poles in cartesian coordinates.

    Returns
    -------
    np.ndarray
        Array [len(zplane_poles), 3] of values h[0], h[1], h[2]
    """

    # prerequisites
    gain_adj, r, phi = gain_adjustment(zplane_poles, return_polar_coords=True)

    # calculate h[0]
    h0 = gain_adj

    # calculate h[1]
    h1 = 2.0 * np.cos(phi) * r * h0

    # calculate h[2]
    h2 = (1.0 + 2.0 * np.cos(2.0 * phi)) * r * r * h0

    # pack and return as an array
    return np.array([h0, h1, h2]).T


# noinspection SpellCheckingInspection,PyPep8Naming
def moment_values(zplane_poles: np.ndarray) -> np.ndarray:
    """Returns the requested moment for each z-plane pole value.

    Parameters
    ----------
    zplane_poles: np.ndarray
        z-plane poles in Cartesian coordinate

    Returns
    -------
    np.ndarray
        Moments [M0, M1, M2]
    """

    # prerequisites
    gain_adj, r, phi = gain_adjustment(zplane_poles, return_polar_coords=True)

    # 0th moment
    M0 = np.ones_like(r)

    # 1st moment -- M1 = H'(1)
    M1 = 2.0 * r * (np.cos(phi) - r) / gain_adj

    # 2nd moment -- M2 = H'(1) + H''(1)
    cos_phi = np.cos(phi)
    cos_2_phi = np.cos(2.0 * phi)

    #   compute H'
    H_p = 2.0 * r * (cos_phi - r) / gain_adj

    #   compute H''
    elmt = 1.0 + 2.0 * cos_2_phi - 6.0 * cos_phi * r + 3.0 * r * r
    H_pp = 2.0 * r * r * elmt / np.power(gain_adj, 2)

    #   compute M2
    M2 = H_pp + H_p

    return np.array([M0, M1, M2]).T


# noinspection SpellCheckingInspection
def full_width_value(zplane_poles: np.ndarray) -> np.ndarray:
    """Calculates the analytic full-width value(s), returns nan when not def'd.

    Parameters
    ----------
    zplane_poles: np.ndarray
        Poles in cartesian coordinates.

    Returns
    -------
    np.ndarray
        Full-width values
    """

    # prerequisites
    gain_adj, r, phi = gain_adjustment(zplane_poles, return_polar_coords=True)

    with np.errstate(invalid="ignore"):

        # compute
        arg = 2.0 * r * ((1.0 + r * r) * np.cos(phi) - 2.0 * r)
        # gain_adj = gain_adjustment(r, phi)

        # return
        return 2.0 * np.sqrt(arg) / gain_adj


# -----------------------------------------------------------------------------
# Mixed analog and digital
# -----------------------------------------------------------------------------

# noinspection SpellCheckingInspection
def tau_minimum_for_analog_pole(splane_poles: np.ndarray) -> np.ndarray:
    """Returns tau_min given an s-plane pole.

    tau_min is defined as the value of tau that leads to

        Re( zplane_pole ) = 1/2 .

    Parameters
    ----------
    splane_poles: np.ndarray
        Either pole of the complex-conjugate pair

    Returns
    -------
    np.ndarray
        minimum tau values
    """

    # by definition
    pdr = 0.5

    # compute and return
    return np.array(
        [
            a2d_tools.solve_for_tau_to_place_splane_pole_onto_zplane_real_axis(
                splane_pole, pdr
            )
            for splane_pole in splane_poles
        ]
    )
