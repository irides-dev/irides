"""
-------------------------------------------------------------------------------

Analog damped-oscillator filter-element design

-------------------------------------------------------------------------------
"""

import numpy as np
import scipy.special

from irides.resources.containers import points


# noinspection SpellCheckingInspection
def unit_peak_times(cartesian_poles: np.ndarray) -> np.ndarray:
    """Returns peak times for a unit-scale impulse response.

    The timepoint where h(t) peaks, with tau = 1, is

              1                             1
        t* = ---- theta csc(theta) = ----------------
              wn                      wn sinc(theta)

    where theta is angle of the reference pole to the
    negative real axis, and wn is the natural frequency.

    Parameters
    ----------
    cartesian_poles: np.ndarray
        Poles in cartesian coordinates.

    Returns
    -------
    np.ndarray
        Unit peak time values.
    """

    # extract wn, theta
    wn, theta = extract_wn_and_theta(cartesian_poles)

    # calc and return
    return 1.0 / (wn * np.sinc(theta / np.pi))


# noinspection SpellCheckingInspection
def unit_peak_values(cartesian_poles: np.ndarray) -> np.ndarray:
    """Returns peak value for unit-scale impulse response.

    The value at h(t*) is the peak value. For unit tau, the
    peak value is

        h(t*) = wn exp(-theta cot(theta))
              = wn exp(-cos(theta) / sinc(theta))

    where theta is the angle of the reference pole to the
    real axis, and wn is the natural frequency.

    Parameters
    ----------
    cartesian_poles: np.ndarray
        Poles in cartesian coordinates.

    Returns
    -------
    np.ndarray
        Unit peak values.
    """

    # extract wn, theta
    wn, theta = extract_wn_and_theta(cartesian_poles)

    # calc and return
    return wn * np.exp(-np.cos(theta) / np.sinc(theta / np.pi))


# noinspection SpellCheckingInspection
def peak_value_coordinates(cartesian_poles: np.ndarray, tau=1.0):
    """Returns (time, value) coordinates for a scaled impulse response.

    Provided that there is a non-zero real part to the reference
    poles, there will be a single peak of the impulse response h(t).

                  ....x...  <---- h(t*)
              ....        ....
           ...                ..
         ..                     ...
        .                          ....
    ----|-------------|-------------------------> t
        0
                     t*

    Parameters
    ----------
    cartesian_poles: np.ndarray
        Poles in cartesian coordinates.
    tau: float
        The impulse-response scale factor (default = 1.0).

    Returns
    -------
    points.ContinuousTimePoints
        Scaled peak coordinates.
    """

    # put together into a point coordinate and return
    return points.ContinuousTimePoints(
        unit_peak_times(cartesian_poles) * tau,
        unit_peak_values(cartesian_poles) / tau,
    )


# noinspection SpellCheckingInspection
def nth_null_times(cartesian_poles: np.ndarray, n=1, tau=1.0) -> np.ndarray:
    """Returns times of the nth zero crossing of the impulse response.

    The null locations are purely governed by the sinusoidal component
    of h(t). Nulls exist only when |theta| > 0, where /theta/ is the
    polar angle of the reference pole.

    The temporally scaled nth null time is

                     n pi
        t_null(n) = ------ csc(|theta|) tau,
                      wn

    where /wn/ and /theta/ are the polar coordinates of /reference_pole/.

    In the case that /theta/ == 0, t_null = inf.

    Parameters
    ----------
    cartesian_poles: np.ndarray
        Poles in cartesian coordinates.
    n: int
        The nth zero crossing (starts with 1).
    tau: float
        The impulse-response scale factor (default = 1.0).

    Returns
    -------
    np.ndarray
        nth null times of a scaled impulse response.
    """

    # extract wn, theta
    wn, theta = extract_wn_and_theta(cartesian_poles)

    # calc and return
    with np.errstate(divide="ignore"):

        return (tau * n * np.pi / wn) / np.sin(np.abs(theta))


# noinspection SpellCheckingInspection,PyUnusedLocal
def moment_value_generator(moment: int, dt=0.0):
    """Returns a function to calculate the requested moment.

    Returns an anonymous function with arguments (tau, wn, theta), where
    wn is the natural frequency of the pole pair and theta is the inclination
    angle from the negative real axis. The returned function captures the
    dt argument in this calling function in order to make a discretization
    adjustment to the moments.

    Note that the generated functions are not purely continuous time.
    Instead, they bridge continuous and discrete time by applying order-dt
    corrections to the moments. This is required because the CT series are
    in fact implemented in DT.

    Parameters
    ----------
    moment: int
        Moment to calculate: [0, 1, 2]
    dt: float
        Time interval for discretization.

    Returns
    -------
    Anonymous function with arguments (tau, wn, theta) as np.ndarrays.
    """

    # generator defs with dt capture
    # noinspection PyUnusedLocal
    def m0_gen(
        wn: np.ndarray, theta: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        return np.ones_like(wn)

    # noinspection PyUnusedLocal
    def m1_gen(
        wn: np.ndarray, theta: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        return 2.0 * tau * np.cos(theta) / wn

    # noinspection PyUnusedLocal
    def m2_gen(
        wn: np.ndarray, theta: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        return 2.0 * (1.0 + 2.0 * np.cos(2.0 * theta)) / np.power(wn / tau, 2)

    # return
    if moment == 0:
        return m0_gen
    elif moment == 1:
        return m1_gen
    elif moment == 2:
        return m2_gen
    else:
        msg = "Moments must be within (0, 1, 2). Calling value: {0}".format(
            moment
        )
        raise IndexError(msg)


# noinspection SpellCheckingInspection,PyUnusedLocal
def full_width_generator(dt=0.0):
    """
    Returns anonymous function for the full-width. `dt` is captured here,
    and the anonymous function has arguments (wn, theta).

    Parameters
    ----------
    dt: float
        Time interval for discretization.

    Returns
    -------
    Anonymous function with arguments (tau, wn, theta).
    """

    # noinspection PyPep8Naming,PyUnusedLocal
    def fw_gen(
        wn: np.ndarray, theta: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        """
        Full width of the damped oscillator is only defined for
        -pi/4 < theta < pi/4  (theta from the negative real axis).
        Returns np.nan otherwise.
        """

        with np.errstate(invalid="ignore"):

            M1 = 2.0 * np.cos(theta) / (wn / tau)
            RHW = np.sqrt(np.cos(2.0 * theta) / (1.0 + np.cos(2.0 * theta)))
            return 2.0 * M1 * RHW

    return fw_gen


# noinspection SpellCheckingInspection
def key_spectral_features(cartesian_poles: np.ndarray, tau=1.0) -> dict:
    """Calculates peak freq (wmax), gain(wmax), phase(wmax), grp-dly(wmax).

    Calculates the peak frequency, and the gain, phase and group delay at this
    frequency, for both over- and underdamped oscillators, and stores the
    results in a dictionary.

    The 'damp-type' key records the damp type for a particular cartesian pole,
    and the value of this record is the key to the array that stores the
    associated (wmax, gain, phase, group delay) for the pole.

    {
     'damp-type': [{"under"|"over"}]
     'under': [wmax, gain, phase, group delay],
     'over': [wmax, gain, phase, group delay],
    }

    Parameters
    ----------
    cartesian_poles: np.ndarray
        Poles in cartesian coordinates.
    tau: float
        The impulse-response scale factor (default = 1.0).

    Returns
    -------
    dict
        Keys: ["damp-type", "over", "under"].
        Values: np.ndarrays.
    """

    # extract wn, theta
    wn, theta = extract_wn_and_theta(cartesian_poles)

    # build dict container
    n = wn.shape[0]
    d = {
        "damp-type": np.empty([n], dtype=object),
        "under": np.zeros((n, 4)),
        "over": np.zeros((n, 4)),
    }

    def eval_under_over(this_theta: float) -> str:
        if 0 <= np.abs(np.arctan(np.tan(this_theta))) < np.pi / 4:
            return "over"
        else:
            return "under"

    # apply and build iterator
    map_under_over = map(eval_under_over, theta)

    # iter over types
    for i, damp_type in enumerate(map_under_over):

        # record type
        d["damp-type"][i] = damp_type

        # locals
        wn_ = wn[i]
        th_ = theta[i]

        # overdamped calcs
        wmax_ = 0.0
        go_ = 1.0
        pho_ = 0.0
        gdo_ = 2.0 * np.cos(th_) / wn_ * tau

        d["over"][i, :] = np.array([wmax_, go_, pho_, gdo_])

        # underdamped calcs
        if damp_type == "under":

            wmax_ = wn_ / tau * np.sqrt(np.cos(np.pi - 2.0 * np.abs(th_)))
            gu_ = 1.0 / np.sin(2.0 * np.abs(th_))
            phu_ = -np.arctan(np.sqrt(np.power(np.tan(th_), 2) - 1.0))
            gdu_ = tau / (wn_ * np.cos(th_))

            d["under"][i, :] = np.array([wmax_, gu_, phu_, gdu_])

        else:

            d["under"][i, :] = np.array([np.nan, np.nan, np.nan, np.nan])

    # return
    return d


# noinspection SpellCheckingInspection
def autocorrelation_peak_and_stride_values(
    cartesian_poles: np.ndarray, tau=1.0
) -> dict:
    """Returns peak and stride values for array of poles, scaled by tau.

    The cartesian poles are converted to (wn, theta), where theta is zero
    on the negative real axis.

    For theta => 0,

                 wn sec(theta)
        kh(0) = --------------- .
                    4 tau

    The stride express depends on whether |theta| > 0 or theta == 0.
    For |theta| > 0,

                     sec(theta)
        xi_5pct = - ------------ log( 0.05 sin( |theta| ) ) tau,
                         wn

    while for theta == 0,

                   -1 - W_{-1}( -0.05 e^{-1} )
        xi_5pct = ----------------------------- tau
                               wn

    where W is the lambertw function, with branch index -1.

    Parameters
    ----------
    cartesian_poles: np.ndarray
        Poles in cartesian coordinates.
    tau: float
        The impulse-response scale factor (default = 1.0).

    Returns
    -------
    dict
        Peak value (scaled by 1/tau) and 5% pct stride (scaled by tau)
    """

    # extract wn, theta
    wn, theta = extract_wn_and_theta(cartesian_poles)

    # eval kh0
    kh0 = wn / (4.0 * np.cos(theta))

    # eval xi_5pct
    xi_5pct = np.zeros_like(kh0)
    pct_5 = 0.05
    lam_arg = pct_5 * np.exp(-1.0)

    # it would be much better to build a lambda array and apply
    for i, th in enumerate(theta):

        if np.abs(th) > 0:
            xi_5pct[i] = -np.log(pct_5 * np.sin(np.abs(th))) / (
                wn[i] * np.cos(th)
            )
        else:
            xi_5pct[i] = (
                -1 - np.real(scipy.special.lambertw(-lam_arg, -1))
            ) / wn[i]

    # scale
    kh0 /= tau
    xi_5pct *= tau

    # compose dict and return
    return {"kh0": kh0, "xi_5pct": xi_5pct}


def cast_to_polar_form(cartesian_coords: np.ndarray) -> np.ndarray:
    """Converts complex values in Cartesian form to polar form.

    Polar angles are with respect to the positive real axis (as is standard).

    Parameters
    ----------
    cartesian_coords: np.ndarray
        (n, ) array of complex coordinates in cartesian form.

    Returns
    -------
    np.ndarray
        (n, 2) array of complex coordinates in polar form (abs, angle).
    """

    # allocate
    polar_coords = np.zeros((cartesian_coords.shape[0], 2))

    # convert
    polar_coords[:, 0] = np.abs(cartesian_coords)
    polar_coords[:, 1] = np.angle(cartesian_coords)

    # return
    return polar_coords


def cast_to_cartesian_form(polar_coords: np.ndarray) -> np.ndarray:
    """Converts polar representation of complex coordinates to Cartesian form.

    Polar angles are with respect to the positive real axis (as is standard).

    Parameters
    ----------
    polar_coords: np.ndarray
        (n, 2) array of complex coordinates in polar form (abs, angle).

    Returns
    -------
    np.ndarray
        (n, ) array of complex values in Cartesian form.
    """

    # convert and return
    return np.array(
        polar_coords[:, 0] * np.cos(polar_coords[:, 1])
        + 1j * polar_coords[:, 0] * np.sin(polar_coords[:, 1])
    )


def convert_reference_angle_to_negative_real_axis(
    polar_coords: np.ndarray,
) -> np.ndarray:
    r"""Converts the angle reference from positive to negative real axis.

    The angle of a complex coordinate is naturally taken with respect to the
    positive real axis. This angle is labeled phi. For the damped oscillator,
    it is convenient to reference angles the negative real axis (labeled
    theta) so that a "damping angle" of zero refers to a purely real pole
    location, instead of a "damping angle" of +/- pi.


     s-plane         ^  imag                s-plane         ^  imag
                     |                                      |
               x     +  pi                            x     +  pi
                \    |                                 \    |
                 \ <------ wn                           \ <------ wn
                  \  |                                   \--|-
                  /\ |                                    \ |  \  <---- phi
     theta --->  |  \|                                     \|   |
        -------+--\--+-------> real    <-->    -------+-----+---|----> real
              pr     |                               pr     |
                     |                                      |
                     |                                      |

    Applying this function twice recovers the original angle.

    Parameters
    ----------
    polar_coords: np.ndarray
        (n, 2) array of complex coordinates in polar form (abs, angle).

    Returns
    -------
    np.ndarray
        (n, 2) array where the angles (array[:, 1]) have been converted.

    """

    # convert and return
    polar_coords[:, 1] = np.pi - (polar_coords[:, 1] % (2 * np.pi))

    return polar_coords


def extract_wn_and_theta(cartesian_poles: np.ndarray):
    """Extracts natural frequency wn and inclination angle theta from coords.

    Parameters
    ----------
    cartesian_poles: np.ndarray
        Poles in cartesian coordinates.

    Returns
    -------
    tuple
        (wn, theta) as np.ndarrays.
    """

    # convert to polar form
    polar_poles = convert_reference_angle_to_negative_real_axis(
        cast_to_polar_form(cartesian_poles)
    )

    return polar_poles[:, 0], polar_poles[:, 1]
