"""
-------------------------------------------------------------------------------

Tools to convert analog poles to digital, provide for pole-location scaling,
and offer return in either Cartesian or polar form.

-------------------------------------------------------------------------------
"""

import numpy as np
import scipy.optimize

from typing import Union


# noinspection SpellCheckingInspection,PyPep8Naming
def convert_analog_poles_to_digital_poles(
    tau: float, splane_poles: np.ndarray, T: float = 1
) -> np.ndarray:
    """Converts s-plane poles via z = e^(p T / tau) to digital poles.

    Parameters
    ----------
    tau: float
        Scale factor, generally tau > 0
    splane_poles: np.ndarray
        Poles in the s-plane
    T: float
        Sampling period, use default T=1 except for special cases.

    Returns
    -------
    np.ndarray
        Converted z-plane poles (Cartesian coords).
    """

    # convert via exponential map and return
    return np.exp(splane_poles * T / tau)


# noinspection SpellCheckingInspection,PyPep8Naming
def convert_digital_poles_to_analog_poles(
    zplane_poles: np.ndarray, T: float = 1
) -> np.ndarray:
    """Converts z-plane poles via s = log(p / T) to analog poles.

    Parameters
    ----------
    zplane_poles: np.ndarray
        Poles in the z-plane
    T: float
        Sampling period, use default T=1 except for special cases.

    Returns
    -------
    np.ndarray
        Converted z-plane poles (Cartesian coords).
    """

    # convert via logarithmic map and return
    return np.log(zplane_poles / T)


# noinspection SpellCheckingInspection
def first_moment_from_zplane_pole(
    zplane_poles: Union[complex, np.ndarray],
) -> Union[float, np.ndarray]:
    """Infer M1 values from z-plane pole coords |p|.

    Parameters
    ----------
    zplane_poles: np.ndarray
        Poles in the z-plane

    Returns
    -------
    np.ndarray
        inferred M1-d values
    """

    # compute and return
    p_abs = np.abs(zplane_poles)
    return p_abs / (1.0 - p_abs)


# noinspection SpellCheckingInspection,PyPep8Naming
def zplane_poles_from_analog_design(
    design_alg, mu: float, filter_order: int, T: float = 1
) -> np.ndarray:
    """Returns zplane poles provided a given design.

    There is a common pattern where
        splane-poles    <- design
        tau             <- calc_from(mu, splane-poles)
        zplane-poles    <- calc_from(splane-poles, tau),

    and so this function captures it.

    Parameters
    ----------
    design_alg:
        A design module that has designs() as a function.
    mu: float
        Digital scale parameter
    filter_order: int
        Filter order
    T: float
        Sampling period (defaults to 1)

    Returns
    -------
    np.ndarray
        Array of zplane poles
    """

    # run through the sequence
    splane_poles = design_alg.designs(filter_order)["poles"]
    tau = solve_for_tau_from_mu_and_splane_poles(mu, splane_poles, T)
    zplane_poles = convert_analog_poles_to_digital_poles(tau, splane_poles)

    # return
    return zplane_poles


# noinspection SpellCheckingInspection,PyPep8Naming
def solve_for_tau_to_place_splane_pole_onto_zplane_real_axis(
    splane_pole: complex, zplane_real_coordinate: float, T: float = 1
) -> float:
    """Solves for `tau` s.t. Re(e^(pa T / tau)) == rho.

    Solves f(tau) == 0

    where

        f(tau) = exp(par T / tau) cos(pai T / tau) - rho

    where rho = zplane_real_coordinate = real(pd). Invokes the `brentq` solver.

    `pa` is an s-plane pole,
    `pd` is the corresponding z-plane pole.

    Edge cases are treated analytically.

    Parameters
    ----------
    splane_pole: complex
        The s-plane pole to scale
    zplane_real_coordinate: float
        The value `rho` on the real z-plane axis, rho in [0, 1)
    T: float
        The sampling rate (default = 1)

    Returns
    -------
    float
        The scaling `tau` value that attains the goal, `tau` > 0.
    """

    # test that the zplane-coord is achievable
    if zplane_real_coordinate < 0 or zplane_real_coordinate >= 1:
        msg = (
            "The real-part of the z-plane coord must be on [0, 1), "
            "but was called with {0}".format(zplane_real_coordinate)
        )
        raise RuntimeError(msg)

    # pull of real and imaginary parts of the s-plane pole
    parT = T * np.real(splane_pole)
    paiT = T * np.abs(np.imag(splane_pole))

    # bounds on tau
    tau_upper = np.power(10, 9)
    tau_lower = paiT / (np.pi / 2.0)

    # analytic solutions:

    # imag(splane-pole) == 0 case (real-valued pa pole)
    if paiT <= 1e-12:
        soln = (
            parT / np.log(zplane_real_coordinate)
            if zplane_real_coordinate > 0
            else 0.0
        )
        return soln

    # rho == 0 case
    if np.abs(zplane_real_coordinate) <= 1e-12:
        return tau_lower

    # numeric solution:

    # the function for which f(tau) == 0 for optimal tau
    def f(tau: float) -> float:
        return np.exp(parT / tau) * np.cos(paiT / tau) - zplane_real_coordinate

    # call the solver
    tau_rho, r = scipy.optimize.brentq(
        f, tau_lower, tau_upper, full_output=True
    )

    return tau_rho


# noinspection PyPep8Naming
def compute_mu_from_tau_and_splane_poles(
    tau: Union[float, np.ndarray], splane_poles: np.ndarray, T: float = 1
) -> np.ndarray:
    """Compute `mu` from `tau` and analog poles.

    Parameters
    ----------
    tau: np.ndarray
        Temporal scales
    splane_poles: np.ndarray
        Analog poles in the s-plane
    T: float
        Sampling period (default = 1)

    Returns
    -------
    np.ndarray
        Value of `mu` to achieve target `tau`
    """

    # compute and return
    return np.sum(
        np.real(1.0 / (np.exp(-np.outer(splane_poles * T, 1.0 / tau)) - 1.0)),
        axis=0,
    )


# noinspection PyPep8Naming
def solve_for_tau_from_mu_and_splane_poles(
    mu: float, splane_poles: np.ndarray, T: float = 1
) -> float:
    """Computes `tau` such that M1(H(z)) = mu.

    This function computes an inflated `tau` value so that the target `mu`
    value is achieved.

    In general M1( H(z; mu) ) < M1( H(s; tau) ). The interface for digital
    filters uses scale-factor `mu` but the synthesis of these filters relies
    on the continuous-time design where the interface is in `tau`.

    Parameters
    ----------
    mu: float
        Temporal scale
    splane_poles: np.ndarray
        Analog poles in the s-plane
    T: float
        Sampling period (default = 1)

    Returns
    -------
    float
        Value of `tau` to achieve target `mu`
    """

    # set up
    filter_order = splane_poles.shape[0]

    # central function to drive to zero
    def f(tau: float) -> float:
        return compute_mu_from_tau_and_splane_poles(tau, splane_poles, T) - mu

    # setup bounds
    tau_lower = max(1.0, mu)
    tau_upper = filter_order + tau_lower

    # compute optimum tau
    tau_star, r = scipy.optimize.brentq(
        f, tau_lower, tau_upper, full_output=True
    )

    # return
    return tau_star


# noinspection SpellCheckingInspection,PyPep8Naming
def solve_for_tau_min_from_splane_poles(
    splane_poles: np.ndarray, T: float = 1
) -> dict:
    """Computes `tau_min` for a constellation of splane poles in a simple way.

    In this calculation, the outermost complex pole is placed in the zplane
    so that its projection onto the real axis lies as Re(pd) = 1/2. All poles
    inside of the outer pole will have real-axis projections that are larger.

    Parameters
    ----------
    splane_poles: np.ndarray
        Analog poles in the s-plane
    T: float
        Sampling period (default = 1)

    Returns
    -------
    dict
        Result dict, keys -- "tau_min" and "mu_min"
    """

    # identify pole with maximum imaginary part
    outer_splane_pole = splane_poles[np.argmax(np.imag(splane_poles))]

    # solve for tau value to place the real part of outer_splane_pole
    #         at Re(z) = 1/2.
    tau_min = solve_for_tau_to_place_splane_pole_onto_zplane_real_axis(
        outer_splane_pole, 0.5, T
    )

    # compute associated mu-min
    mu_min = compute_mu_from_tau_and_splane_poles(tau_min, splane_poles)[0]

    # return
    return {
        "tau_min": tau_min,
        "mu_min": mu_min,
    }


# noinspection SpellCheckingInspection,PyPep8Naming
def solve_for_min_tau_min_rho_tuple_from_splane_poles(
    splane_poles: np.ndarray, T: float = 1
) -> dict:
    """Solves for min(tau, a) tuple s.t. real-parts of z-plane poles are equal.

    The mapping of a constellation of s-plane poles, such as those of a
    classical, rational-function IIR filter, to the z-plane generally leads
    to a constellation with rightward or leftward concavity. This solver
    find min(tau) and min(a) such that the real-parts of the z-plane poles
    fall on a vertical line. This in turn produces the minimum scale-factor
    `tau` for which the discrete-time filter is not significantly aliased.

    From the s-plane constellation, the poles with the least and greatest
    imaginary parts are identified, and these are labeled p-inner and p-outer,
    respectively.

    Then, the iterative solution cycles through the sequence

        M_outer -> rho_outer --(p-outer)-->
                                tau
                                        --(p-inner)--> rho_inner -> M_inner.

    After a pass, M_outer' = M_inner, and the cycle repeats. Convergence is
    attained when |M_outer / M_inner - 1| < TOL. The sequence is initialized
    with a large M_outer value.

    Parameters
    ----------
    splane_poles: np.ndarray
        Constellation of s-plane poles.
    T: float
        The sampling rate (default = 1)

    Returns
    -------
    dict
        `tau_min`: the minimum tau value
        `rho_min`: the minimum rho value, the real-part of z-plane poles in soln
        `M_min`: the M-value related to a_min
        `success`: success bool,
        `iter_count`: the # of iterations to solve,
        `message`: rho message about the solution,
    """

    # validate entry
    if splane_poles.shape[0] <= 2:
        msg = (
            "This solver must have 3 or more complex-conjugate poles and "
            "at most one real-valued pole in order to formulate "
            "the objective. The current pole-count is {0}".format(
                splane_poles.shape[0]
            )
        )
        raise RuntimeError(msg)

    # step 1: identify poles with (min, max) imaginary parts
    working_spoles = {
        "inner": splane_poles[np.argmin(np.abs(np.imag(splane_poles)))],
        "outer": splane_poles[np.argmax(np.imag(splane_poles))],
    }

    # helpers
    #   brings p: (0, 1) to the linear scale M: (0, inf)
    p2M = lambda p: p / (1.0 - p)
    M2p = lambda M: M / (1.0 + M)

    #   returns the rho-value for the inner pole given a tau input
    calc_rho_inner = lambda tau: np.real(
        convert_analog_poles_to_digital_poles(tau, working_spoles["inner"], T)
    )

    # step 2: initialize, iterate until convergence of max-iter
    M_outer = 1e6
    max_iter = 1000
    curr_iter = 0
    rel_tol = 1e-3
    success = False

    while True:

        # sweep through sequence
        rho_outer = M2p(M_outer)
        tau_min = solve_for_tau_to_place_splane_pole_onto_zplane_real_axis(
            working_spoles["outer"], rho_outer
        )
        rho_inner = calc_rho_inner(tau_min)
        M_inner = p2M(rho_inner)

        # test for convergence, or setup next step
        if np.abs(M_outer / M_inner - 1.0) < rel_tol:
            msg = "converged after {0} iterations".format(curr_iter)
            success = True
            break
        elif curr_iter == max_iter:  # pragma: no cover
            msg = "Convergence not achieved. M: ({0},{1}), tau: {2}".format(
                M_outer, M_inner, tau_min
            )
            break
        else:
            M_outer = M_inner
            curr_iter += 1

    # pack for return
    solution = {
        "tau_min": tau_min,
        "rho_min": rho_outer,
        "M_min": p2M(rho_outer),
        "success": success,
        "iter_count": curr_iter,
        "message": msg,
    }

    return solution
