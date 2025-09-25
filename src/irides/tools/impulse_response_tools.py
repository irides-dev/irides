"""
-------------------------------------------------------------------------------

Tools to compute realized moments and autocorrelation functions from a
numerically calculated impulse response.

-------------------------------------------------------------------------------
"""

import numpy as np
from scipy import signal


# noinspection SpellCheckingInspection,PyPep8Naming
def calculate_impulse_response_moment(
    impulse_response: np.array, moment: int
) -> float:
    """
    Calculate a moment of the input impulse response.

        M_k = sum( n^moment * h ) * dt.

    Parameters
    ----------
    impulse_response: np.array
        Impulse response, where [:, 0] is the time axis and [:, 1] is the
        weight axis.
    moment: int
        The moment to calculate.

    Returns
    -------
    moment_value: float
        The value of the `moment` moment.
    """

    # references
    t = impulse_response[:, 0]
    h = impulse_response[:, 1]

    # infer dt, the increment of time
    dt = min(np.diff(t))

    # calculate moment
    mk = np.power(t, moment).transpose().dot(h) * dt

    # return
    return mk


# noinspection SpellCheckingInspection,PyPep8Naming
def calculate_impulse_response_full_width(impulse_response: np.array) -> float:
    """
    Calculates the full width of the input impulse response.

        FW = 2 sqrt(M2 - M1^2).

    Parameters
    ----------
    impulse_response: np.array
        Impulse response, where [:, 0] is the time axis and [:, 1] is the
        weight axis.

    Returns
    -------
    fw: float
        The full width of the impulse response.
    """

    # calculate the location and M2
    M1 = calculate_impulse_response_moment(impulse_response, 1)
    M2 = calculate_impulse_response_moment(impulse_response, 2)

    # calculate the
    arg = M2 - np.power(M1, 2)
    fw = 2.0 * np.sqrt(arg) if arg >= 0.0 else np.nan

    # return
    return fw


# noinspection SpellCheckingInspection,PyPep8Naming
def calculate_auto_correlation_function(impulse_response: np.array) -> np.array:
    """
    Calculates the system autocorrelation spectrum from the input impulse
    response.

        SACF = conv(h[-t], h[t]) * dt.

    Parameters
    ----------
    impulse_response: np.array
        Impulse response, where [:, 0] is the time axis and [:, 1] is the
        weight axis.

    Returns
    -------
    sacf: np.array
        The system autocorrelation spectrum of the input impulse response.

    """

    # references
    t = impulse_response[:, 0]
    h = impulse_response[:, 1]

    # infer dt, the increment of time
    dt: float = min(np.diff(t))

    # compute number of output points and create output panel
    npts = 2 * t.shape[0] - 1
    acf = np.ndarray([npts, 2], dtype="float")

    # compute the lags axis
    acf[:, 0] = np.array(range(npts)) * dt + (min(t) - max(t))

    # The ACF is the convolution of impulse_response(t)
    # with impulse_response(-t), scaled by dt.
    acf[:, 1] = signal.convolve(h[::-1], h, mode="full", method="direct") * dt

    # return
    return acf


def get_valid_moments() -> np.ndarray:
    """Array of valid moments in this framework.

    Returns
    -------
    np.ndarray
        Array of valid moments.
    """

    return np.array([0, 1, 2])


# noinspection SpellCheckingInspection,PyPep8Naming
def validate_filter_order_or_die(
    design, filter_order: int, strict: bool = False
):
    """
    This function either throws or returns nothing. If it returns, the
    calling code continues knowing that the filter-order is valid.

    Parameters
    ----------
    design:
        An imported design module that impl's get_valid_filter_orders().
    filter_order: int
        The filter order.
    strict: bool
        Whether to enforce strict orders.
    """

    valid_filter_orders = design.get_valid_filter_orders(strict)

    m_min = min(valid_filter_orders)
    m_max = max(valid_filter_orders)

    if not m_min <= filter_order <= m_max:
        msg = "filter order must be within [1, {0}] inclusive. order: {1}".format(
            m_max, filter_order
        )
        raise IndexError(msg)
