"""
-------------------------------------------------------------------------------

Analog unit-step filter signature:

Implements the temporal and spectral signatures of a continuous-time
unit-step function u(t).

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.tools import transfer_function_tools


def generate_impulse_response(tstart, tend, dt):
    r"""
    Returns an np array, first column is the (discrete) time axis,
    second column is the impulse response.

    The impulse response of a unit step is

                /  0, t <= 0
        u(t) = <
                \  1, t >= 0-

    where 0- captures any transition at t = 0.

    input:  /tstart/  start time
            /tend/    end time
            /dt/      time increment

    output: /panel/   np array, shape (npts, 2)

    """

    # calculate the number of points to eval on
    npts = int(np.ceil((tend - tstart) / dt))

    # setup the panel return
    panel = np.ndarray([npts, 2], dtype=float)

    # compute the time axis
    panel[:, 0] = [v * dt + tstart for v in range(npts)]

    # compute the impulse response
    panel[:, 1] = [0.0 if t < 0.0 else 1.0 for t in panel[:, 0]]

    # return
    return panel


def generate_transfer_function(fstart, fend, df):
    """
    Returns an np array, first column is the (cyclic) frequency axis,
    second column is the complex-valued transfer function.

    The defining transfer function is

                1
        H(s) = --- .
                s

    In general, s = omega + j w, a complex value. Here, however, omega = 0,
    so /s/ is replaced by /j w/.

    Moreover, the api is for cyclic frequency /f/, measured in cycles per sec,
    but radial frequency /w/, measured in radians per sec, is used internally.

    input:  /fstart/        start frequency, cycles per second
            /fend/          end frequency
            /df/            frequency increment

    output: /panel/         np array, shape (npts, 2)

    """

    # calculate the number of points to eval on
    npts = int(np.ceil((fend - fstart) / df))

    # setup the panel return
    panel = np.ndarray([npts, 2], dtype=complex)

    # compute the frequency axis
    panel[:, 0] = [v * df + fstart for v in range(npts)]

    # compute the complex transfer function along the /jw/ axis
    jw = 1j * 2.0 * np.math.pi * panel[:, 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        panel[:, 1] = 1.0 / jw

    # return
    return panel


def generate_gain_spectrum(fstart, fend, df):
    """
    Generates the gain spectrum of a unit-step filter.

    input:  /fstart/        start frequency, cycles per second
            /fend/          end frequency
            /df/            frequency increment

    output: /panel/         np array, shape (npts, 2)
                                column 1: [fstart, fend]
                                column 2: [gain spectrum]
    """

    # get the transfer function panel
    xfer_fcxn_panel = generate_transfer_function(fstart, fend, df)

    # convert to a gain spectrum
    spectral_panel = transfer_function_tools.calculate_gain_spectrum(
        xfer_fcxn_panel
    )

    # return
    return spectral_panel


def generate_phase_spectrum(fstart, fend, df):
    """
    Generates the phase spectrum of a unit-step filter.

    input:  /fstart/        start frequency, cycles per second
            /fend/          end frequency
            /df/            frequency increment

    output: /panel/         np array, shape (npts, 2)
                                column 1: [fstart, fend]
                                column 2: [phase spectrum]
    """

    # get the transfer function panel
    xfer_fcxn_panel = generate_transfer_function(fstart, fend, df)

    # convert to a phase spectrum
    xfer_fcxn_panel = transfer_function_tools.perturb_dc_frequency(
        xfer_fcxn_panel, df
    )
    spectral_panel = transfer_function_tools.calculate_phase_spectrum(
        xfer_fcxn_panel
    )

    # return
    return spectral_panel


def generate_group_delay_spectrum(fstart, fend, df):
    """
    Generates the group-delay spectrum of a unit-step filter.

    input:  /fstart/        start frequency, cycles per second
            /fend/          end frequency
            /df/            frequency increment

    output: /panel/         np array, shape (npts, 2)
                                column 1: [fstart, fend]
                                column 2: [group-delay spectrum]
    """

    # get the transfer function panel with an extra row at fend + df
    xfer_fcxn_panel = generate_transfer_function(fstart, fend + df, df)

    # compute the group-delay spectrum
    spectral_panel = transfer_function_tools.calculate_group_delay_spectrum(
        xfer_fcxn_panel, df
    )

    # return
    return spectral_panel[:-1, :]
