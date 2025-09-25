"""
-------------------------------------------------------------------------------

Tests against an analog box filter signature

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.filter_signatures.analog import unit_step


def test_one_equals_one():
    """Tests that 1 == 1 on the integers."""
    assert 1 == 1


def test_impulse_response_zeros_before_tzero():
    """Test u(t<0) == 0"""

    # generate impulse response panel
    panel = unit_step.generate_impulse_response(-1.0, 1.0, 0.1)

    # ensure that t < 0 for this point
    assert panel[0][0] < 0.0

    # ensure that h(t) == 0 for this point
    assert panel[0][1] == 0.0


def test_impulse_response_one_at_tzero():
    """Test u(t==0) == 1"""

    # generate impulse response panel
    panel = unit_step.generate_impulse_response(0.0, 1.0, 0.1)

    # ensure that t == 0 for this point
    assert panel[0][0] == 0.0

    # ensure that h(t) == 0 for this point
    assert panel[0][1] == 1.0


def test_impulse_response_ones_after_tzero():
    """Test u(t>0) == 1"""

    # generate impulse response panel
    panel = unit_step.generate_impulse_response(-1.0, 1.0, 0.1)

    # ensure that t < 0 for this point
    assert panel[-1][0] > 0.0

    # ensure that h(t) == 1 for this point
    assert panel[-1][1] == 1.0


def test_gain_spectrum():
    """Test specific points along gain spectrum"""

    # generate a gain-spectrum panel
    panel = unit_step.generate_gain_spectrum(0.0, 10.1, 0.1)

    # test at f == 0
    assert np.isclose(panel[0][0].real, 0.0)
    assert panel[0][1].real == np.inf

    # test at f == 0.1
    assert np.isclose(panel[1][0].real, 0.1)
    assert np.isclose(panel[1][1].real, 1.0 / (2.0 * np.pi * 0.1))

    # test at f == 1.
    assert np.isclose(panel[10][0].real, 1.0)
    assert np.isclose(panel[10][1], 1.0 / (2.0 * np.pi * 1.0))

    # test at f == 10.
    assert np.isclose(panel[100][0].real, 10.0)
    assert np.isclose(panel[100][1].real, 1.0 / (2.0 * np.pi * 10.0))
