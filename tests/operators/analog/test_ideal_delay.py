"""
-------------------------------------------------------------------------------

Unit tests for the .ideal_delay module of spqf.operators.analog.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.operators.analog import ideal_delay


# noinspection SpellCheckingInspection,PyPep8Naming
class TestIdealDelay(unittest.TestCase):
    """
    Unit tests to verify the correct operation of the ideal-delay
    time series.
    """

    def test_ideal_delay_series_for_tau_equals_zero(self):
        """The simplest time series: one record only"""

        # defs
        tau = 0.0
        dt = 0.01

        # create
        ts = ideal_delay.generate_impulse_response(tau, dt)

        # test ts length
        self.assertEqual(ts.shape[0], 1)

        # test ts value
        precision = 4
        self.assertAlmostEqual(ts[0, 1], 1.0 / dt, precision)

    def test_ideal_delay_series_for_tau_as_multiple_of_dt(self):
        """Multiple records, tau should be the last entry."""

        # defs
        dt = 0.01
        tau_sample = 99
        tau = tau_sample * dt

        # create
        ts = ideal_delay.generate_impulse_response(tau, dt)

        # test length
        self.assertEqual(ts.shape[0], tau_sample + 1)

        # test ts zeros
        theo_seq = np.zeros(tau_sample)
        test_seq = ts[:tau_sample, 1]
        self.assertSequenceEqual(list(test_seq), list(theo_seq))

        # test delta location and value
        precision = 4
        self.assertGreater(ts[tau_sample, 1], 0.0)
        self.assertAlmostEqual(ts[tau_sample, 1], 1.0 / dt, precision)

    def test_ideal_delay_series_for_tau_as_fractional_multiple_of_dt(self):
        """Multiple records, tau should be the 2nd-to-last entry."""

        """
        notes: 
            ceil(tau / dt) = 100,
            len(ts) = 100 + 1
        """

        # defs
        tau = 0.994
        dt = 0.01

        # create
        ts = ideal_delay.generate_impulse_response(tau, dt)

        # test length
        tau_sample = 100
        self.assertEqual(ts.shape[0], tau_sample + 1)

        # test that last sample is zero
        precision = 4
        self.assertAlmostEqual(ts[-1, 1], 0.0, precision)

        # test that 2nd to last sample is a delta
        self.assertAlmostEqual(ts[-2, 1], 1.0 / dt, precision)

        # test that all earlier samples are zero
        theo_seq = np.zeros(tau_sample - 1)
        test_seq = ts[: tau_sample - 1, 1]
        self.assertSequenceEqual(list(test_seq), list(theo_seq))
