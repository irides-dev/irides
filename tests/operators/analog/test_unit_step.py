"""
-------------------------------------------------------------------------------

Unit tests for the .unit_step module of spqf.operators.analog.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.operators.analog import unit_step


# noinspection SpellCheckingInspection,PyPep8Naming
class TestUnitStep(unittest.TestCase):
    """
    Unit tests to verify the correct operation of the unit-step
    time series.
    """

    def test_unit_step_has_correct_time_bounds(self):
        """Test that the time bounds are as expected."""

        # defs
        t_end = 0.95
        dt = 0.01

        # create
        ts = unit_step.generate_impulse_response(t_end, dt)

        # test
        precision = 4
        self.assertAlmostEqual(ts[0, 0], 0.0, precision)
        self.assertAlmostEqual(ts[-1, 0], 0.95, precision)

    def test_unit_step_has_all_one_entries(self):
        """Test that all series values are one."""

        # defs
        t_end = 0.95
        dt = 0.01

        # create
        ts = unit_step.generate_impulse_response(t_end, dt)

        # test
        theo_seq = np.ones_like(ts[:, 0])
        test_seq = ts[:, 1]
        self.assertSequenceEqual(list(test_seq), list(theo_seq))
