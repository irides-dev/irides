"""
-------------------------------------------------------------------------------

Unit tests for the .test_signals.analog.test_signals module of spqf.sources.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.resources.core_enumerations import TestSignalType
from irides.test_signals.analog import test_signals


# noinspection SpellCheckingInspection
class TestTheTestSignals(unittest.TestCase):
    """Unit tests against the test signals."""

    t_start = -0.5
    t_end = 3.0
    dt = 0.01
    w = 4.0 * (2.0 * np.pi)
    precision = 4

    def test_impulse(self):

        ts, i_t0_plus = test_signals.generate_test_signal(
            self.t_start, self.t_end, self.dt, TestSignalType.IMPULSE
        )

        with self.subTest(msg="t>=0 timespan"):
            self.assertAlmostEqual(
                ts[i_t0_plus, 1], 1.0 / self.dt, self.precision
            )
            self.assertSequenceEqual(
                list(ts[(1 + i_t0_plus) :, 1]),
                list(np.zeros_like(ts[(1 + i_t0_plus) :, 1])),
            )

        test_for_zeros_for_negative_time(self, ts, i_t0_plus)

    def test_step(self):

        ts, i_t0_plus = test_signals.generate_test_signal(
            self.t_start, self.t_end, self.dt, TestSignalType.STEP
        )

        with self.subTest(msg="t>=0 timespan"):
            self.assertSequenceEqual(
                list(ts[i_t0_plus:, 1]), list(np.ones_like(ts[i_t0_plus:, 1]))
            )

        test_for_zeros_for_negative_time(self, ts, i_t0_plus)

    def test_ramp(self):

        ts, i_t0_plus = test_signals.generate_test_signal(
            self.t_start, self.t_end, self.dt, TestSignalType.RAMP
        )

        t_axis = ts[i_t0_plus:, 0]
        x_axis = t_axis
        with self.subTest(msg="t>=0 timespan"):
            self.assertSequenceEqual(list(ts[i_t0_plus:, 1]), list(x_axis))

        test_for_zeros_for_negative_time(self, ts, i_t0_plus)

    def test_parabola(self):

        ts, i_t0_plus = test_signals.generate_test_signal(
            self.t_start, self.t_end, self.dt, TestSignalType.PARABOLA
        )

        t_axis = ts[i_t0_plus:, 0]
        x_axis = 0.5 * np.power(t_axis, 2)
        with self.subTest(msg="t>=0 timespan"):
            self.assertSequenceEqual(list(ts[i_t0_plus:, 1]), list(x_axis))

        test_for_zeros_for_negative_time(self, ts, i_t0_plus)

    def test_alternating(self):

        ts, i_t0_plus = test_signals.generate_test_signal(
            self.t_start,
            self.t_end,
            self.dt,
            TestSignalType.ALTERNATING,
            self.w,
        )

        t_axis = ts[i_t0_plus:, 0]
        x_axis = np.cos(self.w * t_axis)
        with self.subTest(msg="t>=0 timespan"):
            self.assertSequenceEqual(list(ts[i_t0_plus:, 1]), list(x_axis))

        test_for_zeros_for_negative_time(self, ts, i_t0_plus)

    def test_white_noise(self):

        ts, i_t0_plus = test_signals.generate_test_signal(
            self.t_start, self.t_end, self.dt, TestSignalType.WHITE_NOISE
        )

        with self.subTest(msg="t>=0 timespan"):
            self.assertTrue(np.abs(np.mean(ts[i_t0_plus:, 1])) < 0.25, 1)
            self.assertTrue(0.9 <= np.std(ts[i_t0_plus:, 1]) <= 1.1, 1)

        test_for_zeros_for_negative_time(self, ts, i_t0_plus)

    def test_unknown(self):

        ts, i_t0_plus = test_signals.generate_test_signal(
            self.t_start, self.t_end, self.dt, TestSignalType.UNKNOWN
        )

        with self.subTest(msg="t>=0 timespan"):
            self.assertSequenceEqual(
                list(ts[i_t0_plus:, 1]), list(np.zeros_like(ts[i_t0_plus:, 1]))
            )
        test_for_zeros_for_negative_time(self, ts, i_t0_plus)


def test_for_zeros_for_negative_time(this, ts: np.ndarray, i_t0_plus: int):

    with this.subTest(msg="t<0 timespan"):
        this.assertSequenceEqual(
            list(ts[:i_t0_plus, 1]), list(np.zeros_like(ts[:i_t0_plus, 1]))
        )
