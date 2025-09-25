"""
-------------------------------------------------------------------------------

Unit tests for the .test_signals.discrete.test_signals module of spqf.sources.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.resources.core_enumerations import TestSignalType
from irides.test_signals.digital import test_signals
from irides.resources.containers.discrete_sequence import DiscreteSequence


# noinspection SpellCheckingInspection
class TestTheTestSignals(unittest.TestCase):
    """Unit tests against the test signals."""

    n_start = -4
    n_end = 20
    precision = 4

    def test_impulse(self):

        ds = test_signals.generate_test_signal(
            self.n_start, self.n_end, TestSignalType.IMPULSE
        )

        with self.subTest(msg="n>=0 span"):
            self.assertAlmostEqual(ds.v_axis[ds.i_n_zero], 1.0, self.precision)
            test_values = ds.v_axis[1 + ds.i_n_zero :]
            self.assertSequenceEqual(
                list(test_values), list(np.zeros_like(test_values)),
            )

        test_for_zeros_on_negative_indices(self, ds)

    def test_unit_step(self):

        ds = test_signals.generate_test_signal(
            self.n_start, self.n_end, TestSignalType.STEP
        )

        with self.subTest(msg="n>=0 span"):
            test_values = ds.v_axis[ds.i_n_zero :]
            self.assertSequenceEqual(
                list(test_values), list(np.ones_like(test_values))
            )

        test_for_zeros_on_negative_indices(self, ds)

    def test_ramp(self):

        ds = test_signals.generate_test_signal(
            self.n_start, self.n_end, TestSignalType.RAMP
        )

        with self.subTest(msg="n>=0 span"):
            soln_seq = ds.n_axis[ds.i_n_zero :] + 1.0
            test_values = ds.v_axis[ds.i_n_zero :]
            self.assertSequenceEqual(list(test_values), list(soln_seq))

        test_for_zeros_on_negative_indices(self, ds)

    def test_parabola(self):

        ds = test_signals.generate_test_signal(
            self.n_start, self.n_end, TestSignalType.PARABOLA
        )

        with self.subTest(msg="n>=0 span"):
            n_axis = ds.n_axis[ds.i_n_zero :]
            soln_seq = 0.5 * (n_axis + 1.0) * (n_axis + 2.0)
            test_values = ds.v_axis[ds.i_n_zero :]
            self.assertSequenceEqual(list(test_values), list(soln_seq))

        test_for_zeros_on_negative_indices(self, ds)

    def test_alternating(self):

        ds = test_signals.generate_test_signal(
            self.n_start, self.n_end, TestSignalType.ALTERNATING
        )

        with self.subTest(msg="n>=0 span"):
            # n even -> +1
            test_values = ds.v_axis[ds.i_n_zero :: 2]
            self.assertSequenceEqual(
                list(test_values), list(np.ones_like(test_values)),
            )
            # n odd -> -1
            test_values = ds.v_axis[1 + ds.i_n_zero :: 2]
            self.assertSequenceEqual(
                list(test_values), list(-1.0 * np.ones_like(test_values)),
            )

        test_for_zeros_on_negative_indices(self, ds)

    def test_white_noise(self):

        n_end_whitenoise = 1000
        ds = test_signals.generate_test_signal(
            self.n_start, n_end_whitenoise, TestSignalType.WHITE_NOISE
        )

        with self.subTest(msg="n>=0 span"):
            test_values = ds.v_axis[ds.i_n_zero :]
            self.assertTrue(np.abs(np.mean(test_values)) < 0.1)
            self.assertTrue(0.9 <= np.std(test_values) <= 1.1)

        test_for_zeros_on_negative_indices(self, ds)

    def test_unknown(self):

        ds = test_signals.generate_test_signal(
            self.n_start, self.n_end, TestSignalType.UNKNOWN
        )

        with self.subTest(msg="n>=0 span"):
            test_values = ds.v_axis[ds.i_n_zero :]
            self.assertSequenceEqual(
                list(test_values), list(np.zeros_like(test_values)),
            )

        test_for_zeros_on_negative_indices(self, ds)


def test_for_zeros_on_negative_indices(this, ds: DiscreteSequence):

    values_on_neg_indices = ds.v_axis[: ds.i_n_zero]
    with this.subTest(msg="n<0 span"):
        this.assertSequenceEqual(
            list(values_on_neg_indices),
            list(np.zeros_like(values_on_neg_indices)),
        )
