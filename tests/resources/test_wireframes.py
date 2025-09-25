"""
-------------------------------------------------------------------------------

Unit tests for the .containers.wireframes module of spqf.sources.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.resources.core_enumerations import FilterDesignType
from irides.resources.containers.wireframes import WireframeContinuousTime


# noinspection SpellCheckingInspection
class TestWireframeContinuousTimeContainer(unittest.TestCase):
    """
    Unit tests against the wireframes container
    """

    def test_constructor_does_not_throw(self):
        """Tests clean instantiation."""

        with self.subTest(msg="level ctor"):
            try:
                WireframeContinuousTime(FilterDesignType.LEVEL, np.array([1.0]))
            except RuntimeError:
                self.fail(msg="level wireframe ctor is broken.")

        with self.subTest(msg="slope ctor"):
            try:
                WireframeContinuousTime(
                    FilterDesignType.SLOPE, np.array([1.0, 2.0])
                )
            except RuntimeError:
                self.fail(msg="slope wireframe ctor is broken.")

        with self.subTest(msg="curvature ctor"):
            try:
                WireframeContinuousTime(
                    FilterDesignType.CURVE, np.array([1.0, 2.0, 3.0])
                )
            except RuntimeError:
                self.fail(msg="curvature wireframe ctor is broken.")

    def test_constructor_throws_for_unset_design_type(self):

        with self.assertRaises(RuntimeError):
            WireframeContinuousTime(FilterDesignType.UNKNOWN, np.array([]))

    def test_constructor_throws_for_inconsistent_config(self):

        with self.subTest(msg="level ctor"):
            positions = np.array([1.0, 2.0, 3.0])
            # noinspection PyUnusedLocal
            for e in [1, 2]:
                with self.assertRaises(RuntimeError):
                    WireframeContinuousTime(FilterDesignType.LEVEL, positions)
                np.delete(positions, 0)

            with self.subTest(msg="slope ctor"):
                positions = np.array([1.0, 2.0, 3.0])
                # noinspection PyUnusedLocal
                for e in [1, 2]:
                    with self.assertRaises(RuntimeError):
                        WireframeContinuousTime(
                            FilterDesignType.LEVEL, positions
                        )
                    np.delete(positions, 0)
                    np.delete(positions, 0)

            with self.subTest(msg="curvature ctor"):
                positions = np.array([1.0, 2.0])
                # noinspection PyUnusedLocal
                for e in [1, 2]:
                    with self.assertRaises(RuntimeError):
                        WireframeContinuousTime(
                            FilterDesignType.LEVEL, positions
                        )
                    np.delete(positions, 0)

    def test_level_wireframe_has_correct_weights_by_spot_check(self):

        wf = WireframeContinuousTime(FilterDesignType.LEVEL, np.array([1.0]))
        weights_theo = np.array([1.0])
        precision = 4
        rebase = np.power(10, precision)
        self.assertSequenceEqual(
            list(rebase * np.round(weights_theo, precision)),
            list(rebase * np.round(wf.weights, precision)),
        )

    def test_slope_wireframe_has_correct_weights_by_spot_check(self):

        wf = WireframeContinuousTime(
            FilterDesignType.SLOPE, np.array([1.0, 3.0])
        )
        weights_theo = np.array([1.0, -1.0]) / 2.0
        precision = 4
        rebase = np.power(10, precision)
        self.assertSequenceEqual(
            list(rebase * np.round(weights_theo, precision)),
            list(rebase * np.round(wf.weights, precision)),
        )

    def test_curvature_wireframe_has_correct_weights_by_spot_check(self):

        wf = WireframeContinuousTime(
            FilterDesignType.CURVE, np.array([1.0, 2.0, 3.0])
        )
        weights_theo = np.array([1.0, -2.0, 1.0]) / np.power(2.0 / 2.0, 2)
        precision = 4
        rebase = np.power(10, precision)
        self.assertSequenceEqual(
            list(rebase * np.round(weights_theo, precision)),
            list(rebase * np.round(wf.weights, precision)),
        )

    def test_level_wireframe_has_correct_change_interval_and_midpoint(self):

        wf = WireframeContinuousTime(FilterDesignType.LEVEL, np.array([1.0]))
        ci_theo = 0.0
        ci_test = wf.change_interval
        self.assertAlmostEqual(ci_test, ci_theo, 4)

        mp_theo = 1.0
        mp_test = wf.mid_point
        self.assertAlmostEqual(mp_test, mp_theo, 4)

    def test_slope_wireframe_has_correct_change_interval_and_midpoint(self):

        wf = WireframeContinuousTime(
            FilterDesignType.SLOPE, np.array([1.0, 3.0])
        )
        ci_theo = 2.0
        ci_test = wf.change_interval
        self.assertAlmostEqual(ci_test, ci_theo, 4)

        mp_theo = 2.0
        mp_test = wf.mid_point
        self.assertAlmostEqual(mp_test, mp_theo, 4)

    def test_curvature_wireframe_has_correct_change_interval_and_midpoint(self):

        wf = WireframeContinuousTime(
            FilterDesignType.CURVE, np.array([1.0, 2.0, 3.0])
        )
        ci_theo = 2.0
        ci_test = wf.change_interval
        self.assertAlmostEqual(ci_test, ci_theo, 4)

        mp_theo = 2.0
        mp_test = wf.mid_point
        self.assertAlmostEqual(mp_test, mp_theo, 4)

    def test_instance_comparison(self):

        wf_1 = WireframeContinuousTime(
            FilterDesignType.SLOPE, np.array([1.0, 2.0])
        )
        wf_2 = WireframeContinuousTime(
            FilterDesignType.SLOPE, np.array([1.0, 2.0])
        )
        self.assertTrue(wf_1 == wf_2)
