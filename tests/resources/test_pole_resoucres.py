"""
-------------------------------------------------------------------------------

Unit tests for the .containers.pole_resources module of spqf.resources.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.resources.containers import pole_resources
from irides.design.analog import bessel_level as bssl_dsgn


# noinspection SpellCheckingInspection
class TestPoleResources(unittest.TestCase):
    """
    Unit tests against the poles container and free functions
    """

    def test_constructor_does_not_throw_for_ndarray_input(self):
        """Test no-throw for proper input"""

        poles = bssl_dsgn.designs(4)["poles"]

        try:
            pole_resources.PoleContainer(poles)
        except RuntimeError:
            self.fail(msg="pole-container ctor should accept ndarray")

    def test_constructor_does_throw_for_non_ndarray_input(self):
        """Test does-throw for bad input"""

        dsgns = bssl_dsgn.designs(4)  # note: this is a dict

        with self.assertRaises(RuntimeError):
            pole_resources.PoleContainer(dsgns)

    def test_cartesian_pole_interface(self):
        """Tests the cartesian-pole interface"""

        poles = bssl_dsgn.designs(4)["poles"]
        pole_container = pole_resources.PoleContainer(poles)
        cart_poles = pole_container.cartesian_poles()

        self.assertAlmostEqual(np.linalg.norm(cart_poles - poles), 0.0, 4)

    def test_polar_pole_interface(self):
        """Tests the polar-pole interface"""

        # setup
        poles = bssl_dsgn.designs(4)["poles"]
        pole_container = pole_resources.PoleContainer(poles)

        with self.subTest(msg="positive real axis"):

            # answer
            radii = np.abs(poles)
            angles = np.arctan2(np.imag(poles), np.real(poles))
            polar_ans = np.array([radii, angles]).T

            # candidate
            polar_cand = pole_container.polar_poles(True)

            self.assertAlmostEqual(
                np.linalg.norm(polar_ans - polar_cand), 0.0, 4
            )

        with self.subTest(msg="negative real axis"):

            # answer
            radii = np.abs(poles)
            angles = np.arctan2(np.imag(poles), -np.real(poles))
            polar_ans = np.array([radii, angles]).T

            # candidate
            polar_cand = pole_container.polar_poles(False)

            # test
            self.assertAlmostEqual(
                np.linalg.norm(polar_ans - polar_cand), 0.0, 4
            )
