"""
-------------------------------------------------------------------------------

Unit tests for the .design_tools module of spqf.tools.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.tools import design_tools


# noinspection SpellCheckingInspection,PyPep8Naming
class TestDesignTools(unittest.TestCase):
    """
    Unit tests to verify the correct operation of design_tools
    under src.spqf.tools.
    """

    def test_get_minimum_valid_filter_order_is_correct(self):
        """Quick sanity test"""

        self.assertEqual(
            design_tools.get_minimum_valid_filter_order(
                design_tools.FilterDesignType.LEVEL
            ),
            1,
        )
        self.assertEqual(
            design_tools.get_minimum_valid_filter_order(
                design_tools.FilterDesignType.SLOPE
            ),
            2,
        )
        self.assertEqual(
            design_tools.get_minimum_valid_filter_order(
                design_tools.FilterDesignType.CURVE
            ),
            3,
        )

    def test_get_valid_filter_orders_with_nonstrict(self):
        """Quick sanity test"""

        ans_level = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        ans_slope = np.array([2, 3, 4, 5, 6, 7, 8])
        ans_curv = np.array([3, 4, 5, 6, 7, 8])

        self.assertSequenceEqual(
            list(
                design_tools.get_valid_filter_orders(
                    design_tools.FilterDesignType.LEVEL
                )
            ),
            list(ans_level),
        )
        self.assertSequenceEqual(
            list(
                design_tools.get_valid_filter_orders(
                    design_tools.FilterDesignType.SLOPE
                )
            ),
            list(ans_slope),
        )
        self.assertSequenceEqual(
            list(
                design_tools.get_valid_filter_orders(
                    design_tools.FilterDesignType.CURVE
                )
            ),
            list(ans_curv),
        )

    def test_get_valid_filter_orders_with_strict(self):
        """Quick sanity test"""

        ans_level = np.array([2, 3, 4, 5, 6, 7, 8])
        ans_slope = np.array([3, 4, 5, 6, 7, 8])
        ans_curv = np.array([4, 5, 6, 7, 8])

        self.assertSequenceEqual(
            list(
                design_tools.get_valid_filter_orders(
                    design_tools.FilterDesignType.LEVEL, strict=True
                )
            ),
            list(ans_level),
        )
        self.assertSequenceEqual(
            list(
                design_tools.get_valid_filter_orders(
                    design_tools.FilterDesignType.SLOPE, strict=True
                )
            ),
            list(ans_slope),
        )
        self.assertSequenceEqual(
            list(
                design_tools.get_valid_filter_orders(
                    design_tools.FilterDesignType.CURVE, strict=True
                )
            ),
            list(ans_curv),
        )
