"""
-------------------------------------------------------------------------------

Scaffold unit tests

-------------------------------------------------------------------------------
"""

import unittest
from irides import get_name


class TestProject(unittest.TestCase):
    """Scaffold test cases"""

    def test_get_project_name(self):
        """Test that get_name prints the correct message."""

        # def names
        name_test = get_name()
        name_ans = "I am irides"

        # Check the captured output
        self.assertEqual(name_test, name_ans)
