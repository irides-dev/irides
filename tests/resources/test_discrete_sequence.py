"""
-------------------------------------------------------------------------------

Unit tests for the .containers.discrete_sequence module of spqf.resources.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.resources.containers.discrete_sequence import DiscreteSequence


# noinspection SpellCheckingInspection
class TestDiscreteSequenceObject(unittest.TestCase):
    """
    Unit tests against the discrete-sequence container.
    """

    def test_constructor_with_default_column_count(self):
        """Tests the constructor using the default #col setting"""

        # setup
        n_start = 0
        n_end = 20
        ds = DiscreteSequence(n_start, n_end)

        # test
        self.assertEqual(ds.n_axis.ndim, 1)
        self.assertEqual(ds.v_axes.ndim, 2)
        self.assertEqual(ds.v_axes.shape[1], 1)

    def test_constructor_with_user_defined_column_count(self):
        """Tests the constructor using a user-def'd number of cols"""

        # setup
        n_start = 0
        n_end = 20
        n_cols = 4
        ds = DiscreteSequence(n_start, n_end, n_cols)

        # test
        self.assertEqual(ds.n_axis.ndim, 1)
        self.assertEqual(ds.v_axes.ndim, 2)
        self.assertEqual(ds.v_axes.shape[1], n_cols)

    def test_copy_classmethod_copies_an_instance(self):
        """Tests that a copy of an instance can be made using copy()"""

        # setup
        ds_org = DiscreteSequence(0, 6)
        ds_org.v_axis = -2 * np.arange(0, 6)
        ds_copy = DiscreteSequence.copy(ds_org)

        # test
        self.assertSequenceEqual(list(ds_org.n_axis), list(ds_copy.n_axis))
        self.assertSequenceEqual(list(ds_org.v_axis), list(ds_copy.v_axis))

    def test_naxis_and_vaxes_have_same_length(self):
        """Tests that n-axis and v-axes are conformal"""

        # setup
        n_start = -4
        n_end = 13
        ds = DiscreteSequence(n_start, n_end)

        len_naxis = ds.n_axis.shape[0]
        len_vaxes = ds.v_axes.shape[0]

        # test
        self.assertEqual(len_naxis, len_vaxes)

    def test_step_size_returns_correct_value(self):
        """Tests that the step-size interface is correct"""

        # setup
        n_start = -4
        n_end = 13
        ds = DiscreteSequence(n_start, n_end)

        ss_ans = 1
        ss_test = ds.step_size

        # test
        self.assertEqual(ss_test, ss_ans)

    def test_len_returns_the_correct_length_of_internal_axes(self):
        """Tests that len instance property returns correct length"""

        # setup
        n_start = -4
        n_end = 13
        ds = DiscreteSequence(n_start, n_end)

        len_ans = ds.n_axis.shape[0]
        len_test = ds.len

        # test
        self.assertEqual(len_test, len_ans)

    def test_n_axis_and_zero_index_value(self):
        """Tests that n_axis is correctly def'd and i_n_zero is correct"""

        # test 1
        n_start = 0
        n_end = 20
        ds = DiscreteSequence(n_start, n_end)

        with self.subTest("n-start = 0"):

            self.assertEqual(ds.n_axis[0], n_start)
            self.assertEqual(ds.n_axis[-1], n_end - 1)
            self.assertEqual(ds.i_n_zero, 0)

        # test 2
        n_start = -10
        ds = DiscreteSequence(n_start, n_end)

        with self.subTest("n-start = -10"):

            self.assertEqual(ds.n_axis[0], n_start)
            self.assertEqual(ds.i_n_zero, 0 - n_start)

    def test_v_axes_is_assignable(self):
        """Tests that ds.v_axes is assignable in both getter and setter form"""

        # setup
        n_start = -10
        n_end = 20
        ds = DiscreteSequence(n_start, n_end)

        # assignment against getter
        v_axis_setting = -np.arange(n_start, n_end - ds.i_n_zero)
        ds.v_axes[ds.i_n_zero :] = v_axis_setting[:, np.newaxis]

        with self.subTest("assignment against getter"):
            v1_test = ds.v_axes[: ds.i_n_zero, 0]
            v1_ans = np.zeros_like(v1_test)
            self.assertSequenceEqual(list(v1_test), list(v1_ans))
            self.assertSequenceEqual(
                list(ds.v_axes[ds.i_n_zero :, 0]), list(v_axis_setting)
            )

        # assignment against setter
        v_axis_setting = -2.0 * np.arange(n_start, n_end)
        ds.v_axes = v_axis_setting

        with self.subTest("assignment against setter"):
            self.assertSequenceEqual(list(ds.v_axes), list(v_axis_setting))

    def test_v_axis_returns_v_axes_zeroth_column(self):
        """Tests that .v_axis returns .v_axes[:, 0]"""

        # setup
        n_start = 0
        n_end = 20
        n_cols = 4
        ds = DiscreteSequence(n_start, n_end, n_cols)

        # load columns
        for col in range(n_cols):
            ds.v_axes[:, col] = np.power(-2, col) * ds.n_axis

        # test zeroth-column size and values
        self.assertEqual(ds.v_axis.shape[0], ds.v_axes.shape[0])
        self.assertSequenceEqual(list(ds.v_axis), list(ds.n_axis))

    def test_v_axis_is_assignable(self):
        """Tests that ds.v_axis = <values> assigns values"""

        # setup
        n_start = 0
        n_end = 20
        n_cols = 4
        ds = DiscreteSequence(n_start, n_end, n_cols)

        # assign
        asgn_array = -2 * ds.n_axis
        ds.v_axis = asgn_array

        # test
        self.assertSequenceEqual(list(ds.v_axis), list(asgn_array))

    def test_to_ndarray_correctly_merges_n_and_v_axes(self):
        """Tests that to_ndarray() correctly merges n- and v-axes and dims"""

        # setup
        n_start = -10
        n_end = 20
        n_cols = 3
        ds = DiscreteSequence(n_start, n_end, n_cols)
        array = ds.to_ndarray()

        # n-axis answer
        n_axis_answer = np.arange(n_start, n_end, dtype="int64")

        # test
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.shape[0], ds.n_axis.shape[0])
        self.assertEqual(array.shape[1], n_cols + 1)
        self.assertSequenceEqual(list(ds.n_axis), list(n_axis_answer))

    def test_index_at_correctly_returns_an_index_or_none(self):
        """Tests that index_at(n) returns the correct index or None.

        Expecting index_test to look like (within the np.ndarray)
            [None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, None, None, None].
        """

        # setup
        n_start = -3
        n_end = 5
        ds = DiscreteSequence(n_start, n_end)
        ns_wide_array = np.arange(n_start - 3, n_end + 3)

        # answers
        correct_indices = np.arange(0, n_end - n_start)

        # create test results
        test_indices = np.array([ds.index_at(n) for n in ns_wide_array])

        # test
        ns_truth = np.isin(ns_wide_array, ds.n_axis)
        self.assertSequenceEqual(
            list(test_indices[ns_truth]), list(correct_indices)
        )
        self.assertTrue(np.all(test_indices[ns_truth == False]) == None)
