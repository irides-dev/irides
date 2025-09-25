"""
-------------------------------------------------------------------------------

Unit tests for the .containers.transfer_functions module of spqf.resources.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np
from numpy.polynomial import Polynomial

from irides.design.analog import mbox_level as dsgn_a_l_mbox
from irides.design.analog import bessel_level as dsgn_a_l_bssl
from irides.tools import digital_design_tools as dd_tools
from irides.resources.containers.transfer_functions import (
    IIRTransferFunctionDiscreteTime,
    MboxTransferFunctionDiscreteTime,
)
from irides.resources.core_enumerations import FilterDesignType


# noinspection SpellCheckingInspection
class TestTransferFunctions(unittest.TestCase):
    """
    Unit tests against the transfer-function containers.
    """

    def test_iir_transfer_function_object_ctor_and_interface_using_an_example(
        self,
    ):
        # from digital bessel filter, m = 3, mu = 10
        filter_order = 3
        num_frags = np.array(
            [Polynomial([0.05375797]), Polynomial([0.20722714])]
        )
        den_frags = np.array(
            [
                Polynomial([1.0, -1.63851049, 0.69226845]),
                Polynomial([1.0, -0.79277286]),
            ]
        )

        Hz = IIRTransferFunctionDiscreteTime(filter_order, num_frags, den_frags)

        # order
        self.assertEqual(filter_order, Hz.order())

        # initial value
        initial_value_ans = 0.011140110375305804
        initial_value_test = Hz.initial_value()
        self.assertAlmostEqual(initial_value_test, initial_value_ans, 4)

        # DC value
        dc_value_ans = 1.0000001860189636
        dc_value_test = Hz.value(1.0)
        self.assertAlmostEqual(dc_value_test, dc_value_ans, 4)

        # again using array of z values
        z_values = np.ones(4, dtype=float)  # just the DC value repeated
        dc_values_ans = dc_value_ans * np.ones_like(z_values)
        dc_value_test = Hz.value(z_values)
        self.assertAlmostEqual(
            np.linalg.norm(dc_values_ans - dc_value_test), 0.0, 4
        )

    def test_mbox_level_transfer_function_z_one_values_across_params(self):
        """Tests H_level(z=+1) across filter orders and N_stage = {even|odd}"""

        # setup
        filter_orders = dsgn_a_l_mbox.get_valid_filter_orders()
        n_stages = np.array([10, 11])
        z_plus_one = 1.0
        test_answer = 1.0
        precision = 6
        z_ones_array = np.ones(4, dtype=float)  # just the DC value repeated
        test_answers = test_answer * np.ones_like(z_ones_array)

        # run tests
        for m in filter_orders:
            for n_stage in n_stages:
                Hmbox = MboxTransferFunctionDiscreteTime(m, n_stage)
                self.assertEqual(m, Hmbox.order())

                with self.subTest(msg=f"order: {m}, n_stage: {n_stage}"):
                    self.assertAlmostEqual(
                        test_answer, Hmbox.value(z_plus_one), precision
                    )

                # again using array of z values
                test_values = Hmbox.value(z_ones_array)
                self.assertAlmostEqual(
                    np.linalg.norm(test_answers - test_values), 0.0, precision
                )

    def test_mbox_level_transfer_function_z_neg_one_values_across_params(self):
        """Tests H_level(z=-1) across filter orders and N_stage = {even|odd}"""

        # setup
        filter_orders = dsgn_a_l_mbox.get_valid_filter_orders()
        n_stages = np.array([10, 11])
        z_neg_one = -1.0
        test_answer = 0.0
        precision = 6

        # run tests
        for m in filter_orders:
            for n_stage in n_stages:
                Hmbox = MboxTransferFunctionDiscreteTime(m, n_stage)
                self.assertEqual(m, Hmbox.order())

                with self.subTest(msg=f"order: {m}, n_stage: {n_stage}"):
                    if (n_stage % 2) == 0:  # even
                        self.assertAlmostEqual(
                            test_answer, Hmbox.value(z_neg_one), precision
                        )

                    else:  # odd
                        self.assertNotEqual(
                            test_answer, Hmbox.value(z_neg_one), precision
                        )

    def test_mbox_level_transfer_function_hn0_values_across_params(self):
        """Tests h_level[0] values across orders and n_stage = {even|odd}"""

        # set up (answers are in effect regression values)
        filter_orders = dsgn_a_l_mbox.get_valid_filter_orders()
        n_stages = np.array([10, 11])
        precision = 6
        hn0_answers = np.array(
            [
                [
                    np.power(10.0, -v)
                    for v in np.arange(1, filter_orders.shape[0] + 1)
                ],
                [
                    9.09090909e-2,
                    8.26446281e-3,
                    7.51314801e-4,
                    6.83013455e-5,
                    6.20921323e-6,
                    5.64473930e-7,
                    5.13158118e-8,
                    4.66507380e-9,
                ],
            ]
        )

        # test
        for i, n_stage in enumerate(n_stages):
            for j, m in enumerate(filter_orders):
                Hmbox = MboxTransferFunctionDiscreteTime(m, n_stage)
                hn0_test = Hmbox.initial_value()
                self.assertAlmostEqual(hn0_answers[i, j], hn0_test, precision)

    def test_mbox_across_classes_z_one_and_neg_one_values_across_params(self):
        """Tests H(1) and H(-1) for l|s|c filter classes with N_stage = odd.

        We have
            H_class = (1 - zeta)^{0|1|2} H_level
        so
            H_class(z=-1) = (2**{0|1|2}} H_level(z=-1).

        This test is for the mbox filter type.
        """

        # setup
        filter_orders = dsgn_a_l_mbox.get_valid_filter_orders()
        filter_classes = np.array(
            [
                FilterDesignType.SLOPE,
                FilterDesignType.CURVE,
            ]
        )
        n_stage = 11  # odd
        z_pos_one = 1.0
        z_neg_one = -1.0
        precision = 6

        for filter_class in filter_classes:
            for m in filter_orders:
                H_level = MboxTransferFunctionDiscreteTime(
                    m, n_stage, FilterDesignType.LEVEL
                )
                H_level_z_neg_one = H_level.value(z_neg_one)

                H_class = MboxTransferFunctionDiscreteTime(
                    m, n_stage, filter_class
                )
                H_class_z_neg_one = H_class.value(z_neg_one)

                # test H(z=1) value
                self.assertAlmostEqual(0.0, H_class.value(z_pos_one))

                with self.subTest(
                    msg=f"class: {filter_class.name.lower()}, m: {m}"
                ):
                    self.assertAlmostEqual(
                        H_class_z_neg_one,
                        H_level_z_neg_one * (2**filter_class.value),
                        precision,
                    )

    def test_bssl_across_classes_z_one_and_neg_one_values_across_params(self):
        """Tests H(1) and H(-1) for l|s|c filter classes with N_stage = odd.

        We have
            H_class = (1 - zeta)^{0|1|2} H_level
        so
            H_class(z=-1) = (2**{0|1|2}} H_level(z=-1).

        This test is for the Bessel filter type, but exercises the same
        mechanics used for the polyema filter type.
        """

        # setup
        filter_orders = dsgn_a_l_bssl.get_valid_filter_orders()
        filter_classes = np.array(
            [
                FilterDesignType.SLOPE,
                FilterDesignType.CURVE,
            ]
        )
        mu = 10
        z_pos_one = 1.0
        z_neg_one = -1.0
        precision = 6

        for filter_class in filter_classes:
            for m in filter_orders:
                H_level = dd_tools.construct_transfer_function(
                    mu, m, dsgn_a_l_bssl, filter_class=FilterDesignType.LEVEL
                )
                H_level_z_neg_one = H_level.value(z_neg_one)

                H_class = dd_tools.construct_transfer_function(
                    mu, m, dsgn_a_l_bssl, filter_class=filter_class
                )
                H_class_z_neg_one = H_class.value(z_neg_one)

                # test H(z=1) value
                self.assertAlmostEqual(0.0, H_class.value(z_pos_one))

                with self.subTest(
                    msg=f"class: {filter_class.name.lower()}, m: {m}"
                ):
                    self.assertAlmostEqual(
                        H_class_z_neg_one,
                        H_level_z_neg_one * (2**filter_class.value),
                        precision,
                    )
