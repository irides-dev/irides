"""
-------------------------------------------------------------------------------

Unit tests for the .tools.analog_to_digital_conversion_tools module of
spqf.resources.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.tools import analog_to_digital_conversion_tools as a2d_tools
from irides.design.analog import bessel_level as dsgn_alg_bssl
from irides.resources.special_designs import chebyshev as dsgn_alg_cheb


# noinspection SpellCheckingInspection
class TestAnalogToDigitalConversionTools(unittest.TestCase):
    """Unit tests against the a-to-d conversion toolset"""

    # noinspection PyPep8Naming
    def test_convert_analog_poles_to_digital_poles(self):
        """Tests the exponential relationship between s- and z-plane poles"""

        # setup
        splane_poles = dsgn_alg_bssl.designs(5)["poles"]
        tau = 13.0  # nice prime number
        T = 1.0

        # compute test and answer
        ans_poles = np.exp(splane_poles * T / tau)
        test_poles = a2d_tools.convert_analog_poles_to_digital_poles(
            tau, splane_poles
        )

        # test as a whole
        self.assertAlmostEqual(np.linalg.norm(ans_poles - test_poles), 0.0, 4)

    def test_convert_digital_poles_to_analog_poles(self):
        """Tests the logarithmic relationship b/w s- and z-plane poles"""

        # setup
        splane_poles = dsgn_alg_bssl.designs(5)["poles"]
        tau = 13.0  # nice prime number

        # scale s-plane poles
        splane_poles_scaled = splane_poles / tau
        # splane_poles_scaled = a2d_tools.scale_analog_poles(tau, splane_poles)

        # convert original s-plane poles to z-plane
        zplane_poles = a2d_tools.convert_analog_poles_to_digital_poles(
            tau, splane_poles
        )

        # convert back
        splane_poles_from_z = a2d_tools.convert_digital_poles_to_analog_poles(
            zplane_poles
        )

        # test as a whole
        self.assertAlmostEqual(
            np.linalg.norm(splane_poles_from_z - splane_poles_scaled), 0.0, 4
        )

    def test_first_moment_from_zplane_pole(self):
        """Test that M1 values inferred from z-plane poles are ~ correct"""

        # setup
        splane_poles = dsgn_alg_bssl.designs(5)["poles"]
        tau = 13.0  # nice prime number

        # convert to z-plane poles
        zplane_poles = a2d_tools.convert_analog_poles_to_digital_poles(
            tau, splane_poles
        )

        # compute answer and test
        p_abs = np.abs(zplane_poles)
        answers = p_abs / (1.0 - p_abs)
        test_values = a2d_tools.first_moment_from_zplane_pole(zplane_poles)

        # test as a whole
        self.assertAlmostEqual(np.linalg.norm(answers - test_values), 0.0, 4)

    def test_zplane_poles_from_analog_design(self):
        """Tests that zplane poles are correct for a given design and scale"""

        # setup
        order = 5
        mu = 10
        zplane_poles_answer = np.array(
            [
                0.74519252 + 0.0j,
                0.75559658 - 0.10690179j,
                0.75559658 + 0.10690179j,
                0.79489348 - 0.23548136j,
                0.79489348 + 0.23548136j,
            ]
        )

        # fetch zplane poles
        zplane_poles_test = a2d_tools.zplane_poles_from_analog_design(
            dsgn_alg_bssl, mu, order
        )

        for i in range(zplane_poles_answer.shape[0]):

            self.assertAlmostEqual(
                zplane_poles_test[i], zplane_poles_answer[i], 4
            )

    def test_solve_for_tau_from_mu_and_splane_poles(self):
        """Tests tau values calc'd from mu against pre-calcd solns"""

        # setup
        filter_orders = dsgn_alg_bssl.get_valid_filter_orders()
        mu_target = 20.0

        # pre-calc'd answers
        tau_answers = {
            1: 20.495934314288192,
            2: 20.988088481920833,
            3: 21.47671771396506,
            4: 21.96205117215398,
            5: 22.444295503966426,
            6: 22.923637786005948,
            7: 23.400247992230494,
            8: 23.874281077404106,
        }

        for filter_order in filter_orders:

            splane_poles = dsgn_alg_bssl.designs(filter_order)["poles"]
            tau_test = a2d_tools.solve_for_tau_from_mu_and_splane_poles(
                mu_target, splane_poles
            )

            with self.subTest("order: {0}".format(filter_order)):

                self.assertAlmostEqual(tau_answers[filter_order], tau_test, 4)

    def test_solve_for_tau_with_error_case_of_unsuitable_rho_value(self):
        """Test that the fcxn throws for a bad rho-value"""

        splane_pole = -1.0 + 0.0j
        rho_invalid_low = -1.0
        rho_invalid_high = 2.0
        rho_invalid_edge = 1.0

        # fmt: off
        with self.subTest("rho < 0"):
            with self.assertRaises(RuntimeError):
                a2d_tools.\
                    solve_for_tau_to_place_splane_pole_onto_zplane_real_axis(
                        splane_pole, rho_invalid_low
                    )

        with self.subTest("rho > 1"):
            with self.assertRaises(RuntimeError):
                a2d_tools.\
                    solve_for_tau_to_place_splane_pole_onto_zplane_real_axis(
                        splane_pole, rho_invalid_high
                    )

        with self.subTest("rho == 1.0"):
            with self.assertRaises(RuntimeError):
                a2d_tools.\
                    solve_for_tau_to_place_splane_pole_onto_zplane_real_axis(
                        splane_pole, rho_invalid_edge
                    )
        # fmt: on

    # noinspection PyPep8Naming
    def test_solve_for_tau_with_edge_case_of_real_valued_splane_pole(self):
        """Test for analytic soln when s-plane pole is purely real"""

        splane_pole = -1.0 + 0.0j
        rho_intermediate = 0.5
        rho_zero = 0.0
        T = 1

        with self.subTest("0 < rho < 1"):

            tau_ans = (
                T * np.real(np.array([splane_pole])) / np.log(rho_intermediate)
            )
            # fmt: off
            tau_test = a2d_tools.\
                solve_for_tau_to_place_splane_pole_onto_zplane_real_axis(
                    splane_pole, rho_intermediate
                )
            # fmt: on

            self.assertAlmostEqual(tau_ans, tau_test, 4)

        with self.subTest("rho == 0"):

            tau_ans = 0.0
            # fmt: off
            tau_test = a2d_tools.\
                solve_for_tau_to_place_splane_pole_onto_zplane_real_axis(
                    splane_pole, rho_zero
                )
            # fmt: on

            self.assertAlmostEqual(tau_ans, tau_test, 4)

    # noinspection PyPep8Naming
    def test_solve_for_tau_with_edge_case_of_rho_equals_zero(self):
        """Tests that tau leads to the Nyquist condition"""

        # setup
        splane_poles = np.array([-1.0 + 0.0j, -1.0 + 1.0j, -1.0 - 2.0j])
        rho = 0.0
        T = 1

        # iter
        for splane_pole in splane_poles:

            # compute the answer value
            tau_ans = T * np.abs(np.imag(splane_pole)) / (np.pi / 2.0)

            # compute the test value
            # fmt: off
            tau_test = a2d_tools.\
                solve_for_tau_to_place_splane_pole_onto_zplane_real_axis(
                    splane_pole, rho
                )
            # fmt: on

            # test
            self.assertAlmostEqual(tau_ans, tau_test, 4)

    def test_solve_for_tau_using_solver_and_precalcd_solutions(self):
        """Tests tau values from the solver against pre-calcd answers"""

        # setup
        filter_orders = np.array([7, 8])  # for 1 and 0 real-valued pole cases
        rho = 0.75

        # pre-calculated answers
        tau_answers = {
            7: {
                6: 13.28988618788072,
                4: 15.543270198144251,
                2: 16.85263966787174,
                0: 17.28222693,
            },
            8: {
                1: 19.49097044353998,
                3: 18.729543887674094,
                5: 17.178182704113578,
                7: 14.772501387790333,
            },
        }

        # iter
        for filter_order in filter_orders:

            # pull design
            design = dsgn_alg_bssl.designs(filter_order)

            for stage in design["stages"]:

                # look up pole and calculate the test value
                pole_index = stage["indices"][0]
                splane_pole = design["poles"][pole_index]
                # fmt: off
                tau_test = a2d_tools.\
                    solve_for_tau_to_place_splane_pole_onto_zplane_real_axis(
                        splane_pole, rho
                    )
                # fmt: on

                # look up the answer value
                tau_answer = tau_answers[filter_order][pole_index]

                # test
                self.assertAlmostEqual(tau_answer, tau_test, 4)

    def test_solve_for_min_tau_min_rho_tuple_throws_for_less_than_3_poles(self):
        """Confirm that the call throws for #poles == 1 and 2"""

        # setup
        filter_orders = np.array([1, 2])

        # iter
        for filter_order in filter_orders:

            # select constellation
            splane_poles = dsgn_alg_bssl.designs(filter_order)["poles"]

            with self.subTest("order == {0}".format(filter_order)):
                with self.assertRaises(RuntimeError):
                    # fmt: off
                    a2d_tools.\
                        solve_for_min_tau_min_rho_tuple_from_splane_poles(
                            splane_poles
                        )
                    # fmt: on

    def test_solve_for_tau_min_from_splane_poles_bessel_poles(self):
        """Confirm minimum tau values across orders [1, 8]"""

        # setup
        filter_orders = dsgn_alg_bssl.get_valid_filter_orders()

        # pre-calc'd answers
        tau_min_answers = np.array(
            [
                1.4426950408889634,
                2.3950324087595174,
                3.3487313481270853,
                4.302913367060106,
                5.25732894539093,
                6.2118775638638635,
                7.1665102003888395,
                8.121199813729849,
            ]
        )
        answer_offset = -1  # to account for index misalignment

        # iter
        for filter_order in filter_orders:

            # select constellation
            splane_poles = dsgn_alg_bssl.designs(filter_order)["poles"]

            # call solver
            res = a2d_tools.solve_for_tau_min_from_splane_poles(splane_poles)

            # test
            with self.subTest("filter-order: {0}".format(filter_order)):

                self.assertAlmostEqual(
                    tau_min_answers[filter_order + answer_offset],
                    res["tau_min"],
                    4,
                )

    def test_solve_for_min_tau_min_rho_tuple_for_bessel_poles(self):
        """Using Bessel constellations m: [3,8] test solns match expectation"""

        # setup
        filter_orders = np.arange(3, 8 + 1)

        # pre-calc'd answers
        tau_min_answers = np.array(
            [
                3.3524635697518392,
                4.303872342035766,
                5.254815648469996,
                6.209445016309111,
                7.16452125708545,
                8.116191794787076,
            ]
        )
        rho_min_answers = np.array(
            [
                0.5004742630798171,
                0.5001033432038307,
                0.4997631152349994,
                0.49979560954069524,
                0.499848913807619,
                0.49965222163215545,
            ]
        )
        answer_offset = -3  # to account for index misalignment

        # iter
        for filter_order in filter_orders:

            # select constellation
            splane_poles = dsgn_alg_bssl.designs(filter_order)["poles"]

            # call solver
            # fmt: off
            res = a2d_tools.\
                solve_for_min_tau_min_rho_tuple_from_splane_poles(
                    splane_poles
                )
            # fmt: on

            # test
            with self.subTest("filter-order: {0}".format(filter_order)):

                self.assertAlmostEqual(
                    tau_min_answers[filter_order + answer_offset],
                    res["tau_min"],
                    4,
                )
                self.assertAlmostEqual(
                    rho_min_answers[filter_order + answer_offset],
                    res["rho_min"],
                    4,
                )

    def test_solve_for_min_tau_min_rho_tuple_for_chebyshev_poles(self,):
        """Using Cheby constellations m: [3,8] test solns match expectations

        What's interesting here is that the rho_min values are not all ~0.5,
        like the Bessel case. Moreover, rho_min ~0.975 for higher orders,
        so the Chebyshev constellations stress the solver more.
        """

        # setup higher-order filters
        filter_orders = np.arange(3, 8 + 1)

        # pre-calc'd answers
        tau_min_answers = np.array(
            [
                10.363081793670293,
                7.190630764580221,
                25.143545021533555,
                15.82103788950372,
                46.322351458530775,
                27.68017155568881,
            ]
        )
        rho_min_answers = np.array(
            [
                0.8988445632016356,
                0.9296789487599604,
                0.9551240135176623,
                0.9655583789312321,
                0.9748768034528569,
                0.9796212301148094,
            ]
        )
        answer_offset = -3  # to account for index misalignment

        # iter
        for filter_order in filter_orders:

            # select constellation
            splane_poles = dsgn_alg_cheb.designs(filter_order)["poles"]

            # call solver
            # fmt: off
            res = a2d_tools.\
                solve_for_min_tau_min_rho_tuple_from_splane_poles(
                    splane_poles
                )
            # fmt: on

            # test
            with self.subTest("filter-order: {0}".format(filter_order)):

                self.assertAlmostEqual(
                    tau_min_answers[filter_order + answer_offset],
                    res["tau_min"],
                    4,
                )
                self.assertAlmostEqual(
                    rho_min_answers[filter_order + answer_offset],
                    res["rho_min"],
                    4,
                )
