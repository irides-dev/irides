"""
-------------------------------------------------------------------------------

Unit tests for the .digital.damped_oscillator module of spqf.filter_signatures.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.design.analog import bessel_level as dsgn_alg_bssl
from irides.tools import analog_to_digital_conversion_tools as a2d_tools
from irides.tools import design_tools
from irides.tools import impulse_response_tools

from irides.design.digital import damped_oscillator as dsgn_dig_dosc
from irides.filter_signatures.digital import (
    damped_oscillator as fsig_dig_dosc,
)


# noinspection SpellCheckingInspection
class TestDigitalFilterSignatureDampedOscillator(unittest.TestCase):
    """
    Unit tests against the digital damped-oscillator filter.
    """

    def test_impulse_response_initial_values(self):
        """Tests the realized h[0..2] values against the design values"""

        # setup arrays over which to perform tests
        filter_orders = dsgn_alg_bssl.get_valid_filter_orders(strict=True)
        tau_v = np.array([1.0, 100.0])

        # iter on scale
        for tau in tau_v:

            # iter on filter order
            for filter_order in filter_orders:

                # retrieve concrete design
                bessel_design = dsgn_alg_bssl.designs(filter_order)
                stages = bessel_design["stages"]

                n_start = 0
                n_end = 3

                # iter over distinct stages (avoiding +/- imag-part pairs)
                for i, stage in enumerate(stages):

                    # reject if ema stage
                    if stage["type"] == design_tools.StageTypes.EMA.value:
                        continue

                    # identify the +imag pole index, extract
                    i_poles = stage["indices"]
                    splane_poles = bessel_design["poles"][i_poles]

                    # convert to zplane
                    zplane_poles = (
                        a2d_tools.convert_analog_poles_to_digital_poles(
                            tau, splane_poles
                        )
                    )

                    # design answer
                    hn_answers = dsgn_dig_dosc.initial_hn_values(zplane_poles)[
                        0
                    ]

                    # impulse-response test
                    ds = fsig_dig_dosc.generate_impulse_response(
                        n_start, n_end, zplane_poles
                    )
                    hn_tests = np.array(
                        [
                            ds.v_axis[ds.i_n_zero],
                            ds.v_axis[ds.i_n_zero + 1],
                            ds.v_axis[ds.i_n_zero + 2],
                        ]
                    )

                    # test
                    for index in range(3):
                        with self.subTest(
                            (
                                "tau: {0}, order: {1}, stage: {2}, "
                                "stage-type: {3}, h[{4}]"
                            ).format(tau, filter_order, i, stage["type"], index)
                        ):
                            self.assertAlmostEqual(
                                hn_answers[index], hn_tests[index], 4
                            )

    def test_impulse_response_moments_and_full_widths(self):
        """Tests the realized h[n] moments and FW against design values"""

        # setup arrays over which to perform tests
        filter_orders = dsgn_alg_bssl.get_valid_filter_orders(strict=True)
        tau_v = np.array([20.0])
        moments = np.array([0, 1, 2])

        # iter on scale
        for tau in tau_v:

            # iter on filter order
            for filter_order in filter_orders:

                # retrieve concrete design
                bessel_design = dsgn_alg_bssl.designs(filter_order)
                stages = bessel_design["stages"]

                n_start = 0

                # iter over distinct stages (avoiding +/- imag-part pairs)
                for i, stage in enumerate(stages):

                    # reject if ema stage
                    if stage["type"] == design_tools.StageTypes.EMA.value:
                        continue

                    # identify the +imag pole index, extract
                    i_poles = stage["indices"]
                    splane_poles = bessel_design["poles"][i_poles]

                    # convert to zplane
                    zplane_poles = (
                        a2d_tools.convert_analog_poles_to_digital_poles(
                            tau, splane_poles
                        )
                    )

                    # h[n] generation
                    # compute n_end dynamically based on z-plane pole coord
                    m1_inferred = a2d_tools.first_moment_from_zplane_pole(
                        zplane_poles
                    )[0]
                    n_end = 20 * int(np.ceil(m1_inferred / 20.0) * 20.0)

                    # impulse-response test
                    ds = fsig_dig_dosc.generate_impulse_response(
                        n_start, n_end, zplane_poles
                    )

                    # iter and test on moments
                    hn_answers = dsgn_dig_dosc.moment_values(zplane_poles)[0]
                    for moment in moments:

                        hn_test = impulse_response_tools.calculate_impulse_response_moment(
                            ds.to_ndarray(), moment
                        )
                        hn_ans = hn_answers[moment]

                        # test
                        with self.subTest(
                            (
                                "moment: tau: {0}, order: {1}, stage: {2}, "
                                "stage-type: {3}"
                            ).format(tau, filter_order, i, stage["type"])
                        ):
                            self.assertAlmostEqual(
                                np.abs(hn_test / hn_ans - 1.0), 0.0, 4
                            )

                    # calculate full width
                    fw_test = impulse_response_tools.calculate_impulse_response_full_width(
                        ds.to_ndarray()
                    )
                    fw_ans = dsgn_dig_dosc.full_width_value(zplane_poles)[0]

                    # test
                    with self.subTest(
                        (
                            "fw: tau: {0}, order: {1}, stage: {2}, "
                            "stage-type: {3}"
                        ).format(tau, filter_order, i, stage["type"])
                    ):
                        if np.isnan(fw_ans):
                            self.assertTrue(np.isnan(fw_test))
                        else:
                            self.assertAlmostEqual(
                                np.abs(fw_test / fw_ans - 1.0), 0.0, 4
                            )
