"""
-------------------------------------------------------------------------------

Unit tests for the .tools.analog_to_digital_conversion_tools module of
spqf.resources.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.tools import digital_design_tools as dd_tools
from irides.tools.digital_design_tools import FrequencyBand
from irides.tools import analog_to_digital_conversion_tools as a2d_tools
from irides.resources.containers.discrete_sequence import DiscreteSequence

from irides.design.analog import bessel_level as dsgn_a_l_bssl
from irides.design.digital import bessel_level as dsgn_d_l_bssl
from irides.design.analog import polyema_level as dsgn_a_l_pema
from irides.design.digital import polyema_level as dsgn_d_l_pema
from irides.filter_signatures.digital import polyema_level as fsig_d_l_pema


class TestDigitalDesignTools(unittest.TestCase):
    """Unit tests against the digital-design toolset"""

    def setUp(self) -> None:

        # setup
        self.filter_order = 8
        self.mu = 10
        self.zplane_poles = a2d_tools.zplane_poles_from_analog_design(
            dsgn_a_l_bssl, self.mu, self.filter_order
        )
        self.signs_apply = np.array([-1, 1, 1, -1, -1, -1, 1, 1])
        self.zplane_poles_signed = self.signs_apply * self.zplane_poles

    def test_identify_frequency_bands_correctly_identifies_freq_bands(self):
        """Tests that this fcxn correctly identifies the freq band roots"""

        # call fcxn
        freq_band_test = dd_tools.identify_frequency_bands_of_zplane_roots(
            self.zplane_poles_signed
        )

        # setup answers
        freq_band_ans = np.array(
            [
                FrequencyBand.LOW if v > 0 else FrequencyBand.HIGH
                for v in self.signs_apply
            ]
        )

        # test
        for bands in zip(freq_band_test, freq_band_ans):
            self.assertEqual(bands[0].value, bands[1].value)

    def test_frequency_band_converter_converts_to_low_freq(self):
        """Tests that output is all low freq given randomly signed input"""

        zplane_poles_converted = dd_tools.convert_frequency_band(
            self.zplane_poles_signed, FrequencyBand.LOW
        )

        # test
        for zplane_pair in zip(zplane_poles_converted, self.zplane_poles):

            self.assertAlmostEqual(zplane_pair[0], zplane_pair[1], 4)

    def test_frequency_band_converter_converts_to_high_freq(self):
        """Tests that output is all high freq given randomly signed input"""

        zplane_poles_converted = dd_tools.convert_frequency_band(
            self.zplane_poles_signed, FrequencyBand.HIGH
        )

        # test
        hf_sign_adj = -1
        for zplane_pair in zip(zplane_poles_converted, self.zplane_poles):
            self.assertAlmostEqual(
                zplane_pair[0], hf_sign_adj * zplane_pair[1], 4
            )

    def test_qz_coefs_from_zplane_poles(self):
        """Spot test that the Q(z) polynomial result is correct"""

        # setup
        order = 5
        mu = 10
        qz_coefs_answer = np.array(
            [1.0, -3.84617264, 5.98296367, -4.70092154, 1.86416669, -0.29826791]
        )

        # fetch zplane poles
        zplane_poles = a2d_tools.zplane_poles_from_analog_design(
            dsgn_a_l_bssl, mu, order
        )

        # fetch Q(z) coefs
        qz_coefs_test = dd_tools.qz_coefs_from_zplane_poles(zplane_poles)

        for i in range(qz_coefs_answer.shape[0]):

            self.assertAlmostEqual(qz_coefs_test[i], qz_coefs_answer[i], 4)

    def test_zir_coefs_from_pema_design(self):
        """Spot test zir-coef construction for pema filters"""

        # setup
        mu = 20
        order = 8
        zir_coefs_answer = np.array(
            [
                0.0,
                1010.82884165,
                -4824.41038059,
                9888.54301613,
                -11281.5718934,
                7735.93501262,
                -3187.8853074,
                730.9084479,
                -71.91883416,
            ]
        )

        zir_coefs_test = dd_tools.zir_coefs_at_fixed_horizon(
            dsgn_a_l_pema, dsgn_d_l_pema, fsig_d_l_pema, mu, order
        )
        print(zir_coefs_test)

        for i in range(zir_coefs_answer.shape[0]):

            self.assertAlmostEqual(zir_coefs_test[i], zir_coefs_answer[i], 4)

    def test_hzir_for_variable_horizon_alpha_ic_zero_reproduces_pema_gain(self):
        """Tests that the h_zir[0] entry leads to unit gain for unit ic's"""

        self.common_hzir_for_variable_horizon_alpha_ic_zero_reproduces_gain(
            dsgn_a_l_pema, dsgn_d_l_pema
        )

    def test_hzir_for_variable_horizon_sum_matches_mu_value_for_pema(self):
        """Tests that sum(h_zir) = mu"""

        self.common_hzir_for_variable_horizon_sum_matches_mu_value(
            dsgn_a_l_pema, dsgn_d_l_pema
        )

    def test_hzir_for_variable_horizon_alpha_ic_zero_reproduces_bessel_gain(
        self,
    ):
        """Tests that the h_zir[0] entry leads to a unit gain for unit ic's"""

        self.common_hzir_for_variable_horizon_alpha_ic_zero_reproduces_gain(
            dsgn_a_l_bssl, dsgn_d_l_bssl
        )

    def test_hzir_for_variable_horizon_sum_matches_mu_value_for_bessel(self):
        """Tests that sum(h_zir) = mu"""

        self.common_hzir_for_variable_horizon_sum_matches_mu_value(
            dsgn_a_l_bssl, dsgn_d_l_bssl
        )

    def common_hzir_for_variable_horizon_alpha_ic_zero_reproduces_gain(
        self,
        design_analog_level,
        design_digital_level,
    ):
        """Common test that h_zir[0] entry leads to a unit gain for unit ic's"""

        # setup
        mu = 20
        orders = design_analog_level.get_valid_filter_orders()
        H_zir_dc_gain_answer = 1.0

        # iter over orders
        for i, order in enumerate(orders):

            # build mock stream of all ones
            yn_mock_stream = np.ones(order + 1)

            # make the impulse response, pull off the zeroth order term
            ds_hzir = dd_tools.hzir_for_variable_horizon(
                yn_mock_stream,
                design_analog_level,
                design_digital_level,
                mu,
                order,
            )

            # fetch the associated gain adjustment
            gain_adj = design_digital_level.gain_adjustment(mu, order)

            # reconstruct the sum of a_coefs
            h_zir_zero = ds_hzir.v_axis[0]
            a_coef_sum = -gain_adj * h_zir_zero

            # claim
            H_zir_dc_gain_test = gain_adj / (a_coef_sum + 1)

            # test
            self.assertAlmostEqual(
                H_zir_dc_gain_test, H_zir_dc_gain_answer, places=4
            )

    def common_hzir_for_variable_horizon_sum_matches_mu_value(
        self,
        design_analog_level,
        design_digital_level,
    ):
        """Common test that sum(h_zir) = mu for unit ic's"""

        # setup
        mu = 20
        orders = design_analog_level.get_valid_filter_orders()
        hzir_sum_answer = mu

        # iter over orders
        for i, order in enumerate(orders):
            # build mock stream of all ones
            yn_mock_stream = np.ones(order + 1)

            # make the impulse response, pull off the zeroth order term
            ds_hzir = dd_tools.hzir_for_variable_horizon(
                yn_mock_stream,
                design_analog_level,
                design_digital_level,
                mu,
                order,
            )

            # claim
            hzir_sum_test = ds_hzir.v_axis.sum()

            # test
            self.assertAlmostEqual(hzir_sum_test, hzir_sum_answer, places=4)

    def test_upsample_sequence_correctly_inserts_zeros_into_sequence(self):
        """Upsampling a sequence is to insert zeros between adjacent points"""

        # setup an original sequence
        n_start = 0
        n_end = 13
        ds_orig = DiscreteSequence(n_start, n_end)
        ds_orig.v_axis = np.arange(n_end)

        # upsample
        upsample_strides = np.array([1, 2, 3, 4])
        for upsample_stride in upsample_strides:

            ds_up = dd_tools.upsample_sequence(ds_orig, upsample_stride)

            with self.subTest(
                msg="upsample stride: {0}".format(upsample_stride)
            ):

                # test lengths
                self.assertEqual(ds_up.len, ds_orig.len * upsample_stride)

                # test that original values are in correct places
                self.assertSequenceEqual(
                    list(ds_orig.v_axis), list(ds_up.v_axis[::upsample_stride])
                )

                # test that zeros exist in between original values
                for i in np.arange(upsample_stride - 1):
                    self.assertSequenceEqual(
                        list(np.zeros(ds_orig.len)),
                        list(ds_up.v_axis[(i + 1) :: upsample_stride]),
                    )
