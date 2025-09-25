"""
-------------------------------------------------------------------------------

Parameterized unit tests for mirrored level-filter signatures.

-------------------------------------------------------------------------------
"""

import numpy as np
import parameterized

import unittest

from irides.design.analog import polyema_level as dsgn_alg_l_pema
from irides.design.analog import bessel_level as dsgn_alg_l_bssl
from irides.design.digital import polyema_level as dsgn_dig_l_pema
from irides.design.digital import bessel_level as dsgn_dig_l_bssl

from irides.filter_signatures.digital import polyema_level as fsig_dig_l_pema
from irides.filter_signatures.digital import bessel_level as fsig_dig_l_bssl

from irides.resources.core_enumerations import FilterDesignType
from irides.tools.digital_design_tools import FrequencyBand
from irides.tools import digital_design_tools as dd_tools


def make_impulse_response(this, filter_order, freq_band):
    """Helper that builds h[n] consistently"""

    # setup
    n_start = 0
    mu_min = this.dsgn_dig_l.minimum_mu_value(filter_order)
    mu_trg = 5 * mu_min
    n_end = 20 * mu_trg

    ds = this.f_sig.generate_impulse_response(
        n_start, n_end, mu_trg, filter_order, freq_band
    )

    return ds, mu_trg


# noinspection SpellCheckingInspection,PyPep8Naming
# setup to iter over filter types
@parameterized.parameterized_class(
    ("f_type", "dsgn_alg_l", "dsgn_dig_l", "f_sig"),
    [
        ("pema", dsgn_alg_l_pema, dsgn_dig_l_pema, fsig_dig_l_pema),
        ("bessel", dsgn_alg_l_bssl, dsgn_dig_l_bssl, fsig_dig_l_bssl),
    ],
)
class TestMirroredLevelFilters(unittest.TestCase):
    """Parameterized, common unit tests for mirrored level filters"""

    def test_high_freq_impulse_response_gains_match_xfer_fcxn_gains(self):
        """Tests that sum(h[n]) == H(1) for high-frequency responses"""

        # setup
        filter_orders = self.dsgn_alg_l.get_valid_filter_orders(strict=False)

        for filter_order in filter_orders:
            # make hf impulse response
            ds, mu_trg = make_impulse_response(
                self, filter_order, FrequencyBand.HIGH
            )

            # make transfer function
            tf = dd_tools.construct_transfer_function(
                mu_trg,
                filter_order,
                self.dsgn_alg_l,
                FilterDesignType.LEVEL,
                FrequencyBand.HIGH,
            )

            # empirical and transfer-function gains
            gain_hn = np.sum(ds.v_axis)
            gain_tf = tf.value(1)

            # test
            with self.subTest(msg="order: {0}".format(filter_order)):
                self.assertAlmostEqual(gain_hn / gain_tf, 1.0, 4)

    def test_dealternating_of_high_freq_impulse_response_matches_low_freq(self):
        """Tests that (-1)^n h_hf[n] == h_lf[n]"""

        # setup
        filter_orders = self.dsgn_alg_l.get_valid_filter_orders(strict=False)

        for filter_order in filter_orders:
            # make lf and hf impulse responses
            ds_lf, mu_trg = make_impulse_response(
                self, filter_order, FrequencyBand.LOW
            )
            ds_hf, mu_trg = make_impulse_response(
                self, filter_order, FrequencyBand.HIGH
            )

            # make alternating sequence
            alt_seq = np.power(-1, ds_lf.n_axis)

            # down-convert hf to lf
            ds_hf.v_axis *= alt_seq

            # test || alt * hn_hf - hn_lf || < err
            with self.subTest(msg="order: {0}".format(filter_order)):
                self.assertAlmostEqual(
                    np.linalg.norm(ds_hf.v_axis - ds_lf.v_axis),
                    0.0,
                    4,
                )
