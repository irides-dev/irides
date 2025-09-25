"""
-------------------------------------------------------------------------------

Parameterized unit tests for inline-curvature filter designs.

-------------------------------------------------------------------------------
"""

import parameterized

import unittest
import numpy as np

from irides.design.analog import polyema_inline_curvature as design_pema
from irides.design.analog import bessel_inline_curvature as design_bessel
from irides.design.analog import mbox_inline_curvature as design_mbox
from irides.tools import design_tools

from tests.design.analog import common_tooling


# noinspection SpellCheckingInspection,PyPep8Naming
# setup to iter over filter types
@parameterized.parameterized_class(
    ("f_type", "design"),
    [("pema", design_pema), ("bessel", design_bessel), ("mbox", design_mbox)],
)
class TestInlineCurvatureDesigns(unittest.TestCase):
    """
    Parameterized, common unit tests for inline curvature filter designs.
    """

    def test_design_id_is_correct(self):
        """
        Tests that the correct design-id is coded.
        """

        design_id_theo = datastore(self.f_type, "design_id")
        common_tooling.design_id_test(self, self.design, design_id_theo)

    def test_design_type_is_correct(self):
        """Sanity test to confirm we have a slope filter type"""

        design_type_theo = design_tools.FilterDesignType.CURVE
        common_tooling.design_type_test(self, self.design, design_type_theo)

    def test_get_valid_filter_orders_is_as_expected(self):
        """Tests that the valid design orders match that of the reference."""

        valid_orders_theo = datastore(self.f_type, "valid_orders")
        common_tooling.loose_valid_filter_orders_are_as_expected(
            self, self.design, valid_orders_theo
        )

    def test_impulse_response_t0_value_is_correct_for_valid_orders(self):
        """
        Test that the h(t=0) values are correct, orders [3..8] only.
        """

        tau = 2.0
        dt = 0.001
        filter_orders = self.design.get_valid_filter_orders(strict=False)

        h_t0_theo = np.zeros(filter_orders.shape[0])
        h0_order_3_coef = datastore(
            self.f_type, "ir_t0_value", "h0_order_3_coef"
        )
        tau_power_law = datastore(self.f_type, "ir_t0_value", "tau_power_law")
        h_t0_theo[0] = h0_order_3_coef / np.power(tau, tau_power_law)

        common_tooling.impulse_response_t0_value_is_correct_for_valid_orders(
            self, self.design, h_t0_theo, tau, dt
        )

    def test_impulse_response_lobe_details_are_correct_for_valid_orders(self):
        """Tests that the two crossing times, and lobe imbalance, are right."""

        tau = 2.0
        txl_seq = datastore(self.f_type, "lobe_details", "txl") * tau
        txr_seq = datastore(self.f_type, "lobe_details", "txr") * tau
        imb_seq = np.array(datastore(self.f_type, "lobe_details", "imb"))

        # test crossings
        with self.subTest(msg="txl crossing"):
            common_tooling.impulse_response_lobe_details_are_correct(
                self, self.design, txl_seq, "tx1", tau
            )
        with self.subTest(msg="txr crossing"):
            common_tooling.impulse_response_lobe_details_are_correct(
                self, self.design, txr_seq, "tx2", tau
            )

        # lobe imbalance
        with self.subTest(msg="lobe imbalance"):
            common_tooling.impulse_response_lobe_details_are_correct(
                self, self.design, imb_seq, "imbalance", tau
            )

    def test_moment_value_generators_are_correct(self):
        """
        Test that the anomymous functions returned by moment_value_generator()
        are correct.
        """

        # defs
        tau = 2.0
        dt = 0.005
        M0 = 0
        M1 = 1
        M2 = 2
        strict = False

        # theo values as arrays
        m0_theo = datastore(self.f_type, "moment_values", "M0")
        m1_theo = datastore(self.f_type, "moment_values", "M1")
        m2_theo = datastore(self.f_type, "moment_values", "M2")

        # corrections (if necy)
        if self.f_type == "pema":
            m0_theo[0] += (27.0 / 2.0) * dt / np.power(tau, 3)

        with self.subTest(msg="M0"):
            # fmt: off
            common_tooling. \
                moment_value_generators_for_a_given_moment_are_correct(
                    self, self.design, m0_theo, M0, tau, dt, strict
                )
            # fmt: on

        with self.subTest(msg="M1"):
            # fmt: off
            common_tooling. \
                moment_value_generators_for_a_given_moment_are_correct(
                    self, self.design, m1_theo, M1, tau, dt, strict
                )
            # fmt: on

        with self.subTest(msg="M2"):
            # fmt: off
            common_tooling. \
                moment_value_generators_for_a_given_moment_are_correct(
                    self, self.design, m2_theo, M2, tau, dt, strict
                )
            # fmt: on

    def test_moment_value_generator_throws_on_out_of_bounds_moment(self):
        """
        Test that the target function throws.
        """

        self.assertRaises(
            IndexError, lambda: self.design.moment_value_generator(10)
        )

    def test_autocorrelation_peak_and_stride_values_are_correct(self):
        """
        Test that the sacf zero-lag and 5% stride values are correct.
        Only orders 2 and higher are tested.
        """

        tau = 2.0
        dt = 0.001
        strict = False

        sacf_pairs_theo: dict = datastore(self.f_type, "sacf_values")
        kh0_correction_coef = datastore(
            self.f_type, "kh0_discretization_correction"
        )

        # scale by tau and kh0
        kh0_power_law = datastore(self.f_type, "kh0_power_law")
        kh0_scale = 1.0 / np.power(tau, kh0_power_law)
        xi_scale = tau
        for m, v in sacf_pairs_theo.items():
            v["kh0"] += kh0_correction_coef[m] * dt / tau
            v["kh0"] *= kh0_scale
            v["xi_5pct"] *= xi_scale

        common_tooling.autocorrelation_peak_and_stride_values_are_correct(
            self, self.design, sacf_pairs_theo, tau, dt, strict
        )

    def test_autocorrelation_peak_and_stride_values_scale_with_tau(self):
        """
        Test that kh(0) scales with 1 / tau^5 and that xi_5pct scales with tau.
        """

        kh0_power_law = datastore(self.f_type, "kh0_power_law")

        common_tooling.autocorrelation_peak_and_stride_values_scale_with_tau(
            self, self.design, kh0_power_law
        )

    def test_scale_correlation_strides_are_correct(self):
        """Validates that the scale-decorrelation strides are correct."""

        # defs
        strides_theo = datastore(self.f_type, "sscf_values")

        # send to test
        common_tooling.scale_correlation_strides(
            self, self.design, strides_theo
        )

    def test_gain_phase_and_gd_at_dc_frequency(self):
        """
        Test that the gain, phase and group-delay are correct at DC.
        """

        tau = 2.0

        ans_theo_dict: dict = datastore(self.f_type, "dc_spectral_features")
        ans_theo_dict["group-delay"] *= tau

        common_tooling.gain_phase_gd_at_dc_frequency(
            self, self.design, ans_theo_dict, tau
        )

    def test_frequency_of_peak_gain_is_correct(self):
        """
        Tests frequency values at peak gain are correct.
        """

        tau = 2.0
        seq_theo = datastore(self.f_type, "freq_of_gain_peak", "values")
        tau_power_law = datastore(
            self.f_type, "freq_of_gain_peak", "tau_power_law"
        )
        freq_theo = seq_theo * np.power(tau, tau_power_law)

        common_tooling.peak_spectral_features(
            self,
            self.design,
            freq_theo,
            self.design.frequency_of_peak_gain,
            tau,
            "frequency",
        )

    def test_gain_at_peak_frequency_is_correct(self):
        """
        Tests gain values at peak-frequency values are correct.
        """

        tau = 2.0
        seq_theo = datastore(self.f_type, "gain_at_peak_freq", "values")
        tau_power_law = datastore(
            self.f_type, "gain_at_peak_freq", "tau_power_law"
        )
        gain_theo = seq_theo * np.power(tau, tau_power_law)

        common_tooling.peak_spectral_features(
            self,
            self.design,
            gain_theo,
            self.design.gain_at_peak_frequency,
            tau,
            "gain",
        )

    def test_phase_at_peak_frequency_is_correct(self):
        """
        Tests phase values at peak-frequency values are correct.
        """

        tau = 2.0
        seq_theo = datastore(self.f_type, "phase_at_peak_freq", "values")
        tau_power_law = datastore(
            self.f_type, "phase_at_peak_freq", "tau_power_law"
        )
        phase_theo = seq_theo * np.power(tau, tau_power_law)

        common_tooling.peak_spectral_features(
            self,
            self.design,
            phase_theo,
            self.design.phase_at_peak_frequency,
            tau,
            "phase",
        )

    def test_group_delay_at_peak_frequency_is_correct(self):
        """
        Tests group-delay values at peak-frequency values are correct.
        """

        tau = 2.0
        seq_theo = datastore(self.f_type, "gd_at_peak_freq", "values")
        tau_power_law = datastore(
            self.f_type, "gd_at_peak_freq", "tau_power_law"
        )
        group_delay_theo = seq_theo * np.power(tau, tau_power_law)

        common_tooling.peak_spectral_features(
            self,
            self.design,
            group_delay_theo,
            self.design.group_delay_at_peak_frequency,
            tau,
            "group-delay",
        )

    def test_wireframe_is_correct(self):
        """Validates the Fourier wireframe design."""

        # defs
        tau = 2.0
        wfs_theo = datastore(self.f_type, "wireframes")

        # send to test
        common_tooling.wireframe_curvature_features(
            self, self.design, wfs_theo, tau
        )

    def test_wavenumber_is_correct(self):
        """Validates the Fourier wireframe wavenumber."""

        # defs
        wns_theo = datastore(self.f_type, "wavenumbers")

        # send to test
        common_tooling.wavenumber_features(self, self.design, wns_theo)

    def test_uncertainty_product_is_correct(self):
        """Validates that the uncertainty products are correct."""

        # defs
        wns_theo = datastore(self.f_type, "uncertainty_products")

        # send to test
        common_tooling.uncertainty_products(self, self.design, wns_theo)


# noinspection SpellCheckingInspection
def datastore(f_type: str, test_name: str, details_key: str = None):
    """..."""

    # master data store
    data = {
        "pema": {
            "design_id": "pema-inline-curvature",
            "valid_orders": np.array([3, 4, 5, 6, 7, 8]),
            "ir_t0_value": {"h0_order_3_coef": 27.0, "tau_power_law": 3},
            "lobe_details": {
                "txl": np.array(
                    [0.195262, 0.316987, 0.400000, 0.460655, 0.507216, 0.544281]
                ),
                "txr": np.array(
                    [1.13807, 1.18301, 1.20000, 1.20601, 1.20707, 1.20572]
                ),
                "imb": np.array(
                    [2.90281, 2.29374, 2.02215, 1.86347, 1.75752, 1.68087]
                ),
            },
            "moment_values": {
                "M0": np.array([0.0, 0, 0, 0, 0, 0]),
                "M1": np.array([0.0, 0, 0, 0, 0, 0]),
                "M2": np.array([2.0, 2, 2, 2, 2, 2]),
            },
            "sacf_values": {
                3: {
                    "kh0": 45.5625,
                    "xi_5pct": 1.85385,
                    "residual_acf": 0.78433,
                },
                4: {
                    "kh0": 32.0000,
                    "xi_5pct": 1.69782,
                    "residual_acf": 2.24215,
                },
                5: {
                    "kh0": 36.6211,
                    "xi_5pct": 1.56878,
                    "residual_acf": 3.41986,
                },
                6: {
                    "kh0": 45.5625,
                    "xi_5pct": 1.46340,
                    "residual_acf": 4.35727,
                },
                7: {
                    "kh0": 57.4458,
                    "xi_5pct": 1.37610,
                    "residual_acf": 5.11265,
                },
                8: {
                    "kh0": 72.0000,
                    "xi_5pct": 1.30253,
                    "residual_acf": 5.73130,
                },
            },
            "kh0_discretization_correction": {
                3: 729.0 / 2.0,
                4: 0.0,
                5: 0.0,
                6: 0.0,
                7: 0.0,
                8: 0.0,
            },
            "kh0_power_law": 5,
            "sscf_values": np.array(
                [3.07384, 3.54070, 2.69320, 2.95804, 2.89426, 2.78014]
            ),
            "dc_spectral_features": {
                "gain": 0.0,
                "phase": np.pi,
                "group-delay": 1,
            },
            "freq_of_gain_peak": {
                "values": np.array(
                    [4.242641, 4.0, 4.082483, 4.242641, 4.427189, 4.618802]
                ),
                "tau_power_law": -1,
            },
            "gain_at_peak_freq": {
                "values": np.array(
                    [3.464102, 4.0, 4.64758, 5.333333, 6.036816, 6.75]
                ),
                "tau_power_law": -2,
            },
            "phase_at_peak_freq": {
                "values": np.array(
                    [0.275643, 0.0, -0.282003, -0.551286, -0.806006, -1.047198]
                ),
                "tau_power_law": 0,
            },
            "gd_at_peak_freq": {
                "values": np.array(
                    [0.333333, 0.5, 0.6, 0.666667, 0.714286, 0.75]
                ),
                "tau_power_law": 1,
            },
            "wireframes": {
                3: {"tposl": -0.0649696, "tneg": 0.675511, "tposr": 1.41599},
                4: {"tposl": 0.0, "tneg": 0.785398, "tposr": 1.57080},
                5: {"tposl": 0.069076, "tneg": 0.838606, "tposr": 1.60814},
                6: {"tposl": 0.129939, "tneg": 0.870420, "tposr": 1.61090},
                7: {"tposl": 0.182058, "tneg": 0.891672, "tposr": 1.60129},
                8: {"tposl": 0.226725, "tneg": 0.906900, "tposr": 1.58707},
            },
            "wavenumbers": np.array(
                [
                    0.50644018,
                    0.65761864,
                    0.70048551,
                    0.71784811,
                    0.72791012,
                    0.73608002,
                ]
            ),
            "uncertainty_products": np.array(
                [np.inf, 2.95804, 2.20479, 2.02073, 1.93649, 1.88807]
            ),
        },
        "bessel": {
            "design_id": "bessel-inline-curvature",
            "valid_orders": np.array([3, 4, 5, 6, 7, 8]),
            "ir_t0_value": {"h0_order_3_coef": 15.0, "tau_power_law": 3},
            "lobe_details": {
                "txl": np.array(
                    [0.267707, 0.428704, 0.529771, 0.597575, 0.645709, 0.681470]
                ),
                "txr": np.array(
                    [1.36620, 1.38202, 1.36538, 1.34178, 1.31829, 1.29685]
                ),
                "imb": np.array(
                    [2.17617, 1.62999, 1.39585, 1.26717, 1.18758, 1.13492]
                ),
            },
            "moment_values": {
                "M0": np.array([0.0, 0, 0, 0, 0, 0]),
                "M1": np.array([0.0, 0, 0, 0, 0, 0]),
                "M2": np.array([2.0, 2, 2, 2, 2, 2]),
            },
            "sacf_values": {
                3: {
                    "kh0": 22.5000,
                    "xi_5pct": 2.14021,
                    "residual_acf": 2.27118,
                },
                4: {"kh0": 22.5000, "xi_5pct": 2.16504, "residual_acf": 5.0},
                5: {"kh0": 34.9999, "xi_5pct": 1.99730, "residual_acf": 5.0},
                6: {"kh0": 55.9711, "xi_5pct": 1.78948, "residual_acf": 5.0},
                7: {"kh0": 86.4564, "xi_5pct": 1.61198, "residual_acf": 5.0},
                8: {"kh0": 127.745, "xi_5pct": 1.46765, "residual_acf": 5.0},
            },
            "kh0_discretization_correction": {
                3: 0.0,
                4: 0.0,
                5: 0.0,
                6: 0.0,
                7: 0.0,
                8: 0.0,
            },
            "kh0_power_law": 5,
            "sscf_values": np.array(
                [2.85810, 2.89022, 2.73510, 2.65474, 2.49171, 2.34451]
            ),
            "dc_spectral_features": {
                "gain": 0.0,
                "phase": np.pi,
                "group-delay": 1,
            },
            "freq_of_gain_peak": {
                "values": np.array(
                    [
                        3.096475897,  # m = 3
                        3.276767292,  # m = 4
                        3.675631324,  # m = 5
                        4.121800308,  # m = 6
                        4.568617084,  # m = 7
                        4.996291251,  # m = 8
                    ]
                ),
                "tau_power_law": -1,
            },
            "gain_at_peak_freq": {
                "values": np.array(
                    [
                        3.146314744,  # m = 3
                        4.286691821,  # m = 4
                        5.647797444,  # m = 5
                        7.109325059,  # m = 6
                        8.610656060,  # m = 7
                        10.12018502,  # m = 8
                    ]
                ),
                "tau_power_law": -2,
            },
            "phase_at_peak_freq": {
                "values": np.array(
                    [
                        0.3753478245,  # m = 3
                        -0.0295578201,  # m = 4
                        -0.4929680867,  # m = 5
                        -0.9637886427,  # m = 6
                        -1.4207345620,  # m = 7
                        -1.8524816260,  # m = 8
                    ]
                ),
                "tau_power_law": 0,
            },
            "gd_at_peak_freq": {
                "values": np.array(
                    [
                        0.5781508094,  # m = 3
                        0.8078464881,  # m = 4
                        0.9119181817,  # m = 5
                        0.9610324290,  # m = 6
                        0.9839163584,  # m = 7
                        0.9939684121,  # m = 8
                    ]
                ),
                "tau_power_law": 1,
            },
            "wireframes": {
                3: {"tposl": -0.121218, "tneg": 0.893353, "tposr": 1.90792},
                4: {"tposl": 0.0090204, "tneg": 0.967768, "tposr": 1.92652},
                5: {"tposl": 0.134118, "tneg": 0.988826, "tposr": 1.84353},
                6: {"tposl": 0.233827, "tneg": 0.996017, "tposr": 1.75821},
                7: {"tposl": 0.310977, "tneg": 0.998623, "tposr": 1.68627},
                8: {"tposl": 0.370771, "tneg": 0.999556, "tposr": 1.62834},
            },
            "wavenumbers": np.array(
                [
                    0.56484587,
                    0.75055984,
                    0.79969197,
                    0.82082189,
                    0.83757011,
                    0.84829495,
                ]
            ),
            "uncertainty_products": np.array(
                [np.inf, 2.73338, 2.09664, 1.95647, 1.89726, 1.86429]
            ),
        },
        "mbox": {
            "design_id": "mbox-inline-curvature",
            "valid_orders": np.array([3, 4, 5, 6, 7, 8]),
            "ir_t0_value": {"h0_order_3_coef": 27.0 / 8.0, "tau_power_law": 3},
            "lobe_details": {
                "txl": np.array(
                    [0.666667, 0.666667, 0.723607, 0.746901, 0.768765, 0.785215]
                ),
                "txr": np.array(
                    [1.33333, 1.33333, 1.27639, 1.25310, 1.23124, 1.21479]
                ),
                "imb": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            },
            "moment_values": {
                "M0": np.array([0.0, 0, 0, 0, 0, 0]),
                "M1": np.array([0.0, 0, 0, 0, 0, 0]),
                "M2": np.array([2.0, 2, 2, 2, 2, 2]),
            },
            "sacf_values": {
                3: {"kh0": 45.5625, "xi_5pct": 1.80000, "residual_acf": 5.0},
                4: {"kh0": 85.3333, "xi_5pct": 1.53584, "residual_acf": 5.0},
                5: {"kh0": 154.622, "xi_5pct": 1.36319, "residual_acf": 5.0},
                6: {"kh0": 251.100, "xi_5pct": 1.23588, "residual_acf": 5.0},
                7: {"kh0": 377.015, "xi_5pct": 1.13857, "residual_acf": 5.0},
                8: {"kh0": 534.810, "xi_5pct": 1.06105, "residual_acf": 5.0},
            },
            "kh0_discretization_correction": {
                3: 0.0,
                4: 0.0,
                5: 0.0,
                6: 0.0,
                7: 0.0,
                8: 0.0,
            },
            "kh0_power_law": 5,
            "sscf_values": np.array(
                [2.24613, 2.28198, 2.15137, 2.04397, 1.95425, 1.87980]
            ),
            "dc_spectral_features": {
                "gain": 0.0,
                "phase": np.pi,
                "group-delay": 1,
            },
            "freq_of_gain_peak": {
                "values": np.array(
                    [
                        3.972583340,  # m = 3
                        4.662244733,  # m = 4
                        5.263971483,  # m = 5
                        5.804415831,  # m = 6
                        6.299067303,  # m = 7
                        6.757846685,  # m = 8
                    ]
                ),
                "tau_power_law": -1,
            },
            "gain_at_peak_freq": {
                "values": np.array(
                    [
                        6.198217320,  # m = 3
                        8.400985824,  # m = 4
                        10.60556937,  # m = 5
                        12.81105670,  # m = 6
                        15.01705861,  # m = 7
                        17.22338119,  # m = 8
                    ]
                ),
                "tau_power_law": -2,
            },
            "phase_at_peak_freq": {
                "values": np.array(
                    [
                        -0.830990687,  # m = 3
                        -1.520652079,  # m = 4
                        -2.122378829,  # m = 5
                        -2.662823177,  # m = 6
                        -3.157474649,  # m = 7
                        -3.616254031,  # m = 8
                    ]
                ),
                "tau_power_law": 0,
            },
            "gd_at_peak_freq": {
                "values": np.array(
                    [
                        1.0,  # m = 3
                        1.0,  # m = 4
                        1.0,  # m = 5
                        1.0,  # m = 6
                        1.0,  # m = 7
                        1.0,  # m = 8
                    ]
                ),
                "tau_power_law": 1,
            },
            "wireframes": {
                3: {"tposl": 0.209181, "tneg": 1.0, "tposr": 1.79082},
                4: {"tposl": 0.326163, "tneg": 1.0, "tposr": 1.67384},
                5: {"tposl": 0.403190, "tneg": 1.0, "tposr": 1.59681},
                6: {"tposl": 0.458758, "tneg": 1.0, "tposr": 1.54124},
                7: {"tposl": 0.501261, "tneg": 1.0, "tposr": 1.49874},
                8: {"tposl": 0.535119, "tneg": 1.0, "tposr": 1.46488},
            },
            "wavenumbers": np.array(
                [
                    0.78942125,
                    0.83758044,
                    0.83068636,
                    0.80615684,
                    0.77351056,
                    0.74670987,
                ]
            ),
            "uncertainty_products": np.array(
                [411.927, 1.93649, 1.76532, 1.748, 1.74087, 1.7361]
            ),
        },
    }

    # index into the data store
    if details_key is not None:
        res = data[f_type][test_name][details_key]
    else:
        res = data[f_type][test_name]

    return res
