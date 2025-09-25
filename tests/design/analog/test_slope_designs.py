"""
-------------------------------------------------------------------------------

Parameterized unit tests for inline-slope filter designs.

-------------------------------------------------------------------------------
"""

import parameterized

import unittest
import numpy as np

from irides.design.analog import polyema_inline_slope as design_pema
from irides.design.analog import bessel_inline_slope as design_bessel
from irides.design.analog import mbox_inline_slope as design_mbox
from irides.design.analog import polyema_composite_slope as design_pema_comp

from irides.design.analog import bessel_level as design_bessel_level
from irides.tools import design_tools

from tests.design.analog import common_tooling


# noinspection SpellCheckingInspection,PyPep8Naming
# setup to iter over filter types
@parameterized.parameterized_class(
    ("f_type", "design"),
    [
        ("pema", design_pema),
        ("bessel", design_bessel),
        ("mbox", design_mbox),
        ("pema-comp", design_pema_comp),
    ],
)
class TestInlineSlopeDesigns(unittest.TestCase):
    """
    Parameterized, common unit tests for inline slope filter designs.
    """

    def test_design_id_is_correct(self):
        """
        Tests that the correct design-id is coded.
        """

        design_id_theo = datastore(self.f_type, "design_id")
        common_tooling.design_id_test(self, self.design, design_id_theo)

    def test_design_type_is_correct(self):
        """Sanity test to confirm we have a slope filter type"""

        design_type_theo = design_tools.FilterDesignType.SLOPE
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
        filter_orders = self.design.get_valid_filter_orders()

        h_t0_theo = np.zeros(filter_orders.shape[0])
        h0_order_2_coef = datastore(
            self.f_type, "ir_t0_value", "h0_order_2_coef"
        )
        tau_power_law = datastore(self.f_type, "ir_t0_value", "tau_power_law")
        h_t0_theo[0] = h0_order_2_coef / np.power(tau, tau_power_law)

        common_tooling.impulse_response_t0_value_is_correct_for_valid_orders(
            self, self.design, h_t0_theo, tau, dt
        )

    def test_zero_crossing_time_is_correct_for_valid_orders(self):
        """Tests that the crossing times are right."""

        tau = 2.0
        tx_theo = datastore(self.f_type, "tx_values") * tau

        common_tooling.zero_crossing_time_is_correct_for_valid_orders(
            self, self.design, tx_theo, tau
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
        m2_theo = datastore(self.f_type, "moment_values", "M2") * tau

        # corrections (as necy)
        if self.f_type == "pema":
            tau_power_law = datastore(
                self.f_type, "ir_t0_value", "tau_power_law"
            )
            m0_theo[0] += 2.0 * dt / np.power(tau, tau_power_law)

        if self.f_type == "bessel":
            tau_power_law = 2
            correction_order = 2
            design_config = design_bessel_level.designs(correction_order)
            p = design_config["poles"][0]
            A = design_config["residues"][0]
            coef = np.real(A.real * p.real - A.imag * p.imag)
            m0_theo[0] += coef * dt / np.power(tau, tau_power_law)

        if self.f_type == "pema-comp":
            tau_power_law = datastore(
                self.f_type, "ir_t0_value", "tau_power_law"
            )
            m0_theo[0] += 9.0 / 4.0 * dt / np.power(tau, tau_power_law)

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
        common_tooling.wireframe_slope_features(
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
            "design_id": "polyema-inline-slope",
            "valid_orders": np.array([2, 3, 4, 5, 6, 7, 8]),
            "ir_t0_value": {"h0_order_2_coef": 4.0, "tau_power_law": 2},
            "tx_values": np.array([(m - 1.0) / m for m in np.arange(2, 8 + 1)]),
            "moment_values": {
                "M0": np.array([0.0, 0, 0, 0, 0, 0, 0]),
                "M1": np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                "M2": np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]),
            },
            "sacf_values": {
                2: {"kh0": 2.00000, "xi_5pct": 2.06997},
                3: {"kh0": 1.68750, "xi_5pct": 2.18896},
                4: {"kh0": 2.00000, "xi_5pct": 2.03310},
                5: {"kh0": 2.44141, "xi_5pct": 1.87929},
                6: {"kh0": 2.95313, "xi_5pct": 1.74845},
                7: {"kh0": 3.51709, "xi_5pct": 1.63877},
                8: {"kh0": 4.12500, "xi_5pct": 1.54614},
            },
            "kh0_discretization_correction": {
                2: 8.0,
                3: 0.0,
                4: 0.0,
                5: 0.0,
                6: 0.0,
                7: 0.0,
                8: 0.0,
            },
            "kh0_power_law": 3,
            "sscf_values": np.array(
                [6.39659, 5.89306, 5.47517, 5.18021, 4.72284, 4.33074, 4.01296]
            ),
            "dc_spectral_features": {
                "gain": 0.0,
                "phase": np.pi / 2.0,
                "group-delay": 1,
            },
            "freq_of_gain_peak": {
                "values": np.array(
                    [
                        2.00000,  # m = 2
                        2.12132,  # m = 3
                        2.30940,  # m = 4
                        2.50000,  # m = 5
                        2.68328,  # m = 6
                        2.85774,  # m = 7
                        3.02372,  # m = 8
                    ]
                ),
                "tau_power_law": -1,
            },
            "gain_at_peak_freq": {
                "values": np.array(
                    [
                        np.sqrt(v)
                        for v in [
                            1.0,  # m = 2
                            4.0 / 3,  # m = 3
                            27.0 / 16,  # m = 4
                            256.0 / 125,  # m = 5
                            3125.0 / 1296,  # m = 6
                            46656.0 / 16807,  # m = 7
                            823543.0 / 262144,  # m = 8
                        ]
                    ]
                ),
                "tau_power_law": -1,
            },
            "phase_at_peak_freq": {
                "values": np.array(
                    [
                        np.pi / 2.0 - v
                        for v in [
                            np.pi / 2.0,  # m = 2
                            1.84644,  # m = 3
                            2.09440,  # m = 4
                            2.31824,  # m = 5
                            2.52321,  # m = 6
                            2.71318,  # m = 7
                            2.89094,  # m = 8
                        ]
                    ]
                ),
                "tau_power_law": 0,
            },
            "gd_at_peak_freq": {
                "values": np.array(
                    [
                        1.0 / 2.0,  # m = 2
                        2.0 / 3.0,  # m = 3
                        3.0 / 4.0,  # m = 4
                        4.0 / 5.0,  # m = 5
                        5.0 / 6.0,  # m = 6
                        6.0 / 7.0,  # m = 7
                        7.0 / 8.0,  # m = 8
                    ]
                ),
                "tau_power_law": 1,
            },
            "wireframes": {
                2: {"tpos": 0.000000, "tneg": 1.57080},
                3: {"tpos": 0.129939, "tneg": 1.61090},
                4: {"tpos": 0.226725, "tneg": 1.58707},
                5: {"tpos": 0.298977, "tneg": 1.55561},
                6: {"tpos": 0.354942, "tneg": 1.52574},
                7: {"tpos": 0.399750, "tneg": 1.49908},
                8: {"tpos": 0.436595, "tneg": 1.47558},
            },
            "wavenumbers": np.array(
                [
                    0.56986212,
                    0.70527775,
                    0.73673970,
                    0.74872477,
                    0.75584547,
                    0.76189466,
                    0.76761364,
                ]
            ),
            "uncertainty_products": np.array(
                [np.inf, 2.59808, 1.93649, 1.77482, 1.70084, 1.65831, 1.63067]
            ),
        },
        "bessel": {
            "design_id": "bessel-inline-slope",
            "valid_orders": np.array([2, 3, 4, 5, 6, 7, 8]),
            "ir_t0_value": {"h0_order_2_coef": 3.0, "tau_power_law": 2},
            "tx_values": np.array(
                [
                    0.6045997881,  # m = 2
                    0.8134998949,  # m = 3
                    0.9005481030,  # m = 4
                    0.9425714677,  # m = 5
                    0.9649848592,  # m = 6
                    0.9778024596,  # m = 7
                    0.9855139594,  # m = 8
                ]
            ),
            "moment_values": {
                "M0": np.array([0.0, 0, 0, 0, 0, 0, 0]),
                "M1": np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                "M2": np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]),
            },
            "sacf_values": {
                2: {"kh0": 1.49468, "xi_5pct": 2.46176},
                3: {"kh0": 1.49843, "xi_5pct": 2.29374},
                4: {"kh0": 2.13784, "xi_5pct": 1.93200},
                5: {"kh0": 3.05729, "xi_5pct": 1.65981},
                6: {"kh0": 4.19589, "xi_5pct": 1.46236},
                7: {"kh0": 5.48266, "xi_5pct": 1.32120},
                8: {"kh0": 6.94603, "xi_5pct": 1.21256},
            },
            "kh0_discretization_correction": {
                2: 0.0,
                3: 0.0,
                4: 0.0,
                5: 0.0,
                6: 0.0,
                7: 0.0,
                8: 0.0,
            },
            "kh0_power_law": 3,
            "sscf_values": np.array(
                [5.93470, 4.80664, 4.84400, 4.28553, 3.79060, 3.41999, 3.14215]
            ),
            "dc_spectral_features": {
                "gain": 0.0,
                "phase": np.pi / 2.0,
                "group-delay": 1,
            },
            "freq_of_gain_peak": {
                "values": np.array(
                    [
                        1.732050808,
                        2.001732855,
                        2.378885838,
                        2.757342866,
                        3.108535773,
                        3.426226098,
                        3.714335374,
                    ]
                ),
                "tau_power_law": -1,
            },
            "gain_at_peak_freq": {
                "values": np.array(
                    [
                        1.000000000,  # m = 2
                        1.262110453,  # m = 3
                        1.517338302,  # m = 4
                        1.746638378,  # m = 5
                        1.950576314,  # m = 6
                        2.134070905,  # m = 7
                        2.301949193,  # m = 8
                    ]
                ),
                "tau_power_law": -1,
            },
            "phase_at_peak_freq": {
                "values": np.array(
                    [
                        0.00000000000,  # m = 2
                        -0.3898549443,  # m = 3
                        -0.7960272586,  # m = 4
                        -1.1831948640,  # m = 5
                        -1.5369255350,  # m = 6
                        -1.8552614240,  # m = 7
                        -2.1435091590,  # m = 8
                    ]
                ),
                "tau_power_law": 0,
            },
            "gd_at_peak_freq": {
                "values": np.array(
                    [
                        0.6666666667,  # m = 2
                        0.8863324030,  # m = 3
                        0.9621533147,  # m = 4
                        0.9885851933,  # m = 5
                        0.9970335598,  # m = 6
                        0.9993473592,  # m = 7
                        0.9998773101,  # m = 8
                    ]
                ),
                "tau_power_law": 1,
            },
            "wireframes": {
                2: {"tpos": 0.000000, "tneg": 1.81380},
                3: {"tpos": 0.194759, "tneg": 1.76420},
                4: {"tpos": 0.334622, "tneg": 1.65524},
                5: {"tpos": 0.429107, "tneg": 1.56846},
                6: {"tpos": 0.494421, "tneg": 1.50506},
                7: {"tpos": 0.541488, "tneg": 1.45841},
                8: {"tpos": 0.577091, "tneg": 1.42289},
            },
            "wavenumbers": np.array(
                [
                    0.60892898,
                    0.77474505,
                    0.81632936,
                    0.83570741,
                    0.84774214,
                    0.85137154,
                    0.84539188,
                ]
            ),
            "uncertainty_products": np.array(
                [np.inf, 2.26627, 1.72481, 1.60938, 1.56455, 1.5427, 1.53045]
            ),
        },
        "mbox": {
            "design_id": "mbox-inline-slope",
            "valid_orders": np.array([2, 3, 4, 5, 6, 7, 8]),
            "ir_t0_value": {"h0_order_2_coef": 1.0, "tau_power_law": 2},
            "tx_values": np.array(
                [
                    1.0,  # m = 2
                    1.0,  # m = 3
                    1.0,  # m = 4
                    1.0,  # m = 5
                    1.0,  # m = 6
                    1.0,  # m = 7
                    1.0,  # m = 8
                ]
            ),
            "moment_values": {
                "M0": np.array([0.0, 0, 0, 0, 0, 0, 0]),
                "M1": np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                "M2": np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]),
            },
            "sacf_values": {
                2: {"kh0": 2.00000, "xi_5pct": 1.9000000000},
                3: {"kh0": 3.37500, "xi_5pct": 1.5537113670},
                4: {"kh0": 5.33333, "xi_5pct": 1.3393447220},
                5: {"kh0": 7.59550, "xi_5pct": 1.1953057840},
                6: {"kh0": 10.1125, "xi_5pct": 1.0894011890},
                7: {"kh0": 12.8595, "xi_5pct": 1.0073948210},
                8: {"kh0": 15.8186, "xi_5pct": 0.9414763172},
            },
            "kh0_discretization_correction": {
                2: -4.0,
                3: 0.0,
                4: 0.0,
                5: 0.0,
                6: 0.0,
                7: 0.0,
                8: 0.0,
            },
            "kh0_power_law": 3,
            "sscf_values": np.array(
                [1.87194, 3.31446, 3.21686, 2.95635, 2.73096, 2.55439, 2.41530]
            ),
            "dc_spectral_features": {
                "gain": 0.0,
                "phase": np.pi / 2.0,
                "group-delay": 1,
            },
            "freq_of_gain_peak": {
                "values": np.array(
                    [
                        2.331122356,  # m = 2
                        2.902207933,  # m = 3
                        3.378923335,  # m = 4
                        3.796538379,  # m = 5
                        4.172699669,  # m = 6
                        4.517719693,  # m = 7
                        4.838240217,  # m = 8
                    ]
                ),
                "tau_power_law": -1,
            },
            "gain_at_peak_freq": {
                "values": np.array(
                    [
                        1.449222708,  # m = 2
                        1.789626825,  # m = 3
                        2.075053083,  # m = 4
                        2.325758038,  # m = 5
                        2.551973194,  # m = 6
                        2.759718734,  # m = 7
                        2.952891372,  # m = 8
                    ]
                ),
                "tau_power_law": -1,
            },
            "phase_at_peak_freq": {
                "values": np.array(
                    [
                        -0.760326029,  # m = 2
                        -1.331411606,  # m = 3
                        -1.808127008,  # m = 4
                        -2.225742052,  # m = 5
                        -2.601903342,  # m = 6
                        -2.946923366,  # m = 7
                        -3.267443890,  # m = 8
                    ]
                ),
                "tau_power_law": 0,
            },
            "gd_at_peak_freq": {
                "values": np.array(
                    [
                        1.0,  # m = 2
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
                2: {"tpos": 0.326163, "tneg": 1.67384},
                3: {"tpos": 0.458758, "tneg": 1.54124},
                4: {"tpos": 0.535119, "tneg": 1.46488},
                5: {"tpos": 0.586256, "tneg": 1.41374},
                6: {"tpos": 0.623554, "tneg": 1.37645},
                7: {"tpos": 0.652303, "tneg": 1.34770},
                8: {"tpos": 0.675337, "tneg": 1.32466},
            },
            "wavenumbers": np.array(
                [
                    0.80573771,
                    0.83797329,
                    0.83308778,
                    0.82091467,
                    0.79784648,
                    0.76948133,
                    0.74256534,
                ]
            ),
            "uncertainty_products": np.array(
                [427.838, 1.64317, 1.51186, 1.50396, 1.50244, 1.50174, 1.50131]
            ),
        },
        "pema-comp": {
            "design_id": "polyema-composite-slope",
            "valid_orders": np.array([1, 2, 3, 4, 5, 6, 7, 8]),
            "ir_t0_value": {"h0_order_2_coef": 4.5, "tau_power_law": 2},
            "tx_values": np.array(
                [0.924196 * m / (m + 1.0) for m in np.arange(1, 8 + 1)]
            ),
            "moment_values": {
                "M0": np.array([0.0, 0, 0, 0, 0, 0, 0, 0]),
                "M1": np.array(
                    [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
                ),
                "M2": np.array(
                    [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
                ),
            },
            "sacf_values": {
                1: {"kh0": 2.25000, "xi_5pct": 1.91744},
                2: {"kh0": 1.79297, "xi_5pct": 2.13314},
                3: {"kh0": 2.00926, "xi_5pct": 2.04505},
                4: {"kh0": 2.32178, "xi_5pct": 1.94042},
                5: {"kh0": 2.66156, "xi_5pct": 1.84766},
                6: {"kh0": 3.00749, "xi_5pct": 1.76889},
                7: {"kh0": 3.35042, "xi_5pct": 1.70218},
                8: {"kh0": 3.68571, "xi_5pct": 1.64526},
            },
            "kh0_discretization_correction": {
                1: 81.0 / 8.0,
                2: 0.0,
                3: 0.0,
                4: 0.0,
                5: 0.0,
                6: 0.0,
                7: 0.0,
                8: 0.0,
            },
            "kh0_power_law": 3,
            "sscf_values": np.array(
                [
                    29.5134,
                    44.9399,
                    21.3885,
                    14.0121,
                    10.6693,
                    8.80714,
                    7.63184,
                    6.82619,
                ]
            ),
            "dc_spectral_features": {
                "gain": 0.0,
                "phase": np.pi / 2.0,
                "group-delay": 1,
            },
            "freq_of_gain_peak": {
                "values": np.array(
                    [
                        2.12132,  # m = 1
                        2.17439,  # m = 2
                        2.30366,  # m = 3
                        2.43475,  # m = 4
                        2.55634,  # m = 5
                        2.66689,  # m = 6
                        2.76701,  # m = 7
                        2.85781,  # m = 8
                    ]
                ),
                "tau_power_law": -1,
            },
            "gain_at_peak_freq": {
                "values": np.array(
                    [
                        1.00000,
                        1.12577,
                        1.23961,
                        1.33919,
                        1.42674,
                        1.50441,
                        1.57393,
                        1.63663,
                    ]
                ),
                "tau_power_law": -1,
            },
            "phase_at_peak_freq": {
                "values": np.array(
                    [
                        0.00000,
                        -0.23888,
                        -0.44046,
                        -0.61179,
                        -0.75981,
                        -0.88945,
                        -1.00429,
                        -1.10693,
                    ]
                ),
                "tau_power_law": 0,
            },
            "gd_at_peak_freq": {
                "values": np.array(
                    [
                        0.44444,
                        0.60142,
                        0.68240,
                        0.73221,
                        0.76620,
                        0.79104,
                        0.81009,
                        0.82526,
                    ]
                ),
                "tau_power_law": 1,
            },
            "wireframes": {
                1: {"tpos": 0.00000, "tneg": 1.48096},
                2: {"tpos": 0.10986, "tneg": 1.55468},
                3: {"tpos": 0.19120, "tneg": 1.55494},
                4: {"tpos": 0.25128, "tneg": 1.54159},
                5: {"tpos": 0.29723, "tneg": 1.52617},
                6: {"tpos": 0.33352, "tneg": 1.51152},
                7: {"tpos": 0.36295, "tneg": 1.49833},
                8: {"tpos": 0.38734, "tneg": 1.48663},
            },
            "wavenumbers": np.array(
                [
                    0.55424844,
                    0.68040373,
                    0.70865678,
                    0.71848684,
                    0.72331669,
                    0.72664995,
                    0.72956863,
                    0.73242194,
                ]
            ),
            "uncertainty_products": np.array(
                [
                    np.inf,
                    2.76168,
                    2.06448,
                    1.89782,
                    1.82433,
                    1.78436,
                    1.76036,
                    1.74528,
                ]
            ),
        },
    }

    # index into the data store
    if details_key is not None:
        res = data[f_type][test_name][details_key]
    else:
        res = data[f_type][test_name]

    return res
