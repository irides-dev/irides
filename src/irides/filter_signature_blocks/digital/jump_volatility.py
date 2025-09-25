"""
-------------------------------------------------------------------------------

A filter-signature block for the volatility of isolated jumps.

-------------------------------------------------------------------------------
"""

import numpy as np
import json

from irides.tools import analog_to_digital_conversion_tools as a2d_tools
from irides.tools import digital_design_tools as dd_tools
from irides.tools.digital_design_tools import FrequencyBand

from irides.resources.containers.discrete_sequence import DiscreteSequence
from irides.resources.core_enumerations import FilterType

from irides.design.analog import (
    polyema_level as dsgn_a_l_pema,
    bessel_level as dsgn_a_l_bssl,
)
from irides.design.digital import (
    polyema_jumpvol_level as dsgn_d_jv_pema,
    bessel_jumpvol_level as dsgn_d_jv_bssl,
)
from irides.filter_signatures.digital import (
    polyema_level as fsig_d_l_pema,
    bessel_level as fsig_d_l_bssl,
)


class JumpVolatility(object):
    """Implementation of a convolution-based jump-volatility filter."""

    def __init__(
        self,
        jv_filter_type: FilterType,
        jv_filter_mu: float,
        jv_filter_order: int,
        n_end_scale_factor: int = 8,
    ):
        """Constructs a jump-volatility filter object.

        Parameters
        ----------
        jv_filter_type: FilterType
            filter type {BESSEL|PEMA}
        jv_filter_mu: float
            First moment of the jump-vol filter
        jv_filter_order: int
            Order of the jump-vol filter
        n_end_scale_factor: int
            Technical argument that relates to internal h[n] vector length
        """

        # set up the input transform
        self._input_transform = lambda x: np.log(x)

        # set number of internal stages
        self._n_pipline_stages = 4 + 1  # +1 to store the input

        # set filter types
        self._jv_filter_type = jv_filter_type

        if jv_filter_type == FilterType.PEMA:
            self._dsgn_a_l = dsgn_a_l_pema
            self._dsgn_d_jv = dsgn_d_jv_pema
            self._fsig_d_l = fsig_d_l_pema
        elif jv_filter_type == FilterType.BESSEL:
            self._dsgn_a_l = dsgn_a_l_bssl
            self._dsgn_d_jv = dsgn_d_jv_bssl
            self._fsig_d_l = fsig_d_l_bssl
        else:
            raise ValueError("Invalid filter type")

        # verify the filter order
        valid_filter_orders = self._dsgn_d_jv.get_valid_filter_orders()
        if jv_filter_order not in valid_filter_orders:
            msg = f"filter order must be within {valid_filter_orders}."
            msg += f" order: {jv_filter_order}"
            raise IndexError(msg)
        self._jv_filter_order = jv_filter_order

        # mu and tau values
        self._jv_filter_mu = jv_filter_mu

        splane_poles = self._dsgn_a_l.designs(jv_filter_order)["poles"]
        self._jv_filter_tau = a2d_tools.solve_for_tau_from_mu_and_splane_poles(
            self._jv_filter_mu, splane_poles
        )

        # set up the Hhf(z) - a Hlf(z) impulse response
        #   calc the a_coef using the high-freq point from a low-freq xfer fcxn
        Hz = dd_tools.construct_transfer_function(
            jv_filter_mu, jv_filter_order, self._dsgn_a_l
        )
        self._a_coef = Hz.value(-1)

        #   compute component impulse responses
        n_start = 0
        n_end = np.round(n_end_scale_factor * self._jv_filter_mu).astype(int)
        self._hn_lf_level = self._fsig_d_l.generate_impulse_response(
            n_start,
            n_end,
            jv_filter_mu,
            jv_filter_order,
            frequency_band=FrequencyBand.LOW,
        )
        self._hn_hf_level = self._fsig_d_l.generate_impulse_response(
            n_start,
            n_end,
            jv_filter_mu,
            jv_filter_order,
            frequency_band=FrequencyBand.HIGH,
        )

        #   compute linear combination
        self._hn_jv_level = DiscreteSequence(n_start, n_end)
        self._hn_jv_level.v_axis = (
            self._hn_hf_level.v_axis - self._a_coef * self._hn_lf_level.v_axis
        )

        # set the interval scale
        self._interval_scale = self._jv_filter_mu

        # set the kij[0] correction
        self._kij0_value = self._dsgn_d_jv.designs(jv_filter_order)[
            "kij_zero_at_tau_inf"
        ]
        self._inv_sqrt_kij0_correction = 1 / np.sqrt(self._kij0_value)

        # set the crossover and warmup durations
        self._crossover_duration = np.round(self._jv_filter_mu).astype(int)
        self._warmup_duration = (
            self._dsgn_a_l.autocorrelation_peak_and_stride_values(
                self._jv_filter_tau, jv_filter_order
            )["xi_5pct"]
        )

    def warmup_duration(self) -> int:
        """Returns the duration of the warmup period."""

        return self._warmup_duration

    def crossover_duration(self) -> int:
        """Returns the duration of the crossover, where jv approx'ly peaks."""

        return self._crossover_duration

    def report(self) -> str:
        """Returns json representation of this object's fields."""

        return json.dumps(self.__dict__, default=str, indent=2)

    def apply(
        self,
        xn: DiscreteSequence,
        test: bool = False,
    ) -> DiscreteSequence:
        """Computes the jump volatility of an input signal `xn`.

        The pipeline sequence is

            xn -> log -> hf-block -> scalings -> abs -> jv

        Parameters
        ----------
        xn: DiscreteSequence
            The input signal
        test: bool
            Skips the input log transform and input conditioning
            (for testing -- default is False)

        Returns
        -------
        DiscreteSequence
            The output signal along with outputs from each stage
        """

        # set up working sequence panel
        yn = DiscreteSequence(0, xn.len, self._n_pipline_stages)
        i_stage = 0

        # copy the input
        yn.v_axes[:, i_stage] = xn.v_axis

        # apply input transform
        i_stage += 1
        if not test:
            yn.v_axes[:, i_stage] = self._input_transform(
                yn.v_axes[:, i_stage - 1]
            )
        else:
            yn.v_axes[:, i_stage] = yn.v_axes[:, i_stage - 1]

        # apply the high-frequency level filter
        i_stage += 1
        x_0 = yn.v_axes[0, i_stage - 1] if not test else 0.0
        cand = np.convolve(
            self._hn_jv_level.v_axis, yn.v_axes[:, i_stage - 1] - x_0
        )
        yn.v_axes[:, i_stage] = cand[: yn.len]

        # scale by root-interval and apply bias correction
        i_stage += 1
        scalings = (
            np.sqrt(self._interval_scale) * self._inv_sqrt_kij0_correction
        )
        yn.v_axes[:, i_stage] = scalings * yn.v_axes[:, i_stage - 1]

        # rectify the signal
        i_stage += 1
        yn.v_axes[:, i_stage] = np.abs(yn.v_axes[:, i_stage - 1])

        # return
        return yn
