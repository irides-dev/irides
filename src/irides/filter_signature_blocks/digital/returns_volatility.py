"""
-------------------------------------------------------------------------------

A filter-signature block for the volatility of returns.

-------------------------------------------------------------------------------
"""

import numpy as np
import scipy.special
import json
from irides.resources.core_enumerations import FilterType

from irides.tools import impulse_response_tools as ir_tools
from irides.tools import analog_to_digital_conversion_tools as a2d_tools

from irides.resources.containers.discrete_sequence import DiscreteSequence

from irides.design.analog import (
    mbox_level as dsgn_a_l_mbox,
    polyema_level as dsgn_a_l_pema,
    bessel_level as dsgn_a_l_bssl,
)
from irides.design.analog import (
    mbox_inline_slope as dsgn_a_s_mbox,
    polyema_inline_slope as dsgn_a_s_pema,
    bessel_inline_slope as dsgn_a_s_bssl,
)
from irides.filter_signatures.digital import (
    mbox_level as fsig_d_l_mbox,
    mbox_inline_slope as fsig_d_s_mbox,
)
from irides.filter_signatures.digital import (
    polyema_level as fsig_d_l_pema,
    polyema_inline_slope as fsig_d_s_pema,
)
from irides.filter_signatures.digital import (
    bessel_level as fsig_d_l_bssl,
    bessel_inline_slope as fsig_d_s_bssl,
)


class ReturnsVolatility(object):
    """Implementation of a convolution-based volatility of returns filter."""

    def __init__(
        self,
        returns_filter_type: FilterType,
        returns_filter_mu: float,
        returns_filter_order: int,
        averaging_filter_type: FilterType,
        averaging_filter_mu_scale_factor: float,
        averaging_filter_order: int,
        n_end_scale_factor: int = 8,
    ):
        """Constructs a returns-volatility filter object.

        Parameters
        ----------
        returns_filter_type: FilterType
            filter type {BESSEL|PEMA|MBOX}
        returns_filter_mu: float
            First moment of the returns filter
        returns_filter_order: int
            Order of the returns filter
        averaging_filter_type: FilterType
            filter type {BESSEL|PEMA|MBOX}
        averaging_filter_mu_scale_factor: int
            Scale factor between returns and averaging filter (eg n = 6)
        averaging_filter_order: int
            Order of the averaging filter
        n_end_scale_factor: int
            Technical argument that relates to internal h[n] vector length
        """

        # set number of internal stages
        self._n_pipline_stages = 7 + 1  # +1 to store the input

        # set filter types
        self._returns_filter_type = returns_filter_type
        self._averaging_filter_type = averaging_filter_type

        if returns_filter_type == FilterType.MBOX:
            self._fsig_d_s = fsig_d_s_mbox
            self._dsgn_a_s = dsgn_a_s_mbox
        elif returns_filter_type == FilterType.PEMA:
            self._fsig_d_s = fsig_d_s_pema
            self._dsgn_a_s = dsgn_a_s_pema
        elif returns_filter_type == FilterType.BESSEL:
            self._fsig_d_s = fsig_d_s_bssl
            self._dsgn_a_s = dsgn_a_s_bssl
        else:
            raise ValueError("Invalid filter type")

        if averaging_filter_type == FilterType.MBOX:
            self._fsig_d_l = fsig_d_l_mbox
            self._dsgn_a_l = dsgn_a_l_mbox
        elif averaging_filter_type == FilterType.PEMA:
            self._fsig_d_l = fsig_d_l_pema
            self._dsgn_a_l = dsgn_a_l_pema
        elif averaging_filter_type == FilterType.BESSEL:
            self._fsig_d_l = fsig_d_l_bssl
            self._dsgn_a_l = dsgn_a_l_bssl
        else:
            raise ValueError("Invalid filter type")

        # verify filter orders
        ir_tools.validate_filter_order_or_die(
            self._dsgn_a_s, returns_filter_order
        )
        self._returns_filter_order = returns_filter_order
        ir_tools.validate_filter_order_or_die(
            self._dsgn_a_l, averaging_filter_order
        )
        self._averaging_filter_order = averaging_filter_order

        # mu and tau values
        self._returns_filter_mu = returns_filter_mu
        self._averaging_filter_mu = (
            averaging_filter_mu_scale_factor * returns_filter_mu
        )
        self._averaging_filter_mu_scale_factor = (
            averaging_filter_mu_scale_factor
        )

        if returns_filter_type in [FilterType.PEMA, FilterType.BESSEL]:
            splane_poles = self._dsgn_a_l.designs(returns_filter_order)["poles"]
            returns_filter_tau = (
                a2d_tools.solve_for_tau_from_mu_and_splane_poles(
                    self._returns_filter_mu, splane_poles
                )
            )
        else:
            returns_filter_tau = self._returns_filter_mu

        if averaging_filter_type in [FilterType.PEMA, FilterType.BESSEL]:
            splane_poles = self._dsgn_a_l.designs(averaging_filter_order)[
                "poles"
            ]
            averaging_filter_tau = (
                a2d_tools.solve_for_tau_from_mu_and_splane_poles(
                    self._averaging_filter_mu, splane_poles
                )
            )
        else:
            averaging_filter_tau = self._averaging_filter_mu

        self._returns_filter_tau = returns_filter_tau
        self._averaging_filter_tau = averaging_filter_tau

        # set up the inner, returns filter
        n_start = 0
        n_end = np.round(n_end_scale_factor * self._averaging_filter_mu).astype(
            int
        )
        self._hn_returns = self._fsig_d_s.generate_impulse_response(
            n_start, n_end, self._returns_filter_mu, returns_filter_order
        )

        # set up the outer, averaging filter
        self._hn_averaging = self._fsig_d_l.generate_impulse_response(
            n_start, n_end, self._averaging_filter_mu, averaging_filter_order
        )

        # set up the input transform
        self._input_transform = lambda x: np.log(x)

        # set interval scaling to the slope-filter scale factor
        self._interval_scale = returns_filter_mu

        # set up output bias correction
        n = averaging_filter_mu_scale_factor
        c4_n = np.sqrt(2.0 / (n - 1)) * np.exp(
            scipy.special.loggamma(n / 2) - scipy.special.loggamma((n - 1) / 2)
        )
        self._convexity_correction = c4_n
        self._inv_convexity_correction = 1.0 / c4_n

        # set up the kh(0) correction
        tau_unit = 1.0
        self._kh0_level_value_of_returns_filter = (
            self._dsgn_a_l.autocorrelation_peak_and_stride_values(
                tau_unit, returns_filter_order
            )["kh0"]
        )
        self._inv_sqrt_kh0_correction = 1.0 / np.sqrt(
            self._kh0_level_value_of_returns_filter
        )

        # set crossover duration
        self._crossover_duration = np.round(
            self._returns_filter_mu + self._averaging_filter_mu
        ).astype(int)

        # set warmup duration
        if returns_filter_type in [FilterType.PEMA, FilterType.BESSEL]:
            returns_warmup_duration = (
                self._dsgn_a_s.autocorrelation_peak_and_stride_values(
                    returns_filter_tau, returns_filter_order
                )["xi_5pct"]
            )
        else:
            returns_warmup_duration = returns_filter_mu

        if averaging_filter_type in [FilterType.PEMA, FilterType.BESSEL]:
            averaging_warmup_duration = (
                self._dsgn_a_l.autocorrelation_peak_and_stride_values(
                    averaging_filter_tau, averaging_filter_order
                )["xi_5pct"]
            )
        else:
            averaging_warmup_duration = self._averaging_filter_mu

        self._warmup_duration = np.round(
            returns_warmup_duration + averaging_warmup_duration
        ).astype(int)

    def warmup_duration(self) -> int:
        """Returns the duration of the warmup period."""

        return self._warmup_duration

    def crossover_duration(self) -> int:
        """Returns the duration of the crossover, where rv approx'ly peaks."""

        return self._crossover_duration

    def report(self) -> str:
        """Returns json representation of this object's fields."""

        return json.dumps(self.__dict__, default=str, indent=2)

    def apply(
        self,
        xn: DiscreteSequence,
        test: bool = False,
    ) -> DiscreteSequence:
        """Computes the returns volatility of an input signal `xn`.

        Parameters
        ----------
        xn: DiscreteSequence
            The input signal
        test: bool
            Skips the input log transform and input conditioning (for testing -- default is False)

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

        # impl of pipeline
        try:
            i_stage += 1

            # apply input transform
            if not test:
                yn.v_axes[:, i_stage] = self._input_transform(
                    yn.v_axes[:, i_stage - 1]
                )
            else:
                yn.v_axes[:, i_stage] = yn.v_axes[:, i_stage - 1]

            # apply the returns filter
            i_stage += 1
            x_0 = yn.v_axes[0, i_stage - 1] if not test else 0.0
            cand = np.convolve(
                self._hn_returns.v_axis, yn.v_axes[:, i_stage - 1] - x_0
            )
            yn.v_axes[:, i_stage] = cand[: yn.len]

            # square the output
            i_stage += 1
            yn.v_axes[:, i_stage] = yn.v_axes[:, i_stage - 1] ** 2

            # scale the interval
            i_stage += 1
            yn.v_axes[:, i_stage] = (
                self._interval_scale * yn.v_axes[:, i_stage - 1]
            )

            # apply the averaging filter
            i_stage += 1
            cand = np.convolve(
                self._hn_averaging.v_axis, yn.v_axes[:, i_stage - 1]
            )
            yn.v_axes[:, i_stage] = cand[: yn.len]

            # take the square root
            i_stage += 1
            yn.v_axes[:, i_stage] = np.sqrt(np.abs(yn.v_axes[:, i_stage - 1]))

            # apply khl(0) and convexity corrections
            i_stage += 1
            corrections = (
                self._inv_convexity_correction * self._inv_sqrt_kh0_correction
            )
            yn.v_axes[:, i_stage] = corrections * yn.v_axes[:, i_stage - 1]

        except Exception as e:
            print(f"stage: {i_stage}, e: {e}")
            return yn

        # return
        return yn
