"""
-------------------------------------------------------------------------------

Digital Kiyasu--Thomson Bessel filter signature.

-------------------------------------------------------------------------------
"""

import numpy as np

from irides.design.analog import bessel_level as dsgn_a_l_bssl
from irides.filter_signatures.digital import polyema_level as fsig_d_l_pema
from irides.filter_signatures.digital import (
    damped_oscillator as fsig_d_dosc,
)

from irides.tools import impulse_response_tools as ir_tools
from irides.tools import analog_to_digital_conversion_tools as a2d_tools
from irides.resources.containers.discrete_sequence import DiscreteSequence
from irides.tools import digital_design_tools as dd_tools
from irides.tools.design_tools import StageTypes
from irides.tools.digital_design_tools import FrequencyBand


# noinspection SpellCheckingInspection
def generate_impulse_response(
    n_start: int,
    n_end: int,
    mu: float,
    filter_order: int,
    frequency_band: FrequencyBand = FrequencyBand.LOW,
    design=dsgn_a_l_bssl,
) -> DiscreteSequence:
    """Creates a discrete-time impulse response for the KT-Bessel filter.

    Parameters
    ----------
    n_start: int
        Start sample
    n_end: int
        End sample (exclusive)
    mu: float
        First moment of the impulse response
    filter_order: int
        Filter order
    frequency_band: FrequencyBand
        Specifies LOW or HIGH frequency (default = LOW)
    design:
        The name of the associated analog design module

    Returns
    -------
    DiscreteSequence
        Contains n-axis and h-axis
    """

    # validate
    ir_tools.validate_filter_order_or_die(design, filter_order)

    # fetch the analog design config for this filter order
    design_config = design.designs(filter_order)

    # aliases
    stages = design_config["stages"]
    splane_poles = design_config["poles"]

    # calc the number of stages
    n_stgs = len(design_config["stages"])

    # set up a discrete sequence to work with
    ds = DiscreteSequence(n_start, n_end, 1 + n_stgs)
    n_0plus_vec = ds.n_axis[ds.i_n_zero :]

    # correct analog-tau to achieve target digital-mu
    tau = a2d_tools.solve_for_tau_from_mu_and_splane_poles(mu, splane_poles)

    # create h[n]-generating fcxns
    # noinspection SpellCheckingInspection
    ema_filter_order = 1

    def hn_ema_gen(splane_poles_for_stage: np.ndarray) -> DiscreteSequence:
        pd = a2d_tools.convert_analog_poles_to_digital_poles(
            tau, splane_poles_for_stage
        )
        mu_stg = a2d_tools.first_moment_from_zplane_pole(pd)[0]
        return fsig_d_l_pema.generate_impulse_response(
            n_start, n_end, mu_stg, ema_filter_order, frequency_band
        )

    # noinspection SpellCheckingInspection
    def hn_dosc_gen(splane_poles_for_stage: np.ndarray) -> DiscreteSequence:
        pd = a2d_tools.convert_analog_poles_to_digital_poles(
            tau, splane_poles_for_stage
        )
        pd_signed = dd_tools.convert_frequency_band(pd, frequency_band)
        return fsig_d_dosc.generate_impulse_response(n_start, n_end, pd_signed)

    # pack into a dict
    hn_gen_dict = {
        StageTypes.EMA.value: hn_ema_gen,
        StageTypes.DOSC.value: hn_dosc_gen,
    }

    # iter across stages
    hn_cand = np.array([1.0])  # starting point for cascade
    for i, stage in enumerate(stages):
        # gen h[n] for this stage (dosc or ema)
        hn_stage = hn_gen_dict[stage["type"]](splane_poles[stage["indices"]])

        # convolve in new h[n] with running candidate
        hn_cand = np.convolve(hn_stage.v_axis, hn_cand)[: n_0plus_vec.shape[0]]

        # persist stage result (starting from second value column)
        ds.v_axes[ds.i_n_zero :, 1 + i] = hn_cand

    # persist the final cascade
    ds.v_axes[ds.i_n_zero :, 0] = hn_cand

    # return
    return ds
