"""
-------------------------------------------------------------------------------

Analog Bessel filter design:

Captures the reference pole locations and partial-fraction expansion residues
for analog Bessel filters, and also captures pole groupings for multistage
design.

Wavenumbers are calculated numerically in Python in `calculate_wavenumber.py`.

-------------------------------------------------------------------------------
"""

import numpy as np
import sys

from irides.resources.containers import points
from irides.tools import design_tools
from irides.tools.design_tools import StageTypes


# noinspection SpellCheckingInspection
def design_id() -> str:
    """
    Since `design` modules are passed around as objects in this framework,
    a `design_id` gives a unique name.

    Returns
    -------
    str
        The id of this design.
    """

    return "bessel-level"


def design_type() -> design_tools.FilterDesignType:
    """This is a level filter"""

    return design_tools.FilterDesignType.LEVEL


# noinspection SpellCheckingInspection,PyUnusedLocal
def get_valid_filter_orders(strict: bool = False) -> np.ndarray:
    """Returns valid filter orders for this filter design.

    Parameters
    ----------
    strict: bool
        When true, only orders where h(0)=0 are included.

    Returns
    -------
    np.ndarray
        Array of valid filter orders
    """

    return design_tools.get_valid_filter_orders(
        design_tools.FilterDesignType.LEVEL, strict
    )


"""
-------------------------------------------------------------------------------

Filter-specific definitions

-------------------------------------------------------------------------------
"""


# noinspection SpellCheckingInspection
def designs(filter_order: int) -> dict:
    r"""
    The mth order Bessel filter (Laplace) transfer function is defined by

                   a_{0,m}
        Hm(s) = -------------- ,
                 theta^(m)(s)

    where theta^(m)(s) is an mth order reverse Bessel polynomial defined by

                                     (2m - k)!      s^k
        theta^(m)(s) = sum_{k=0}^m ------------- ---------
                                    k! (m - k)!   2^{m-k}

    and
                   (2m)!
        a_{0,m} = -------- .
                   2^m m!

    Numeric values of roots and residues are computed by Wolfram Alpha, located
    at https://www.wolframalpha.com/. An example root evaluation string is:

        roots s^2 + 3 s + 3.

    The concept of stages stems from the observation that the a transfer
    function, such as

                          15
        H(s) = -------------------------,
                s^3 + 6 s^2 + 15 s + 15

    can be rewritten as

                  pr^2 + pi^2         pa
        H(s) = ----------------- . --------.
                (s+pr)^2 + pi^2     pa + s

    The inverse Laplace transform yields the multistage construction:

        h(t) = L^(-1){ H_dosc(s) } * L^(-1){ H_ema(s) }

    where * is the convolution operator.

    The stage definitions below adhere to the multistage method
    of construction.

    An alternative construction, which is not recommendedjn practice but
    will help with aspects of design, is a full partial-fraction expansion
    of H(s). Forjnstance, as per above,

                  A1         A2         A3
        H(s) = -------- + -------- + --------,
                s + p1     s + p2     s + p3

    where {p_k} are the pole locations and {A_k} are the residues. A residue
    is naturally associated with a specific pole. See file bessel.nb for the
    calculation of pole and residue values.

    Parameters
    ----------
    filter_order: int
        The order of the filter (note: user is required to validate that
        the order is supported).

    Returns
    -------
    dict
        Design dictionary with keys 'stages', 'poles', 'residues'.
    """

    # designs
    design = {}

    r"""
    1st order transfer function:

                  1
        H(s) = -------
                s + 1

    stage design:

             -------
        ---->| ema |---->
             -------

    """

    design[1] = {
        "stages": [{"type": StageTypes.EMA.value, "indices": np.array([0])}],
        "poles": np.array([-1.0 + 0j]),
        "residues": np.array([1]),
        "poly_coeffs": np.array([1, 1]),
    }

    r"""
    2nd order transfer function:

                      3
        H(s) = ---------------
                s^2 + 3 s + 3

    stage design:

             ---------
        ---->| d-osc |---->
             ---------

    """

    design[2] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([1, 0])}
        ],
        "poles": np.array(
            [
                -1.5000000000000000000 - 0.8660254037844386468j,
                -1.5000000000000000000 + 0.8660254037844386468j,
            ]
        ),
        "residues": np.array(
            [0.0 + 1.7320508075688772935j, 0.0 - 1.7320508075688772935j]
        ),
        "poly_coeffs": np.array([1, 3, 3]),
    }

    r"""
    3rd order transfer function:

                         15
        H(s) = -------------------------
                s^3 + 6 s^2 + 15 s + 15

    stage design:

             ---------     -------
        ---->| d-osc |---->| ema |---->
             ---------     -------

    """

    design[3] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([2, 1])},
            {"type": StageTypes.EMA.value, "indices": np.array([0])},
        ],
        "poles": np.array(
            [
                -2.3221853546260855929,
                -1.8389073226869572035 - 1.7543809597837216610j,
                -1.8389073226869572035 + 1.7543809597837216610j,
            ]
        ),
        "residues": np.array(
            [
                4.5297921600971586592,
                -2.2648960800485793296 + 0.62390925642938012061j,
                -2.2648960800485793296 - 0.62390925642938012061j,
            ]
        ),
        "poly_coeffs": np.array([1, 6, 15, 15]),
    }

    r"""
    4th order transfer function:

                               105
        H(s) = -------------------------------------
                s^4 + 10 s^3 + 45 s^2 + 105 s + 105

    stage design:

             ---------     ---------
        ---->| d-osc |---->| d-osc |---->
             ---------     ---------

    """

    design[4] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([3, 2])},
            {"type": StageTypes.DOSC.value, "indices": np.array([1, 0])},
        ],
        "poles": np.array(
            [
                -2.8962106028203721684 - 0.8672341289345037518j,
                -2.8962106028203721684 + 0.8672341289345037518j,
                -2.1037893971796278316 - 2.6574180418567527169j,
                -2.1037893971796278316 + 2.6574180418567527169j,
            ]
        ),
        "residues": np.array(
            [
                1.6633914159371089178 + 8.3963003152960348792j,
                1.6633914159371089178 - 8.3963003152960348792j,
                -1.6633914159371089178 - 2.2440773205454007801j,
                -1.6633914159371089178 + 2.2440773205454007801j,
            ]
        ),
        "poly_coeffs": np.array([1, 10, 45, 105, 105]),
    }

    r"""
    5th order transfer function:

                                   945
        H(s) = ------------------------------------------------
                s^5 + 15 s^4 + 105 s^3 + 420 s^2 + 945 s + 945

    stage design:

             ---------     ---------     -------
        ---->| d-osc |---->| d-osc |---->| ema |---->
             ---------     ---------     -------

    """

    design[5] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([4, 3])},
            {"type": StageTypes.DOSC.value, "indices": np.array([2, 1])},
            {"type": StageTypes.EMA.value, "indices": np.array([0])},
        ],
        "poles": np.array(
            [
                -3.6467385953296432597,
                -3.3519563991535331430 - 1.7426614161831977227j,
                -3.3519563991535331430 + 1.7426614161831977227j,
                -2.3246743031816452271 - 3.5710229203379764004j,
                -2.3246743031816452271 + 3.5710229203379764004j,
            ]
        ),
        "residues": np.array(
            [
                20.863330821504189555,
                -11.935117225355992568 + 6.3430888965861925994j,
                -11.935117225355992568 - 6.3430888965861925994j,
                1.5034518146038977909 - 2.6668146309020763827j,
                1.5034518146038977909 + 2.6668146309020763827j,
            ]
        ),
        "poly_coeffs": np.array([1, 15, 105, 420, 945, 945]),
    }

    r"""
    6th order transfer function:

                                        10395
        H(s) = ----------------------------------------------------------------
                s^6 + 21 s^5 + 210 s^4 + 1260 s^3 + 4725 s^2 + 10395 s + 10395

    stage design:

             ---------     ---------     ---------
        ---->| d-osc |---->| d-osc |---->| d-osc |---->
             ---------     ---------     ---------

    """

    design[6] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([5, 4])},
            {"type": StageTypes.DOSC.value, "indices": np.array([3, 2])},
            {"type": StageTypes.DOSC.value, "indices": np.array([1, 0])},
        ],
        "poles": np.array(
            [
                -4.2483593958633639449 - 0.8675096732313656064j,
                -4.2483593958633639449 + 0.8675096732313656064j,
                -3.7357083563258146679 - 2.6262723114471256405j,
                -3.7357083563258146679 + 2.6262723114471256405j,
                -2.5159322478108213871 - 4.4926729536539425359j,
                -2.5159322478108213871 + 4.4926729536539425359j,
            ]
        ),
        "residues": np.array(
            [
                10.959228792516907373 + 39.425157113160190552j,
                10.959228792516907373 - 39.425157113160190552j,
                -14.126767991748670862 - 12.701164165560234053j,
                -14.126767991748670862 + 12.701164165560234053j,
                3.1675391992317634894 + 0.20245898404171038568j,
                3.1675391992317634894 - 0.20245898404171038568j,
            ]
        ),
        "poly_coeffs": np.array([1, 21, 210, 1260, 4725, 10395, 10395]),
    }

    r"""
    7th order transfer function:

                                        135135
        H(s) = -----------------------------------------------------------------
                s^7 + 28 s^6 + 378 s^5 + 3150 s^4 + 17325 s^3 
                                                + 62370 s^2 + 135135 s + 135135

    stage design:

             ---------     ---------     ---------     -------
        ---->| d-osc |---->| d-osc |---->| d-osc |---->| ema |---->
             ---------     ---------     ---------     -------

    """

    design[7] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([6, 5])},
            {"type": StageTypes.DOSC.value, "indices": np.array([4, 3])},
            {"type": StageTypes.DOSC.value, "indices": np.array([2, 1])},
            {"type": StageTypes.EMA.value, "indices": np.array([0])},
        ],
        "poles": np.array(
            [
                -4.9717868585279356779,
                -4.7582905281546289452 - 1.7392860611305365429j,
                -4.7582905281546289452 + 1.7392860611305365429j,
                -4.0701391636381374717 - 3.5171740477097531658j,
                -4.0701391636381374717 + 3.5171740477097531658j,
                -2.6856768789432657441 - 5.4206941307167488958j,
                -2.6856768789432657441 + 5.4206941307167488958j,
            ]
        ),
        "residues": np.array(
            [
                96.448703466230703258,
                -57.792465959616429334 + 38.884808124790946028j,
                -57.792465959616429334 - 38.884808124790946028j,
                8.3218236759203368582 - 23.140605908290046126j,
                8.3218236759203368582 + 23.140605908290046126j,
                1.2462905505807408470 + 2.9043702547805305311j,
                1.2462905505807408470 - 2.9043702547805305311j,
            ]
        ),
        "poly_coeffs": np.array(
            [1, 28, 378, 3150, 17325, 62370, 135135, 135135]
        ),
    }

    r"""
    8th order transfer function:

                                        2027025
        H(s) = -----------------------------------------------------------------
                s^8 + 36 s^7 + 630 s^6 + 6930 s^5 + 51975 s^4 + 270270 s^3 
                                              + 945945 s^2 + 2027025 s + 2027025

    stage design:

             ---------     ---------     ---------     ---------
        ---->| d-osc |---->| d-osc |---->| d-osc |---->| d-osc |---->
             ---------     ---------     ---------     ---------


    """

    design[8] = {
        "stages": [
            {"type": StageTypes.DOSC.value, "indices": np.array([7, 6])},
            {"type": StageTypes.DOSC.value, "indices": np.array([5, 4])},
            {"type": StageTypes.DOSC.value, "indices": np.array([3, 2])},
            {"type": StageTypes.DOSC.value, "indices": np.array([1, 0])},
        ],
        "poles": np.array(
            [
                -5.5878860432630851990 - 0.8676144453527864598j,
                -5.5878860432630851990 + 0.8676144453527864598j,
                -5.2048407906368819183 - 2.6161751526425274287j,
                -5.2048407906368819183 + 2.6161751526425274287j,
                -4.3682892172024024070 - 4.4144425004715390836j,
                -4.3682892172024024070 + 4.4144425004715390836j,
                -2.8389839488976304757 - 6.3539112986048768221j,
                -2.8389839488976304757 + 6.3539112986048768221j,
            ]
        ),
        "residues": np.array(
            [
                59.136190765097912790 + 183.86040218976815559j,
                59.136190765097912790 - 183.86040218976815559j,
                -87.076721410343559811 - 62.455631732253360265j,
                -87.076721410343559811 + 62.455631732253360265j,
                29.865102171946156214 - 2.0276026150302011702j,
                29.865102171946156214 + 2.0276026150302011702j,
                -1.9245715267005091935 + 2.3681579888539021834j,
                -1.9245715267005091935 - 2.3681579888539021834j,
            ]
        ),
        "poly_coeffs": np.array(
            [1, 36, 630, 6930, 51975, 270270, 945945, 2027025, 2027025]
        ),
    }

    # check and return
    return design[filter_order]


"""
-------------------------------------------------------------------------------

Standard features

-------------------------------------------------------------------------------
"""


# noinspection SpellCheckingInspection
def unit_peak_time(filter_order: int) -> float:
    """
    The timepoint where a unit-scale impulse response h(t)
    reaches its first peak. See bessel_inline_slope.nb for calculation.

    Parameters
    ----------
    filter_order: int
        The order of the filter (note: user is required to validate that
        the order is supported).

    Returns
    -------
    float
        The timepoint where h(t) peaks, for unit delay.
    """

    # values calculated from Mathematica
    unit_peak_times = np.array(
        [
            0.000000000000000,  # m = 1
            0.604599788078073,  # m = 2
            0.813499894915707,  # m = 3
            0.900548102972202,  # m = 4
            0.942571467655463,  # m = 5
            0.964984859216068,  # m = 6
            0.977802459643315,  # m = 7
            0.985513959386379,  # m = 8
        ]
    )

    # return
    return unit_peak_times[filter_order - 1]


# noinspection SpellCheckingInspection
def unit_peak_value(filter_order: int) -> float:
    """
    The value of h(t_peak), the first and principal peak, for a unit-scale
    impulse response. See bessel_inline_slope.nb for calculation.

    Parameters
    ----------
    filter_order: int
        The order of the filter (note: user is required to validate that
        the order is supported).

    Returns
    -------
    float
        The value of h(t_peak), for unit delay.
    """

    # values calculated from Mathematica
    unit_peak_values = np.array(
        [
            1.000000000000000,  # m = 1
            0.699357279553692,  # m = 2
            0.816382165302566,  # m = 3
            0.952223206957527,  # m = 4
            1.085517577490184,  # m = 5
            1.212016261684917,  # m = 6
            1.331053039720972,  # m = 7
            1.442974600373275,  # m = 8
        ]
    )

    # return
    return unit_peak_values[filter_order - 1]


# noinspection SpellCheckingInspection,PyUnusedLocal
def peak_value_coordinate(tau: float, filter_order: int, dt=0.0):
    """
    The impulse response of a low-pass level Bessel filter
    has a principal mode for t > 0, and then rings slightly in the
    tail. The peak time /t*/ and peak value /h(t*)/, which look
    roughly like this,

                  ....x...  <---- h(t*)
              ....        ....
           ...                ..
         ..                     ...
        .                          ....
    ----|-------------|-------------------------> t
        0
                     t*

    are calculated numerically. See bessel_inline_slope.nb for the
    calculations. Bessel filters differ from polyemas in that polyemas
    have analytic expressions for any filter order.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        The order of the filter (note: user is required to validate that
        the order is supported).
    dt: float
        Temporal discretization interval.

    Returns
    -------
    ContinuousTimePoint
        Object with .time and .value fields.
    """

    # aliases
    m = filter_order

    # put together into a point coordinate
    peak_coord = points.ContinuousTimePoint(
        unit_peak_time(m) * tau, unit_peak_value(m) / tau
    )

    # return
    return peak_coord


# noinspection SpellCheckingInspection,PyUnusedLocal
def moment_value_generator(moment: int, tau: float = 1.0, dt: float = 0.0):
    """
    Returns an anonymous function with the single argument (order: int),
    where order is a `filter_order`. The returned function captures the `tau`
    and `dt` arguments in this calling function.

    Note that the generated functions are not purely continuous time.
    Instead, they bridge continuous and discrete time by applying order-dt
    corrections to the moments. This is required because the CT series are
    in fact implemented in DT. Use dt=0.0 to recover pure CT moments.

    The empirically determined polynomial functions in filter-order were
    estimated by calculating the raw M* moments, comparing them with the
    theoretic moments, and fitting a polynomial through the difference.
    Specifically,

        x = (order + 1) / 2
        corrections(x) = (raw-moment(x) - theo-moment(x)) / (dt / 2)
        poly(x) = fit(x, corrections(x), poly-order).

    Note that only odd filter orders require a dt-order correction; even
    orders require no correction. This is because odd orders blend in an
    ema stage, and leading edge of the ema induces a dt-order correction.

    Parameters
    ----------
    moment: int
        Moment to calculate: [0, 1, 2]
    tau: float
        Temporal scale of the filter.
    dt: float
        Time interval for discretization.

    Returns
    -------
    Anonymous function with argument (order: int).
    """

    # generator defs with dt capture
    # noinspection PyUnusedLocal
    def m0_gen(order: int) -> float:
        """
        A dt correction is required for odd orders, but not even orders.
        The dt correction was determined empirically and is expressed as
        a linear function of filter order.
        """
        x = (order + 1.0) / 2.0
        dt_scale_correction = (-0.322 + x * 1.3224) if (order % 2 == 1) else 0.0

        return 1.0 + ((dt / tau) / 2.0) * dt_scale_correction

    # noinspection PyUnusedLocal
    def m1_gen(order: int) -> float:
        """
        Like M0, there is a dt-order correction to M1 for odd orders.
        The correction, while not analytic, was determined empirically
        and fit to a linear function of filter order.
        """
        x = (order + 1.0) / 2.0
        dt_scale_correction = (-1.322 + x * 1.3224) if (order % 2 == 1) else 0.0

        return tau + (dt / 2.0) * dt_scale_correction

    # noinspection PyUnusedLocal
    def m2_gen(order: int) -> float:
        """
        M2 is best fit with a cubic expression in filter order `order`.
        """
        x = (order + 1.0) / 2.0
        dt_scale_correction = (
            (0.1280003 + x * (-0.732667 + x * (0.678 - x * 0.0733333)))
            if (order % 2 == 1)
            else 0.0
        )

        return (
            2.0 * order / (2.0 * order - 1.0) * np.power(tau, 2)
            + (dt / 2.0) * dt_scale_correction
        )

    # return
    if moment == 0:
        return m0_gen
    elif moment == 1:
        return m1_gen
    elif moment == 2:
        return m2_gen
    else:
        msg = "Moments must be within (0, 1, 2). Calling value: {0}".format(
            moment
        )
        raise IndexError(msg)


# noinspection SpellCheckingInspection
def autocorrelation_peak_and_stride_values(
    tau: float, filter_order: int
) -> dict:
    """
    System autocorrelation peak and stride values. Peak value is kh(0),
    and the stride comes from kh(xi_5pct) / kh(0) = 5%.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Order of the filter (note: user is required to validate that
        the order is supported).

    Returns
    -------
    dict
        Peak value (scaled by 1/tau) and 5% pct stride (scaled by tau)
    """

    # values from Mathematica
    unit_sacf_values = {
        1: {"kh0": 0.5000, "xi_5pct": 2.9957},
        2: {"kh0": 0.4994, "xi_5pct": 2.1882},
        3: {"kh0": 0.5999, "xi_5pct": 1.6556},
        4: {"kh0": 0.7034, "xi_5pct": 1.3604},
        5: {"kh0": 0.8027, "xi_5pct": 1.1733},
        6: {"kh0": 0.8949, "xi_5pct": 1.0473},
        7: {"kh0": 0.9779, "xi_5pct": 0.9594},
        8: {"kh0": 1.0563, "xi_5pct": 0.8909},
    }

    # fetch values for order and scale
    sacf_values = unit_sacf_values[filter_order]
    sacf_values["kh0"] /= tau
    sacf_values["xi_5pct"] *= tau

    # return
    return sacf_values


# noinspection SpellCheckingInspection,PyPep8Naming
def scale_correlation_stride_and_residual_values(
    tau: float, filter_order: int
) -> dict:
    """Scale decorrelation strides and residual correlations.

    Parameters
    ----------
    tau: float
        The first moment of the filter.
    filter_order: int
        Order of the filter.

    Returns
    -------
    dict
        Scale decorrelation (scales as `tau`) and residual correlation
        (independent of `tau`).
    """

    # values from Mathematica
    unit_sscf_values = {
        1: {"sti_5pct": 1598.00, "residual": 0.05},
        2: {"sti_5pct": 22.9564, "residual": 0.05},
        3: {"sti_5pct": 8.64193, "residual": 0.05},
        4: {"sti_5pct": 5.46444, "residual": 0.05},
        5: {"sti_5pct": 4.17151, "residual": 0.05},
        6: {"sti_5pct": 3.48665, "residual": 0.05},
        7: {"sti_5pct": 3.06588, "residual": 0.05},
        8: {"sti_5pct": 2.78165, "residual": 0.05},
    }

    # fetch values for order and scale
    sscf_values = unit_sscf_values[filter_order]
    sscf_values["sti_5pct"] *= tau

    # return
    return sscf_values


# noinspection SpellCheckingInspection,PyPep8Naming,PyUnusedLocal
def full_width_generator(tau: float = 1.0, dt: float = 0.0):
    """
    Returns anonymous function for the full-width. `tau` and `dt` are
    captured here, and the anonymous function has one argument, (order: int),
    the filter order.

    Parameters
    ----------
    tau: float
        Temporal scale of the filter.
    dt: float
        Time interval for discretization.

    Returns
    -------
    Anonymous function with argument (order: int).
    """

    # noinspection PyUnusedLocal
    def fw_gen(order: int) -> float:
        """
        A quadratic function of filter order is sufficient to capture
        the order-based dt-scale correction. Only odd orders, where an ema
        stage exists, requires a correction. The functional form was determined
        empirically.
        """
        x = (order + 1.0) / 2.0
        dt_scale_correction = (
            (2.836 + x * (-2.16 - x * 0.676)) if (order % 2 == 1) else 0.0
        )

        return (
            2.0 * np.sqrt(1.0 / (2.0 * order - 1.0))
            + (dt / 2.0) * dt_scale_correction
        )

    return fw_gen


# noinspection PyUnusedLocal
def gain_at_dc(tau: float) -> float:
    """Filter gain at DC frequency. This is independent of filter order.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.

    Returns
    -------
    float
        The gain at DC frequency.
    """

    return 1.0


# noinspection PyUnusedLocal
def phase_at_dc(tau: float) -> float:
    """Filter phase at DC frequency. This is independent of filter order.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.

    Returns
    -------
    float
        The gain at DC frequency.
    """

    return 0.0


def group_delay_at_dc(tau: float) -> float:
    """Filter group delay at DC frequency. This is independent of filter order.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.

    Returns
    -------
    float
        The gain at DC frequency.
    """

    return tau


# noinspection SpellCheckingInspection
def cutoff_frequency(tau: float, filter_order: int) -> float:
    """
    At w_cutoff,

        |H(w_cutoff; tau, m)|^2 = 1/2.

    The solution is not analytic so the values for order 1-8 are reported here
    for tau = 1. The frequencies scale as 1 / tau.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Order of the filter  (note: user is required to validate that
        the order is supported).

    Returns
    -------
    float
        Radian frequency at cutoff.
    """

    # cutoff freqs (from Mathematica bessel_level.nb, tau=1)
    cutoff_frequencies = {
        1: 1.00,
        2: 1.36,
        3: 1.76,
        4: 2.11,
        5: 2.43,
        6: 2.70,
        7: 2.95,
        8: 3.18,
    }

    # check and return
    return cutoff_frequencies[filter_order] / tau


# noinspection PyUnusedLocal
def gain_at_cutoff(tau: float, filter_order: int) -> float:
    """
    The cutoff gain is always sqrt(1/2). The purpose of this function is
    to adhere to the phase- and group-delay-cutoff api so that any of the
    gain/phase/group-delay values @ cutoff can be determined uniformly.
    """

    return np.sqrt(1.0 / 2.0)


# noinspection PyUnusedLocal
def phase_at_cutoff(tau: float, filter_order: int) -> float:
    """
    The phase at cutoff frequency `wc`. Phase at cutoff is independent
    of temporal scale `tau`.

    Parameters
    ----------
    tau: float
        Unused but present for a consistent api
    filter_order: int
        Order of the filter  (note: user is required to validate that
        the order is supported).

    Returns
    -------
    float
        Phase at cutoff frequency, tau = 1.
    """

    # phase at wc (from Mathematica bessel_level.nb, tau=1)
    phases_at_cutoff = {
        1: -0.785,
        2: -1.297,
        3: -1.736,
        4: -2.109,
        5: -2.426,
        6: -2.703,
        7: -2.952,
        8: -3.180,
    }

    # check and return
    return phases_at_cutoff[filter_order]


# noinspection SpellCheckingInspection
def group_delay_at_cutoff(tau: float, filter_order: int) -> float:
    """
    The group delay at cutoff frequency `wc`, and the group delay scales with
    the `tau`. The unit-scale group-delay values are evaluated by
    Mathematica.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        Order of the filter (note: user is required to validate that
        the order is supported).

    Returns
    -------
    float
        Group delay at cutoff.
    """

    # cutoff freqs (from Mathematica bessel_level.nb, tau=1)
    group_delays_at_cutoff = {
        1: 0.500,
        2: 0.809,
        3: 0.935,
        4: 0.982,
        5: 0.996,
        6: 0.999,
        7: 1.000,
        8: 1.000,
    }

    # check and return
    return group_delays_at_cutoff[filter_order] * tau


# noinspection SpellCheckingInspection
def wireframe(tau: float, filter_order: int):
    """Wireframe timepoint for this level filter.

    Parameters
    ----------
    tau: float
        Temporal scaling, which ='s the first moment of the final filter.
    filter_order: int
        The order of the filter (unused)

    Returns
    -------
    WireframeContinuousTime
        Wireframe object
    """

    return design_tools.generate_level_wireframe_from_spectra(
        sys.modules[__name__], tau, filter_order
    )


# noinspection SpellCheckingInspection
def wavenumber(filter_order: int) -> float:
    """Report the wavenumber for this filter as a function of order.

    Parameters
    ----------
    filter_order: int
        The order of the filter.

    Returns
    -------
    float
        The wavenumber.
    """

    wavenumbers = {
        1: 0.64778095,
        2: 0.75316630,
        3: 0.77811737,
        4: 0.78250289,
        5: 0.76528238,
        6: 0.74164804,
        7: 0.71391692,
        8: 0.68803663,
    }

    return wavenumbers[filter_order]


# noinspection SpellCheckingInspection
def uncertainty_product(filter_order: int) -> dict:
    """Dict of TW, BW, and UCP for this filter order.

    Parameters
    ----------
    filter_order: int
        The order of the filter

    Returns
    -------
    dict
        TW is temporal width, BW is spectral width, and UCP is the
        uncertainty product.
    """

    uncertainty_products = {
        1: {"TW": 0.5, "BW": np.inf, "UCP": np.inf},
        2: {"TW": 0.440959, "BW": 1.73205, "UCP": 0.763763},
        3: {"TW": 0.360416, "BW": 1.58114, "UCP": 0.569868},
        4: {"TW": 0.304435, "BW": 1.74456, "UCP": 0.531105},
        5: {"TW": 0.265071, "BW": 1.95133, "UCP": 0.517241},
        6: {"TW": 0.23622, "BW": 2.16293, "UCP": 0.510926},
        7: {"TW": 0.214274, "BW": 2.36861, "UCP": 0.507532},
        8: {"TW": 0.197061, "BW": 2.56501, "UCP": 0.505463},
    }

    return uncertainty_products[filter_order]
