"""
-------------------------------------------------------------------------------

Unit tests for the .impulse_response_builder_tools module of spqf.tools.

-------------------------------------------------------------------------------
"""

import unittest
import numpy as np

from irides.tools import impulse_response_builder_tools


# noinspection SpellCheckingInspection,PyPep8Naming
def make_time_series_template(
    dt: float, t_start: float = 0.0, t_end: float = 1.0
) -> np.ndarray:
    """Alloc a time-series array and set the t-axis."""

    t_end_adjust = t_end + dt
    ts = np.zeros((np.arange(t_start, t_end_adjust, dt).shape[0], 2))
    ts[:, 0] = np.arange(t_start, t_end_adjust, dt)

    return ts


# noinspection SpellCheckingInspection,PyPep8Naming
class TestImpulseResponseBuilderTools(unittest.TestCase):
    """
    Unit tests to verify the correct operation of the impulse-response tools
    under src.spqf.tools.impulse_response_builder_tools.
    """

    def test_1st_diff_is_correct_for_padded_impulse_when_noncausal(self):
        """
        Here I expect h[0-dt] = +, h[0+dt] = -, and the rest 0.
        """

        dt = 0.1
        time_series = make_time_series_template(dt, -0.5, 1.0)
        i_t0 = impulse_response_builder_tools.protected_index_search(
            time_series[:, 0], dt, 0.0
        )
        time_series[i_t0, 1] = 1.0

        # fmt: off
        call_fcxn = impulse_response_builder_tools. \
            compute_noncausal_symmetric_first_difference
        # fmt: on
        test_values = {"-1": 5.0, "0": 0.0, "1": -5.0, "last": 0.0}

        common_difference_series_keypoint_test(
            "1st diff: padded impulse",
            call_fcxn,
            test_values,
            self,
            time_series,
            i_t0,
            enforce_causality=False,
        )

    def test_1st_diff_is_correct_for_unpadded_impulse(self):
        """
        Like the padded above, but trucated to t >= 0.
        """

        dt = 0.1
        time_series = make_time_series_template(dt)
        time_series[0, 1] = 1.0

        # fmt: off
        impulse_response_builder_tools. \
            compute_noncausal_symmetric_first_difference(time_series)
        # fmt: on

        # test
        precision = 4
        self.assertAlmostEqual(time_series[0 - 0, 1], 0.0, precision)
        self.assertAlmostEqual(time_series[0 + 1, 1], -5.0, precision)

    def test_1st_diff_is_correct_for_padded_unit_step_when_noncausal(self):
        """
        Here I expect [0, 0, 5, 5, 0, ..., -5]
                                ^
                                | --->  t = 0
        """

        dt = 0.1
        time_series = make_time_series_template(dt, -0.5, 1.0)
        i_t0 = impulse_response_builder_tools.protected_index_search(
            time_series[:, 0], dt, 0.0
        )
        time_series[i_t0:, 1] = 1.0

        # fmt: off
        call_fcxn = impulse_response_builder_tools. \
            compute_noncausal_symmetric_first_difference
        # fmt: on
        test_values = {"-1": 5.0, "0": 5.0, "1": 0.0, "last": 0.0}

        common_difference_series_keypoint_test(
            "1st diff: padded unit step",
            call_fcxn,
            test_values,
            self,
            time_series,
            i_t0,
            enforce_causality=False,
        )

    def test_1st_diff_is_correct_for_unpadded_unit_step(self):
        """
        Same as above, just truncated to t >= 0.
        """

        dt = 0.1
        time_series = make_time_series_template(dt, 0.0, 1.0)
        i_t0 = impulse_response_builder_tools.protected_index_search(
            time_series[:, 0], dt, 0.0
        )
        time_series[i_t0:, 1] = 1.0

        # fmt: off
        call_fcxn = impulse_response_builder_tools. \
            compute_noncausal_symmetric_first_difference
        # fmt: on
        test_values = {"-1": None, "0": 5.0, "1": 0.0, "last": 0.0}

        common_difference_series_keypoint_test(
            "1st diff: unpadded unit step",
            call_fcxn,
            test_values,
            self,
            time_series,
            i_t0,
            enforce_causality=True,
        )

    def test_1st_diff_causality_enforcement_for_padded_unit_step(self):
        """
        Same as above, but here there is padding for t < 0, and those
        values are force-set to zero.
        """

        dt = 0.1
        time_series = make_time_series_template(dt, -0.5, 1.0)
        i_t0 = impulse_response_builder_tools.protected_index_search(
            time_series[:, 0], dt, 0.0
        )
        time_series[i_t0:, 1] = 1.0

        # fmt: off
        call_fcxn = impulse_response_builder_tools. \
            compute_noncausal_symmetric_first_difference
        # fmt: on
        test_values = {"-1": 0.0, "0": 5.0, "1": 0.0, "last": 0.0}

        common_difference_series_keypoint_test(
            "1st diff: padded unit step, causality enforcement",
            call_fcxn,
            test_values,
            self,
            time_series,
            i_t0,
            enforce_causality=True,
        )

    def test_2nd_diff_is_correct_for_padded_impulse_when_noncausal(self):
        """Expect nonzero values only about the impulse"""

        dt = 0.1
        time_series = make_time_series_template(dt, -0.5, 1.0)
        i_t0 = impulse_response_builder_tools.protected_index_search(
            time_series[:, 0], dt, 0.0
        )
        time_series[i_t0, 1] = 1.0

        # fmt: off
        call_fcxn = impulse_response_builder_tools. \
            compute_noncausal_symmetric_second_difference
        # fmt: on
        test_values = {"-1": 100.0, "0": -200.0, "1": 100.0, "last": 0.0}

        common_difference_series_keypoint_test(
            "2nd diff: padded impulse",
            call_fcxn,
            test_values,
            self,
            time_series,
            i_t0,
            enforce_causality=False,
        )

    def test_2nd_diff_is_correct_for_padded_unit_step_when_noncausal(self):
        """2nd diff on a step has a zero-value in steady state"""

        dt = 0.1
        time_series = make_time_series_template(dt, -0.5, 1.0)
        i_t0 = impulse_response_builder_tools.protected_index_search(
            time_series[:, 0], dt, 0.0
        )
        time_series[i_t0:, 1] = 1.0

        # fmt: off
        call_fcxn = impulse_response_builder_tools. \
            compute_noncausal_symmetric_second_difference
        # fmt: on
        test_values = {"-1": 100.0, "0": -100.0, "1": 0.0, "last": 0.0}

        common_difference_series_keypoint_test(
            "unit step",
            call_fcxn,
            test_values,
            self,
            time_series,
            i_t0,
            enforce_causality=False,
        )

    def test_2nd_diff_is_correct_for_padded_ramp_when_noncausal(self):
        """2nd diff on a ramp has a zero-value in steady state"""

        dt = 0.1
        time_series = make_time_series_template(dt, -0.5, 1.0)
        i_t0 = impulse_response_builder_tools.protected_index_search(
            time_series[:, 0], dt, 0.0
        )

        time_series[i_t0:, 1] = np.arange(time_series.shape[0] - i_t0)

        # fmt: off
        call_fcxn = impulse_response_builder_tools. \
            compute_noncausal_symmetric_second_difference
        # fmt: on
        test_values = {"-1": 0.0, "0": 100.0, "1": 0.0, "last": 0.0}

        common_difference_series_keypoint_test(
            "unit ramp",
            call_fcxn,
            test_values,
            self,
            time_series,
            i_t0,
            enforce_causality=False,
        )

    def test_2nd_diff_is_correct_for_padded_parabola_when_noncausal(self):
        """2nd diff on a parabola will achieve a steady-state value"""

        dt = 0.1
        time_series = make_time_series_template(dt, -0.5, 1.0)
        i_t0 = impulse_response_builder_tools.protected_index_search(
            time_series[:, 0], dt, 0.0
        )

        time_series[i_t0:, 1] = np.power(
            np.arange(time_series.shape[0] - i_t0), 2
        )

        # fmt: off
        call_fcxn = impulse_response_builder_tools. \
            compute_noncausal_symmetric_second_difference
        # fmt: on
        test_values = {"-1": 0.0, "0": 100.0, "1": 200.0, "last": 200.0}

        common_difference_series_keypoint_test(
            "unit parabola",
            call_fcxn,
            test_values,
            self,
            time_series,
            i_t0,
            enforce_causality=False,
        )


# noinspection SpellCheckingInspection
def common_difference_series_keypoint_test(
    name: str,
    call_function,
    test_values: dict,
    this,
    time_series: np.ndarray,
    i_t0: int,
    enforce_causality: bool,
):
    np.set_printoptions(precision=2, suppress=True)

    call_function(time_series, enforce_causality)

    precision = 4

    if i_t0 > 0:
        this.assertAlmostEqual(
            time_series[i_t0 - 1, 1], test_values["-1"], precision
        )
    this.assertAlmostEqual(
        time_series[i_t0 - 0, 1], test_values["0"], precision
    )
    this.assertAlmostEqual(
        time_series[i_t0 + 1, 1], test_values["1"], precision
    )
    this.assertAlmostEqual(time_series[-1, 1], test_values["last"], precision)
