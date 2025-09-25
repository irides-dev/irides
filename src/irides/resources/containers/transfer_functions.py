"""
-------------------------------------------------------------------------------

Container for transfer functions as encoded by the numpy polynomial class

-------------------------------------------------------------------------------
"""

import numpy as np
from typing import Union

from irides.resources.core_enumerations import FilterDesignType


class IIRTransferFunctionDiscreteTime(object):
    """Container for IIR H(z) rational functions."""

    def __init__(
        self,
        filter_order: int,
        numerator_polynomials: np.ndarray,
        denominator_polynomials: np.ndarray,
        filter_class: FilterDesignType = FilterDesignType.LEVEL,
    ):
        """Initializes a Hz object.

        Instances of Polynomial objects that are passed in are to be expressed
        in terms of zeta = z^-1. The .value(z) method internally evaluates the
        polynomials at (1/z).

        Parameters
        ----------
        filter_order: int
            The filter order
        numerator_polynomials: np.ndarray
            Array of Polynomial objects
        denominator_polynomials: np.ndarray
            Array of Polynomial objects
        filter_class: FilterDesignType
            Class of filter {level|slope|curve}
        """

        # store locally
        self._filter_order = filter_order
        self._numerator_polynomials = numerator_polynomials
        self._denominator_polynomials = denominator_polynomials
        self._filter_class = filter_class

        # create capture
        exp = filter_class.value

        def eval_lead_coef_capture(
            zeta_value: Union[float, complex]
        ) -> Union[float, complex]:
            return (1 - zeta_value) ** exp

        self._eval_lead_coef = eval_lead_coef_capture

    def order(self) -> int:
        """Returns the order of H(z)"""

        return self._filter_order

    def value(
        self, z: Union[float, complex, np.ndarray]
    ) -> Union[float, complex, np.ndarray]:
        """Returns H(z) evaluated at point(s) z

        The method is

            H_impl(z) = (1 - z^(-1))^{0|1|2} x exp( log( H(z) ) ),

        which is significantly more numerically stable than just H(z).

        Parameters
        ----------
        z: Union[float, complex, np.ndarray]
            Value(s) of z to evaluate H(z).

        Returns
        -------
        Union[float, np.ndarray]
            Value(s) of H(z)
        """

        def single_value(zeta_value: Union[float, complex]) -> float:
            return self._eval_lead_coef(zeta_value) * np.exp(
                np.sum(
                    np.log([n(zeta_value) for n in self._numerator_polynomials])
                    - np.log(
                        [d(zeta_value) for d in self._denominator_polynomials]
                    )
                )
            )

        if np.isscalar(z):
            return single_value(1.0 / z)

        else:
            return np.array([single_value(1.0 / z_value) for z_value in z])

    def initial_value(self) -> float:
        """Returns value of h[0] computed from H(z)"""

        # calculate and return
        return self.value(np.inf)


# noinspection PyPep8Naming
class MboxTransferFunctionDiscreteTime(object):
    """Container for mbox H(z) transfer function."""

    def __init__(
        self,
        filter_order: int,
        N_stage: int,
        filter_class: FilterDesignType = FilterDesignType.LEVEL,
    ):
        """Initializes the Hz object.

        Parameters
        ----------
        filter_order: int
            The filter order
        N_stage: int
            The number of samples per stage (= N / m)
        filter_class: FilterDesignType
            Class of filter {level|slope|curve}
        """

        # store locally
        self._filter_order = filter_order
        self._n_stage = N_stage
        self._filter_class = filter_class

        # local calculations
        self._n_stage_is_even = (N_stage % 2) == 0
        self._log_gain_adj_per_stage = -np.log(N_stage)

        # create capture
        exp = filter_class.value

        def eval_lead_coef_capture(
            zeta_value: Union[float, complex]
        ) -> Union[float, complex]:
            return (1 - zeta_value) ** exp

        self._eval_lead_coef = eval_lead_coef_capture

    def order(self) -> int:
        """Returns the order of H(z)"""

        return self._filter_order

    def value(
        self, z: Union[float, complex, np.ndarray]
    ) -> Union[float, complex, np.ndarray]:
        """Returns H(z) evaluated at point(s) z.

        The method is

            H_impl(zeta) = (1 - zeta)^{0|1|2} x
                exp(
                    filter_order x (
                       log(gadj_per_stage)
                       + log(1 - zeta^{n_stage})
                       - log(1 - zeta)
                    )
                )

        which is fast and numerically stable. The degree of the lead (1 - zeta)
        polynomial depends on the filter class: level = 0, slope = 1, curve = 2.

        Parameters
        ----------
        z: Union[float, complex, np.ndarray]
            Value(s) of z to evaluate H(z).

        Returns
        -------
        Union[float, np.ndarray]
            Value(s) of H(z)
        """

        def single_value(zeta_value: Union[float, complex]) -> complex:
            if zeta_value == 1.0:
                v = self._eval_lead_coef(zeta_value) * np.exp(
                    self._filter_order
                    * (self._log_gain_adj_per_stage + np.log(self._n_stage))
                )
            elif zeta_value == -1.0 and self._n_stage_is_even:
                v = 0.0
            else:
                v = self._eval_lead_coef(zeta_value) * np.exp(
                    self._filter_order
                    * (
                        self._log_gain_adj_per_stage
                        + np.log(1 - zeta_value**self._n_stage)
                        - np.log(1 - zeta_value)
                    )
                )
            return v

        if np.isscalar(z):
            return single_value(1.0 / z)

        else:
            return np.array([single_value(1.0 / z_value) for z_value in z])

    def initial_value(self) -> float:
        """Returns value of h[0] computed from H(z)"""

        # calculate and return
        return self.value(np.inf)
