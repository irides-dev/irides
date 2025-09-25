"""
-------------------------------------------------------------------------------

A container for cartesian_poles. Ctor consumes Cartesian form and the
interface provides for Cartesian or polar return values.

-------------------------------------------------------------------------------
"""

import numpy as np


class PoleContainer(object):
    """Container for nd-array of complex-valued cartesian_poles."""

    def __init__(self, cartesian_poles: np.ndarray):
        """Ctor to the PoleContainer.

        Parameters
        ----------
        cartesian_poles: np.ndarray
            An nd-array of complex-valued poles.
            dtype=complex (accepts float)
        """

        if isinstance(cartesian_poles, np.ndarray):
            self._cartesian_poles = cartesian_poles
        else:
            msg = "ctor argument must be an ndarray of pole coords."
            raise RuntimeError(msg)

    def cartesian_poles(self) -> np.ndarray:
        """Retrieve cartesian_poles in Cartesian form.

        Returns
        -------
        np.ndarray
            Poles in Cartesian form
        """

        return self._cartesian_poles

    def polar_poles(self, from_positive_real_axis: bool = True) -> np.ndarray:
        """Retrieve cartesian_poles in polar form,

        Parameters
        ----------
        from_positive_real_axis: bool
            Whether or not to reference polar angle to pos (T) or neg (F) axis

        Returns
        -------
        np.ndarray
            Poles in polar form
        """

        # manage real-axis angle
        sgn = 1.0 if from_positive_real_axis else -1.0

        # convert Cartesian to polar
        radius = np.abs(self._cartesian_poles)
        angle = np.arctan2(
            np.imag(self._cartesian_poles), sgn * np.real(self._cartesian_poles)
        )

        # pack and return
        return np.array([radius, angle]).T
