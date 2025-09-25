"""
-------------------------------------------------------------------------------

Container for a discrete-time sequence wherein the n-axis is int-type while
value(s) are float-type.

-------------------------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
import copy


class DiscreteSequence(object):
    """Container of a discrete-time sequence"""

    def __init__(
        self, n_start: int, n_end: int, n_cols: int = 1, v_type: str = "float"
    ):
        """Initializes a discrete-time sequence.

        Parameters
        ----------
        n_start: int
            Start sample.
        n_end: int
            End sample (exclusive)
        n_cols: int
            Number of value columns (default = 1)
        """

        # initialize and save locally
        self._n_axis = np.arange(n_start, n_end, dtype="int64")
        self._v_axes = np.zeros((self._n_axis.shape[0], n_cols), dtype=v_type)
        self._step_size = 1

        # find index into n == 0 (if it exists)
        candidate = np.where(self._n_axis == 0)[0]
        self._i_n_zero = candidate[0] if candidate.shape[0] > 0 else None

    @classmethod
    def copy(cls, obj) -> DiscreteSequence:
        """Returns of deep copy of an instance of this object"""

        return copy.deepcopy(obj)

    @property
    def step_size(self) -> int:
        """Returns the step size of the sequence"""

        return self._step_size

    @property
    def len(self) -> int:
        """Returns the length of the sequence"""

        return self._n_axis.shape[0]

    @property
    def n_axis(self):
        """Returns the index axis."""

        return self._n_axis

    @property
    def i_n_zero(self):
        """Returns the index into n_axis where n == 0, or None"""

        return self._i_n_zero

    @property
    def v_axis(self):
        """Returns the zeroth column of values"""

        return self._v_axes[:, 0]

    @v_axis.setter
    def v_axis(self, new_v_axis_values):
        """Sets the zeroth column of values"""

        self._v_axes[:, 0] = new_v_axis_values

    @property
    def v_axes(self):
        """Returns the value axes"""

        return self._v_axes

    @v_axes.setter
    def v_axes(self, new_v_axes_values):
        """Sets the value axes (subject to throw if sizes don't match)"""

        self._v_axes = new_v_axes_values

    def to_ndarray(self):
        """Returns a single np.ndarray where n_axis (now float) is 1st col."""

        return np.hstack((self._n_axis[:, np.newaxis], self._v_axes))

    def index_at(self, n_value: int) -> int:
        """Returns index i such that ds.n_axis[i] == n_value"""

        candidate = np.where(self._n_axis == n_value)[0]
        return candidate[0] if candidate.shape[0] > 0 else None
