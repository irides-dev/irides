"""
-------------------------------------------------------------------------------

Simple containers to structure points.

-------------------------------------------------------------------------------
"""

import numpy as np


class ContinuousTimePoint(object):
    """
    Container for a continuous-time time point and associated
    series value.
    """

    def __init__(self, time=None, value=None):
        """Constructor takes time and value; defaults to None."""

        self.time = time
        self.value = value


class ContinuousTimePoints(object):
    def __init__(self, time: np.ndarray, value: np.ndarray):

        self.points = np.array([time, value]).T

    def times(self) -> np.ndarray:

        return self.points[:, 0]

    def values(self) -> np.ndarray:

        return self.points[:, 1]


class ContinuousFrequencyPoint(object):
    """
    Container for a continuous-frequency frequency point
    and associated value.
    """

    def __init__(self, frequency=None, value=None):
        """Constructor takes time and value; defaults to None."""

        self.frequency = frequency
        self.value = value


class DiscreteTimePoint(object):
    """
    Container for a discrete-time time point and associated
    series value.
    """

    def __init__(self, index=None, value=None):
        """Constructor takes index and value; defaults to None."""

        self.index = index
        self.value = value
