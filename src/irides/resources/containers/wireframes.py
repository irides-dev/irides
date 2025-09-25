"""
-------------------------------------------------------------------------------

Container for wireframes.

-------------------------------------------------------------------------------
"""

import numpy as np
from dataclasses import dataclass

from irides.resources.core_enumerations import FilterDesignType


# noinspection SpellCheckingInspection
@dataclass
class WireframeContinuousTime(object):
    """Container for level, slope, or curvature wireframes.

    type: FilterDesignType -- {LEVEL|SLOPE|CURVE}
    timepoints: np.ndarray -- 1, 2, or 3 timepoints
    weights: np.ndarray -- timepoint weights so that the * gain is unity
    change_interval: float -- timespan over which the wireframe covers
    mid_point: float -- mean of the timepoints
    """

    # locals
    type = FilterDesignType.UNKNOWN
    timepoints = np.array([])
    weights = np.array([])
    change_interval = None
    mid_point = None

    def __init__(
        self, design_type: FilterDesignType, wireframe_timepoints: np.ndarray
    ):
        """Initializes an instance.

        Parameters
        ----------
        design_type: FilterDesignType
            Type of wireframe that this instance refers to
        wireframe_timepoints: np.ndarray
            Wireframe timepoints
        """

        # validate type
        if design_type == FilterDesignType.UNKNOWN:
            raise RuntimeError("Wireframe cannot be `unknown`.")

        # capture type
        self.type = design_type

        # capture timepoints
        self.timepoints = np.asarray(wireframe_timepoints)

        # map to calculation functions
        set_weights = set_weights_functions[design_type]
        change_function = change_functions[design_type]
        midpoint_function = midpoint_functions[design_type]

        # exec
        set_weights(self)
        self.change_interval = change_function(self)
        self.mid_point = midpoint_function(self)

        # validate config
        if design_type == FilterDesignType.LEVEL:
            validate_check_or_die(self, 1, "level")
        if design_type == FilterDesignType.SLOPE:
            validate_check_or_die(self, 2, "slope")
        if design_type == FilterDesignType.CURVE:
            validate_check_or_die(self, 3, "curvature")


"""
-------------------------------------------------------------------------------

Supporting functions, specific to wireframe type

-------------------------------------------------------------------------------
"""


def set_level_weights(self):
    self.weights = np.zeros(1)
    self.weights[0] = 1.0


def set_slope_weights(self):
    h = self.timepoints[1] - self.timepoints[0]
    self.weights = np.zeros(2)
    self.weights[0] = 1.0 / h
    self.weights[1] = -1.0 / h


def set_curvature_weights(self):
    h = self.timepoints[2] - self.timepoints[0]
    self.weights = np.zeros(3)
    self.weights[0] = 1.0 / np.power(h / 2.0, 2)
    self.weights[1] = -2.0 / np.power(h / 2.0, 2)
    self.weights[2] = 1.0 / np.power(h / 2.0, 2)


set_weights_functions = {
    FilterDesignType.LEVEL: set_level_weights,
    FilterDesignType.SLOPE: set_slope_weights,
    FilterDesignType.CURVE: set_curvature_weights,
}


change_functions = {
    FilterDesignType.LEVEL: lambda self: 0.0,
    FilterDesignType.SLOPE: lambda self: self.timepoints[1]
    - self.timepoints[0],
    FilterDesignType.CURVE: lambda self: self.timepoints[2]
    - self.timepoints[0],
}

midpoint_functions = {
    FilterDesignType.LEVEL: lambda self: self.timepoints[0],
    FilterDesignType.SLOPE: lambda self: np.mean(
        [self.timepoints[0], self.timepoints[1]]
    ),
    FilterDesignType.CURVE: lambda self: np.mean(
        [self.timepoints[0], self.timepoints[2]]
    ),
}


def validate_check_or_die(self, count: int, label: str):
    if self.timepoints.shape[0] != count:
        raise RuntimeError(
            "{0} filter must have {1} timepoints".format(label, count)
        )
