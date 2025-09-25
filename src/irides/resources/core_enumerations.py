"""
-------------------------------------------------------------------------------

Core enumerations for the irides package.

-------------------------------------------------------------------------------
"""

from enum import Enum


class FilterType(Enum):
    """Universe of filter types."""

    MBOX = 0
    PEMA = 1
    BESSEL = 2
    UNKNOWN = 3


# noinspection SpellCheckingInspection,PyPep8Naming
class FilterDesignType(Enum):
    """Universe of labels that represent implemented filter-operation types."""

    LEVEL = 0
    SLOPE = 1
    CURVE = 2
    UNKNOWN = 3


class TestSignalType(Enum):
    """Universe of test-signal types."""

    IMPULSE = 0
    STEP = 1
    RAMP = 2
    PARABOLA = 3
    ALTERNATING = 4
    WHITE_NOISE = 5
    UNKNOWN = 6
