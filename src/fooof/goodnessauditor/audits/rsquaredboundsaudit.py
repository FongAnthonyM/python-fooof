""" rsquaredboundsaudit.py

"""
# Package Header #
from .....header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #

# Third-Party Packages #
from baseobjects import singlekwargdispatchmethod
import numpy as np

# Local Packages #
from ...dataclasses import FitSpectrumCurve, FitSpectrumCurves
from .goodnessauditinterface import GoodnessAuditInterface


# Definitions #
# Classes #
class RSquaredBoundsAudit(GoodnessAuditInterface):
    """A goodness audit which uses R squared to determine goodness, particularly within a bounded R squared range.

    The goodness values are R squared values rescaled with the minimum and maximum values being 0 and 1 respectively.
    Any R squared values outside the bounds are clipped to the minimum and maximum values.

    Class Attributes:
        default_minimum: The default lowest bound of the R squared to be considered 0 goodness.
        default_maximum: The default highest bound of the R squared to be considered 1 goodness.

    Attributes:
        minimum: The lowest bound of the R squared to be considered 0 goodness.
        maximum: The highest bound of the R squared to be considered 1 goodness.

    Args:
        min_: The minimum bound of the R squared range for goodness.
        max_: The maximum bound of the R squared range for goodness.
        init: Determines if this object will be constructed.
    """
    default_minimum: float = 0.50
    default_maximum: float = 0.55

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, min_: float | None = None, max_: float | None = None, init: bool = True) -> None:
        # New Attributes #
        self.minimum: float = self.default_minimum
        self.maximum: float = self.default_maximum

        # Object Construction #
        if init:
            self.construct(min_=min_, max_=max_)

    # Instance Methods #
    # Constructors/Destructors
    def construct(self, min_: float | None, max_: float | None = None,) -> None:
        """Constructs this object.

        Args:
            min_: The minimum bound of the r_squared range for goodness.
            max_: The maximum bound of the r_squared range for goodness.
        """
        if min_ is not None:
            self.minimum = min_

        if max_ is not None:
            self.maximum = max_

    # Auditing
    @singlekwargdispatchmethod("info")
    def run_audit(self, info: float | np.ndarray | FitSpectrumCurve | FitSpectrumCurves) -> float | np.ndarray:
        """Runs the audit to determine the goodness of the fit.

        Args:
            info: Either a FitSepctrumCurve data object or the R squared values of the curves.

        Returns:
            The goodness values of the curves.
        """
        raise ValueError(f"{self.__class__} cannot run an audit with a {type(info)}.")

    @run_audit.register
    def __run_audit(self, info: float) -> float:
        """Runs the audit to determine the goodness of the fit.

        Args:
            info: The R squared value to determine its goodness.

        Returns:
            The goodness value of the curve.
        """
        return self.single_goodness_audit(r_squared=info)

    @run_audit.register
    def __run_audit(self, info: np.ndarray) -> np.ndarray:
        """Runs the audit to determine the goodness of the fits.

        Args:
            info: The R squared values to determine their goodness.

        Returns:
            The goodness values of the curves.
        """
        return self.multiple_goodness_audit(r_squared=info)

    @run_audit.register
    def __run_audit(self, info: FitSpectrumCurve) -> float:
        """Runs the audit to determine the goodness of the fit.

        Args:
            info: The curve and its metrics that will determine its goodness.

        Returns:
            The goodness values of the curve.
        """
        return self.single_goodness_audit(r_squared=info.r_squared)

    @run_audit.register
    def __run_audit(self, info: FitSpectrumCurves) -> np.ndarray:
        """Runs the audit to determine the goodness of the fits.

        Args:
            info: The curves and their metrics that will determine their goodness.

        Returns:
            The goodness values of the curves.
        """
        return self.multiple_goodness_audit(r_squared=info.r_squared)

    def single_goodness_audit(self, r_squared: float) -> float:
        """Evaluates the goodness for a single fit based on the r_squared value.

        Args:
            r_squared: A single R squared value to evaluate the goodness.

        Returns:
            The goodness value as a scalar between 0 and 1.
        """
        if r_squared <= self.minimum:
            return 0.0
        elif r_squared >= self.maximum:
            return 1.0
        else:
            return (r_squared - self.minimum) / (self.maximum - self.minimum)

    def multiple_goodness_audit(self, r_squared: np.ndarray) -> np.ndarray:
        """Evaluates the goodness for multiple fits based on the r_squared values.

        Args:
            r_squared: R squared values to evaluate the goodness of each fit.

        Returns:
            The goodness values as an array of scalars between 0 and 1.
        """
        return (np.clip(r_squared, self.minimum, self.maximum) - self.minimum) / (self.maximum - self.minimum)
