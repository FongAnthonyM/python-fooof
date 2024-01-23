""" dataclasses.py

"""
# Package Header #
from ...header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Standard Libraries #
from collections.abc import Callable, Sequence
from typing import NamedTuple

# Third-Party Packages #
import numpy as np

# Local Packages #


# Definitions #
# Classes #
class MeanErrors(NamedTuple):
    """A data class for containing mean errors."""
    mae: np.ndarray
    mse: np.ndarray
    rmse: np.ndarray


class PowerSpectra(NamedTuple):
    """A data class containing a power spectra and its frequencies."""
    spectra: np.ndarray
    frequencies: np.ndarray


class FitSpectrumCurve(NamedTuple):
    """A data class for storing a curve fit and its metrics.

    Attributes:
        curve: The fit curve
        parameters: The parameters used to create the curve.
        method: The method to use create the curve.
        spectra: The original spectra comparing the curve.
        r_squared: The r squared value of the curve and original spectra.
        normal_entropy: The normalized spectral entropy of the flattened signal.
        mae: Mean Absolute Error
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
    """
    curve: np.ndarray
    parameters: np.ndarray
    method: Callable[..., np.ndarray]
    spectra: np.ndarray
    frequencies: np.ndarray
    r_squared: float
    normal_entropy: float
    mae: np.ndarray
    mse: np.ndarray
    rmse: np.ndarray


class FitSpectrumCurves(NamedTuple):
    """A data class for storing fit curves and their metrics.

    Attributes:
        curves: The fit curve
        parameters: The parameters used to create the curve.
        method: The method to use create the curve.
        spectra: The original spectra comparing the curve.
        r_squared: The r squared value of the curve and original spectra.
        normal_entropy: The normalized spectral entropy of the flattened signal.
        mae: Mean Absolute Error
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
    """
    curves: np.ndarray
    parameters: np.ndarray
    method: Callable[..., np.ndarray]
    frequencies: np.ndarray
    spectra: np.ndarray
    r_squared: np.ndarray
    normal_entropy: np.ndarray
    mae: np.ndarray
    mse: np.ndarray
    rmse: np.ndarray
