""" fitspectrumcurverecords.py

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
import datetime

# Third-Party Packages #
from baseobjects import BaseObject
import h5py
from hdf5objects import HDF5Map, HDF5Group
from hdf5objects.datasets import TimeSeriesMap
import numpy as np

# Local Packages #
from .dataclasses import FitSpectrumCurves


# Definitions #
# Classes #
class FitSpectrumCurveRecords(BaseObject):
    # Magic Methods #
    # Construction/Destruction
    def __init__(self, init=True) -> None:
        # Attributes #
        self.data: np.ndarray | None = None
        self.spectra: np.ndarray | None = None
        self.curves: np.ndarray | None = None
        self.frequencies: np.ndarray | None = None
        self.parameters: np.ndarray | None = None
        self.fit_method: np.ndarray | None = None
        self.r_squared: np.ndarray | None = None
        self.mae: np.ndarray | None = None
        self.mse: np.ndarray | None = None
        self.rmse: np.ndarray | None = None

        # Object Construction #
        if init:
            self.construct()

    # Instance Methods #
    # Constructors/Destructors
    def construct(self) -> None:
        pass

    def find_record(
        self,
        timestamp: datetime.datetime | float,
        approx: bool = False,
        tails: bool = False,
    ) -> FitSpectrumCurves:
        pass

    def find_record_range(
        self,
        start: datetime.datetime | float | None = None,
        stop: datetime.datetime | float | None = None,
        step: int | float | datetime.timedelta | None = None,
        approx: bool = False,
        tails: bool = False,
    ) -> FitSpectrumCurves:
        pass


class FitSpectrumCurveRecordsMap(HDF5Map):
    """A map of a group which holds the records of fit spectrum curves."""
    default_attribute_names = {"default_fit_method": "default_fit_method"}
    default_attributes = {"default_fit_method": ""}
    default_map_names = {
        "data": "data",
        "spectra": "spectra",
        "curves": "curves",
        "frequencies": "frequencies",
        "parameters": "parameters",
        "fit_method": "fit_method",
        "r_squared": "r_squared",
        "mae": "mae",
        "mse": "mse",
        "rmse": "rmse",
    }
    default_maps = {
        "data": TimeSeriesMap(shape=(0, 0, 0), maxshape=(None, None, None), dtype='f8'),
        "spectra": TimeSeriesMap(shape=(0, 0, 0), maxshape=(None, None, None), dtype='f8'),
        "curves": TimeSeriesMap(shape=(0, 0, 0), maxshape=(None, None, None), dtype='f8'),
        "frequencies": TimeSeriesMap(shape=(0, 0, 0), maxshape=(None, None, None), dtype='f8'),
        "parameters": TimeSeriesMap(shape=(0, 0, 0), maxshape=(None, None, None), dtype='f8'),
        "fit_method": TimeSeriesMap(shape=(0, 0, 0), maxshape=(None, None), dtype=h5py.string_dtype(encoding='utf-8')),
        "r_squared": TimeSeriesMap(shape=(0, 0), maxshape=(None, None), dtype='f8'),
        "mae": TimeSeriesMap(shape=(0, 0), maxshape=(None, None), dtype='f8'),
        "mse": TimeSeriesMap(shape=(0, 0), maxshape=(None, None), dtype='f8'),
        "rmse": TimeSeriesMap(shape=(0, 0), maxshape=(None, None), dtype='f8'),
    }


class HDF5FitSpectrumCurveRecords(HDF5Group):
    default_map = FitSpectrumCurveRecordsMap()

    @property
    def default_fit_method(self) -> float | h5py.Empty:
        """The default fit method of the curves."""
        return self.attributes["default_fit_method"]

    @default_fit_method.setter
    def default_fit_method(self, value: int | float) -> None:
        self.attributes.set_attribute("default_fit_method", value)


    
# Assign Cyclic Definitions
FitSpectrumCurveRecordsMap.default_type = HDF5FitSpectrumCurveRecords
