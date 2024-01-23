""" ooffitter.py
A data processing object which fits a one over f curve to given data or spectra.
"""
# Package Header #
from typing import Tuple

from numpy import ndarray

from ...header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Standard Libraries #
from collections.abc import Callable, Sequence
import warnings

# Third-Party Packages #
from baseobjects import BaseObject, singlekwargdispatchmethod, search_sentinel
from fooof import FOOOF
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, welch
from scipy.stats import entropy

# Local Packages #
from .dataclasses import MeanErrors, PowerSpectra, FitSpectrumCurve, FitSpectrumCurves


# Definitions #
class FitError(Exception):
    """Error for a failure to fit."""


def iterdim(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """Iterates over a given axis of an array.

    Args:
        a: The array to iterate through.
        axis: The axis to iterate over.

    Returns:
        The data at an element of the axis.
    """
    slices = (slice(None),) * axis
    for i in range(a.shape[axis]):
        yield a[slices + (i,)]


def calculate_mean_errors(a: np.ndarray, b: np.ndarray, axis: int | None = None) -> MeanErrors:
    """Calculates the mean errors between two arrays.

    Args:
        a: The primary array.
        b: The secondary array.
        axis: The axis to calculate the mean along.

    Returns:.
        The mean errors between the two arrays
    """
    difference = a - b

    mae = np.abs(difference).mean(axis)
    mse = (difference ** 2).mean(axis)
    rmse = np.sqrt(mse)

    return MeanErrors(mae, mse, rmse)


def remove_zero_frequency(spectra: np.ndarray, freqs: np.ndarray, axis: int = 0, copy_: bool = True) -> PowerSpectra:
    """Removes the zero frequency from spectra.

    Args:
        spectra: The power spectra to trim as an 1D or 2D array
        freqs: Frequency values for the power spectrum as an 1D array.
        axis: The frequencies' axis number on the spectra.
        copy_: Determines if the output arrays will be copies.

    Returns:
        The new trimmed power spectra and frequencies.
    """
    if freqs[0] == 0.0:
        slices = (slice(None),) * axis + (slice(1, None),)
        trimmed_frequencies = freqs[1:]
        trimmed_spectra = spectra[1:] if spectra.ndim == 1 else spectra[slices]
        return PowerSpectra(trimmed_spectra, trimmed_frequencies)
    elif copy_:
        return PowerSpectra(spectra.copy(), freqs.copy())
    else:
        return PowerSpectra(spectra, freqs)


def trim_spectra(
    spectra: np.ndarray,
    freqs: np.ndarray,
    f_range: Sequence[float, float],
    axis: int = 0,
) -> PowerSpectra:
    """Extract a frequency range from power spectra.

    This function extracts frequency ranges >= f_low and <= f_high.
    It does not round to below or above f_low and f_high, respectively.

    Args:
        spectra: The power spectra to trim as an 1D or 2D array
        freqs: Frequency values for the power spectrum as an 1D array.
        f_range: Frequency range to restrict to, as [lowest_freq, highest_freq].
        axis: The frequencies' axis number on the spectra.

    Returns:
        The new trimmed power spectra and frequencies.
    """
    # Create mask to index only requested frequencies
    f_mask = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])

    # Restrict freqs & spectra to requested range
    slices = (slice(None), f_mask) if axis == 0 else (f_mask, slice(None))
    trimmed_frequencies = freqs[f_mask]
    trimmed_spectra = spectra[f_mask] if spectra.ndim == 1 else spectra[slices]

    return PowerSpectra(trimmed_spectra, trimmed_frequencies)


def exponential_knee_fitting(xs: np.ndarray, *args: float) -> np.ndarray:
    """Exponential fitting function, for fitting one over f component with a 'knee'.

    NOTE: this function requires linear frequency (not log).

    Args:
        xs: Input x-axis values as an 1D array.
        *args: Parameters (offset, knee, exp) that define Lorentzian function: y = 10^offset * (1/(knee + x^exp))

    Returns:
        Output values for exponential function.
    """

    ys = np.zeros_like(xs)

    offset, knee, exp = args

    ys = ys + offset - np.log10(knee + xs**exp)

    return ys


def exponential_fitting(xs: np.ndarray, *args: float) -> np.ndarray:
    """Exponential fitting function, for fitting aperiodic component without a 'knee'.

    NOTE: this function requires linear frequency (not log).

    Args:
        xs: Input x-axis values as an 1D array.
        *args: Parameters (offset, exp) that define Lorentzian function: y = 10^off * (1/(x^exp))

    Returns:
        Output values for exponential function, without a knee.
    """

    ys = np.zeros_like(xs)

    offset, exp = args

    ys = ys + offset - np.log10(xs**exp)

    return ys


# Classes #
class OOFFitter(BaseObject):
    """An object that can take time series or power spectra and fit a one over f curve to them.

    Class Attributes:
        fitting_methods: The fitting methods and their associated names.

    Attributes:
        axis: The axis to run the fitting across.
        sample_rate: The sample rate of the incoming signal.
        lower_frequency: The lower limit to run the fitting for.
        upper_frequency: The upper limit to run the fitting for.
        _oof_percentile_thresh: Percentile threshold, to select points from a flat spectrum for an initial aperiodic fit
        _oof_guess: Guess parameters for aperiodic fitting, [offset, knee, exponent]
            If offset guess is None, the first value of the power spectrum is used as offset guess
            If exponent guess is None, the abs(log-log slope) of first & last points is used
        _oof_bounds: The bounds for the fitting ((offset_low_bound, knee_low_bound, exp_low_bound),
                                                 (offset_high_bound, knee_high_bound, exp_high_bound))
        _maxfev: The maximum number of calls to the curve fitting function
        _fitting_mode: The type of fitting to use.
        _fitting_method: The method to use for fitting.

    Args:
        sample_rate: The sample rate of the incoming signal.
        axis: The axis to run the fitting across.
        init: Determines if this object will construct.
    """
    fitting_methods: dict[str, Callable] = {"fixed": exponential_fitting, "knee": exponential_knee_fitting}

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, sample_rate: float | None = None, axis: int | None = None, init: bool = True) -> None:
        # New Attributes #
        # Signal Information
        self.axis: int = 0
        self.channel_axis: int = 1
        self.sample_rate: float | None = None

        # Power Spectra
        self.lower_frequency: float | None = None
        self.upper_frequency: float | None = None
        self._spectra_method: Callable[..., np.ndarray] = self.fft

        # Fitting
        self._oof_percentile_thresh: float = 0.025
        self._oof_guess = (None, 0, None)
        # Bounds for aperiodic fitting, as: ((offset_low_bound, knee_low_bound, exp_low_bound),
        #                                    (offset_high_bound, knee_high_bound, exp_high_bound))
        # By default, aperiodic fitting is unbound, but can be restricted here, if desired
        #   Even if fitting without knee, leave bounds for knee (they are dropped later)
        self._oof_bounds: tuple = ((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf))
        self._maxfev: int = 5000

        self._fitting_mode: str = "fixed"
        self._fitting_method: Callable[..., np.ndarray] = exponential_fitting

        # Object Construction #
        if init:
            self.construct(sample_rate=sample_rate, axis=axis)

    # Instance Methods #
    # Constructors/Destructors
    def construct(self, sample_rate: float | None = None, axis: int | None = None) -> None:
        """Construct this object.

        Args:
            sample_rate: The sample rate of the incoming signal
            axis: The axis to run the fitting across.
        """
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if axis is not None:
            self.axis = axis

    def set_spectra_method(self, method: Callable | str) -> None:
        """Sets the __get__ method to another function or a method name within this object can be given to select it.

        Args:
            method: The function to set the __get__ method to.
        """
        if isinstance(method, str):
            self._spectra_method = getattr(self, method)
        else:
            self._spectra_method = method

    @property
    def fitting_mode(self) -> str:
        """The type of fitting to use."""
        return self._fitting_mode

    @fitting_mode.setter
    def fitting_mode(self, value: str) -> None:
        self._fitting_mode = value
        self.set_fitting_method(value)

    @property
    def fitting_method(self) -> Callable[..., np.ndarray]:
        """The method to use for fitting."""
        return self._fitting_method

    @fitting_method.setter
    def fitting_method(self, value: Callable[..., np.ndarray] | str) -> None:
        self.set_fitting_method(value)

    # Setters
    @singlekwargdispatchmethod("method")
    def set_fitting_method(self, method: Callable | str) -> None:
        """Sets the fitting method of this object.

        Args:
            method: The method or name of method to use for fitting.
        """
        raise TypeError(f"{type(method)} can not be used to set the fitting method of {type(self)}")

    @set_fitting_method.register(Callable)
    def _(self, method: Callable[..., np.ndarray]) -> None:
        """Sets the fitting method of this object.

        Args:
            method: The method or name of method to use for fitting.
        """
        self._fitting_method = method

    @set_fitting_method.register
    def _(self, method: str) -> None:
        """Sets the fitting method of this object.

        Args:
            method: The method or name of method to use for fitting.
        """
        method = self.fitting_methods.get(method, search_sentinel)
        if method is search_sentinel:
            raise ValueError("The provide method name is not a fitting method.")
        else:
            self._fitting_method = method

    # Data Preparation
    def fft(self, data: np.ndarray, sample_rate: float | None =None, axis: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Uses the fast fourier transform to calculate the power spectra of the given signals in a numpy array.

        Args:
            data: The data to find the power spectra of.
            axis: The axis over which to compute the power spectra.

        Returns:
            The power spectra of the given data.
        """
        axis = self.axis if axis is None else axis
        sample_rate = self.sample_rate if sample_rate is None else sample_rate
        f_transform = np.fft.rfft(data, axis=axis)
        spectra = np.square(np.abs(f_transform))
        freqs = np.linspace(0, sample_rate / 2, spectra.shape[axis])
        return freqs, spectra

    def welch(self, data: np.ndarray, sample_rate: float | None = None, axis: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        axis = self.axis if axis is None else axis
        sample_rate = self.sample_rate if sample_rate is None else sample_rate
        return welch(data, sample_rate, nperseg=2048, axis=axis)

    def _prepare_timeseries(
        self,
        data: np.ndarray,
        sample_rate: float | None = None,
        f_range: Sequence[float, float] | None = None,
        axis: int | None = None,
        smoothing: bool = False,
    ) -> PowerSpectra:
        """Prepares a set of timeseries for fitting the one over f curve.

        Args:
            data: The timeseries to prepare.
            sample_rate: The sample rate of the data.
            f_range: Frequency range to restrict to, as [lowest_freq, highest_freq].
            axis: The axis to get power spectra of.
            smoothing: Determines if smoothing will be done on the spectra.

        Returns:
            The prepared power spectra for one over f fitting.
        """
        axis = self.axis if axis is None else axis
        sample_rate = self.sample_rate if sample_rate is None else sample_rate

        # Create Power Spectra and Frequencies
        freqs, spectra = self._spectra_method(data, sample_rate, axis)

        # Limit Frequency Range
        if f_range is not None:
            lower_limit = int(np.searchsorted(freqs, f_range[0], side="right") - 1)
            lower_limit = 1 if lower_limit < 1 else lower_limit
        elif self.lower_frequency is not None:
            lower_limit = int(np.searchsorted(freqs, self.lower_frequency, side="right") - 1)
            lower_limit = 1 if lower_limit < 1 else lower_limit
        else:
            lower_limit = 1

        if f_range is not None:
            upper_limit = int(np.searchsorted(freqs, f_range[1], side="right") - 1)
        elif self.upper_frequency is not None:
            upper_limit = int(np.searchsorted(freqs, self.upper_frequency, side="right"))
        else:
            upper_limit = freqs.shape[axis]

        spectra = spectra[(slice(None),) * axis + (slice(lower_limit, upper_limit),)]
        freqs = freqs[lower_limit:upper_limit]

        # Put Spectra in Log Space
        spectra = np.log10(spectra)

        if smoothing:
            smooth_spectra = np.empty(spectra.shape)
            c_slices = (slice(None),) * self.channel_axis

            for i, spectrum in enumerate(iterdim(spectra, self.channel_axis)):
                curve_slices = c_slices + (i,)
                smooth_spectra[curve_slices] = savgol_filter(spectrum, 12, 5)
            spectra = smooth_spectra

        return PowerSpectra(spectra, freqs)

    def _prepare_spectra(
        self,
        spectra: np.ndarray,
        freqs: np.ndarray,
        freq_range: Sequence[int, int] | None = None,
        axis: int | None = None,
    ) -> PowerSpectra:
        """Prepare power spectra for fitting.

        Args:
            spectra : Power values, which must be input in linear space as an 1D or 2D array.
            freqs: Frequency values for the power spectrum, in linear space as an 1D array.
            freq_range: Frequency range to restrict power spectrum to. If None, keeps the entire range.
            axis: The frequencies' axis number on the spectra.

        Returns:
            The prepared power spectra.

        Raises:
            ValueError: If there is an issue with the input spectra or frequencies.
        """
        axis = self.axis if axis is None else axis

        # Validation #
        # Check that data have the right dimensionality
        if freqs.ndim != 1:
            raise ValueError("Inputs are not the right dimensions.")

        # Check that data sizes are compatible
        if freqs.shape[-1] != spectra.shape[-1]:
            raise ValueError("The input frequencies and power spectra are not consistent sizes.")

        # Check if power values are complex
        if np.iscomplexobj(spectra):
            raise ValueError("Input power spectra are complex values which are not supported")

        # Data Modification #
        # Force data to be dtype of float64
        if freqs.dtype != 'float64':
            freqs = freqs.astype('float64')
        if spectra.dtype != 'float64':
            spectra = spectra.astype('float64')

        # Remove Zero Frequency
        spectra, freqs = remove_zero_frequency(spectra, freqs, axis, copy_=False)

        # Check frequency range, trim the power_spectrum range if requested
        if freq_range is not None:
            spectra, freqs = trim_spectra(spectra, freqs, freq_range, axis)

        # Log power values
        spectra = np.log10(spectra)

        # Check if there are any infs / nans, and raise an error if so
        if np.any(np.isinf(spectra)) or np.any(np.isnan(spectra)):
            raise ValueError("The input power spectra data, after logging, contains NaNs or Infs."
                             "This will cause the fitting to fail. "
                             "One reason this can happen is if inputs are already logged. "
                             "Inputs data should be in linear spacing, not log.")

        return PowerSpectra(spectra, freqs)

    # Curve Generation
    def generate_oof(self, freqs: np.ndarray, *fit_params: float) -> np.ndarray:
        """Generate one over f curve values.

        Args:
            freqs: Frequency values for the power spectrum, in linear space as an 1D array.
            fit_params: Parameters that define the one over f curve.

        Returns:
            The one over f curve, in log10 spacing.
        """
        return self._fitting_method(freqs, *fit_params)

    # Fitting
    def _simple_oof_fit(self, spectrum: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Fit the one over f of the power spectrum.

        Args:
            spectrum: Power values, which must be input in linear space as an 1D array.
            freqs: Frequency values for the power spectrum, in linear space as an 1D array.

        Returns:
            The parameter estimates for aperiodic fit as an 1D array.

        Raises:
            FitError: If the fitting encounters an error.
        """

        # Get the guess parameters and/or calculate from the data, as needed.
        # Note that these are collected as lists, to concatenate with or without knee later
        off_guess = [spectrum[0] if not self._oof_guess[0] else self._oof_guess[0]]
        kne_guess = [self._oof_guess[1]] if self._fitting_mode == 'knee' else []
        exp_guess = [np.abs(spectrum[-1] - spectrum[0] / np.log10(freqs[-1]) - np.log10(freqs[0]))
                     if not self._oof_guess[2] else self._oof_guess[2]]
        guess = np.array([off_guess + kne_guess + exp_guess])

        # Ignore warnings that are raised in curve_fit.
        # A runtime warning can occur while exploring parameters in curve fitting.
        # This doesn't affect outcome - it won't settle on an answer that does this.
        # It happens if / when b < 0 & |b| > x**2, as it leads to log of a negative number.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof_params, _ = curve_fit(
                    self._fitting_method,
                    freqs,
                    spectrum,
                    p0=guess,
                    maxfev=self._maxfev,
                    bounds=self._oof_bounds,
                )
        except RuntimeError:
            raise FitError("Model fitting failed due to not finding parameters in "
                           "the simple one over f fit.")

        return oof_params
    
    def _robust_oof_fit(self, spectrum: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Fit the one over f spectrum robustly, ignoring outliers.

        Args:
            spectrum: Power values, which must be input in linear space as an 1D array.
            freqs: Frequency values for the power spectrum, in linear space as an 1D array.

        Returns:
            The parameter estimates for aperiodic fit as an 1D array.

        Raises:
            FitError: If the fitting encounters an error.
        """

        # Do a quick, initial aperiodic fit
        popt = self._simple_oof_fit(spectrum, freqs)
        initial_fit = self._fitting_method(freqs, *popt)

        # Flatten power_spectrum based on initial aperiodic fit
        flatspec = spectrum - initial_fit
        flatspec[flatspec < 0] = 0  # Flatten outliers that drop below 0

        # Use percentile threshold, in terms of # of points, to extract and re-fit
        perc_thresh = np.percentile(flatspec, self._oof_percentile_thresh)
        perc_mask = flatspec <= perc_thresh
        freqs_ignore = freqs[perc_mask]
        spectrum_ignore = spectrum[perc_mask]

        # Second aperiodic fit - using results of first fit as guess parameters
        # Ignore warnings that are raised in curve_fit.
        # A runtime warning can occur while exploring parameters in curve fitting.
        # This doesn't affect outcome - it won't settle on an answer that does this.
        # It happens if / when b < 0 & |b| > x**2, as it leads to log of a negative number.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof_params, _ = curve_fit(
                    self._fitting_method,
                    freqs_ignore,
                    spectrum_ignore,
                    p0=popt,
                    maxfev=self._maxfev,
                    bounds=self._oof_bounds,
                )
        except RuntimeError:
            raise FitError("Model fitting failed due to not finding "
                           "parameters in the robust one over f fit.")
        except TypeError:
            raise FitError("Model fitting failed due to sub-sampling in the robust one over f fit.")

        return oof_params

    def single_fit_power(self, spectrum: np.ndarray, freqs: np.ndarray) -> FitSpectrumCurve:
        """Fit a power spectrum to a one over f signal, without data checking.

        Args:
            spectrum: Power values, which must be input in linear space as an 1D array.
            freqs: Frequency values for the power spectrum, in linear space as an 1D array.

        Returns
            The fit curve and its metrics.
        """
        # Fitting
        oof_params = self._robust_oof_fit(spectrum, freqs)
        oof_curve = self._fitting_method(freqs, *oof_params)

        # Calculate Fitting Statistics
        r_val = np.corrcoef(spectrum, oof_curve)
        r_squared = r_val[0][1] ** 2

        errors = calculate_mean_errors(spectrum, oof_curve)

        curve_removed = spectrum - oof_curve
        curve_2 = curve_removed ** 2
        prob = curve_2 / np.sum(curve_2, axis=0)
        entro = entropy(prob)
        normal_entropy = entro / np.log(prob.shape[0])

        return FitSpectrumCurve(
            curve=oof_curve,
            parameters=oof_params,
            method=self._fitting_method,
            frequencies=freqs,
            r_squared=r_squared,
            spectra=spectrum,
            normal_entropy=normal_entropy,
            mae=errors.mae,
            mse=errors.mse,
            rmse=errors.rmse,
        )

    def multiple_fit_power(
        self,
        spectra: np.ndarray,
        freqs: np.ndarray,
        c_axis: int | None = None,
    ) -> FitSpectrumCurves:
        """Fit multiple power spectra to a one over f signal, without data checking.

        Args:
            spectra: Power values, which must be input in linear space as a 2D array.
            freqs: Frequency values for the power spectrum, in linear space as an 1D array.
            c_axis: The axis of channels in the spectra.

        Returns:
            The fit curves and their metrics.
        """
        c_axis = self.channel_axis if c_axis is None else c_axis
        shape = spectra.shape
        n_channels = spectra.shape[c_axis]
        oof_shape = [None, None]
        oof_shape[0] = n_channels if c_axis == 0 else 3 if self._fitting_mode == 'knee' else 2
        oof_shape[1] = n_channels if c_axis > 0 else 3 if self._fitting_mode == 'knee' else 2

        oof_params = np.empty(tuple(oof_shape))
        oof_curves = np.empty(shape)
        r_squared = np.empty((n_channels,))
        normal_entropy = np.empty((n_channels,))
        mae = np.empty((n_channels,))
        mse = np.empty((n_channels,))
        rmse = np.empty((n_channels,))

        param_slices = [slice(None)] * 2
        c_slices = (slice(None),) * c_axis

        for i, spectrum in enumerate(iterdim(spectra, c_axis)):
            param_slices[c_axis] = i

            curve_slices = c_slices + (i,)

            oof_params[tuple(param_slices)] = oof_args = self._robust_oof_fit(spectrum, freqs)
            oof_curves[curve_slices] = oof_curve = self._fitting_method(freqs, *oof_args)

            # R Squared
            r_val = np.corrcoef(spectrum, oof_curve)
            r_squared[i] = r_val[0][1] ** 2

            difference = spectrum - oof_curve
            # Normalized Spectral Entropy
            curve_2 = difference ** 2
            prob = curve_2 / np.sum(curve_2, axis=0)
            entro = entropy(prob)
            normal_entropy[i] = entro / np.log(prob.shape[0])

            # Mean Errors
            mae[i] = np.abs(difference).mean()
            mse[i] = (difference ** 2).mean()
            rmse[i] = np.sqrt(mse[i])

        return FitSpectrumCurves(
            curves=oof_curves,
            parameters=oof_params,
            method=self._fitting_method,
            frequencies=freqs,
            r_squared=r_squared,
            spectra=spectra,
            normal_entropy=normal_entropy,
            mae=mae,
            mse=mse,
            rmse=rmse
        )

    def fit_power(
        self,
        spectra: np.ndarray,
        freqs: np.ndarray,
        f_range: Sequence[int, int] | None = None,
        axis: int | None = None,
    ) -> FitSpectrumCurve | tuple[FitSpectrumCurve]:
        """Fit a power spectrum to an one over f signal.

        Args:
            spectra: Power values, which must be input in linear space.
            freqs: Frequency values for the power spectrum, in linear space as an 1D array.
            f_range: Frequency range to restrict to, as [lowest_freq, highest_freq].
            axis: The axis over which to fit one over f curves.

        Returns:
            The fit one over f curves.
        """
        spectra, freqs = self._prepare_spectra(spectra, freqs, f_range, axis)

        if spectra.ndim == 1:
            return self.single_fit_power(spectra, freqs)
        else:
            curves = [None] * spectra.shape[axis]
            for i, spectrum in enumerate(iterdim(spectra, axis)):
                curves[i] = self.single_fit_power(spectra, freqs)

            return tuple(curves)

    def fit_timeseries(
        self,
        data: np.ndarray,
        sample_rate: float | None = None,
        f_range: Sequence[float, float] | None = None,
        axis: int | None = None,
        c_axis: int | None = None,
    ) -> FitSpectrumCurve | FitSpectrumCurves:
        """Fit a time series to a one over f signal, without data checking.

        Args:
            data: The timeseries to prepare.
            sample_rate: The sample rate of the data.
            f_range: Frequency range to restrict to, as [lowest_freq, highest_freq].
            axis: The axis to get power spectra of.
            c_axis: The axis of channels in the spectra.

        Returns:
            The fit one over f curves.
        """
        spectra, freqs = self._prepare_timeseries(data, sample_rate, axis, f_range)

        if spectra.ndim == 1:
            return self.single_fit_power(spectrum=spectra, freqs=freqs)
        else:
            return self.multiple_fit_power(spectra=spectra, freqs=freqs, c_axis=c_axis)
