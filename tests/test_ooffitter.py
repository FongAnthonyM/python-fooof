""" test_ooffitter.py

"""
# Package Header #
from src.spikedetection.header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Standard Libraries #
import abc
import cProfile
import datetime
import importlib
import itertools
import io
import os
import pathlib
import pickle
import pstats
from pstats import Stats, f8, func_std_string
import random
import time
import warnings
from typing import NamedTuple

# Third-Party Packages #
from dspobjects.plot import Figure, TimeSeriesPlot, SpectraPlot, BarPlot
from fooof.sim.gen import gen_aperiodic
from fooof import FOOOF, FOOOFGroup
import hdf5objects
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.io import loadmat
from scipy.stats import entropy
from scipy.signal import savgol_filter
import sklearn
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import toml
import torch
from torch import nn
from xltektools.hdf5framestructure import XLTEKStudyFrame

# Local Packages #
from src.spikedetection.artifactrejection.fooof.goodnessauditor import GoodnessAuditor, RSquaredBoundsAudit, SVMAudit
from src.spikedetection.artifactrejection.fooof.ooffitter import OOFFitter, iterdim


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# Definitions #
# Classes #
class R2Map(hdf5objects.basehdf5.BaseHDF5Map):
    default_attribute_names = {"file_type": "FileType",
                               "file_version": "FileVersion",
                               "subject_id": "subject_id",
                               "start": "start",
                               "end": "end",
                               "window_size": "window_size"}
    default_map_names = {"data": "data"}
    default_maps = {"data": hdf5objects.datasets.TimeSeriesMap()}


class R2HDF5(hdf5objects.BaseHDF5):
    FILE_TYPE = "R2OverTime"
    default_map = R2Map()


class StatsMicro(Stats):
    def print_stats(self, *amount):
        for filename in self.files:
            print(filename, file=self.stream)
        if self.files:
            print(file=self.stream)
        indent = " " * 8
        for func in self.top_level:
            print(indent, func_get_function_name(func), file=self.stream)

        print(indent, self.total_calls, "function calls", end=" ", file=self.stream)
        if self.total_calls != self.prim_calls:
            print("(%d primitive calls)" % self.prim_calls, end="  ", file=self.stream)
        print("in %.3f microseconds" % (self.total_tt * 1000000), file=self.stream)
        print(file=self.stream)
        width, list = self.get_print_list(amount)
        if list:
            self.print_title()
            for func in list:
                self.print_line(func)
            print(file=self.stream)
            print(file=self.stream)
        return self

    def print_title(self):
        print('          ncalls          tottime      percall      cumtime      percall', end=' ', file=self.stream)
        print('filename:lineno(function)', file=self.stream)

    def print_line(self, func):  # hack: should print percentages
        cc, nc, tt, ct, callers = self.stats[func]
        c = str(nc)
        if nc != cc:
            c = c + "/" + str(cc)
        print(c.rjust(20), end=" ", file=self.stream)
        print(str(f8(tt * 1000000)).rjust(12), end=" ", file=self.stream)
        if nc == 0:
            print(" " * 8, end=" ", file=self.stream)
        else:
            print(str(f8(tt / nc * 1000000)).rjust(12), end=" ", file=self.stream)
        print(str(f8(ct * 1000000)).rjust(12), end=" ", file=self.stream)
        if cc == 0:
            print(" " * 8, end=" ", file=self.stream)
        else:
            print(str(f8(ct / cc * 1000000)).rjust(12), end=" ", file=self.stream)
        print(func_std_string(func), file=self.stream)


# Data Classes
class ElectrodeLead(NamedTuple):
    name: str
    type: str
    contacts: dict


# Functions #
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def closest_square(n):
    n = int(n)
    i = int(np.ceil(np.sqrt(n)))
    while True:
        if (n % i) == 0:
            break
        i += 1
    assert n == (i * (n // i))
    return i, n // i


def get_lead_groups(el_label, el_type):
    assert len(el_label) == len(el_type)

    LEAD_NAME_NOID = np.array([''.join(map(lambda c: '' if c in '0123456789' else c, ll))
        for ll in el_label])
    CONTACT_IX = np.arange(len(el_label))
    LEAD_NAME = np.unique(LEAD_NAME_NOID)

    lead_group = {}
    for l_name in LEAD_NAME:
        lead_group[l_name] = \
            {'Contacts': el_label[np.flatnonzero(LEAD_NAME_NOID == l_name)],
             'IDs': CONTACT_IX[np.flatnonzero(LEAD_NAME_NOID == l_name)],
             'Type': np.unique(el_type[np.flatnonzero(LEAD_NAME_NOID == l_name)])}
        assert len(lead_group[l_name]['Type']) == 1

        lead_group[l_name]['Type'] = lead_group[l_name]['Type'][0]

    return lead_group


def make_bipolar(lead_group):
    for l_name in lead_group:
        sel_lead = lead_group[l_name]
        n_contact = len(sel_lead['IDs'])
        if 'grid' in sel_lead['Type']:
            n_row, n_col = closest_square(n_contact)
        else:
            n_row, n_col = [n_contact, 1]

        CA = np.arange(len(sel_lead['Contacts'])).reshape((n_row, n_col), order='F')

        lead_group[l_name]['Contact_Pairs_ix'] = []

        if n_row > 1:
            for bp1, bp2 in zip(CA[:-1, :].flatten(), CA[1:, :].flatten()):
                lead_group[l_name]['Contact_Pairs_ix'].append(
                        (sel_lead['IDs'][bp1],
                         sel_lead['IDs'][bp2]))

        if n_col > 1:
            for bp1, bp2 in zip(CA[:, :-1].flatten(), CA[:, 1:].flatten()):
                lead_group[l_name]['Contact_Pairs_ix'].append(
                        (sel_lead['IDs'][bp1],
                         sel_lead['IDs'][bp2]))

        """
        if (n_row > 1) & (n_col > 1):
            for bp1, bp2 in zip(CA[:-1, :-1].flatten(), CA[1:, 1:].flatten()):
                lead_group[l_name]['Contact_Pairs_ix'].append(
                        (sel_lead['IDs'][bp1],
                         sel_lead['IDs'][bp2]))
        lead_group[l_name]['Contact_Pairs_ix'] = np.array(
            lead_group[l_name]['Contact_Pairs_ix'])

        lead_group[l_name]['Contact_Pairs_ix'] = \
            lead_group[l_name]['Contact_Pairs_ix'][
                np.argsort(lead_group[l_name]['Contact_Pairs_ix'][:, 0])]
        """

    return lead_group


def make_bipolar_elecs_all(eleclabels, eleccoords):

    lead_group = get_lead_groups(eleclabels[:, 1], eleclabels[:, 2])
    lead_group = make_bipolar(lead_group)

    bp_elecs_all = {
            'IDX': [],
            'Anode': [],
            'Cathode': [],
            'Lead': [],
            'Contact': [],
            'Contact_Abbr': [],
            'Type': [],
            'x': [],
            'y': [],
            'z': []}

    for l_name in lead_group:
        for el_ix, el_iy in lead_group[l_name]['Contact_Pairs_ix']:
            bp_elecs_all['IDX'].append((el_ix, el_iy))
            bp_elecs_all['Anode'].append(el_ix)
            bp_elecs_all['Cathode'].append(el_iy)

            bp_elecs_all['Lead'].append(l_name)
            bp_elecs_all['Contact'].append('{}-{}'.format(eleclabels[el_ix, 1], eleclabels[el_iy, 1]))
            bp_elecs_all['Contact_Abbr'].append('{}-{}'.format(eleclabels[el_ix, 0], eleclabels[el_iy, 0]))
            bp_elecs_all['Type'].append(lead_group[l_name]['Type'])

            try:
                coord = (eleccoords[el_ix] + eleccoords[el_iy]) / 2
            except:
                coord = [np.nan, np.nan, np.nan]
            bp_elecs_all['x'].append(coord[0])
            bp_elecs_all['y'].append(coord[1])
            bp_elecs_all['z'].append(coord[2])

    bp_elecs_all = pd.DataFrame(bp_elecs_all)
    if np.core.numeric.dtype is None:
        importlib.reload(np.core.numeric)
    return bp_elecs_all.sort_values(by=['Anode', 'Cathode']).reset_index(drop=True)


def get_ECoG_sample(study_frame, time_start, time_end):
    natus_data = {}

    # Get the Sample Rate
    if study_frame.validate_sample_rate():
        natus_data['fs'] = 1024  #
    else:
        natus_data['fs'] = 1024

    # Get the minimum number of channels present in all recordings
    natus_data['min_valid_chan'] = min([shape[1] for shape in study_frame.get_shapes()])

    natus_data['data'] = study_frame.find_data_range(time_start, time_end, approx=True)

    return natus_data


def convert_ECoG_BP(natus_data, BP_ELECS):
    natus_data['data'] = (natus_data['data'].data[:, BP_ELECS['Anode'].values] -
                          natus_data['data'].data[:, BP_ELECS['Cathode'].values])

    return natus_data


def half_life(duration, fs_state):
    samples = duration / fs_state
    return np.exp(-(1/samples)*np.log(2))


def plot_time_stacked(sig, fs, wsize=10, color='k', labels=None, zscore=True, scale=3, ax=None):
    """
    Plot of the normalized signal in a stacked montage.
    Parameters
    ----------
    sig: np.ndarray, shape: [n_sample, n_ch]
        Time series signal.
    fs: float
        Sampling frequency of the signal (in Hz)
    wsize: float
        Window size in seconds.
    color: str
        Color of the plot lines.
    labels: array-like, len(n_ch)
        Labels corresponding to the channel names.
    scale: float, default=3.0
        Standard deviations of signal fluctuation by which the montage is
        vertically spaced for each channel.
    ax: matplotlib axis
        For updating the plot post-hoc.
    """
    plt.figure()
    sig = sig[...]
    n_s, n_ch = sig.shape
    ts = np.arange(0, n_s) / fs
    if labels is None:
        labels = ['Ch{}'.format(ix + 1) for ix in range(n_ch)]

    if ax is None:
        plt.figure(figsize=(24, 12))
        ax = plt.subplot(111)

    if zscore:
        sig_Z = (sig - np.nanmean(sig, axis=0)) / np.nanstd(sig, axis=0)
    else:
        sig_Z = sig

    offset = np.arange(n_ch) * scale

    for ch, sig_ch in enumerate(sig_Z.T):
        ax.plot(ts, sig_ch + offset[ch], color=color, alpha=0.5, linewidth=0.5)

        #ax.hlines(offset[ch], ts[0], ts[-1], color='k', alpha=1.0, linewidth=0.2)

    ax.set_yticks(offset)
    ax.set_yticklabels(labels)

    ax.set_xlim([ts[0], ts[0] + wsize])
    ax.set_ylim([np.min(offset) - scale, np.max(offset) + scale])

    plt.tight_layout()

    return ax


def plot_spectrum_stacked(spectra, freqs, color='k', labels=None, zscore=True, scale=3, ax=None):
    """
    Plot of the normalized signal in a stacked montage.
    Parameters
    ----------
    sig: np.ndarray, shape: [n_sample, n_ch]
        Time series signal.
    fs: float
        Sampling frequency of the signal (in Hz)
    wsize: float
        Window size in seconds.
    color: str
        Color of the plot lines.
    labels: array-like, len(n_ch)
        Labels corresponding to the channel names.
    scale: float, default=3.0
        Standard deviations of signal fluctuation by which the montage is
        vertically spaced for each channel.
    ax: matplotlib axis
        For updating the plot post-hoc.
    """
    plt.figure()
    sig = spectra[...]
    n_s, n_ch = sig.shape
    if labels is None:
        labels = ['Ch{}'.format(ix + 1) for ix in range(n_ch)]

    if ax is None:
        plt.figure(figsize=(24, 12))
        ax = plt.subplot(111)

    if zscore:
        sig_Z = (sig - np.nanmean(sig, axis=0)) / np.nanstd(sig, axis=0)
    else:
        sig_Z = sig

    offset = np.arange(n_ch) * scale

    for ch, sig_ch in enumerate(sig_Z.T):
        ax.plot(freqs, sig_ch + offset[ch], color=color, alpha=0.5, linewidth=0.5)

        #ax.hlines(offset[ch], freqs[0], freqs[-1], color='k', alpha=1.0, linewidth=0.2)

    ax.set_yticks(offset)
    ax.set_yticklabels(labels)

    ax.set_xlim([freqs[0], freqs[-1]])
    ax.set_ylim([np.min(offset) - scale, np.max(offset) + scale])

    plt.tight_layout()

    return ax


def do_fitting(foo, freqs, spectrum, freq_range):
    foo.add_data(freqs, spectrum, freq_range)
    aperiodic_params_ = foo._robust_ap_fit(freqs, spectrum)
    ap_fit = gen_aperiodic(freqs, aperiodic_params_)
    r_val = np.corrcoef(spectrum, ap_fit)
    return r_val[0][1] ** 2


def do_fittings(foo, freqs, spectra, freq_range):
    r_sq = []
    for spectrum in spectra:
        r_sq.append(do_fitting(foo, freqs, spectrum, freq_range))

    return r_sq


# Functions #
@pytest.fixture
def tmp_dir(tmpdir):
    """A pytest fixture that turn the tmpdir into a Path object."""
    return pathlib.Path(tmpdir)


# Classes #
class ClassTest(abc.ABC):
    """Default class tests that all classes should pass."""
    class_ = None
    timeit_runs = 100000
    speed_tolerance = 200

    def test_instance_creation(self):
        pass


class TestOOFitter:
    STUDY_PATH = pathlib.Path('/common/xltek/subjects')
    SUBJECT_ID = "EC212"
    WINDOW_SIZE = 10.0
    OUT_DIR = pathlib.Path(f"/home/anthonyfong/ProjectData/EpilepsySpikeDetection/{SUBJECT_ID}/artifact")
    OUT_R_PATH = OUT_DIR.joinpath(f"R2_{WINDOW_SIZE}seconds.h5")
    PLOT_WINDOW = True

    IMG_PATH = pathlib.Path("/common/imaging/subjects/")
    IMAGING_PATH = IMG_PATH.joinpath(SUBJECT_ID, "elecs", "clinical_TDT_elecs_all.mat")
    EXCLUDE_ELECTRODES = {'EKG', 'REF', 'Reference'}

    DURATION = datetime.timedelta(minutes=1)
    SVM_PATH = pathlib.Path.cwd().joinpath("all_metric_svm.obj")

    def load_electrode_map(self):
        # Load Electrode Montage #
        elecs_file = loadmat(self.IMAGING_PATH.as_posix(), squeeze_me=True)

        # Generate Bipolar Montage
        bipolar_electrodes = make_bipolar_elecs_all(elecs_file['eleclabels'], elecs_file['elecmatrix'])

        # Remove common bad labels
        return bipolar_electrodes.loc[~bipolar_electrodes['Lead'].isin(self.EXCLUDE_ELECTRODES)]

    def test_r_squared(self):
        # Load Electrode Map
        bipolar_electrode_map = self.load_electrode_map()

        # Load Study #
        study_frame = XLTEKStudyFrame(s_id=self.SUBJECT_ID, studies_path=self.STUDY_PATH)

        # Time Information
        T0 = study_frame.frames[1].start_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        t0 = T0
        tF = T0 + self.DURATION
        tD = datetime.timedelta(seconds=self.WINDOW_SIZE)

        # Artifact Stuff
        oof = OOFFitter(sample_rate=1024.0, axis=0)
        oof.lower_frequency = 3
        oof.upper_frequency = 500

        auditor = GoodnessAuditor(r_squared_bounded=RSquaredBoundsAudit())
        auditor.set_audit("r_squared_bounded")

        fg = FOOOFGroup(peak_width_limits=[4, 8], min_peak_height=0.05, max_n_peaks=0, verbose=True)

        # Iterate over windows of the study frame
        while t0 <= (tF - tD):
            t0 += tD
            t1 = t0 + tD

            # print('\r{} -- {}'.format(t0, t1), end='')
            ### Try to grab current frame
            raw_ecog = get_ECoG_sample(study_frame, t0, t1)
            raw_ecog = convert_ECoG_BP(raw_ecog, bipolar_electrode_map)
            if len(raw_ecog['data']) < (raw_ecog['fs'] * self.WINDOW_SIZE):
                raise Exception('Unable to grab full ECoG window.')

            # Spectra Rejection

            # OOF
            out = oof.fit_timeseries(data=raw_ecog["data"])

            goodness = auditor.run_audit(info=out)

            plot_spectrum_stacked(out.spectra[:, 175:185], out.frequencies)
            plot_spectrum_stacked(out.curves[:, 175:185], out.frequencies)

            # FOOF
            s_start = time.perf_counter()
            f_transform = np.fft.rfft(raw_ecog["data"], axis=0)
            spectra = np.square(np.abs(f_transform))

            freqs = np.linspace(0, 1024.0 / 2, spectra.shape[0])

            f_start = time.perf_counter()

            fg.fit(freqs=freqs, power_spectra=spectra.T, freq_range=[3, 500], n_jobs=1)

            r_squared_vector = np.array([c[2] for c in fg.group_results], ndmin=2)

            if self.PLOT_WINDOW:
                # Plot
                plot_time_stacked(raw_ecog["data"], 1024)
                fig2 = plt.figure()
                ax1 = fig2.add_subplot(211)
                ax1.plot(out.mse, linestyle="", marker="o")

                fig1 = plt.figure()
                ax1 = fig1.add_subplot(211)
                ax1.set_ylim([-0.1, 1.1])
                ax1.plot(r_squared_vector.T, linestyle="", marker="o")
                ax1.plot(out.r_squared, linestyle="", marker="o")
                ax1.set_title('R2 & Goodness Values')
                ax1.set_ylabel('R Squared')

                ax2 = fig1.add_subplot(212)
                ax2.set_ylim([-0.1, 1.1])
                ax2.plot(goodness, linestyle="", marker="o")
                ax2.set_xlabel('Channels')
                ax2.set_ylabel('Goodness')

                plt.show()

    def test_svm(self):
        # Load Electrode Map
        bipolar_electrode_map = self.load_electrode_map()

        # Load Study #
        study_frame = XLTEKStudyFrame(s_id=self.SUBJECT_ID, studies_path=self.STUDY_PATH)

        # Time Information
        T0 = study_frame.frames[1].start_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        t0 = T0
        tF = T0 + self.DURATION
        tD = datetime.timedelta(seconds=self.WINDOW_SIZE)

        # Artifact Stuff
        oof = OOFFitter(sample_rate=1024.0, axis=0)
        oof.lower_frequency = 3
        oof.upper_frequency = 500

        svm = SVMAudit(path=self.SVM_PATH, probability=True)

        auditor = GoodnessAuditor(svm=svm)
        auditor.set_audit("svm")

        # Iterate over windows of the study frame
        while t0 <= (tF - tD):
            t0 += tD
            t1 = t0 + tD

            # print('\r{} -- {}'.format(t0, t1), end='')
            ### Try to grab current frame
            raw_ecog = get_ECoG_sample(study_frame, t0, t1)
            raw_ecog = convert_ECoG_BP(raw_ecog, bipolar_electrode_map)
            if len(raw_ecog['data']) < (raw_ecog['fs'] * self.WINDOW_SIZE):
                raise Exception('Unable to grab full ECoG window.')

            # OOF
            out = oof.fit_timeseries(data=raw_ecog["data"])

            goodness = auditor.run_audit(info=out)



            if self.PLOT_WINDOW:
                # Plot
                plot_time_stacked(raw_ecog["data"], 1024)
                plot_spectrum_stacked(out.spectra[:, 174:211], out.frequencies)

                fig1 = plt.figure()
                ax1 = fig1.add_subplot(211)
                ax1.set_ylim([-0.1, 1.1])

                ax1.plot(goodness[:, 0], linestyle="", marker="o")
                ax1.set_ylabel('Goodness')

                ax2 = fig1.add_subplot(212)
                ax2.set_ylim([-0.1, 1.1])

                ax2.plot(goodness[:, 1], linestyle="", marker="o")

                ax2.set_xlabel('Channels')
                ax1.set_ylabel('Badness')

                plt.show()


class TestArtifact:
    SVM_PATH = pathlib.Path.cwd().joinpath("all_metric_svm.obj")
    ARTIFACT_DIR = pathlib.Path("/home/anthonyfong/ProjectData/EpilepsySpikeDetection/Artifact_Review/")
    ARTIFACT_INFO = ARTIFACT_DIR.joinpath("Artifact_Info.toml")
    ARTIFACT_FILES = ARTIFACT_DIR.glob("*.mat")
    OUT_DIR = pathlib.Path("/home/anthonyfong/ProjectData/EpilepsySpikeDetection/Artifact_Review/Images")
    TIME_AXIS = 0
    CHANNEL_AXIS = 1
    LOWER_FREQUENCY = 1
    UPPER_FREQUENCY = 250
    METRICS = {"r_squared", "normal_entropy", "mae", "mse", "rmse", "curve_offset", "curve_exp"}
    BEST_METRICS = {"r_squared", "normal_entropy", "mae", "rmse"}

    def plot_r_squared(self, r_squared, ax=None):
        if ax is None:
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
        ax.set_ylim([-0.1, 1.1])
        ax.set_title('R2 Values')
        ax.set_ylabel('R Squared')
        ax.plot(r_squared, linestyle="", marker="o")
        return ax

    def plot_entropy(self, entro, ax=None):
        if ax is None:
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
        ax.set_ylim([-0.1, 1.1])
        ax.set_title('Entropy Values')
        ax.set_ylabel('Entropy')
        ax.plot(entro, linestyle="", marker="o")
        return ax

    def plot_entropyXr_squared(self, entro, ax=None):
        if ax is None:
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
        ax.set_ylim([-0.1, 1.1])
        ax.set_title('Entropy x R Squared Values')
        ax.set_ylabel('Entropy x R Squared')
        ax.plot(entro, linestyle="", marker="o")
        return ax

    def load_data(self):
        artifact_info = toml.load(self.ARTIFACT_INFO.as_posix())["raters"]
        artifact_data = {}

        for file in self.ARTIFACT_FILES:
            name_parts = file.name.split('.')
            subject_id = name_parts[0]
            file_number = int(name_parts[2])
            artifact_file = loadmat(file.as_posix(), squeeze_me=True)

            clip_data = {
                "sample_rate": artifact_file["fs"],
                "channel_labels": artifact_file["channels"],
                "time_axis": artifact_file["timestamp vector"],
                "data": artifact_file["data"],
            }

            if subject_id not in artifact_data:
                artifact_data[subject_id] = [None] * 10

            artifact_data[subject_id][file_number] = clip_data

        return artifact_data, artifact_info

    def aggregate_data(self, artifact_data=None, artifact_info=None, spec_method="fft"):
        if artifact_data is None or artifact_info is None:
            artifact_data, artifact_info = self.load_data()

        # One Over F Fitter
        oof = OOFFitter(axis=self.TIME_AXIS)
        oof.lower_frequency = self.LOWER_FREQUENCY
        oof.upper_frequency = self.UPPER_FREQUENCY
        oof.set_spectra_method(spec_method)

        ag_reviews = {reviewer["name"]: [] for reviewer in artifact_info}
        ag_reviews |= {"Reviewer Intersection": [], "Reviewer Union": []}
        ag_metrics = {m: [] for m in self.METRICS}

        artifact_metrics = {}
        aggregate_data = {
            "reviews": ag_reviews,
            "metrics": ag_metrics,
        }

        for subject_id, data in artifact_data.items():
            artifact_metrics[subject_id] = [None] * len(data)
            for i, artifact_clip in enumerate(data):
                review_channels = {}
                for reviewer in artifact_info:
                    zero_index = tuple(np.array(reviewer["review_channels"][subject_id][i]) - 1)
                    review_channels[reviewer["name"]] = zero_index
                review_union = set()
                review_intersect = set(np.array(artifact_info[0]["review_channels"][subject_id][i]) - 1)
                for rv in review_channels.values():
                    review_union |= (set(rv))
                    review_intersect.intersection_update(set(rv))
                review_union = tuple(review_union)
                review_intersect = tuple(review_intersect)
                reviews = review_channels.copy()
                reviews.update({"Reviewer Intersection": review_intersect, "Reviewer Union": review_union})

                oof.sample_rate = artifact_clip["sample_rate"]

                fit_curves = oof.fit_timeseries(data=artifact_clip["data"])

                curve_removed = fit_curves.spectra - fit_curves.curves
                curve_2 = curve_removed ** 2

                prob = curve_2 / np.sum(curve_2, axis=0)
                entro = entropy(prob)
                normal_entropy = entro / np.log(prob.shape[0])

                artifact_metric = {
                    "raw_data": artifact_clip["data"],
                    "fit_curves": fit_curves,
                    "r_squared": fit_curves.r_squared,
                    "mae": fit_curves.mae,
                    "mse": fit_curves.mse,
                    "rmse": fit_curves.rmse,
                    "normal_entropy": normal_entropy,
                    "curve_offset": fit_curves.parameters[0, :],
                    "curve_exp": fit_curves.parameters[1, :],
                    "reviews": reviews,
                }

                artifact_metrics[subject_id][i] = artifact_metric

                ag_metrics["r_squared"] += list(fit_curves.r_squared)
                ag_metrics["normal_entropy"] += list(normal_entropy)
                ag_metrics["mae"] += list(fit_curves.mae)
                ag_metrics["mse"] += list(fit_curves.mse)
                ag_metrics["rmse"] += list(fit_curves.rmse)
                ag_metrics["curve_offset"] += list(fit_curves.parameters[0, :])
                ag_metrics["curve_exp"] += list(fit_curves.parameters[1, :])
                for reviewer, channels in reviews.items():
                    good_channels = np.zeros((fit_curves.spectra.shape[self.CHANNEL_AXIS],))
                    good_channels[channels,] = 1
                    aggregate_data["reviews"][reviewer] += list(good_channels)

        review_dataframe = pd.DataFrame.from_dict(ag_reviews)
        metrics_dataframe = pd.DataFrame.from_dict(ag_metrics)
        for name, metric_ in ag_metrics.items():
            ag_metrics[name] = np.array(metric_)

        importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
        metrics_all_scaled = scale(metrics_dataframe.to_numpy())
        metrics_all_scaled = pd.DataFrame(metrics_all_scaled, columns=metrics_dataframe.columns)

        return review_dataframe, metrics_dataframe, metrics_all_scaled, artifact_metrics, aggregate_data
    
    def create_datasets(self):
        shuffles = 200
        review_dataframe, metrics_dataframe, metrics_all_scaled, artifact_metrics, aggregate_data = self.aggregate_data()

        # Create Train and Test Datasets
        datasets = {}

        for t_set in range(1, 3):
            datasets[t_set] = {}
            datasets[t_set]["training"] = training = [None] * shuffles
            datasets[t_set]["review_training"] = review_training = [None] * shuffles
            datasets[t_set]["testing"] = testing = [None] * shuffles
            datasets[t_set]["review_testing"] = review_testing = [None] * shuffles

            n_artifacts = len(list(artifact_metrics.values())[0])
            art_sets = list(itertools.combinations(range(0, n_artifacts), t_set))
            all_combos = itertools.product(*((art_sets,) * len(artifact_metrics)))
            combos = random.sample(list(all_combos), k=shuffles)

            for n_shuffle, trains in enumerate(combos):

                train_metrics = {m: [] for m in self.METRICS}
                train_reviews = {reviewer: [] for reviewer in list(artifact_metrics.values())[0][0]["reviews"].keys()}

                test_metrics = {m: [] for m in self.METRICS}
                test_reviews = {reviewer: [] for reviewer in list(artifact_metrics.values())[0][0]["reviews"].keys()}

                for subject, train in zip(artifact_metrics.values(), trains):

                    test = [i for i in range(0, n_artifacts) if i not in set(train)]

                    for artifact in (subject[i] for i in train):
                        train_metrics["r_squared"] += list(artifact["r_squared"])
                        train_metrics["normal_entropy"] += list(artifact["normal_entropy"])
                        train_metrics["mae"] += list(artifact["mae"])
                        train_metrics["mse"] += list(artifact["mse"])
                        train_metrics["rmse"] += list(artifact["rmse"])
                        for reviewer, channels in artifact["reviews"].items():
                            bad_channels = np.zeros((len(artifact["r_squared"]),))
                            bad_channels[channels, ] = 1
                            train_reviews[reviewer] += list(bad_channels)

                    for artifact in (subject[i] for i in test):
                        test_metrics["r_squared"] += list(artifact["r_squared"])
                        test_metrics["normal_entropy"] += list(artifact["normal_entropy"])
                        test_metrics["mae"] += list(artifact["mae"])
                        test_metrics["mse"] += list(artifact["mse"])
                        test_metrics["rmse"] += list(artifact["rmse"])
                        for reviewer, channels in artifact["reviews"].items():
                            bad_channels = np.zeros((len(artifact["r_squared"]),))
                            bad_channels[channels, ] = 1
                            test_reviews[reviewer] += list(bad_channels)

                # Create Data Frames
                metrics_train = pd.DataFrame.from_dict(train_metrics)
                review_training[n_shuffle] = pd.DataFrame.from_dict(train_reviews)
                metrics_test = pd.DataFrame.from_dict(test_metrics)
                review_testing[n_shuffle] = pd.DataFrame.from_dict(test_reviews)
                
                metrics_train_columns = metrics_train.columns
                metrics_test_columns = metrics_test.columns
                importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
                
                # Scale Data
                metrics_train_scaled = scale(metrics_train.to_numpy())
                metrics_test_scaled = scale(metrics_test.to_numpy())
                training[n_shuffle] = pd.DataFrame(metrics_train_scaled, columns=metrics_train_columns)
                testing[n_shuffle] = pd.DataFrame(metrics_test_scaled, columns=metrics_test_columns)
                importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....

        return datasets, review_dataframe, metrics_all_scaled

    def create_half_datasets(self):
        shuffles = 200
        review_dataframe, metrics_dataframe, metrics_all_scaled, artifact_metrics, aggregate_data = self.aggregate_data()

        # Create Train and Test Datasets
        datasets = {}

        for t_set in range(1, 3):
            datasets[t_set] = {}
            datasets[t_set]["training"] = training = [None] * shuffles
            datasets[t_set]["review_training"] = review_training = [None] * shuffles
            datasets[t_set]["testing"] = testing = [None] * shuffles
            datasets[t_set]["review_testing"] = review_testing = [None] * shuffles

            n_artifacts = len(list(artifact_metrics.values())[0])
            art_sets = list(itertools.combinations(range(0, n_artifacts), t_set))
            all_combos = itertools.product(*((art_sets,) * (len(artifact_metrics)//2)))
            combos_set = set(itertools.chain(*(itertools.permutations(c + (tuple(),) * (len(artifact_metrics)//2)) for c in all_combos)))
            combos = random.sample(combos_set, k=shuffles)

            for n_shuffle, trains in enumerate(combos):

                train_metrics = {m: [] for m in self.METRICS}
                train_reviews = {reviewer: [] for reviewer in list(artifact_metrics.values())[0][0]["reviews"].keys()}

                test_metrics = {m: [] for m in self.METRICS}
                test_reviews = {reviewer: [] for reviewer in list(artifact_metrics.values())[0][0]["reviews"].keys()}

                for subject, train in zip(artifact_metrics.values(), trains):

                    test = [i for i in range(0, n_artifacts) if i not in set(train)]

                    for artifact in (subject[i] for i in train):
                        train_metrics["r_squared"] += list(artifact["r_squared"])
                        train_metrics["normal_entropy"] += list(artifact["normal_entropy"])
                        train_metrics["mae"] += list(artifact["mae"])
                        train_metrics["mse"] += list(artifact["mse"])
                        train_metrics["rmse"] += list(artifact["rmse"])
                        for reviewer, channels in artifact["reviews"].items():
                            bad_channels = np.zeros((len(artifact["r_squared"]),))
                            bad_channels[channels,] = 1
                            train_reviews[reviewer] += list(bad_channels)

                    for artifact in (subject[i] for i in test):
                        test_metrics["r_squared"] += list(artifact["r_squared"])
                        test_metrics["normal_entropy"] += list(artifact["normal_entropy"])
                        test_metrics["mae"] += list(artifact["mae"])
                        test_metrics["mse"] += list(artifact["mse"])
                        test_metrics["rmse"] += list(artifact["rmse"])
                        for reviewer, channels in artifact["reviews"].items():
                            bad_channels = np.zeros((len(artifact["r_squared"]),))
                            bad_channels[channels,] = 1
                            test_reviews[reviewer] += list(bad_channels)

                # Create Data Frames
                metrics_train = pd.DataFrame.from_dict(train_metrics)
                review_training[n_shuffle] = pd.DataFrame.from_dict(train_reviews)
                metrics_test = pd.DataFrame.from_dict(test_metrics)
                review_testing[n_shuffle] = pd.DataFrame.from_dict(test_reviews)

                metrics_train_columns = metrics_train.columns
                metrics_test_columns = metrics_test.columns
                importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....

                # Scale Data
                metrics_train_scaled = scale(metrics_train.to_numpy())
                metrics_test_scaled = scale(metrics_test.to_numpy())
                training[n_shuffle] = pd.DataFrame(metrics_train_scaled, columns=metrics_train_columns)
                testing[n_shuffle] = pd.DataFrame(metrics_test_scaled, columns=metrics_test_columns)
                importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....

        return datasets, review_dataframe, metrics_all_scaled

    def test_artifact_review(self):
        artifact_data, artifact_info = self.load_data()

        # One Over F Fitter
        oof = OOFFitter(axis=self.TIME_AXIS)
        oof.lower_frequency = self.LOWER_FREQUENCY
        oof.upper_frequency = self.UPPER_FREQUENCY

        for subject_id, data in artifact_data.items():
            for i, artifact_clip in enumerate(data):
                review_channels = {}
                for reviewer in artifact_info:
                    zero_index = tuple(np.array(reviewer["review_channels"][subject_id][i]) - 1)
                    review_channels[reviewer["name"]] = zero_index
                review_union = set()
                review_intersect = set(np.array(artifact_info[0]["review_channels"][subject_id][i]) - 1)
                for rv in review_channels.values():
                    review_union |= (set(rv))
                    review_intersect.intersection_update(set(rv))
                review_union = tuple(review_union)
                review_intersect = tuple(review_intersect)
                oof.sample_rate = artifact_clip["sample_rate"]

                fit_curves = oof.fit_timeseries(data=artifact_clip["data"])

                curve_removed = fit_curves.spectra - fit_curves.curves
                curve_2 = curve_removed ** 2

                prob = curve_2 / np.sum(curve_2, axis=0)
                entro = entropy(prob)
                normal_entro = entro / np.log(prob.shape[0])
                rxe = normal_entro * fit_curves.r_squared

                # 2D
                fig = plt.figure()
                ax = plt.axes()
                ax.scatter(fit_curves.r_squared, normal_entro)
                ax.scatter(fit_curves.r_squared[review_channels["Joline"],], normal_entro[review_channels["Joline"],])
                ax.scatter(fit_curves.r_squared[review_channels["Jon"],], normal_entro[review_channels["Jon"],])
                ax.scatter(fit_curves.r_squared[review_intersect,], normal_entro[review_intersect,])



                # 3D
                # channel_axis = np.arange(0, fit_curves.spectra.shape[self.CHANNEL_AXIS])
                # fig = plt.figure()
                # ax = plt.axes(projection='3d')
                # ax.scatter(channel_axis, fit_curves.r_squared, normal_entro)
                # ax.scatter(review_channels["Joline"], fit_curves.r_squared[review_channels["Joline"], ], normal_entro[review_channels["Joline"], ])
                # ax.scatter(review_channels["Jon"], fit_curves.r_squared[review_channels["Jon"], ], normal_entro[review_channels["Jon"], ])
                # ax.scatter(review_intersect, fit_curves.r_squared[review_intersect, ], normal_entro[review_intersect, ])
                #
                # plt.show()

                # ROC
                reviews = review_channels.copy()
                reviews |= {"Reviewer Intersection": review_intersect, "Reviewer Union": review_union}
                fig, axs = plt.subplots(1, 4, figsize=(20, 7))
                fig.suptitle(f"{subject_id}: File {i}", fontsize=30)
                for k, (reviewer, channels) in enumerate(reviews.items()):
                    good_channels = np.ones((fit_curves.spectra.shape[self.CHANNEL_AXIS],))
                    good_channels[channels, ] = 0  # np.zeros((len(channels),))

                    fpr = [None] * 3
                    tpr = [None] * 3
                    roc_auc = [None] * 3

                    for index, metric_ in enumerate((fit_curves.r_squared, normal_entro, rxe)):
                        fpr[index], tpr[index], thresholds = metrics.roc_curve(good_channels, metric_)
                        roc_auc[index] = metrics.auc(fpr[index], tpr[index])

                    # Plot all ROC curves
                    axs[k].plot(
                        fpr[2],
                        tpr[2],
                        label="Entropy ROC curve (area = {0:0.2f})".format(roc_auc[2]),
                        color="aqua",
                        linewidth=4,
                    )

                    axs[k].plot(
                        fpr[0],
                        tpr[0],
                        label="R Squared ROC curve (area = {0:0.2f})".format(roc_auc[0]),
                        color="deeppink",
                        linestyle=":",
                        linewidth=4,
                    )

                    axs[k].plot(
                        fpr[1],
                        tpr[1],
                        label="Entropy ROC curve (area = {0:0.2f})".format(roc_auc[1]),
                        color="navy",
                        linestyle=":",
                        linewidth=4,
                    )

                    axs[k].plot([0, 1], [0, 1], "k--", lw=4)
                    axs[k].set_xlim([0.0, 1.0])
                    axs[k].set_ylim([0.0, 1.05])
                    axs[k].set_xlabel("False Positive Rate")
                    axs[k].set_ylabel("True Positive Rate")
                    axs[k].set_title(f"ROC of Metrics vs {reviewer}")
                    axs[k].legend(loc="lower right")

                # Plotting
                fig, axs = plt.subplots(2, 3, figsize=(30, 15), gridspec_kw={'height_ratios': [3, 1]})
                fig.suptitle(f"{subject_id}: File {i}", fontsize=60)

                # Time Series
                if review_union:
                    labels = [artifact_clip["channel_labels"][i] for i in review_union]
                    raw_ax = plot_time_stacked(artifact_clip["data"][:, review_union], artifact_clip["sample_rate"], labels=labels, ax=axs[0, 0], scale=10)
                else:
                    raw_ax = plot_time_stacked(artifact_clip["data"], artifact_clip["sample_rate"], labels=artifact_clip["channel_labels"], ax=axs[0, 0])

                raw_ax.set_title('Time Series Data')

                # R Squared
                r_ax = self.plot_r_squared(fit_curves.r_squared, ax=axs[1, 0])
                r_ax.plot(review_channels["Joline"], fit_curves.r_squared[review_channels["Joline"], ], linestyle="", marker="o")
                r_ax.plot(review_channels["Jon"], fit_curves.r_squared[review_channels["Jon"], ], linestyle="", marker="o")
                r_ax.plot(review_intersect, fit_curves.r_squared[review_intersect, ], linestyle="", marker="o")

                # Entropy
                e_ax = self.plot_entropy(normal_entro, ax=axs[1, 1])
                e_ax.plot(review_channels["Joline"], normal_entro[review_channels["Joline"], ], linestyle="", marker="o")
                e_ax.plot(review_channels["Jon"], normal_entro[review_channels["Jon"], ], linestyle="", marker="o")
                e_ax.plot(review_intersect, normal_entro[review_intersect, ], linestyle="", marker="o")

                # R Squared X Entropy
                rxe_ax = self.plot_entropyXr_squared(rxe, ax=axs[1, 2])
                rxe_ax.plot(review_channels["Joline"], rxe[review_channels["Joline"], ], linestyle="", marker="o")
                rxe_ax.plot(review_channels["Jon"], rxe[review_channels["Jon"], ], linestyle="", marker="o")
                rxe_ax.plot(review_intersect, rxe[review_intersect, ], linestyle="", marker="o")

                if review_union:
                    labels = [artifact_clip["channel_labels"][i] for i in review_union]
                    s_ax = plot_spectrum_stacked(fit_curves.spectra[:, review_union], fit_curves.frequencies, labels=labels, ax=axs[0, 1])
                    s_ax.set_title('Power Spectra of Bad Channels')
                    c_ax = plot_spectrum_stacked(curve_removed[:, review_union], fit_curves.frequencies, labels=labels, ax=axs[0, 2])
                    c_ax.set_title('One Over F Removed')

                plt.show()

    def test_artifact_missing_feature(self):
        target_precision = 0.95
        shuffles = 100
        artifact_data, artifact_info = self.load_data()
        review_dataframe, metrics_dataframe, metrics_all_scaled, artifact_metrics, aggregate_data = self.aggregate_data(artifact_data, artifact_info, "welch")

        flat_metrics = []
        for name, clips in artifact_metrics.items():
            for clip, metrics_ in enumerate(clips):
                for channel in range(0, len(metrics_["r_squared"])):
                    single_metrics = {
                        "subject": name,
                        "clip": clip,
                        "channel": channel,
                        "raw_data": metrics_["raw_data"][:, channel],
                        "spectra": metrics_["fit_curves"].spectra[:, channel],
                        "frequencies": metrics_["fit_curves"].frequencies,
                        "r_squared": metrics_["r_squared"][channel],
                        "normal_entropy": metrics_["normal_entropy"][channel],
                        "mae": metrics_["mae"][channel],
                        "mse": metrics_["mse"][channel],
                        "rmse": metrics_["rmse"][channel],
                        "curve_offset": metrics_["curve_offset"][channel],
                        "curve_exp": metrics_["curve_exp"][channel],
                    }

                    flat_metrics.append(single_metrics)

        svms = [None] * shuffles
        for t_set in range(0, shuffles):
            # Create Train and Test Datasets
            data_split = train_test_split(metrics_dataframe, review_dataframe, random_state=None)
            metrics_train, metrics_test, review_train, review_test = data_split
            metrics_train_columns = metrics_train.columns
            metrics_test_columns = metrics_test.columns

            # Scale data
            importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
            metrics_train_scaled = scale(metrics_train.to_numpy())
            metrics_test_scaled = scale(metrics_test.to_numpy())

            metrics_train_scaled = pd.DataFrame(metrics_train_scaled, columns=metrics_train_columns)
            metrics_test_scaled = pd.DataFrame(metrics_test_scaled, columns=metrics_test_columns)
            importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....

            svm = {}
            metrics_list = list(self.METRICS)
            all_set = metrics_all_scaled[metrics_list]
            training_set = metrics_train_scaled[metrics_list]
            testing_set = metrics_test_scaled[metrics_list]

            importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
            svm["classifier"] = classifier = SVC()  # class_weight="balanced", random_state=42
            classifier.fit(training_set.to_numpy(), review_train["Reviewer Union"])

            review_all_decision = classifier.decision_function(all_set)
            review_test_decision = classifier.decision_function(testing_set)

            svm["all_fpr"], svm["all_tpr"], _ = metrics.roc_curve(review_dataframe["Reviewer Union"],
                                                                  review_all_decision)
            svm["test_fpr"], svm["test_tpr"], _ = metrics.roc_curve(review_test["Reviewer Union"],
                                                                    review_test_decision)
            svm["all_roc_auc"] = metrics.auc(svm["all_fpr"], svm["all_tpr"])
            svm["test_roc_auc"] = metrics.auc(svm["test_fpr"], svm["test_tpr"], )

            svm["all_pr"], svm["all_rec"], thresholds = metrics.precision_recall_curve(review_dataframe["Reviewer Union"],
                                                                              review_all_decision)
            svm["test_pr"], svm["test_rec"], _ = metrics.precision_recall_curve(review_test["Reviewer Union"],
                                                                                review_test_decision)
            svm["all_prc_auc"] = metrics.auc(svm["all_rec"], svm["all_pr"])
            svm["test_prc_auc"] = metrics.auc(svm["test_rec"], svm["test_pr"])
            svm["all_prc_f1"] = 2 * (svm["all_pr"] * svm["all_rec"]) / (svm["all_pr"] + svm["all_rec"])
            svm["test_prc_f1"] = 2 * (svm["all_pr"] * svm["all_rec"]) / (svm["all_pr"] + svm["all_rec"])

            p_index_p = int(np.searchsorted(svm["all_pr"][1:], target_precision, side="right") - 1)
            thresh = thresholds[p_index_p]

            predicted_class = review_all_decision > thresh
            errors = review_dataframe["Reviewer Union"] != predicted_class

            svm["false_positives"] = np.where(errors)[0][predicted_class[errors]]
            svm["false_negatives"] = np.where(errors)[0][(review_dataframe["Reviewer Union"] == 1)[errors]]

            svms[t_set] = svm

        # Plot
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        fig.suptitle("SVM Performance 100 Shuffles", fontsize=30)
        roc_xlabel = "False Positive Rate"
        roc_ylabel = "True Positive Rate"
        pc_xlabel = "Recall"
        pc_ylabel = "Precision"

        for index, set_type in enumerate(("All", "Test")):
            axs[index, 0].set(xlabel=roc_xlabel, ylabel=roc_ylabel, title=f"{set_type} ROC")
            axs[index, 0].set_xticks(np.arange(0, 1.1, 0.1))
            axs[index, 0].set_yticks(np.arange(0, 1.1, 0.1))
            axs[index, 0].grid(color='gray', linestyle='-', linewidth=0.5)
            axs[index, 0].plot([0, 1], [0, 1], 'g--')

            axs[index, 1].set(xlabel=pc_xlabel, ylabel=pc_ylabel, title=f"{set_type} Precision Recall")
            axs[index, 1].set_xticks(np.arange(0, 1.1, 0.1))
            axs[index, 1].set_yticks(np.arange(0, 1.1, 0.1))
            axs[index, 1].grid(color='gray', linestyle='-', linewidth=0.5)
            axs[index, 1].plot([0, 1], [0, 0], 'g--')

        for svm in svms:
            axs[0, 0].plot(svm["all_fpr"], svm["all_tpr"])
            axs[1, 0].plot(svm["test_fpr"], svm["test_tpr"])

            axs[0, 1].plot(svm["all_rec"], svm["all_pr"])
            axs[1, 1].plot(svm["test_rec"], svm["test_pr"])

        importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
        plt.show()

        all_false_positives = np.concatenate([svm["false_positives"] for svm in svms])
        all_false_negatives = np.concatenate([svm["false_negatives"] for svm in svms])

        fp_dict = dict()
        fn_dict = dict()
        fp_dict["id"], fp_dict["counts"] = np.unique(all_false_positives.flatten(), return_counts=True)
        fn_dict["id"], fn_dict["counts"] = np.unique(all_false_negatives.flatten(), return_counts=True)

        fp_df = pd.DataFrame.from_dict(fp_dict).sort_values("counts", ascending=False)
        fn_df = pd.DataFrame.from_dict(fn_dict).sort_values("counts", ascending=False)
        importlib.reload(np.core.numeric)

        fp_metrics = [None] * len(fp_df)
        fp_labels = [""] * len(fp_df)
        for index, id in enumerate(fp_df["id"]):
            fp_metrics[index] = fm = flat_metrics[id]
            c_label = artifact_data[fm["subject"]][fm["clip"]]["channel_labels"][fm["channel"]]
            fp_labels[index] = f"{fm['subject']} Clip {fm['clip'] +1 } Channel {fm['channel'] + 1}: {c_label}"

        fn_metrics = [None] * len(fn_df)
        fn_labels = [""] * len(fn_df)
        for index, id in enumerate(fn_df["id"]):
            fn_metrics[index] = fm = flat_metrics[id]
            c_label = artifact_data[fm["subject"]][fm["clip"]]["channel_labels"][fm["channel"]]
            fn_labels[index] = f"{fm['subject']} Clip {fm['clip'] +1 } Channel {fm['channel'] + 1}: {c_label}"

        fp_raw = np.stack([m["raw_data"] for m in fp_metrics], axis=1)
        fp_spectra = np.stack([m["spectra"] for m in fp_metrics], axis=1)

        fn_raw = np.stack([m["raw_data"] for m in fn_metrics], axis=1)
        fn_spectra = np.stack([m["spectra"] for m in fn_metrics], axis=1)

        raw_fp_ax = plot_time_stacked(fp_raw, 1024, labels=fp_labels, scale=10)
        raw_fn_ax = plot_time_stacked(fn_raw[:, 0:10], 1024, labels=fn_labels[0:10], scale=10)

        fp_fig = Figure()
        fp_fig.update_layout(title="False Positives 100Hz Cutoff", margin=dict(t=60))
        fp_fig.update_layout(TimeSeriesPlot.default_layout_settings)
        fp_fig.set_subplots(1, 3, horizontal_spacing=0.01)
        fp_sub = fp_fig.subplots

        fp_hist_plot = BarPlot(subplot=fp_sub[0][0], x=fp_df["counts"], labels=fp_labels, orientation='h', separated=True)
        fp_hist_plot.update_title(text="Occurrence Histogram")

        fp_ts_plot = TimeSeriesPlot(subplot=fp_sub[0][1], y=fp_raw, sample_rate=1024, labels=fp_labels)
        fp_ts_plot.update_title(text="Preprocessed Time Series")
        fp_ts_plot.set_yaxis(fp_hist_plot.yaxis)
        
        fp_sp_plot = SpectraPlot(subplot=fp_sub[0][2], x=flat_metrics[0]["frequencies"], y=fp_spectra, labels=fp_labels)
        fp_sp_plot.update_title(text="Spectra")
        fp_sp_plot.set_yaxis(fp_hist_plot.yaxis)

        fp_hist_plot.group_same_legend_items(fp_ts_plot, fp_sp_plot)

        fp_fig.show()
        fp_fig.write_html(self.OUT_DIR.joinpath("FP_Plot_100Hz_Cutoff.html").as_posix())

        fn_fig = Figure()
        fn_fig.update_layout(title="False Negatives 100Hz Cutoff", margin=dict(t=60))
        fn_fig.update_layout(TimeSeriesPlot.default_layout_settings)
        fn_fig.set_subplots(1, 3, horizontal_spacing=0.01)
        fn_sub = fn_fig.subplots

        fn_hist_plot = BarPlot(subplot=fn_sub[0][0], x=fn_df["counts"], labels=fn_labels, orientation='h',
                               separated=True)
        fn_hist_plot.update_title(text="Occurrence Histogram")

        fn_ts_plot = TimeSeriesPlot(subplot=fn_sub[0][1], y=fn_raw, sample_rate=1024, labels=fn_labels)
        fn_ts_plot.update_title(text="Preprocessed Time Series")
        fn_ts_plot.set_yaxis(fn_hist_plot.yaxis)

        fn_sp_plot = SpectraPlot(subplot=fn_sub[0][2], x=flat_metrics[0]["frequencies"], y=fn_spectra, labels=fn_labels)
        fn_sp_plot.update_title(text="Spectra")
        fn_sp_plot.set_yaxis(fn_hist_plot.yaxis)

        fn_hist_plot.group_same_legend_items(fn_ts_plot, fn_sp_plot)

        fn_fig.show()
        fn_fig.write_html(self.OUT_DIR.joinpath("FN_Plot_100Hz_Cutoff.html").as_posix())

    def test_view_rxe_aggregate(self):
        _, _, metrics_all_scaled, artifact_metrics, aggregate_data = self.aggregate_data()

        ag_metrics = aggregate_data["metrics"]
        ag_reviews = aggregate_data["reviews"]

        # Plotting
        fig = plt.figure()
        ax = plt.axes()

        ax.scatter(
            ag_metrics["r_squared"][0 == np.array(ag_reviews["Reviewer Union"])],
            ag_metrics["normal_entropy"][0 == np.array(ag_reviews["Reviewer Union"])],
            label="Good Channels",
            color="steelblue",
        )

        ax.scatter(
            ag_metrics["r_squared"][1 == np.array(ag_reviews["Joline"])],
            ag_metrics["normal_entropy"][1 == np.array(ag_reviews["Joline"])],
            label="Joline",
            color="orange",
        )
        ax.scatter(
            ag_metrics["r_squared"][1 == np.array(ag_reviews["Jon"])],
            ag_metrics["normal_entropy"][1 == np.array(ag_reviews["Jon"])],
            label="Jon",
            color="limegreen",
        )
        ax.scatter(
            ag_metrics["r_squared"][1 == np.array(ag_reviews["Reviewer Intersection"])],
            ag_metrics["normal_entropy"][1 == np.array(ag_reviews["Reviewer Intersection"])],
            label="Intersection",
            color="crimson",
        )

        ax.set_xlabel("R Squared")
        ax.set_ylabel("Normalized Spectral Entropy")
        ax.set_title("Entropy vs R Squared (Bad on Top)")
        ax.legend(loc="lower left")

        plt.show()

    def test_svm_metrics(self):
        datasets, review_dataframe, metrics_all_scaled = self.create_datasets()

        # Support Vector Machine
        metric_combinations = set(powerset(self.METRICS))
        metric_combinations.remove(tuple())
        classifiers = {art_n: {metrics_: [] for metrics_ in metric_combinations} for art_n in datasets.keys()}
        
        for art_n, dataset in datasets.items():
            art_classifier = classifiers[art_n]
            for shuf in range(0, len(dataset["training"])):
                metrics_train_scaled = dataset["training"][shuf]
                review_train = dataset["review_training"][shuf]
                metrics_test_scaled = dataset["testing"][shuf]
                review_test = dataset["review_testing"][shuf]
                
                for metrics_, svms in art_classifier.items():
                    svm = {}
                    metrics_list = list(metrics_)
                    all_set = metrics_all_scaled[metrics_list]
                    training_set = metrics_train_scaled[metrics_list]
                    testing_set = metrics_test_scaled[metrics_list]
        
                    importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
                    svm["classifier"] = classifier = SVC()  # class_weight="balanced", random_state=42
                    classifier.fit(training_set.to_numpy(), review_train["Reviewer Union"])
        
                    review_all_prediction = classifier.decision_function(all_set)
                    review_test_prediction = classifier.decision_function(testing_set)
        
                    svm["all_fpr"], svm["all_tpr"], _ = metrics.roc_curve(review_dataframe["Reviewer Union"],
                                                                          review_all_prediction)
                    svm["test_fpr"], svm["test_tpr"], _ = metrics.roc_curve(review_test["Reviewer Union"],
                                                                            review_test_prediction)
                    svm["all_roc_auc"] = metrics.auc(svm["all_fpr"], svm["all_tpr"])
                    svm["test_roc_auc"] = metrics.auc(svm["test_fpr"], svm["test_tpr"], )
        
                    svm["all_pr"], svm["all_rec"], _ = metrics.precision_recall_curve(review_dataframe["Reviewer Union"],
                                                                                      review_all_prediction)
                    svm["test_pr"], svm["test_rec"], _ = metrics.precision_recall_curve(review_test["Reviewer Union"],
                                                                                        review_test_prediction)
                    svm["all_prc_auc"] = metrics.auc(svm["all_rec"], svm["all_pr"])
                    svm["test_prc_auc"] = metrics.auc(svm["test_rec"], svm["test_pr"])
                    svm["all_prc_f1"] = 2 * (svm["all_pr"] * svm["all_rec"]) / (svm["all_pr"] + svm["all_rec"])
                    svm["test_prc_f1"] = 2 * (svm["all_pr"] * svm["all_rec"]) / (svm["all_pr"] + svm["all_rec"])
        
                    svms.append(svm)

            # Plot
            fig, axs = plt.subplots(2, 2, figsize=(20, 20))
            fig.suptitle(f"SVM Performance of Metrics with {art_n} Artifact Samples per Subject", fontsize=30)
            roc_xlabel = "False Positive Rate"
            roc_ylabel = "True Positive Rate"
            pc_xlabel = "Recall"
            pc_ylabel = "Precision"
        
            for index, set_type in enumerate(("All", "Test")):
                axs[index, 0].set(xlabel=roc_xlabel, ylabel=roc_ylabel, title=f"{set_type} ROC")
                axs[index, 0].set_xticks(np.arange(0, 1.1, 0.1))
                axs[index, 0].set_yticks(np.arange(0, 1.1, 0.1))
                axs[index, 0].grid(color='gray', linestyle='-', linewidth=0.5)
                axs[index, 0].plot([0, 1], [0, 1], 'g--')
        
                axs[index, 1].set(xlabel=pc_xlabel, ylabel=pc_ylabel, title=f"{set_type} Precision Recall")
                axs[index, 1].set_xticks(np.arange(0, 1.1, 0.1))
                axs[index, 1].set_yticks(np.arange(0, 1.1, 0.1))
                axs[index, 1].grid(color='gray', linestyle='-', linewidth=0.5)
                axs[index, 1].plot([0, 1], [0, 0], 'g--')
        
            all_roc_aucs = [None] * len(art_classifier)
            test_roc_aucs = [None] * len(art_classifier)
            all_prc_aucs = [None] * len(art_classifier)
            test_prc_aucs = [None] * len(art_classifier)
            all_prc_max_f1 = [None] * len(art_classifier)
            test_prc_max_f1 = [None] * len(art_classifier)
            for index, (metrics_, svms) in enumerate(art_classifier.items()):
                all_roc_auc = np.array([svm["all_roc_auc"] for svm in svms])
                test_roc_auc = np.array([svm["test_roc_auc"] for svm in svms])
                all_prc_auc = np.array([svm["all_prc_auc"] for svm in svms])
                test_prc_auc = np.array([svm["test_prc_auc"] for svm in svms])
                all_prc_f1 = np.array([svm["all_prc_auc"].max() for svm in svms])
                test_prc_f1 = np.array([svm["test_prc_auc"].max() for svm in svms])
        
                svm = svms[all_prc_auc.argmax()]
        
                axs[0, 0].plot(svm["all_fpr"], svm["all_tpr"])
                axs[1, 0].plot(svm["test_fpr"], svm["test_tpr"])
        
                axs[0, 1].plot(svm["all_rec"], svm["all_pr"])
                axs[1, 1].plot(svm["test_rec"], svm["test_pr"])
        
                all_roc_aucs[index] = {"name": metrics_, "auc": np.sort(all_roc_auc)}
                test_roc_aucs[index] = {"name": metrics_, "auc": np.sort(test_roc_auc)}
                all_prc_aucs[index] = {"name": metrics_, "auc": np.sort(all_prc_auc)}
                test_prc_aucs[index] = {"name": metrics_, "auc": np.sort(test_prc_auc)}
                all_prc_max_f1[index] = {"name": metrics_, "f1": np.sort(all_prc_f1)}
                test_prc_max_f1[index] = {"name": metrics_, "f1": np.sort(test_prc_f1)}
        
            aucs = {
                "all_roc": sorted(all_roc_aucs, key=lambda x: np.median(x["auc"])),
                "test_roc": sorted(test_roc_aucs, key=lambda x: np.median(x["auc"])),
                "all_prc": sorted(all_prc_aucs, key=lambda x: np.median(x["auc"])),
                "test_prc": sorted(test_prc_aucs, key=lambda x: np.median(x["auc"])),
                "all_prc_f1": sorted(all_prc_max_f1, key=lambda x: np.median(x["f1"])),
                "test_prc_f1": sorted(test_prc_max_f1, key=lambda x: np.median(x["f1"])),
            }
        
            # Boxplot AUC
            importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
            fig, ax_auc = plt.subplots(2, 2, figsize=(20, 20))
            fig.subplots_adjust(left=0.2, right=0.9, wspace=0.7)
            fig.suptitle(f"Area Under the Curve with {art_n} Artifact Samples per Subject", fontsize=20)
        
            for index, set_type in enumerate(("all", "test")):
                roc_name = f"{set_type}_roc"
                indices = range(1, len(aucs[roc_name]) + 1)
                importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
                ax_auc[index, 0].grid(color='gray', linestyle='-', linewidth=0.5)
                ax_auc[index, 0].set_axisbelow(True)
                ax_auc[index, 0].boxplot([svm["auc"] for svm in aucs[roc_name]], vert=False)
                ax_auc[index, 0].set(xlabel="AUC", yticks=indices, title=f"{set_type.capitalize()} ROC AUC")
                ax_auc[index, 0].set_xticks(np.arange(0, 1.1, 0.1))
                ax_auc[index, 0].set_yticklabels([svm["name"] for svm in aucs[roc_name]])
                ax_auc[index, 0].set_xlim([0, 1])
        
                prc_name = f"{set_type}_prc"
                importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
                ax_auc[index, 1].grid(color='gray', linestyle='-', linewidth=0.5)
                ax_auc[index, 1].set_axisbelow(True)
                ax_auc[index, 1].boxplot([svm["auc"] for svm in aucs[prc_name]], vert=False)
                ax_auc[index, 1].set(xlabel="AUC", yticks=indices,
                                     title=f"{set_type.capitalize()} Precision Recall AUC")
                ax_auc[index, 1].set_xticks(np.arange(0, 1.1, 0.1))
                ax_auc[index, 1].set_yticklabels([svm["name"] for svm in aucs[prc_name]])
                ax_auc[index, 1].set_xlim([0, 1])
        
            # Boxplot F1
            importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
            fig, ax_auc = plt.subplots(2, 1, figsize=(20, 20))
            fig.subplots_adjust(left=0.2, right=0.9, wspace=0.7)
            fig.suptitle(f"F1 with {art_n} Artifact Samples per Subject", fontsize=20)
        
            for index, set_type in enumerate(("all", "test")):
                prc_name = f"{set_type}_prc_f1"
                indices = range(1, len(aucs[prc_name]) + 1)
                importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
                ax_auc[index].grid(color='gray', linestyle='-', linewidth=0.5)
                ax_auc[index].set_axisbelow(True)
                ax_auc[index].boxplot([svm["f1"] for svm in aucs[prc_name]], vert=False)
                ax_auc[index].set(xlabel="F1", yticks=indices, title=f"{set_type.capitalize()} Precision Recall Max F1")
                ax_auc[index].set_xticks(np.arange(0, 1.1, 0.1))
                ax_auc[index].set_yticklabels([svm["name"] for svm in aucs[prc_name]])
                ax_auc[index].set_xlim([0, 1])
        
            #
            # Plot Bottom Area Under Curves
            fig, axs = plt.subplots(2, 2, figsize=(20, 20))
            fig.suptitle(f"SVM Performance of Each Metric with {art_n} Artifact Samples per Subject", fontsize=20)
            roc_xlabel = "False Positive Rate"
            roc_ylabel = "True Positive Rate"
            pc_xlabel = "Recall"
            pc_ylabel = "Precision"
        
            for auc in aucs["all_prc"][0:6]:
                metrics_ = auc["name"]
                auc_values = auc["auc"]
                svms = art_classifier[metrics_]
                svm = svms[auc_values.argmax()]
        
                axs[0, 0].plot(svm["all_fpr"], svm["all_tpr"], label=f"{metrics_}")
                axs[1, 0].plot(svm["test_fpr"], svm["test_tpr"], label=f"{metrics_}")
        
                axs[0, 1].plot(svm["all_rec"], svm["all_pr"], label=f"{metrics_}")
                axs[1, 1].plot(svm["test_rec"], svm["test_pr"], label=f"{metrics_}")
        
            for index, set_type in enumerate(("All", "Test")):
                axs[index, 0].set(xlabel=roc_xlabel, ylabel=roc_ylabel, title=f"{set_type} ROC")
                axs[index, 0].set_xticks(np.arange(0, 1.1, 0.1))
                axs[index, 0].set_yticks(np.arange(0, 1.1, 0.1))
                axs[index, 0].grid(color='gray', linestyle='-', linewidth=0.5)
                axs[index, 0].plot([0, 1], [0, 1], 'g--')
                axs[index, 0].legend(loc="lower right")
        
                axs[index, 1].set(xlabel=pc_xlabel, ylabel=pc_ylabel, title=f"{set_type} Precision Recall")
                axs[index, 1].set_xticks(np.arange(0, 1.1, 0.1))
                axs[index, 1].set_yticks(np.arange(0, 1.1, 0.1))
                axs[index, 1].grid(color='gray', linestyle='-', linewidth=0.5)
                axs[index, 1].plot([0, 1], [0, 0], 'g--')
                axs[index, 1].legend(loc="lower left")
        
            # Plot Top Area Under Curves
            fig, axs = plt.subplots(2, 2, figsize=(20, 20))
            fig.suptitle(f"SVM Performance of Top 6 Metrics with {art_n} Artifact Samples per Subject", fontsize=20)
            roc_xlabel = "False Positive Rate"
            roc_ylabel = "True Positive Rate"
            pc_xlabel = "Recall"
            pc_ylabel = "Precision"
        
            for auc in aucs["all_prc"][-6:]:
                metrics_ = auc["name"]
                auc_values = auc["auc"]
                svms = art_classifier[metrics_]
                svm = svms[auc_values.argmax()]
        
                axs[0, 0].plot(svm["all_fpr"], svm["all_tpr"], label=f"{metrics_}")
                axs[1, 0].plot(svm["test_fpr"], svm["test_tpr"], label=f"{metrics_}")
        
                axs[0, 1].plot(svm["all_rec"], svm["all_pr"], label=f"{metrics_}")
                axs[1, 1].plot(svm["test_rec"], svm["test_pr"], label=f"{metrics_}")
        
            for index, set_type in enumerate(("All", "Test")):
                axs[index, 0].set(xlabel=roc_xlabel, ylabel=roc_ylabel, title=f"{set_type} ROC")
                axs[index, 0].set_xticks(np.arange(0, 1.1, 0.1))
                axs[index, 0].set_yticks(np.arange(0, 1.1, 0.1))
                axs[index, 0].grid(color='gray', linestyle='-', linewidth=0.5)
                axs[index, 0].plot([0, 1], [0, 1], 'g--')
                axs[index, 0].legend(loc="lower right")
        
                axs[index, 1].set(xlabel=pc_xlabel, ylabel=pc_ylabel, title=f"{set_type} Precision Recall")
                axs[index, 1].set_xticks(np.arange(0, 1.1, 0.1))
                axs[index, 1].set_yticks(np.arange(0, 1.1, 0.1))
                axs[index, 1].grid(color='gray', linestyle='-', linewidth=0.5)
                axs[index, 1].plot([0, 1], [0, 0], 'g--')
                axs[index, 1].legend(loc="lower left")
        
            plt.show()

    def test_svm_metrics_aggregate(self):
        review_dataframe, metrics_dataframe, metrics_all_scaled, artifact_metrics, aggregate_data = self.aggregate_data()

        # Support Vector Machine
        metric_combinations = set(powerset(self.METRICS))
        metric_combinations.remove(tuple())
        classifiers = {metrics_: [] for metrics_ in metric_combinations}

        for t_set in range(0, 100):
            # Create Train and Test Datasets
            data_split = train_test_split(metrics_dataframe, review_dataframe, random_state=None)
            metrics_train, metrics_test, review_train, review_test = data_split
            metrics_train_columns = metrics_train.columns
            metrics_test_columns = metrics_test.columns
    
            # Scale data
            importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
            metrics_train_scaled = scale(metrics_train.to_numpy())
            metrics_test_scaled = scale(metrics_test.to_numpy())
    
            metrics_train_scaled = pd.DataFrame(metrics_train_scaled, columns=metrics_train_columns)
            metrics_test_scaled = pd.DataFrame(metrics_test_scaled, columns=metrics_test_columns)
            importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
            
            for metrics_, svms in classifiers.items():
                svm = {}
                metrics_list = list(metrics_)
                all_set = metrics_all_scaled[metrics_list]
                training_set = metrics_train_scaled[metrics_list]
                testing_set = metrics_test_scaled[metrics_list]
    
                importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
                svm["classifier"] = classifier = SVC()  # class_weight="balanced", random_state=42
                classifier.fit(training_set.to_numpy(), review_train["Reviewer Union"])
    
                review_all_prediction = classifier.decision_function(all_set)
                review_test_prediction = classifier.decision_function(testing_set)
    
                svm["all_fpr"], svm["all_tpr"], _ = metrics.roc_curve(review_dataframe["Reviewer Union"], review_all_prediction)
                svm["test_fpr"], svm["test_tpr"], _ = metrics.roc_curve(review_test["Reviewer Union"], review_test_prediction)
                svm["all_roc_auc"] = metrics.auc(svm["all_fpr"], svm["all_tpr"])
                svm["test_roc_auc"] = metrics.auc(svm["test_fpr"], svm["test_tpr"],)
    
                svm["all_pr"], svm["all_rec"], _ = metrics.precision_recall_curve(review_dataframe["Reviewer Union"], review_all_prediction)
                svm["test_pr"], svm["test_rec"], _ = metrics.precision_recall_curve(review_test["Reviewer Union"], review_test_prediction)
                svm["all_prc_auc"] = metrics.auc(svm["all_rec"], svm["all_pr"])
                svm["test_prc_auc"] = metrics.auc(svm["test_rec"], svm["test_pr"])
                svm["all_prc_f1"] = 2 * (svm["all_pr"] * svm["all_rec"]) / (svm["all_pr"] + svm["all_rec"])
                svm["test_prc_f1"] = 2 * (svm["all_pr"] * svm["all_rec"]) / (svm["all_pr"] + svm["all_rec"])

                svms.append(svm)
                # auc_dataframe.loc["".join(f"{m} " for m in metrics_)] = [svm["all_roc_auc"], svm["all_prc_auc"], svm["test_roc_auc"], svm["test_prc_auc"]]

        # Plot
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        fig.suptitle("SVM Performance of Metrics", fontsize=30)
        roc_xlabel = "False Positive Rate"
        roc_ylabel = "True Positive Rate"
        pc_xlabel = "Recall"
        pc_ylabel = "Precision"

        for index, set_type in enumerate(("All", "Test")):
            axs[index, 0].set(xlabel=roc_xlabel, ylabel=roc_ylabel, title=f"{set_type} ROC")
            axs[index, 0].set_xticks(np.arange(0, 1.1, 0.1))
            axs[index, 0].set_yticks(np.arange(0, 1.1, 0.1))
            axs[index, 0].grid(color='gray', linestyle='-', linewidth=0.5)
            axs[index, 0].plot([0, 1], [0, 1], 'g--')

            axs[index, 1].set(xlabel=pc_xlabel, ylabel=pc_ylabel, title=f"{set_type} Precision Recall")
            axs[index, 1].set_xticks(np.arange(0, 1.1, 0.1))
            axs[index, 1].set_yticks(np.arange(0, 1.1, 0.1))
            axs[index, 1].grid(color='gray', linestyle='-', linewidth=0.5)
            axs[index, 1].plot([0, 1], [0, 0], 'g--')

        all_roc_aucs = [None] * len(classifiers)
        test_roc_aucs = [None] * len(classifiers)
        all_prc_aucs = [None] * len(classifiers)
        test_prc_aucs = [None] * len(classifiers)
        all_prc_max_f1 = [None] * len(classifiers)
        test_prc_max_f1 = [None] * len(classifiers)
        for index, (metrics_, svms) in enumerate(classifiers.items()):
            all_roc_auc = np.array([svm["all_roc_auc"] for svm in svms])
            test_roc_auc = np.array([svm["test_roc_auc"] for svm in svms])
            all_prc_auc = np.array([svm["all_prc_auc"] for svm in svms])
            test_prc_auc = np.array([svm["test_prc_auc"] for svm in svms])
            all_prc_f1 = np.array([svm["all_prc_auc"].max() for svm in svms])
            test_prc_f1 = np.array([svm["test_prc_auc"].max() for svm in svms])

            svm = svms[all_prc_auc.argmax()]

            axs[0, 0].plot(svm["all_fpr"], svm["all_tpr"])
            axs[1, 0].plot(svm["test_fpr"], svm["test_tpr"])
            
            axs[0, 1].plot(svm["all_rec"], svm["all_pr"])
            axs[1, 1].plot(svm["test_rec"], svm["test_pr"])

            all_roc_aucs[index] = {"name": metrics_, "auc": np.sort(all_roc_auc)}
            test_roc_aucs[index] = {"name": metrics_, "auc": np.sort(test_roc_auc)}
            all_prc_aucs[index] = {"name": metrics_, "auc": np.sort(all_prc_auc)}
            test_prc_aucs[index] = {"name": metrics_, "auc": np.sort(test_prc_auc)}
            all_prc_max_f1[index] = {"name": metrics_, "f1": np.sort(all_prc_f1)}
            test_prc_max_f1[index] = {"name": metrics_, "f1": np.sort(test_prc_f1)}

        aucs = {
            "all_roc": sorted(all_roc_aucs, key=lambda x: np.median(x["auc"])),
            "test_roc": sorted(test_roc_aucs, key=lambda x: np.median(x["auc"])),
            "all_prc": sorted(all_prc_aucs, key=lambda x: np.median(x["auc"])),
            "test_prc": sorted(test_prc_aucs, key=lambda x: np.median(x["auc"])),
            "all_prc_f1": sorted(all_prc_max_f1, key=lambda x: np.median(x["f1"])),
            "test_prc_f1": sorted(test_prc_max_f1, key=lambda x: np.median(x["f1"])),
        }

        # Boxplot AUC
        importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
        fig, ax_auc = plt.subplots(2, 2, figsize=(20, 20))
        fig.subplots_adjust(left=0.2, right=0.9, wspace=0.7)
        fig.suptitle("Area Under the Curve", fontsize=20)

        for index, set_type in enumerate(("all", "test")):
            roc_name = f"{set_type}_roc"
            indices = range(1, len(aucs[roc_name]) + 1)
            importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
            ax_auc[index, 0].grid(color='gray', linestyle='-', linewidth=0.5)
            ax_auc[index, 0].set_axisbelow(True)
            ax_auc[index, 0].boxplot([svm["auc"] for svm in aucs[roc_name]], vert=False)
            ax_auc[index, 0].set(xlabel="AUC", yticks=indices, title=f"{set_type.capitalize()} ROC AUC")
            ax_auc[index, 0].set_xticks(np.arange(0, 1.1, 0.1))
            ax_auc[index, 0].set_yticklabels([svm["name"] for svm in aucs[roc_name]])
            ax_auc[index, 0].set_xlim([0, 1])

            prc_name = f"{set_type}_prc"
            importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
            ax_auc[index, 1].grid(color='gray', linestyle='-', linewidth=0.5)
            ax_auc[index, 1].set_axisbelow(True)
            ax_auc[index, 1].boxplot([svm["auc"] for svm in aucs[prc_name]], vert=False)
            ax_auc[index, 1].set(xlabel="AUC", yticks=indices, title=f"{set_type.capitalize()} Precision Recall AUC")
            ax_auc[index, 1].set_xticks(np.arange(0, 1.1, 0.1))
            ax_auc[index, 1].set_yticklabels([svm["name"] for svm in aucs[prc_name]])
            ax_auc[index, 1].set_xlim([0, 1])

        # Boxplot F1
        importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
        fig, ax_auc = plt.subplots(2, 1, figsize=(20, 20))
        fig.subplots_adjust(left=0.2, right=0.9, wspace=0.7)
        fig.suptitle("Area Under the Curve", fontsize=20)

        for index, set_type in enumerate(("all", "test")):
            prc_name = f"{set_type}_prc_f1"
            indices = range(1, len(aucs[prc_name]) + 1)
            importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
            ax_auc[index].grid(color='gray', linestyle='-', linewidth=0.5)
            ax_auc[index].set_axisbelow(True)
            ax_auc[index].boxplot([svm["f1"] for svm in aucs[prc_name]], vert=False)
            ax_auc[index].set(xlabel="F1", yticks=indices, title=f"{set_type.capitalize()} Precision Recall Max F1")
            ax_auc[index].set_xticks(np.arange(0, 1.1, 0.1))
            ax_auc[index].set_yticklabels([svm["name"] for svm in aucs[prc_name]])
            ax_auc[index].set_xlim([0, 1])

        #
        # Plot Top Area Under Curves
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        fig.suptitle("SVM Performance of Each Metric", fontsize=20)
        roc_xlabel = "False Positive Rate"
        roc_ylabel = "True Positive Rate"
        pc_xlabel = "Recall"
        pc_ylabel = "Precision"

        for auc in aucs["all_prc"][0:6]:
            metrics_ = auc["name"]
            auc_values = auc["auc"]
            svms = classifiers[metrics_]
            svm = svms[auc_values.argmax()]

            axs[0, 0].plot(svm["all_fpr"], svm["all_tpr"], label=f"{metrics_}")
            axs[1, 0].plot(svm["test_fpr"], svm["test_tpr"], label=f"{metrics_}")

            axs[0, 1].plot(svm["all_rec"], svm["all_pr"], label=f"{metrics_}")
            axs[1, 1].plot(svm["test_rec"], svm["test_pr"], label=f"{metrics_}")

        for index, set_type in enumerate(("All", "Test")):
            axs[index, 0].set(xlabel=roc_xlabel, ylabel=roc_ylabel, title=f"{set_type} ROC")
            axs[index, 0].set_xticks(np.arange(0, 1.1, 0.1))
            axs[index, 0].set_yticks(np.arange(0, 1.1, 0.1))
            axs[index, 0].grid(color='gray', linestyle='-', linewidth=0.5)
            axs[index, 0].plot([0, 1], [0, 1], 'g--')
            axs[index, 0].legend(loc="lower right")

            axs[index, 1].set(xlabel=pc_xlabel, ylabel=pc_ylabel, title=f"{set_type} Precision Recall")
            axs[index, 1].set_xticks(np.arange(0, 1.1, 0.1))
            axs[index, 1].set_yticks(np.arange(0, 1.1, 0.1))
            axs[index, 1].grid(color='gray', linestyle='-', linewidth=0.5)
            axs[index, 1].plot([0, 1], [0, 0], 'g--')
            axs[index, 1].legend(loc="lower left")

        # Plot Top Area Under Curves
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        fig.suptitle("SVM Performance of Top 6 Metrics", fontsize=20)
        roc_xlabel = "False Positive Rate"
        roc_ylabel = "True Positive Rate"
        pc_xlabel = "Recall"
        pc_ylabel = "Precision"

        for auc in aucs["all_prc"][-6:]:
            metrics_ = auc["name"]
            auc_values = auc["auc"]
            svms = classifiers[metrics_]
            svm = svms[auc_values.argmax()]

            axs[0, 0].plot(svm["all_fpr"], svm["all_tpr"], label=f"{metrics_}")
            axs[1, 0].plot(svm["test_fpr"], svm["test_tpr"], label=f"{metrics_}")

            axs[0, 1].plot(svm["all_rec"], svm["all_pr"], label=f"{metrics_}")
            axs[1, 1].plot(svm["test_rec"], svm["test_pr"], label=f"{metrics_}")

        for index, set_type in enumerate(("All", "Test")):
            axs[index, 0].set(xlabel=roc_xlabel, ylabel=roc_ylabel, title=f"{set_type} ROC")
            axs[index, 0].set_xticks(np.arange(0, 1.1, 0.1))
            axs[index, 0].set_yticks(np.arange(0, 1.1, 0.1))
            axs[index, 0].grid(color='gray', linestyle='-', linewidth=0.5)
            axs[index, 0].plot([0, 1], [0, 1], 'g--')
            axs[index, 0].legend(loc="lower right")

            axs[index, 1].set(xlabel=pc_xlabel, ylabel=pc_ylabel, title=f"{set_type} Precision Recall")
            axs[index, 1].set_xticks(np.arange(0, 1.1, 0.1))
            axs[index, 1].set_yticks(np.arange(0, 1.1, 0.1))
            axs[index, 1].grid(color='gray', linestyle='-', linewidth=0.5)
            axs[index, 1].plot([0, 1], [0, 0], 'g--')
            axs[index, 1].legend(loc="lower left")

        plt.show()
        
        
        # PCA
        metrics_train_scaled = pd.DataFrame(metrics_train_scaled, columns=metrics_train_columns)
        metrics_test_scaled = pd.DataFrame(metrics_test_scaled, columns=metrics_test_columns)
        importlib.reload(np.core.numeric)

        pca = PCA()
        metrics_train_pca = pca.fit_transform(metrics_train_scaled)
        
        per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
        labels = [str(x) for x in range(1, len(per_var)+1)]

        importlib.reload(np.core.numeric)
        fig, ax_pca = plt.subplots(figsize=(10, 10))
        ax_pca.bar(x=range(1, len(per_var)+1), height=per_var)
        ax_pca.tick_params(
            axis='x',
            which="both",
            bottom=False,
            top=False,
        )
        ax_pca.set_xticks(labels)
        ax_pca.set_ylabel("Percentage of Explained Variance")
        ax_pca.set_xlabel("Principal Components")
        ax_pca.set_title("Scree Plot")
        
        # Plot of two best PC
        train_pc1_coords = metrics_train_pca[:, 0]
        train_pc2_coords = metrics_train_pca[:, 1]
        
        pca_trained_scaled = scale(np.column_stack((train_pc1_coords, train_pc2_coords)))

        reduced_classifier = SVC(random_state=42)
        reduced_classifier.fit(pca_trained_scaled, review_train["Reviewer Union"])

        metrics_test_pca = pca.transform(metrics_train_scaled)

        test_pc1_coords = metrics_test_pca[:, 0]
        test_pc2_coords = metrics_test_pca[:, 1]
        
        x_min = test_pc1_coords.min() - 1
        x_max = test_pc1_coords.max() + 1

        y_min = test_pc2_coords.min() - 1
        y_max = test_pc2_coords.max() + 1

        xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1), np.arange(start=y_min, stop=y_max, step=0.1))

        z = reduced_classifier.predict(np.column_stack((xx.ravel(), yy.ravel())))
        z = z.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.contourf(xx, yy, z, alpha=0.1)
        cmap = matplotlib.colors.ListedColormap(["#e41a1c", "#4daf4a"])

        scatter_plot = ax.scatter(
            test_pc1_coords,
            test_pc2_coords,
            c=review_train["Reviewer Union"],
            cmap=cmap,
            s=100,
            edgecolors='k',
            alpha=0.7,
        )

        legend = ax.legend(scatter_plot.legend_elements()[0], scatter_plot.legend_elements()[1], loc="upper right")
        legend.get_texts()[0].set_text("Good Channel")
        legend.get_texts()[1].set_text("Bad Channel")

        ax.set_ylabel("PC 2")
        ax.set_xlabel("PC 1")
        ax.set_title("Decision Surface using the PCA Transformed/Projected Features")

        plt.show()

    def test_save_svm_aggregate(self):
        review_dataframe, metrics_dataframe, metrics_all_scaled, artifact_metrics, aggregate_data = self.aggregate_data()

        svm = SVC(probability=True)  # class_weight="balanced", random_state=42
        metrics_list = ["r_squared", "normal_entropy", "mae", "mse", "rmse"]
        metrics_ = metrics_all_scaled[metrics_list]
        importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
        svm.fit(metrics_.to_numpy(), review_dataframe["Reviewer Union"])

        review_all_prediction = svm.decision_function(metrics_)
        d_pr, d_rec, d_thresh = metrics.precision_recall_curve(review_dataframe["Reviewer Union"], review_all_prediction)

        review_all_probability = svm.predict_proba(metrics_)
        p_pr, p_rec, p_prec_thresh = metrics.precision_recall_curve(review_dataframe["Reviewer Union"], review_all_probability[:, 1])
        p_fpr, p_tpr, p_roc_thresh = metrics.roc_curve(review_dataframe["Reviewer Union"], review_all_probability[:, 1])

        pc_xlabel = "Recall"
        pc_ylabel = "Precision"

        importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs[0, 0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"ROC")
        axs[0, 0].set_xticks(np.arange(0, 1.1, 0.1))
        axs[0, 0].set_yticks(np.arange(0, 1.1, 0.1))
        axs[0, 0].grid(color='gray', linestyle='-', linewidth=0.5)
        axs[0, 0].plot([0, 0], [1, 1], 'g--')
        axs[0, 0].plot(p_fpr, p_tpr)

        axs[0, 1].set(xlabel="Threshold", ylabel="False Positive Rate", title=f"ROC Threshold")
        axs[0, 1].set_xticks(np.arange(0, 1.1, 0.1))
        axs[0, 1].set_yticks(np.arange(0, 1.1, 0.1))
        axs[0, 1].grid(color='gray', linestyle='-', linewidth=0.5)
        axs[0, 1].plot(p_roc_thresh[1:], p_fpr[1:])

        axs[1, 0].set(xlabel=pc_xlabel, ylabel=pc_ylabel, title=f"Precision Recall")
        axs[1, 0].set_xticks(np.arange(0, 1.1, 0.1))
        axs[1, 0].set_yticks(np.arange(0, 1.1, 0.1))
        axs[1, 0].grid(color='gray', linestyle='-', linewidth=0.5)
        axs[1, 0].plot([0, 1], [0, 0], 'g--')
        axs[1, 0].plot(p_rec, p_pr)

        axs[1, 1].set(xlabel="Threshold", ylabel=pc_ylabel, title=f"Precision Threshold")
        axs[1, 1].set_xticks(np.arange(0, 1.1, 0.1))
        axs[1, 1].set_yticks(np.arange(0, 1.1, 0.1))
        axs[1, 1].grid(color='gray', linestyle='-', linewidth=0.5)
        axs[1, 1].plot([0, 1], [0, 0], 'g--')
        axs[1, 1].plot(p_prec_thresh, p_pr[1:])



        importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
        plt.show()

        p_index_r = int(np.searchsorted(p_rec[1:], 0.8, side="right") - 1)
        p_index_p = int(np.searchsorted(p_pr[1:], 0.95, side="right") - 1)
        thresh = p_prec_thresh[p_index_r]
        print(thresh)

        s = pickle.dumps(svm)
        with self.SVM_PATH.open("wb") as file_object:
            file_object.write(s)

    def test_unclassified_aggregate(self):

        #
        review_dataframe, metrics_dataframe, metrics_all_scaled, artifact_metrics, aggregate_data = self.aggregate_data()

        # Create Train and Test Datasets
        data_split = train_test_split(metrics_dataframe, review_dataframe, random_state=None)
        metrics_train, metrics_test, review_train, review_test = data_split
        metrics_train_columns = metrics_train.columns
        metrics_test_columns = metrics_test.columns

        # Scale data
        importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
        metrics_train_scaled = scale(metrics_train.to_numpy())
        metrics_test_scaled = scale(metrics_test.to_numpy())

        metrics_train_scaled = pd.DataFrame(metrics_train_scaled, columns=metrics_train_columns)
        metrics_test_scaled = pd.DataFrame(metrics_test_scaled, columns=metrics_test_columns)
        importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....

        metrics_list = list(self.BEST_METRICS)
        all_set = metrics_all_scaled[metrics_list]
        training_set = metrics_train_scaled[metrics_list]
        testing_set = metrics_test_scaled[metrics_list]

        importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
        classifier = SVC()  # class_weight="balanced", random_state=42
        classifier.fit(training_set.to_numpy(), review_train["Reviewer Union"])

        review_all_prediction = classifier.decision_function(all_set)
        review_test_prediction = classifier.decision_function(testing_set)

    def test_torch_metrics_aggregate(self):
        review_dataframe, metrics_dataframe, metrics_all_scaled, artifact_metrics, aggregate_data = self.aggregate_data()

        for t_set in range(0, 100):
            # Create Train and Test Datasets
            data_split = train_test_split(metrics_dataframe, review_dataframe, random_state=None)
            metrics_train, metrics_test, review_train, review_test = data_split
            metrics_train_columns = metrics_train.columns
            metrics_test_columns = metrics_test.columns

            # Scale data
            importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
            metrics_train_scaled = scale(metrics_train.to_numpy())
            metrics_test_scaled = scale(metrics_test.to_numpy())

            metrics_train_scaled = pd.DataFrame(metrics_train_scaled, columns=metrics_train_columns)
            metrics_test_scaled = pd.DataFrame(metrics_test_scaled, columns=metrics_test_columns)
            importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....


# Main #
if __name__ == '__main__':
    pytest.main(["-v", "-s"])
