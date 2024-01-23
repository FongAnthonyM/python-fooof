"""one_s_fp_fn.py
A pipeline for removing artifact.
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
from collections import deque
import importlib
from multiprocessing import Pool
import itertools
import pathlib
import random
from typing import NamedTuple

# Third-Party Packages #
from dspobjects.plot import *
from fooof.sim.gen import gen_aperiodic
from fooof import FOOOF, FOOOFGroup
import hdf5objects
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.io import loadmat
from scipy.stats import entropy
from scipy.signal import savgol_filter, welch
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
from src.spikedetection.artifactrejection.fooof.ooffitter import OOFFitter, iterdim, calculate_mean_errors


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# Definitions #
# Data Classes
class ElectrodeLead(NamedTuple):
    name: str
    type: str
    contacts: dict


# Classes #


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


def load_data(files, info):
    artifact_info = toml.load(info.as_posix())["raters"]
    artifact_data = {}

    for file in files:
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

#%%
# Parameters #
SVM_PATH = pathlib.Path.cwd().joinpath("all_metric_svm.obj")
ARTIFACT_DIR = pathlib.Path("/home/anthonyfong/ProjectData/EpilepsySpikeDetection/Artifact_Review/")
ARTIFACT_INFO = ARTIFACT_DIR.joinpath("Artifact_Info_2.toml")
ARTIFACT_FILES = ARTIFACT_DIR.glob("*.mat")
OUT_DIR = pathlib.Path("/home/anthonyfong/ProjectData/EpilepsySpikeDetection/Artifact_Review/Images")
PLOT_NAME = "Spectra"
TIME_AXIS = 0
CHANNEL_AXIS = 1
LOWER_FREQUENCY = 1
UPPER_FREQUENCY = 250
BEST_METRICS = {"r_squared", "normal_entropy", "mae", "rmse"}
SAMPLE_RATE = 1024.0

window_size = 1.0
nperseg = min([int(window_size*SAMPLE_RATE), int(2 * SAMPLE_RATE)])

target_precision = 0.95
shuffles = 100

#%%
# FOOOF
fo = FOOOF(peak_width_limits=[4, 8], min_peak_height=0.05, max_n_peaks=1, verbose=True)

#%%
# Aggregate Data #
# Load Data
artifact_data, artifact_info = load_data(ARTIFACT_FILES, ARTIFACT_INFO)

# Create Data Structures
ag_reviews = {reviewer["name"]: deque() for reviewer in artifact_info}
ag_reviews |= {"Reviewer Intersection": deque(), "Reviewer Union": deque()}
ag_metrics = []

artifact_metrics = {}
flat_data = {
    "labels": deque(),
    "reviews": ag_reviews,
    "data": deque(),
    "spectra": deque(),
    "frequencies": deque(),
    "metrics": ag_metrics,
}

# Flatten Data
for subject_id, data in artifact_data.items():
    for i, artifact_clip in enumerate(data):
        n_channels = artifact_clip["data"].shape[CHANNEL_AXIS]
        flat_data["labels"] += [f"{subject_id} Clip{i} Channel {j}: {name}" for j, name in enumerate(artifact_clip["channel_labels"])]

        for channel in iterdim(artifact_clip["data"], CHANNEL_AXIS):
            flat_data["data"].append(channel)

        # Format Reviewer Data
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

        for reviewer, channels in reviews.items():
            good_channels = np.zeros((n_channels,))
            good_channels[channels, ] = 1
            flat_data["reviews"][reviewer] += list(good_channels)

#%%
# Create Metrics
for bin_c in flat_data["data"]:
    freqs, spectra = welch(bin_c, fs=SAMPLE_RATE, nperseg=nperseg, axis=TIME_AXIS)

    # Limit Frequency Range
    lower_limit = int(np.searchsorted(freqs, LOWER_FREQUENCY, side="right") - 1)
    upper_limit = int(np.searchsorted(freqs, UPPER_FREQUENCY, side="right"))

    spectra = spectra[(slice(lower_limit, upper_limit),)]

    freqs = freqs[lower_limit:upper_limit]

    spectra = np.log10(spectra)
    flat_data["spectra"].append(spectra)
    flat_data["frequencies"].append(freqs)

    ag_metrics.append(spectra)

# Load Data into Pandas Data Frame
review_dataframe = pd.DataFrame.from_dict(ag_reviews)
metrics_dataframe = pd.DataFrame.from_dict(ag_metrics)
# for name, metric_ in ag_metrics.items():
#     ag_metrics[name] = np.array(metric_)

importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
metrics_all_scaled = scale(metrics_dataframe.to_numpy())
metrics_all_scaled = pd.DataFrame(metrics_all_scaled, columns=metrics_dataframe.columns)


#%%
def create_svm(metrics_df, review_df, mas, rs):
    # Create Train and Test Datasets
    data_split = train_test_split(metrics_df, review_df, random_state=rs)
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
    all_set = mas
    training_set = metrics_train_scaled
    testing_set = metrics_test_scaled

    importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
    svm["classifier"] = classifier = SVC()  # class_weight="balanced", random_state=42
    classifier.fit(training_set.to_numpy(), review_train["Reviewer Union"])

    review_all_decision = classifier.decision_function(all_set)
    review_test_decision = classifier.decision_function(testing_set)

    svm["all_fpr"], svm["all_tpr"], svm["all_roc_thres"] = metrics.roc_curve(
        review_dataframe["Reviewer Union"],
        review_all_decision,
    )
    svm["test_fpr"], svm["test_tpr"], svm["test_roc_thres"] = metrics.roc_curve(
        review_test["Reviewer Union"],
        review_test_decision,
    )
    svm["all_roc_auc"] = metrics.auc(svm["all_fpr"], svm["all_tpr"])
    svm["test_roc_auc"] = metrics.auc(svm["test_fpr"], svm["test_tpr"], )

    svm["all_pr"], svm["all_rec"], thresholds = metrics.precision_recall_curve(
        review_dataframe["Reviewer Union"],
        review_all_decision,
    )
    svm["all_prr_thres"] = thresholds

    svm["test_pr"], svm["test_rec"], svm["test_prr_thres"] = metrics.precision_recall_curve(
        review_test["Reviewer Union"],
        review_test_decision,
    )
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

    return svm


with Pool(processes=50) as pool:
    workers = [
        pool.apply_async(
            create_svm,
            (metrics_dataframe, review_dataframe, metrics_all_scaled, random.randrange(100000000))
        ) for t_set in range(shuffles)
    ]
    svms = [w.get(timeout=3600) for w in workers]

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
    fp_metrics[index] = {"data": flat_data["data"][id], "spectra": flat_data["spectra"][id]}
    fp_labels[index] = flat_data["labels"][id]

fn_metrics = [None] * len(fn_df)
fn_labels = [""] * len(fn_df)
for index, id in enumerate(fn_df["id"]):
    fn_metrics[index] = {"data": flat_data["data"][id], "spectra": flat_data["spectra"][id]}
    fn_labels[index] = flat_data["labels"][id]

fp_raw = np.stack([m["data"] for m in fp_metrics], axis=1)
fp_spectra = np.stack([m["spectra"] for m in fp_metrics], axis=1)

fn_raw = np.stack([m["data"] for m in fn_metrics], axis=1)
fn_spectra = np.stack([m["spectra"] for m in fn_metrics], axis=1)

# %%
all_fpr = [svm["all_fpr"] for svm in svms]
all_tpr = [svm["all_tpr"] for svm in svms]
all_roc_thres = [svm["all_roc_thres"] for svm in svms]
all_rec = [svm["all_rec"] for svm in svms]
all_pr = [svm["all_pr"] for svm in svms]
test_roc_thres = [svm["test_roc_thres"] for svm in svms]

test_fpr = [svm["test_fpr"] for svm in svms]
test_tpr = [svm["test_tpr"] for svm in svms]
all_prr_thres = [svm["all_prr_thres"] for svm in svms]
test_rec = [svm["test_rec"] for svm in svms]
test_pr = [svm["test_pr"] for svm in svms]
test_prr_thres = [svm["test_prr_thres"] for svm in svms]

#%%
# Plot Performance
importlib.reload(np.core.numeric)  # Pandas causes numpy to break which is dumb....
ts_group = ClassifierPerformanceGroup(layout_settings=dict(title=dict(text=f"{PLOT_NAME} Performance")))
ts_group["all"]["roc"].build(x=all_fpr, y=all_tpr, thresholds=all_roc_thres)
ts_group["all"]["precisionrecall"].build(x=all_rec, y=all_pr, thresholds=all_prr_thres)
ts_group["test"]["roc"].build(x=test_fpr, y=test_tpr, thresholds=test_roc_thres)
ts_group["test"]["precisionrecall"].build(x=test_rec, y=test_pr, thresholds=test_prr_thres)
ts_group.figure.show()
ts_group.figure.write_html(OUT_DIR.joinpath(f"{PLOT_NAME}_Performance_Plot.html").as_posix())

#%%
# Plot False Positives
fp_fig = Figure()
fp_fig.update_layout(title=f"{PLOT_NAME} False Positives Peaks", margin=dict(t=60))
fp_fig.update_layout(TimeSeriesPlot.default_layout_settings)
fp_fig.set_subplots(1, 3, horizontal_spacing=0.01, column_widths=[0.1, 0.6, 0.3])
fp_sub = fp_fig.subplots

fp_hist_plot = BarPlot(subplot=fp_sub[0][0], x=fp_df["counts"], labels=fp_labels, orientation='h', separated=True)
fp_hist_plot.update_title(text="Occurrence Histogram")

fp_group = TimeSpectraGroup(figure=fp_fig, locations=dict(timeseries=(0, 1), spectra=(0, 2)))
fp_group["timeseries"].build(y=fp_raw, sample_rate=1024, labels=fp_labels, title=dict(text="Preprocessed Time Series"))
fp_group["spectra"].build(x=flat_data["frequencies"][0], y=fp_spectra, labels=fp_labels)
fp_group.assign_yaxes((("timeseries", fp_hist_plot.yaxis), ("spectra", fp_hist_plot.yaxis)))

fp_hist_plot.group_same_legend_items(fp_group["timeseries"])

fp_fig.show()
fp_fig.write_html(OUT_DIR.joinpath(f"{PLOT_NAME}_FP_Plot.html").as_posix())

# Plot False Negatives
fn_fig = Figure()
fn_fig.update_layout(title=f"{PLOT_NAME} False Negatives Peaks", margin=dict(t=60))
fn_fig.update_layout(TimeSeriesPlot.default_layout_settings)
fn_fig.set_subplots(1, 3, horizontal_spacing=0.01, column_widths=[0.1, 0.6, 0.3])
fn_sub = fn_fig.subplots

fn_hist_plot = BarPlot(subplot=fn_sub[0][0], x=fn_df["counts"], labels=fn_labels, orientation='h', separated=True)
fn_hist_plot.update_title(text="Occurrence Histogram")

fn_ts_plot = TimeSeriesPlot(subplot=fn_sub[0][1], y=fn_raw, sample_rate=1024, labels=fn_labels)
fn_ts_plot.update_title(text="Preprocessed Time Series")
fn_ts_plot.set_yaxis(fn_hist_plot.yaxis)

fn_sp_plot = SpectraPlot(subplot=fn_sub[0][2], x=flat_data["frequencies"][0], y=fn_spectra, labels=fn_labels)
fn_sp_plot.update_title(text="Spectra")
fn_sp_plot.set_yaxis(fn_hist_plot.yaxis)

fn_hist_plot.group_same_legend_items(fn_ts_plot, fn_sp_plot)

fn_fig.show()
fn_fig.write_html(OUT_DIR.joinpath(f"{PLOT_NAME}_FN_Plot.html").as_posix())

print("Done")
