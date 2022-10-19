import argparse
import copy
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from load_external_binning import load_external_binning


def clean_resp(ksp_in, coord_in, dcf_in, resp_in, diagnostics_dir):
    # Copy input data
    ksp = copy.deepcopy(ksp_in)
    coord = copy.deepcopy(coord_in)
    dcf = copy.deepcopy(dcf_in)
    resp = copy.deepcopy(resp_in)
    # Interpolate resp to spline
    # Find Peaks and Valleys
    prominence = 0.05
    peak_idx, p_prop = find_peaks(resp, prominence=prominence)
    valley_idx, v_prop = find_peaks(resp * -1, prominence=prominence)

    if peak_idx.size < valley_idx.size:
        valley_idx = valley_idx[:-1]

    # Compute peak to peak distances, median distance, and interquartile range.
    # periods = np.diff(peak_idx)
    # p_median = np.median(periods)
    # p_iqr = np.subtract(*np.percentile(periods, [75, 25]))
    # max_period = p_median + 4 * p_iqr

    bins = np.zeros_like(resp)
    # Need to check first location is a minima or maxima
    if (peak_idx[0] - valley_idx[0]) < 0:
        s_idx = 0
    else:
        s_idx = 1
    # Filter based on max amplitude ( if smaller than X, throw away)

    # find the amplitude between peak and the base(minima)
    min_amp = 0.2
    max_amp = 8.0
    for k in range(valley_idx.size - 1):
        # Exclude if amplitude is too small or too big
        if (
            (resp[peak_idx[k]] - resp[valley_idx[k + s_idx]] > min_amp)
            & (resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]] > min_amp)
            & (resp[peak_idx[k]] - resp[valley_idx[k + s_idx]] < max_amp)
            & (resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]] < max_amp)
            # & (peak_idx[k + 1] - peak_idx[k] < max_period)
        ):

            # Bin assignment
            bins[peak_idx[k] : peak_idx[k + 1]] = 1

    # Plot binned waveform
    plt.rcParams["figure.figsize"] = (16, 9)
    # plt.rcParams["figure.facecolor"] = "black"
    # plt.rcParams["axes.facecolor"] = "black"
    plt.style.use("dark_background")
    colors = ["r", "g"]
    plt.gca().set_prop_cycle(color=colors)
    resp_sub = resp
    bins_sub = bins
    resp_array = np.ma.masked_where(bins_sub != 0, resp_sub)
    plt.plot(np.arange(resp_sub.size), resp_array, label=f"Excluded", color="r")
    resp_array = np.ma.masked_where(bins_sub != 1, resp_sub)
    plt.plot(np.arange(resp_sub.size), resp_array, label=f"Included", color="g")
    plt.legend()
    plt.title("Filtered Respiratory Data")
    plt.xlabel("Radial Spoke")
    plt.ylabel("Amplitude")
    plt.savefig(diagnostics_dir + "resp_cleaned_full.png", facecolor="black", edgecolor="none")
    # plt.show()
    plt.close()

    # Bin Data
    idx = bins == 1
    ksp = ksp[:, idx]
    coord = coord[idx]
    dcf = dcf[idx]
    resp = resp[idx]

    return ksp, coord, dcf, resp


def bin_periodically(ksp_in, coord_in, dcf_in, resp_in, n_bins, diagnostics_dir):
    if n_bins % 2:
        raise ValueError(f"Number of bins should be even: Current value: {n_bins}!")

    # Copy input data
    ksp = copy.deepcopy(ksp_in)
    coord = copy.deepcopy(coord_in)
    dcf = copy.deepcopy(dcf_in)
    resp = copy.deepcopy(resp_in)

    # Interpolate resp to spline
    # Find Peaks and Valleys
    prominence = 0.5
    peak_idx, p_prop = find_peaks(resp, prominence=prominence)
    valley_idx, v_prop = find_peaks(resp * -1, prominence=prominence)

    if peak_idx.size < valley_idx.size:
        valley_idx = valley_idx[:-1]

    resp_smol = resp[1000:5000]
    peak_idx_smol, _ = find_peaks(resp_smol, prominence=prominence)
    valley_idx_smol, _ = find_peaks(resp_smol * -1, prominence=prominence)
    plt.plot(resp_smol, "#1f77b4")
    plt.scatter(peak_idx_smol, resp_smol[peak_idx_smol], color="red", marker="x", label="peaks")
    plt.scatter(valley_idx_smol, resp_smol[valley_idx_smol], color="gold", marker="x", label="valleys")
    plt.savefig(diagnostics_dir + "resp_peaks_valleys.png")
    plt.legend()
    plt.grid()
    plt.close()
    bins = n_bins * np.ones_like(resp)
    # Need to check first location is a minima or maxima
    if (peak_idx[0] - valley_idx[0]) < 0:
        s_idx = 0
    else:
        s_idx = 1

    # find the amplitude between peak and the base(minima)
    min_amp = 0.2
    max_amp = 8.0
    for k in range(valley_idx.size - 1):
        # Exclude if amplitude is too small or too big
        if (
            (resp[peak_idx[k]] - resp[valley_idx[k + s_idx]] > min_amp)
            & (resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]] > min_amp)
            & (resp[peak_idx[k]] - resp[valley_idx[k + s_idx]] < max_amp)
            & (resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]] < max_amp)
        ):
            amp_left = resp[peak_idx[k]] - resp[valley_idx[k + s_idx]]
            amp_right = resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]]

            # find the number of data points between peak and the base(minima)
            n_left = valley_idx[k + s_idx] - peak_idx[k]
            n_right = peak_idx[k + 1] - valley_idx[k + s_idx]

            # select the area of interest to find the intersection point
            resp_left = resp[peak_idx[k] : peak_idx[k] + n_left]
            resp_right = resp[valley_idx[k + s_idx] : valley_idx[k + s_idx] + n_right]

            # intersection points
            bin_amp = amp_left / (n_bins // 2)  # Bin size for each compartment
            y_left = []
            for b in range(n_bins // 2):
                y_left.append(resp[peak_idx[k]] - (0.5 + b) * bin_amp)

            bin_amp = amp_right / (n_bins // 2)  # Bin size for each compartment
            y_right = []
            for b in range(n_bins // 2):
                y_right.append(resp[valley_idx[k + s_idx]] + (0.5 + b) * bin_amp)

            # find the indices for each partition (left)
            n = [peak_idx[k]]
            m = [valley_idx[k + s_idx]]
            for b in range(n_bins // 2):
                n.append(np.argmin(np.abs(resp_left - y_left[b])) + peak_idx[k])
                m.append(np.argmin(np.abs(resp_right - y_right[b])) + valley_idx[k + s_idx])
            n.append(n_left + peak_idx[k])
            m.append(n_right + valley_idx[k + s_idx])
            # Bin assignment
            # Left
            for b in range(1 + n_bins // 2):
                bins[n[b] : n[b + 1]] = b

            # Right
            for b in range(n_bins // 2):
                bins[m[b] : m[b + 1]] = b + (n_bins // 2)
            bins[m[-2] : m[-1]] = 0

    # Plot binned waveform
    plt.rcParams["figure.figsize"] = (16, 9)
    # plt.rcParams["figure.facecolor"] = "black"
    # plt.rcParams["axes.facecolor"] = "black"
    plt.style.use("dark_background")
    colors = plt.cm.seismic(np.linspace(0, 1, n_bins + 1))
    plt.gca().set_prop_cycle(color=colors)
    resp_sub = resp
    bins_sub = bins
    for b in range(n_bins):
        resp_array = np.ma.masked_where(bins_sub != b, resp_sub)
        plt.plot(np.arange(resp_sub.size), resp_array, label=f"Bin {b}")
    resp_array = np.ma.masked_where(bins_sub != n_bins, resp_sub)
    plt.plot(np.arange(resp_sub.size), resp_array, label=f"Excluded", color="g")
    plt.legend()
    plt.title("Resp Binning")
    plt.xlabel("Radial Spoke")
    plt.ylabel("Amplitude")
    plt.savefig(diagnostics_dir + "resp_binned_external_full.png", facecolor="black", edgecolor="none")
    # plt.show()
    plt.close()

    plt.rcParams["figure.figsize"] = (16, 9)
    # plt.rcParams["figure.facecolor"] = "black"
    # plt.rcParams["axes.facecolor"] = "black"
    plt.style.use("dark_background")
    colors = plt.cm.seismic(np.linspace(0, 1, n_bins + 1))
    plt.gca().set_prop_cycle(color=colors)
    resp_sub = resp[1000:5000]
    bins_sub = bins[1000:5000]
    for b in range(n_bins + 1):
        resp_array = np.ma.masked_where(bins_sub != b, resp_sub)
        plt.plot(np.arange(resp_sub.size), resp_array, label=f"Bin {b}")
    resp_array = np.ma.masked_where(bins_sub != n_bins, resp_sub)
    plt.plot(np.arange(resp_sub.size), resp_array, label=f"Excluded", color="g")
    plt.legend()
    plt.title("Resp Binning")
    plt.xlabel("Radial Spoke")
    plt.ylabel("Amplitude")
    plt.savefig(diagnostics_dir + "resp_binned_external_smol.png", facecolor="black", edgecolor="none")
    # plt.show()
    plt.close()

    # Bin Data
    kspB = []
    coordB = []
    dcfB = []
    # print(bins.shape)
    for b in range(n_bins):
        idx = bins == b
        kspB.append(ksp[:, idx])
        coordB.append(coord[idx])
        dcfB.append(dcf[idx])
    return kspB, coordB, dcfB


def get_consecutive(data, stepsize=1):
    # Splits data into consecutive portions
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


# def clean_resp(resp, diagnostics_dir, N=2000):
#     x_array = np.arange(resp.size)
#     resp = pd.Series(resp)
#     resp_mean = resp.rolling(N).mean()
#     resp_std = resp.rolling(N).std()
#     # Use global stdev of rolling stdev to define threshold
#     zscore = 2.5
#     thresh = np.nanmedian(resp_std) + np.nanstd(resp_std) * zscore  # 4 standard deviations should remove the outliers.
#     # print(thresh)
#     # Find outilers
#     idx = np.squeeze(np.array(np.where(resp_std > thresh)))
#     # print(idx.shape)
#     # Now add window size to edge indices to account for delays from edge effects of rolling statistics.
#     cons = get_consecutive(idx)
#     idx = []
#     for ii in range(len(cons)):
#         # Add window length N to excluded indices
#         idx_t = np.arange(cons[ii].min() - N, cons[ii].max() + N)
#         idx = np.concatenate((idx, idx_t))
#     # Now clean idx (ensure unique indices and within bounds of projection count)
#     idx = np.unique(idx[(idx >= 0) & (idx < resp.size)])
#     idx_keep = np.setdiff1d(np.arange(resp.size), idx)

#     resp_keep = resp_mean.copy()
#     resp_keep[idx] = np.nan
#     # Save a diagnostics plot
#     fig, ax = plt.subplots(4, constrained_layout=True)
#     ax[0].plot(x_array, resp, "k--")
#     ax[0].plot(idx, resp.max().repeat(idx.size), "ro", linestyle="None")
#     ax[1].plot(x_array, resp_mean, "k")
#     ax[1].plot(idx, resp_mean.max().repeat(idx.size), "ro", linestyle="None")
#     ax[2].plot(x_array, resp_std, "b", label="Rolling std dev")
#     ax[2].plot(x_array, np.repeat(thresh, x_array.size), "r", label="{}*sigma".format(zscore))
#     ax[3].plot(x_array, resp_keep, "b")
#     fig.suptitle("Respiratory waveform diagnostics")

#     ax[0].set_title("Original Waveform")
#     ax[1].set_title("{} projection Rolling Mean".format(N))
#     ax[2].set_title("{} projection Rolling Std".format(N))
#     ax[3].set_title("Filtered Resp. Waveform")
#     ax[3].set_xlabel("Projection Number")

#     ax[0].xaxis.set_visible(False)
#     ax[1].xaxis.set_visible(False)
#     ax[2].xaxis.set_visible(False)
#     ax[2].legend()
#     plt.savefig(diagnostics_dir + "resp_diagnostics.png")
#     plt.close()
#     return np.array(resp[idx_keep]), idx_keep


def bin_motion_states(
    ksp_in,
    coord_in,
    dcf_in,
    resp_in,
    n,
    diagnostics_dir,
    filter_bulk=False,
    filter_extremes=False,
    external=False,
    external_path=None,
):
    """Bin kspace, coordinates, and dcf by respiratory motion. 
        This bins for equal motion bins as opposed to equal SNR.

    Args:
        ksp (array): kspace
        coord (array): coordinates.
        dcf (array): density weights.
        resp (array): waveform.
        n (int): number of bins.

    Returns:
        kspB
        coordB
        dcfB
    """

    # Copy input data
    ksp = copy.deepcopy(ksp_in)
    coord = copy.deepcopy(coord_in)
    dcf = copy.deepcopy(dcf_in)
    resp = copy.deepcopy(resp_in)

    if external and external_path is not None:
        bins = load_external_binning(external_path)
        kspB = []
        coordB = []
        dcfB = []
        # print(bins.shape)
        for b in range(n):
            idx = bins == b
            kspB.append(ksp[:, idx])
            coordB.append(coord[idx])
            dcfB.append(dcf[idx])
        # Plot binned waveform
        plt.rcParams["figure.figsize"] = (16, 9)
        # plt.rcParams["figure.facecolor"] = "black"
        # plt.rcParams["axes.facecolor"] = "black"
        plt.style.use("dark_background")
        colors = plt.cm.seismic(np.linspace(0, 1, n + 1))
        plt.gca().set_prop_cycle(color=colors)
        resp_sub = resp[1000:5000]
        bins_sub = bins[1000:5000]
        for b in range(n + 1):
            resp_array = np.ma.masked_where(bins_sub != b, resp_sub)
            plt.plot(np.arange(resp_sub.size), resp_array, label=f"Bin {b}")
        plt.legend()
        plt.title("Resp Binning")
        plt.xlabel("Radial Spoke")
        plt.ylabel("Amplitude")
        plt.savefig(diagnostics_dir + "resp_binned_external.png", facecolor="black", edgecolor="none")
        # plt.show()
        plt.close()
        return kspB, coordB, dcfB

    else:
        if filter_extremes:
            margin = 5  # Remove 1% at both ends
            bins = np.linspace(np.percentile(resp, margin), np.percentile(resp, 100 - margin), n + 1)
        else:
            bins = np.linspace(resp.min(), resp.max(), n + 1)

        kspB = []
        coordB = []
        dcfB = []
        for b in range(n):
            idx = (resp >= bins[b]) & (resp < bins[b + 1])
            # print(idx)
            # idx = respOrder[b * nSpokesB : (b + 1) * nSpokesB]
            kspB.append(ksp[:, idx])
            coordB.append(coord[idx])
            dcfB.append(dcf[idx])
            # print(len(idx))
            # print(ksp[:, idx].shape)
            # plt.plot(idx, resp[idx])
        # plt.show()
        # kspB = np.stack(kspB)
        # coordB = np.stack(coordB)
        # dcfB = np.stack(dcfB)

        # Plot binned waveform
        de_colores = plt.cm.get_cmap("tab20", n)
        plt.plot(resp, "k")
        for b in range(n):
            # idx = respOrder[b * nSpokesB : (b + 1) * nSpokesB]
            plt.plot(np.arange(resp.size), np.ones((resp.size,)) * bins[b], color=de_colores(b))
        plt.plot(np.arange(resp.size), np.ones((resp.size,)) * bins[b + 1], color=de_colores(b + 1))
        plt.savefig(diagnostics_dir + "resp_binned.png")
        plt.close()
        return kspB, coordB, dcfB


if __name__ == "__main__":
    sub_dir = "/home/ltorres/data/rawdata/ipf/"
    ignored = ["Original Subjects"]
    subjects = [x for x in os.listdir(sub_dir) if x not in ignored]
    subjects.sort()
    subjects = ["P179_Exam1"]
    for subject in subjects:
        print("reading data...")
        ksp = np.load(f"/home/ltorres/data/rawdata/nicu/{subject}/ksp.npy")
        coord = np.load(f"/home/ltorres/data/rawdata/nicu/{subject}/coord.npy")
        dcf = np.load(f"/home/ltorres/data/rawdata/nicu/{subject}/dcf.npy")
        resp = np.load(f"/home/ltorres/data/rawdata/nicu/{subject}/resp.npy")
        n = 6
        diagnostics_dir = f"/home/ltorres/data/recon/nicu/{subject}/diagnostics_run_external/"

        print("Binning data...")
        Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)

        kspB, coordB, dcfB = clean_resp(ksp, coord, dcf, resp, diagnostics_dir)
    # print("kspace shape: {}".format(kspB.shape))
    # print("trajectory shape: {}".format(coordB.shape))
    # print("dcf shape: {}".format(dcfB.shape))
