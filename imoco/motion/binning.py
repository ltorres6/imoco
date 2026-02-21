import copy
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def clean_resp(ksp_in, coord_in, dcf_in, resp_in, diagnostics_dir):
    """Remove breathing cycles with abnormal amplitude.

    Excludes entire breathing cycles whose peak-to-valley amplitude falls
    outside [0.2, 8.0] of the z-scored respiratory waveform.

    Reference: Section II-C of the JMRI paper.

    Args:
        ksp_in (ndarray): K-space data of shape (C, num_spokes, num_ro).
        coord_in (ndarray): Coordinates of shape (num_spokes, num_ro, D).
        dcf_in (ndarray): Density compensation of shape (num_spokes, num_ro).
        resp_in (ndarray): Respiratory signal of length num_spokes.
        diagnostics_dir (str): Directory for diagnostic plots.

    Returns:
        tuple: (ksp, coord, dcf, resp) with invalid cycles removed.
    """
    ksp = copy.deepcopy(ksp_in)
    coord = copy.deepcopy(coord_in)
    dcf = copy.deepcopy(dcf_in)
    resp = copy.deepcopy(resp_in)

    prominence = 0.05
    peak_idx, p_prop = find_peaks(resp, prominence=prominence)
    valley_idx, v_prop = find_peaks(resp * -1, prominence=prominence)

    if peak_idx.size < valley_idx.size:
        valley_idx = valley_idx[:-1]

    bins = np.zeros_like(resp)
    if (peak_idx[0] - valley_idx[0]) < 0:
        s_idx = 0
    else:
        s_idx = 1

    min_amp = 0.2
    max_amp = 8.0
    for k in range(valley_idx.size - 1):
        if (
            (resp[peak_idx[k]] - resp[valley_idx[k + s_idx]] > min_amp)
            & (resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]] > min_amp)
            & (resp[peak_idx[k]] - resp[valley_idx[k + s_idx]] < max_amp)
            & (resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]] < max_amp)
        ):
            bins[peak_idx[k] : peak_idx[k + 1]] = 1

    # Plot binned waveform
    plt.rcParams["figure.figsize"] = (16, 9)
    plt.style.use("dark_background")
    colors = ["r", "g"]
    plt.gca().set_prop_cycle(color=colors)
    resp_array = np.ma.masked_where(bins != 0, resp)
    plt.plot(np.arange(resp.size), resp_array, label="Excluded", color="r")
    resp_array = np.ma.masked_where(bins != 1, resp)
    plt.plot(np.arange(resp.size), resp_array, label="Included", color="g")
    plt.legend()
    plt.title("Filtered Respiratory Data")
    plt.xlabel("Radial Spoke")
    plt.ylabel("Amplitude")
    plt.savefig(os.path.join(diagnostics_dir, "resp_cleaned_full.png"), facecolor="black", edgecolor="none")
    plt.close()

    idx = bins == 1
    ksp = ksp[:, idx]
    coord = coord[idx]
    dcf = dcf[idx]
    resp = resp[idx]

    return ksp, coord, dcf, resp


def bin_periodically(ksp_in, coord_in, dcf_in, resp_in, n_bins, diagnostics_dir):
    """Bin k-space data into respiratory motion states using periodic binning.

    Assigns spokes to motion bins based on their position within each
    breathing cycle (peak-to-peak). Spokes outside valid cycles are excluded.

    Reference: Section II-D of the JMRI paper.

    Args:
        ksp_in (ndarray): K-space data of shape (C, num_spokes, num_ro).
        coord_in (ndarray): Coordinates of shape (num_spokes, num_ro, D).
        dcf_in (ndarray): Density compensation of shape (num_spokes, num_ro).
        resp_in (ndarray): Respiratory signal of length num_spokes.
        n_bins (int): Number of motion bins (must be even).
        diagnostics_dir (str): Directory for diagnostic plots.

    Returns:
        tuple: (kspB, coordB, dcfB) lists of length n_bins, each element
            containing the data for that motion state.
    """
    if n_bins % 2:
        raise ValueError(f"Number of bins should be even: Current value: {n_bins}!")

    ksp = copy.deepcopy(ksp_in)
    coord = copy.deepcopy(coord_in)
    dcf = copy.deepcopy(dcf_in)
    resp = copy.deepcopy(resp_in)

    prominence = 0.5
    peak_idx, p_prop = find_peaks(resp, prominence=prominence)
    valley_idx, v_prop = find_peaks(resp * -1, prominence=prominence)

    if peak_idx.size < valley_idx.size:
        valley_idx = valley_idx[:-1]

    # Diagnostic: peaks and valleys
    resp_smol = resp[1000:5000]
    peak_idx_smol, _ = find_peaks(resp_smol, prominence=prominence)
    valley_idx_smol, _ = find_peaks(resp_smol * -1, prominence=prominence)
    plt.plot(resp_smol, "#1f77b4")
    plt.scatter(peak_idx_smol, resp_smol[peak_idx_smol], color="red", marker="x", label="peaks")
    plt.scatter(valley_idx_smol, resp_smol[valley_idx_smol], color="gold", marker="x", label="valleys")
    plt.savefig(os.path.join(diagnostics_dir, "resp_peaks_valleys.png"))
    plt.legend()
    plt.grid()
    plt.close()

    bins = n_bins * np.ones_like(resp)
    if (peak_idx[0] - valley_idx[0]) < 0:
        s_idx = 0
    else:
        s_idx = 1

    min_amp = 0.2
    max_amp = 8.0
    for k in range(valley_idx.size - 1):
        if (
            (resp[peak_idx[k]] - resp[valley_idx[k + s_idx]] > min_amp)
            & (resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]] > min_amp)
            & (resp[peak_idx[k]] - resp[valley_idx[k + s_idx]] < max_amp)
            & (resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]] < max_amp)
        ):
            amp_left = resp[peak_idx[k]] - resp[valley_idx[k + s_idx]]
            amp_right = resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]]

            n_left = valley_idx[k + s_idx] - peak_idx[k]
            n_right = peak_idx[k + 1] - valley_idx[k + s_idx]

            resp_left = resp[peak_idx[k] : peak_idx[k] + n_left]
            resp_right = resp[valley_idx[k + s_idx] : valley_idx[k + s_idx] + n_right]

            bin_amp = amp_left / (n_bins // 2)
            y_left = []
            for b in range(n_bins // 2):
                y_left.append(resp[peak_idx[k]] - (0.5 + b) * bin_amp)

            bin_amp = amp_right / (n_bins // 2)
            y_right = []
            for b in range(n_bins // 2):
                y_right.append(resp[valley_idx[k + s_idx]] + (0.5 + b) * bin_amp)

            n = [peak_idx[k]]
            m = [valley_idx[k + s_idx]]
            for b in range(n_bins // 2):
                n.append(np.argmin(np.abs(resp_left - y_left[b])) + peak_idx[k])
                m.append(np.argmin(np.abs(resp_right - y_right[b])) + valley_idx[k + s_idx])
            n.append(n_left + peak_idx[k])
            m.append(n_right + valley_idx[k + s_idx])

            for b in range(1 + n_bins // 2):
                bins[n[b] : n[b + 1]] = b
            for b in range(n_bins // 2):
                bins[m[b] : m[b + 1]] = b + (n_bins // 2)
            bins[m[-2] : m[-1]] = 0

    # Diagnostic plots
    plt.rcParams["figure.figsize"] = (16, 9)
    plt.style.use("dark_background")
    colors = plt.cm.seismic(np.linspace(0, 1, n_bins + 1))
    plt.gca().set_prop_cycle(color=colors)
    for b in range(n_bins):
        resp_array = np.ma.masked_where(bins != b, resp)
        plt.plot(np.arange(resp.size), resp_array, label=f"Bin {b}")
    resp_array = np.ma.masked_where(bins != n_bins, resp)
    plt.plot(np.arange(resp.size), resp_array, label="Excluded", color="g")
    plt.legend()
    plt.title("Resp Binning")
    plt.xlabel("Radial Spoke")
    plt.ylabel("Amplitude")
    plt.savefig(os.path.join(diagnostics_dir, "resp_binned_external_full.png"), facecolor="black", edgecolor="none")
    plt.close()

    plt.rcParams["figure.figsize"] = (16, 9)
    plt.style.use("dark_background")
    colors = plt.cm.seismic(np.linspace(0, 1, n_bins + 1))
    plt.gca().set_prop_cycle(color=colors)
    resp_sub = resp[1000:5000]
    bins_sub = bins[1000:5000]
    for b in range(n_bins + 1):
        resp_array = np.ma.masked_where(bins_sub != b, resp_sub)
        plt.plot(np.arange(resp_sub.size), resp_array, label=f"Bin {b}")
    resp_array = np.ma.masked_where(bins_sub != n_bins, resp_sub)
    plt.plot(np.arange(resp_sub.size), resp_array, label="Excluded", color="g")
    plt.legend()
    plt.title("Resp Binning")
    plt.xlabel("Radial Spoke")
    plt.ylabel("Amplitude")
    plt.savefig(os.path.join(diagnostics_dir, "resp_binned_external_smol.png"), facecolor="black", edgecolor="none")
    plt.close()

    kspB = []
    coordB = []
    dcfB = []
    for b in range(n_bins):
        idx = bins == b
        kspB.append(ksp[:, idx])
        coordB.append(coord[idx])
        dcfB.append(dcf[idx])
    return kspB, coordB, dcfB


def bin_motion_states(
    ksp_in,
    coord_in,
    dcf_in,
    resp_in,
    n,
    diagnostics_dir,
    filter_bulk=False,
    filter_extremes=False,
):
    """Bin k-space data into equal-amplitude respiratory motion bins.

    Args:
        ksp_in (ndarray): K-space data.
        coord_in (ndarray): Coordinates.
        dcf_in (ndarray): Density compensation.
        resp_in (ndarray): Respiratory waveform.
        n (int): Number of bins.
        diagnostics_dir (str): Directory for diagnostic plots.
        filter_bulk (bool): Filter bulk motion.
        filter_extremes (bool): Remove extreme percentiles before binning.

    Returns:
        tuple: (kspB, coordB, dcfB) lists of length n.
    """
    ksp = copy.deepcopy(ksp_in)
    coord = copy.deepcopy(coord_in)
    dcf = copy.deepcopy(dcf_in)
    resp = copy.deepcopy(resp_in)

    if filter_extremes:
        margin = 5
        bin_edges = np.linspace(np.percentile(resp, margin), np.percentile(resp, 100 - margin), n + 1)
    else:
        bin_edges = np.linspace(resp.min(), resp.max(), n + 1)

    kspB = []
    coordB = []
    dcfB = []
    for b in range(n):
        idx = (resp >= bin_edges[b]) & (resp < bin_edges[b + 1])
        kspB.append(ksp[:, idx])
        coordB.append(coord[idx])
        dcfB.append(dcf[idx])

    de_colores = plt.cm.get_cmap("tab20", n)
    plt.plot(resp, "k")
    for b in range(n):
        plt.plot(np.arange(resp.size), np.ones((resp.size,)) * bin_edges[b], color=de_colores(b))
    plt.plot(np.arange(resp.size), np.ones((resp.size,)) * bin_edges[b + 1], color=de_colores(b + 1))
    plt.savefig(os.path.join(diagnostics_dir, "resp_binned.png"))
    plt.close()
    return kspB, coordB, dcfB
