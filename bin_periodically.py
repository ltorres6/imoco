import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


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
