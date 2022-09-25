import argparse
import numpy as np
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import copy


def get_consecutive(data, stepsize=1):
    # Splits data into consecutive portions
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def clean_resp(resp, diagnostics_dir, tr, T=10, z_score = 2.0):
    x_array = np.arange(resp.size)
    N = int(T // tr)
    N_5 = int(5 // tr)
    N_lag = int(T//(6*tr))
    resp = pd.Series(resp)
    resp_mean = resp.rolling(N).mean()
    resp_std = resp.rolling(N).std()
    # Use global stdev of rolling stdev to define threshold
    thresh = np.nanmedian(resp_std) + np.nanstd(resp_std) * z_score  # 4 standard deviations should remove the outliers.
    # print(thresh)
    # Find outilers
    idx = np.squeeze(np.array(np.where(resp_std > thresh))) - N_lag
    # print(idx.shape)
    # Now add window size to edge indices to account for delays from edge effects of rolling statistics.
    cons = get_consecutive(idx)
    idx = []
    for ii in range(len(cons)):
        # Add window length N//4 to excluded indices
        idx_t = np.arange(cons[ii].min() - N_5, cons[ii].max() + N_5)
        # idx_t = np.arange(cons[ii].min(), cons[ii].max())
        idx = np.concatenate((idx, idx_t))
    # Now clean idx (ensure unique indices and within bounds of projection count)
    idx = np.unique(idx[(idx >= 0) & (idx < resp.size)])
    idx_keep = np.setdiff1d(np.arange(resp.size), idx)

    resp_keep = resp.copy()
    resp_keep[idx] = np.nan
    # Save a diagnostics plot
    fig, ax = plt.subplots(4, constrained_layout=True)
    ax[0].plot(x_array, resp, "k")
    ax[0].plot(idx, resp.max().repeat(idx.size), "ro", linestyle="None")
    ax[1].plot(x_array, resp_mean, "k")
    ax[1].plot(idx, resp_mean.max().repeat(idx.size), "ro", linestyle="None")
    ax[2].plot(x_array, resp_std, "b", label="Rolling std dev")
    ax[2].plot(x_array, np.repeat(thresh, x_array.size), "r", label="{}*sigma".format(z_score))
    ax[3].plot(x_array, resp_keep, "b")
    fig.suptitle("Respiratory waveform diagnostics")

    ax[0].set_title("Original Waveform")
    ax[1].set_title("{} projection Rolling Mean".format(N))
    ax[2].set_title("{} projection Rolling Std".format(N))
    ax[3].set_title("Filtered Resp. Waveform")
    ax[3].set_xlabel("Projection Number")

    ax[0].xaxis.set_visible(False)
    ax[1].xaxis.set_visible(False)
    ax[2].xaxis.set_visible(False)
    ax[2].legend()
    plt.savefig(diagnostics_dir + "resp_diagnostics.png")
    plt.close()
    return np.array(resp[idx_keep]), idx_keep


def filter_bulk(ksp_in, coord_in, dcf_in, resp_in, tr, diagnostics_dir):
    """Filter for irregular respiratory patterns

    Args:
        ksp (array): kspace
        coord (array): coordinates.
        dcf (array): density weights.
        resp (array): waveform.

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
    n_original = resp.size
    resp, idx_keep = clean_resp(resp, diagnostics_dir, tr, T=30, z_score=1.0)
    ksp = ksp[:, idx_keep]
    coord = coord[idx_keep]
    dcf = dcf[idx_keep]
    logging.info("Number of projections excluded: {}".format(n_original - resp.size))
    return ksp, coord, dcf, resp
