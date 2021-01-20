import argparse
import numpy as np
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import copy

# import matplotlib.pyplot as plt


def get_consecutive(data, stepsize=1):
    # Splits data into consecutive portions
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def clean_resp(resp, diagnostics_dir, N=2000):
    x_array = np.arange(resp.size)
    resp = pd.Series(resp)
    resp_mean = resp.rolling(N).mean()
    resp_std = resp.rolling(N).std()
    # Use global stdev of rolling stdev to define threshold
    zscore = 2.5
    thresh = np.nanmedian(resp_std) + np.nanstd(resp_std) * zscore  # 4 standard deviations should remove the outliers.
    # print(thresh)
    # Find outilers
    idx = np.squeeze(np.array(np.where(resp_std > thresh)))
    # print(idx.shape)
    # Now add window size to edge indices to account for delays from edge effects of rolling statistics.
    cons = get_consecutive(idx)
    idx = []
    for ii in range(len(cons)):
        # Add window length N to excluded indices
        idx_t = np.arange(cons[ii].min() - N, cons[ii].max() + N)
        idx = np.concatenate((idx, idx_t))
    # Now clean idx (ensure unique indices and within bounds of projection count)
    idx = np.unique(idx[(idx >= 0) & (idx < resp.size)])
    idx_keep = np.setdiff1d(np.arange(resp.size), idx)

    resp_keep = resp_mean.copy()
    resp_keep[idx] = np.nan
    # Save a diagnostics plot
    fig, ax = plt.subplots(4, constrained_layout=True)
    ax[0].plot(x_array, resp, "k--")
    ax[0].plot(idx, resp.max().repeat(idx.size), "ro", linestyle="None")
    ax[1].plot(x_array, resp_mean, "k")
    ax[1].plot(idx, resp_mean.max().repeat(idx.size), "ro", linestyle="None")
    ax[2].plot(x_array, resp_std, "b", label="Rolling std dev")
    ax[2].plot(x_array, np.repeat(thresh, x_array.size), "r", label="{}*sigma".format(zscore))
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


def bin_motion_states(ksp_in, coord_in, dcf_in, resp_in, n, diagnostics_dir, filter_bulk=False, filter_extremes=False):
    """Bin kspace, coordinates, and dcf by respiratory motion.

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

    if filter_bulk:
        n_original = resp.size
        resp, idx_keep = clean_resp(resp, diagnostics_dir)
        ksp = ksp[:, idx_keep]
        coord = coord[idx_keep]
        dcf = dcf[idx_keep]
        logging.info("Number of projections excluded: {}".format(n_original - resp.size))

    # nSpokes = dcf.shape[0]
    # count = 0
    # while nSpokes % n is not 0:
    #     count += 1
    #     nSpokes -= 1
    # logging.info("Count is {}".format(count))
    # logging.info("nSpokes is {}".format(nSpokes))

    # ksp = ksp[:, :nSpokes]
    # coord = coord[:nSpokes]
    # dcf = dcf[:nSpokes]
    # resp = resp[:nSpokes]
    # nSpokesB = int(nSpokes // n)
    # respOrder = np.argsort(resp)
    # plt.plot(resp[respOrder])

    if filter_extremes:
        margin = 1  # Remove 1% at both ends
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

    parser = argparse.ArgumentParser(description="Bin data based on Respiratory Motion.")

    parser.add_argument("ksp_file", type=str, help="k-space file.")
    parser.add_argument("coord_file", type=str, help="trajectory file.")
    parser.add_argument("dcf_file", type=str, help="dcf file.")
    parser.add_argument("resp_file", type=str, help="Output respiratory signal file.")
    parser.add_argument("ksp_ofile", type=str, help="binned k-space file.")
    parser.add_argument("coord_ofile", type=str, help="binned trajectory file.")
    parser.add_argument("dcf_ofile", type=str, help="binned dcf file.")
    parser.add_argument("--n_bins", type=int, default=6, help="Number of bins")

    args = parser.parse_args()

    ksp = np.load(args.ksp_file)
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)
    resp = np.load(args.resp_file)

    kspB, coordB, dcfB = bin_motion_states(ksp, coord, dcf, resp, args.n_bins)
    print("kspace shape: {}".format(kspB.shape))
    print("trajectory shape: {}".format(coordB.shape))
    print("dcf shape: {}".format(dcfB.shape))

    if os.path.isfile(args.ksp_ofile):
        os.remove(args.ksp_ofile)
        os.remove(args.coord_ofile)
        os.remove(args.dcf_ofile)
    np.save(args.ksp_ofile, kspB)
    np.save(args.coord_ofile, coordB)
    np.save(args.dcf_ofile, dcfB)
