import scipy.io as sio


def load_external_binning(in_path):

    # Load External Bins
    try:
        bins = sio.loadmat(in_path + "SoftGating_8Bins_Mag.mat")
        key = "SoftGating_Mag"
    except FileNotFoundError:
        try:
            bins = sio.loadmat(in_path + "SoftGating_8Bins_Phs.mat")
            key = "SoftGating_Phs"
        except FileNotFoundError:
            pass
    bins = bins[key].squeeze()
    bins[(bins >= 0.0) & (bins < 0.5) | (bins >= 7.5) & (bins < 8.0)] = 0
    bins[(bins >= 0.5) & (bins < 1.5)] = 1
    bins[(bins >= 1.5) & (bins < 2.5)] = 2
    bins[(bins >= 2.5) & (bins < 3.5)] = 3
    bins[(bins >= 3.5) & (bins < 4.5)] = 4
    bins[(bins >= 4.5) & (bins < 5.5)] = 5
    bins[(bins >= 5.5) & (bins < 6.5)] = 6
    bins[(bins >= 6.5) & (bins < 7.5)] = 7
    return bins
