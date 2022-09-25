import scipy.io as sio
import matplotlib.pyplot as plt


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
    bins_t = bins[key].squeeze()
    bins = bins_t.copy()
    bins[(bins_t >= 0.0) & (bins_t < 0.5)] = 0
    bins[(bins_t >= 7.5) & (bins_t < 8.0)] = 0
    bins[(bins_t >= 0.5) & (bins_t < 1.5)] = 1
    bins[(bins_t >= 1.5) & (bins_t < 2.5)] = 2
    bins[(bins_t >= 2.5) & (bins_t < 3.5)] = 3
    bins[(bins_t >= 3.5) & (bins_t < 4.5)] = 4
    bins[(bins_t >= 4.5) & (bins_t < 5.5)] = 5
    bins[(bins_t >= 5.5) & (bins_t < 6.5)] = 6
    bins[(bins_t >= 6.5) & (bins_t < 7.5)] = 7
    bins[(bins_t == 8.0)] = 8
    return bins
