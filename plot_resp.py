import numpy as np

import matplotlib.pyplot as plt

# from scipy.ndimage.filters import uniform_filter1d
import pandas as pd


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


resp = np.load("/home/ltorres/data/recon/nicu/P24/diagnostics/resp.npy")
resp = pd.Series(resp[:])
window = 10  # seconds
N = int(window / 0.00502)
resp_mean = resp.rolling(N).mean()
resp_std = resp.rolling(N).std()
resp_std_mad = np.nanmedian(np.abs(resp_std - np.nanmedian(resp_std))) * 1.4826
# thresh = resp_std_mad * 7
thresh = np.nanstd(resp_std) * 4
# thresh = 0.05
print(thresh)

idx = np.squeeze(np.array(np.where(resp_std > thresh)))
cons = consecutive(idx)
print(len(cons))
idx = []
for ii in range(len(cons)):
    # Add window length N to excluded indices
    idx_t = np.arange(cons[ii].min() - N, cons[ii].max() + N)
    idx = np.concatenate((idx, idx_t))
# Clean idx
idx = np.unique(idx[(idx >= 0) & (idx < resp.size)])
print(idx.size)
resp_keep = resp_mean.copy()
resp_keep[idx] = np.nan
x_array = np.arange(resp.size)
fig, ax = plt.subplots(4, constrained_layout=True)
ax[0].plot(x_array, resp, "k--")
ax[0].plot(idx, resp.max().repeat(idx.size), "ro", linestyle="None")
ax[1].plot(x_array, resp_mean, "k")
ax[1].plot(idx, resp_mean.max().repeat(idx.size), "ro", linestyle="None")
ax[2].plot(x_array, resp_std, "b")
ax[2].plot(x_array, np.repeat(thresh, x_array.size), "r")
ax[3].plot(x_array, resp_keep, "b")
fig.suptitle("Respiratory waveform diagnostics")

ax[0].set_title("Original Waveform")
ax[1].set_title("{}s Rolling Mean".format(window))
ax[2].set_title("{}s Rolling Std".format(window))
ax[3].set_title("Filtered Resp. Waveform")
ax[3].set_xlabel("Projection Number")
cmap = plt.cm.get_cmap("hsv", 8)
ax[0].fill_between(
    x_array, np.ones_like(x_array) * 0.5, np.ones_like(x_array) * 1.0, alpha=0.2, facecolor=cmap(2)
)
plt.savefig("respDiagnostics.png")
plt.close()
ax[0].xaxis.set_visible(False)
ax[1].xaxis.set_visible(False)
ax[2].xaxis.set_visible(False)
plt.show()
# plt.savefig(diagnosticsDir + "/respWaveformFull.png")
# plt.close()
