import numpy as np
import sigpy.plot as plt

ksp = np.load("/home/ltorres/data/rawdata/nicu/P24/ksp.npy")
coord = np.load("/home/ltorres/data/rawdata/nicu/P24/coord.npy")
dcf = np.load("/home/ltorres/data/rawdata/nicu/P24/dcf.npy")
print(ksp.shape)
print(coord.shape)
# plt.ScatterPlot(
#     coord[20:50, :, :2],
#     np.abs(ksp[:, 20:50, :]),
# )
n_spokes = 1000
plt.ScatterPlot(coord[:n_spokes, :, :2], dcf[:n_spokes, :])
