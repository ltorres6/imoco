from scipy.stats.mstats import winsorize
import numpy as np
import sigpy.plot as plt

mrimg = np.abs(
    np.load(
        "/home/ltorres/data/recon/ipf/103-002/mri/20151222/pre_contrast/MotionResolved_test_syn/MotionResolvedLowRes.npy"
    )
)

n_phases = mrimg.shape[0]
plt.ImagePlot(mrimg)

for p in range(n_phases):
    print(p)
    mrimg[p, ...] = winsorize(mrimg[p, ...], limits=[0.1, 0.00])

plt.ImagePlot(mrimg)
