from ute_recon_tools.convert_ute import get_cov, whiten
import numpy as np
import os
import sigpy.plot as plt


input_path = "/home/ltorres/data/new_oe_data/"
noise = np.load(os.path.join(input_path, "noise.npy"))
print(noise.shape)
cov = get_cov(noise)
L = np.linalg.cholesky(cov)
L_inv = np.linalg.inv(L)
plt.ImagePlot(L, colormap="jet")
plt.ImagePlot(L_inv, colormap="jet")

cov1 = L_inv * cov
cov2 = np.linalg.solve(L, cov)

plt.ImagePlot(np.stack([cov1, cov2], axis=0), colormap="jet")
