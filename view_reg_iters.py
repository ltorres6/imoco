import numpy as np
import sigpy.plot as plt

img = np.load("reg_iters_l1.npy")
plt.ImagePlot(img)
del img

img = np.load("reg_iters_l2.npy")
plt.ImagePlot(img)
del img

img = np.load("reg_iters_l3.npy")
plt.ImagePlot(img)
del img

img = np.load("reg_iters_l4.npy")
plt.ImagePlot(img)
del img
