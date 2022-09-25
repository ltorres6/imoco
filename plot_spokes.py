import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

coord = np.load("/home/ltorres/data/rawdata/ipf/103-001/mri/20151218/pre_contrast/coord.npy")

fig = plt.figure()
ax = plt.axes(projection="3d")
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
ax.set_axis_off()

for spoke in range(12):
    ax.plot3D(coord[spoke, :, 0], coord[spoke, :, 1], coord[spoke, :, 2], linewidth=2)
# plt.show()
plt.savefig('kooshball_8th.png', transparent=True)