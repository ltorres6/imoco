import numpy as np

# import sigpy.plot as plt
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm

sub_dir = "/home/ltorres/data/rawdata/nicu/"
ignored = ["Original Subjects", "P006_Exam1"]
subjects = [x for x in os.listdir(sub_dir) if x not in ignored]
subjects.sort()
for subject in subjects:
    ksp = np.load(f"/home/ltorres/data/rawdata/nicu/{subject}/ksp.npy")
    coord = np.load(f"/home/ltorres/data/rawdata/nicu/{subject}/coord.npy")
    dcf = np.load(f"/home/ltorres/data/rawdata/nicu/{subject}/dcf.npy")
    print(ksp.shape)
    n_spokes = 4000
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.squeeze(np.abs(ksp[0, :n_spokes, :])), norm=LogNorm())
    ax[1].imshow(np.squeeze(np.abs(ksp[0, -n_spokes:, :])), norm=LogNorm())
    ax[0].set_title("Beginning of data")
    ax[1].set_title("End of data")
    plt.show()
