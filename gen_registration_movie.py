import numpy as np
import nibabel as nib
import glob
import os

files = glob.glob("Stage1*")
files.sort()
img0 = []
img1 = []
img2 = []
img3 = []
for file in files:
    if "level1" in file:
        img0.append(nib.load(file).get_fdata())
    if "level2" in file:
        img1.append(nib.load(file).get_fdata())
    if "level3" in file:
        img2.append(nib.load(file).get_fdata())
    if "level4" in file:
        img3.append(nib.load(file).get_fdata())

img = np.stack(img0)
np.save("reg_iters_l1.npy", img)
files_1 = glob.glob("*level1*")
[os.remove(file) for file in files_1]

img = np.stack(img1)
np.save("reg_iters_l2.npy", img)
files_2= glob.glob("*level2*")
[os.remove(file) for file in files_2]

img = np.stack(img2)
np.save("reg_iters_l3.npy", img)
files_3 = glob.glob("*level3*")
[os.remove(file) for file in files_3]

img = np.stack(img3)
np.save("reg_iters_l4.npy", img)
files_4 = glob.glob("*level4*")
[os.remove(file) for file in files_4]

# [os.remove(file) for file in files]
