import sys
import os
import time
from skimage.exposure import match_histograms

import matplotlib.pyplot as plt
import torch as th
import numpy as np
from normalize import normalize
import nibabel as nib

import sigpy.plot as plt2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import airlab as al


# start = time.time()

# set the used data type
# dtype = th.float32
# set the device for the computaion to CPU
# device = th.device("cpu")

# In order to use a GPU uncomment the following line. The number is the device index of the used GPU
# Here, the GPU with the index 0 is used.
# device = th.device("cuda:0")
M_fields = nib.load(
    "/home/ltorres/data/recon/ipf/103-002/mri/20151222/pre_contrast/IterativeMoCo_test_syn/diagnostics/M_mr.nii.gz"
).get_fdata()
M_fields = np.flip(M_fields, (0, 1, 2))
M_fields = np.transpose(M_fields, (2, 1, 0, 3, 4))
M_fields = np.moveaxis(M_fields, -1, 0)

# M_fields = np.sqrt(np.sum(np.power(M_fields, 2), -1)).squeeze()
plt2.ImagePlot(M_fields)
