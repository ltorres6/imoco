import numpy as np
import nibabel as nib
import glob
import os

files = glob.glob("Stage1*")
files.sort()
[os.remove(file) for file in files]
