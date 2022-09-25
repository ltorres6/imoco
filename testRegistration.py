import os
from moco import moco
import nibabel as nib
import numpy as np

# Test Registration Technique
inDir = "/export/home/ltorres/data/testMoco/"
mrimgPath = os.path.join(inDir, "MotionResolved.nii.gz")
# MIOut = os.path.join(inDir, "MoCo_MI.nii.gz")
Demons2Out = os.path.join(inDir, "MoCo_Demons2.nii.gz")

# 8) Full Res MoCo Expiratory
imgMoco = moco(mrimgPath, inDir, nRef=-1)
imgMoco = nib.Nifti1Image(imgMoco, np.eye(4))
nib.save(imgMoco, Demons2Out)