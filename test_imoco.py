import os
cores = "1"
os.environ["OMP_NUM_THREADS"] = cores  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = cores  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = cores  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = cores  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = cores  # export NUMEXPR_NUM_THREADS=6
import nibabel as nib
from convertUTE import convertUTE
from autofov import autofov
from estimate_resp import estimate_resp
from binMotionStates import binMotionStates
from xdgrasp import xdgrasp
from imoco import imoco
from moco import moco
import logging
from pathlib import Path
import sigpy as sp
import numpy as np
import time
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
# subject = "103-005"
# visit = "20160114"
# subject = "103-040"
# visit = "20181005"
# subject = "103-041"
# visit = "20181011"
subject = "103-042"
visit = "20190221"
# rootDir = "/scratch_cnxx/ltorres/"+subject+"/mri/"+visit+"/"
# rootDir = "/export/home/ltorres/data/scratch_tmp/ltorres/"+subject+"/mri/"+visit+"/"
# outDir = "/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf/" + subject + "/mri/" + visit + "/"
rootDir = "/home/ltorres/data/scratch_tmp/ltorres/"+subject+"/mri/"+visit+"/"
outDir = "/home/ltorres/data/scratch_tmp/ltorres/"+subject+"/mri/"+visit+"/"
# rootDir = "/home/ltorres/data/nicu"
# outDir = "/home/ltorres/data/nicu"
# set device
device = 0
nBins = 6
nCoils = 8
doLowRes = 1
register = 1
dc_signal = 1
nRef = -1  # Reference Frame
xp = sp.Device(device).xp
try:
    timei = time.time()
    # print(visit)
    motionResolvedDir = os.path.join(outDir, "PreContrastMotionResolved")
    iterativeMocoDir = os.path.join(outDir, "PreContrastIterativeMoCo")
    mocoDir = os.path.join(outDir, "PreContrastMoCo")
    print(rootDir)
    # Set up data paths
    h5Path = os.path.join(rootDir, "MRI_Raw.h5")
    diagnosticsDir = os.path.join(motionResolvedDir, "diagnostics")
    mrimgPath = os.path.join(motionResolvedDir, "MotionResolved.nii.gz")
    mrimgLPath = os.path.join(motionResolvedDir, "MotionResolvedLowRes.nii.gz")
    imgMocoPath = os.path.join(mocoDir, "MoCo.nii.gz")
    imgPath = os.path.join(iterativeMocoDir, "iMoCo.nii.gz")

    # Create outpath if doesn't exist.
    Path(motionResolvedDir).mkdir(parents=True, exist_ok=True)
    Path(iterativeMocoDir).mkdir(parents=True, exist_ok=True)
    Path(diagnosticsDir).mkdir(parents=True, exist_ok=True)
    Path(mocoDir).mkdir(parents=True, exist_ok=True)

    # 7) Full Res MoCo
    mrimg = nib.load(mrimgPath)
    mrimg = mrimg.get_fdata()
    mrimg = np.moveaxis(np.abs(mrimg), -1, 0)
    imgMoco = moco(mrimg, diagnosticsDir, nRef=nRef)
    del mrimg  # not needed anymore
    imgMoco = nib.Nifti1Image(imgMoco, np.eye(4))
    nib.save(imgMoco, imgMocoPath)
    timeF = (time.time() - timei) / 60
    logging.info("Finshed Subject in {} minutes".format(timeF))
    # destroy data every loop
except KeyboardInterrupt:
    print("interrupted!")
