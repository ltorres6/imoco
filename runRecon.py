import os

# cores = "1"
# os.environ["OMP_NUM_THREADS"] = cores  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = cores  # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = cores  # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = cores  # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = cores  # export NUMEXPR_NUM_THREADS=6
import nibabel as nib
from convertUTE import convertUTE
from autofov import autofov
from estimate_resp import estimate_resp
from estimate_respSavitzkyGolay import estimate_respSavitzkyGolay
from binMotionStates import binMotionStates
from xdgrasp import xdgrasp
import logging
from pathlib import Path
import cfl
import sigpy as sp
import numpy as np
import time
import sigpy.plot as plt

logging.basicConfig(level=logging.INFO)
# loop_moco.py
rootDir = "/home/ltorres/data/nicu/"
outDir = "/home/ltorres/data/nicu/"
# set device
device = 0
nBins = 8
nCoils = 1
lambda_tv = 0.05
xp = sp.Device(device).xp
try:
    timei = time.time()
    # print(visit)
    subjectDir = rootDir
    subjectOutDir = os.path.join(outDir, "MotionResolved")
    print(subjectDir)

    fileList = os.listdir(subjectDir)
    if "MRI_Raw.h5" in fileList:
        "File Exists, Begin!"
    else:
        pass
    # Set up data paths
    h5Path = subjectDir + "/MRI_Raw.h5"
    # mpsPath = subjectDir + "/mps.npy"
    # respPath = subjectDir + "/resp.npy"
    mrimgPath = subjectOutDir + "/MotionResolved.nii.gz"
    mrimgPathnpy = subjectOutDir + "/MotionResolved.npy"
    # kspBPath = subjectOutDir + "/kspB.npy"
    # coordBPath = subjectOutDir + "/coordB.npy"
    # dcfBPath = subjectOutDir + "/dcfB.npy"
    diagnosticsDir = subjectOutDir + "/diagnostics"
    # Create outpath if doesn't exist.
    Path(subjectOutDir).mkdir(parents=True, exist_ok=True)
    Path(diagnosticsDir).mkdir(parents=True, exist_ok=True)

    # 1) Convert MRI_Raw.h5 to cfl and read resp waveform.
    logging.info("Running File Conversion...")
    ksp, coord, dcf, resp = convertUTE(h5Path, nCoils)
    print(ksp.shape)
    # 1.5)
    tr = 0.0052  # nicu
    # resp = estimate_resp(ksp[:, :, 0], tr)
    resp = estimate_respSavitzkyGolay(ksp[:, :, 0], tr, 0.5, 3)
    plt.LinePlot(resp, mode="r")
    # 2) AutoFOV to reduce matrix size
    logging.info("Running AutoFOV...")
    thresh = 0.05
    coord = autofov(ksp, coord, dcf, diagnosticsDir, num_ro=150, thresh=thresh, device=device)

    # # 3) Bin Motion States
    logging.info("Running BinMotionStates...")
    ksp, coord, dcf = binMotionStates(ksp, coord, dcf, resp, nBins)
    del resp

    # # 4) xdgrasp recon
    logging.info("Running Reconstruction...")
    img = xdgrasp(ksp, coord, dcf, res_scale=1.0, lambda_tv=lambda_tv, device=device, tv_device=0)
    del ksp, coord, dcf
    # cfl.write_cfl(mrimgPath, img)
    img = np.moveaxis(np.abs(img), 0, -1)
    # np.save(mrimgPathnpy,img)
    img = nib.Nifti1Image(img.astype("f"), np.eye(4))
    nib.save(img, mrimgPath)
    timeF = (timei - time.time()) / 60
    logging.info("Finshed Subject in {} minutes".format(timeF))
    # destroy data every loop
    # del ksp, coord, dcf
except KeyboardInterrupt:
    print("interrupted!")
