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
from estimate_respSavitzkyGolay import estimate_respSavitzkyGolay
from binMotionStates import binMotionStates
from xdgrasp import xdgrasp
from imoco import imoco
import logging
from pathlib import Path
import cfl
import sigpy as sp
import numpy as np
import time
import sigpy.plot as plt

logging.basicConfig(level=logging.INFO)
# loop_moco.py
subject = "103-040"
visit = "20181005"
# subject = "103-041"
# visit = "20181011"
# subject = "103-042"
# visit = "20190221"
# rootDir = "/scratch_cnxx/ltorres/"+subject+"/mri/"+visit+"/"
rootDir = "/export/home/ltorres/data/scratch_tmp/ltorres/"+subject+"/mri/"+visit+"/"
outDir = "/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf/"+subject+"/mri/"+visit+"/"
# set device
device = 0
nBins = 6
nCoils = 8
xdgrasp_lambda = 0.05
imoco_lambda = 0.025
xp = sp.Device(device).xp
try:
    timei = time.time()
    # print(visit)
    motionResolvedDir = os.path.join(outDir, "PreContrastMotionResolved")
    print(rootDir)

    fileList = os.listdir(rootDir)
    if "MRI_Raw.h5" in fileList:
        "File Exists, Begin!"
    else:
        pass
    # Set up data paths
    h5Path = os.path.join(rootDir, "MRI_Raw.h5")
    diagnosticsDir = os.path.join(motionResolvedDir, "diagnostics")
    mrimgPath = os.path.join(motionResolvedDir, "MotionResolved.nii.gz")
    mrimgLPath = os.path.join(motionResolvedDir, "MotionResolvedLowRes.nii.gz")
    mrimgPathnpy = os.path.join(motionResolvedDir, "MotionResolved.npy")
    imgPath = os.path.join(outDir, "PreContrastIterativeMoCo", "iMoCo.nii.gz")
    respPath = os.path.join(diagnosticsDir, "resp.npy")

    # Create outpath if doesn't exist.
    Path(motionResolvedDir).mkdir(parents=True, exist_ok=True)
    Path(diagnosticsDir).mkdir(parents=True, exist_ok=True)

    # 1) Convert MRI_Raw.h5 to cfl and read resp waveform.
    logging.info("Running File Conversion...")
    ksp, coord, dcf, resp = convertUTE(h5Path, nCoils)

    # 1.5)
    # tr = 0.0052  # nicu
    tr = 0.0034  # ipf
    # resp = estimate_resp(ksp[:, :, 0], tr)
    resp = estimate_respSavitzkyGolay(ksp[:, :, 0], tr, 0.05, 3)
    # plt.LinePlot(resp, mode="r")
    # 2) AutoFOV to reduce matrix size
    logging.info("Running AutoFOV...")
    thresh = 0.05
    coord = autofov(ksp, coord, dcf, diagnosticsDir, num_ro=150, thresh=thresh, device=device)

    # # 3) Bin Motion States
    logging.info("Running BinMotionStates...")
    ksp, coord, dcf = binMotionStates(ksp, coord, dcf, resp, nBins)
    np.save(respPath, resp)
    del resp

    # # 4) xdgrasp recon
    logging.info("Running Low Res XDGrasp Reconstruction...")
    # mrimg = xdgrasp(ksp, coord, dcf, res_scale=0.75, lambda_tv=xdgrasp_lambda, device=device, tv_device=-1)
    # mrimgL = sp.resize(mrimg, (nBins, 192, 192, 192))
    # mrimgL = np.moveaxis(np.abs(mrimgL), 0, -1)
    # mrimgL = nib.Nifti1Image(mrimgL, np.eye(4))
    # nib.save(mrimgL, mrimgLPath)
    # del mrimgL
    # # mrimg = nib.load(mrimgLPath).get_fdata()
    # # mrimg = np.moveaxis(mrimg, -1, 0)
    # logging.info("Running iMoCo Reconstruction...")
    # img = imoco(ksp, coord, dcf, mrimg, diagnosticsDir, res_scale=1.0, lambda_tv=imoco_lambda,
    #             inner_iter=15, outer_iter=20, device=device, nRef=-1, reg_flag=1)

    logging.info("Running Full Res XDGrasp Reconstruction...")
    mrimg = xdgrasp(ksp, coord, dcf, res_scale=1.0, lambda_tv=xdgrasp_lambda, device=device, tv_device=-1)
    mrimg = sp.resize(mrimg, (nBins, 256, 256, 256))
    mrimg = np.moveaxis(np.abs(mrimg), 0, -1)
    mrimg = nib.Nifti1Image(mrimg, np.eye(4))
    nib.save(mrimg, mrimgPath)
    del ksp, coord, dcf, mrimg
    # np.save(mrimgPathnpy, img)
    # img = np.load(mrimgPathnpy)
    img = sp.resize(np.abs(img), (256, 256, 256))
    img = nib.Nifti1Image(img, np.eye(4))
    nib.save(img, imgPath)
    timeF = (time.time() - timei) / 60
    logging.info("Finshed Subject in {} minutes".format(timeF))
    # destroy data every loop
    del img
except KeyboardInterrupt:
    print("interrupted!")
