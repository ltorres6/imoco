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
subject = "103-016"
visit = "20161121"
# subject = "103-040"
# visit = "20181005"
# subject = "103-041"
# visit = "20181011"
# subject = "103-042"
# visit = "20190221"
# rootDir = "/scratch_cnxx/ltorres/"+subject+"/mri/"+visit+"/"
# rootDir = "/export/home/ltorres/data/scratch_tmp/ltorres/"+subject+"/mri/"+visit+"/"
# outDir = "/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf/" + subject + "/mri/" + visit + "/"
rootDir = "/home/ltorres/data/ipf/ltorres/"+subject+"/mri/"+visit+"/"
outDir = "/home/ltorres/data/ipf/ltorres/"+subject+"/mri/"+visit+"/"
# rootDir = "/home/ltorres/data/nicu"
# outDir = "/home/ltorres/data/nicu"
# set device
device = 0
nBins = 6
nCoils = 8
ignoreExisting = False
dc_signal = 1
nRef = -1  # Reference Frame (last ie expiratory)
xp = sp.Device(device).xp
try:
    timei = time.time()
    # print(visit)
    motionResolvedDir = os.path.join(outDir, "PreContrastMotionResolved")
    iterativeMocoDir = os.path.join(outDir, "PreContrastIterativeMoCo")
    iterativeMocoInspDir = os.path.join(outDir, "PreContrastIterativeMoCoInsp")
    mocoDir = os.path.join(outDir, "PreContrastMoCo")
    mocoInspDir = os.path.join(outDir, "PreContrastMoCoInsp")
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
    imgMocoPath = os.path.join(mocoDir, "MoCo.nii.gz")
    imgPath = os.path.join(iterativeMocoDir, "iMoCo.nii.gz")
    respPath = os.path.join(diagnosticsDir, "resp.npy")
    imgInspPath = os.path.join(iterativeMocoInspDir, "iMoCo.nii.gz")
    imgMocoInspPath = os.path.join(mocoInspDir, "MoCo.nii.gz")

    # Create outpath if doesn't exist.
    Path(motionResolvedDir).mkdir(parents=True, exist_ok=True)
    Path(iterativeMocoDir).mkdir(parents=True, exist_ok=True)
    Path(diagnosticsDir).mkdir(parents=True, exist_ok=True)
    Path(mocoDir).mkdir(parents=True, exist_ok=True)
    Path(iterativeMocoInspDir).mkdir(parents=True, exist_ok=True)
    Path(mocoInspDir).mkdir(parents=True, exist_ok=True)

    # 1) Convert MRI_Raw.h5 to cfl and read resp waveform.
    logging.info("Running File Conversion...")
    ksp, coord, dcf, resp = convertUTE(h5Path, nCoils)
    dcf **= 0.5

    logging.info("Kspace Shape: {}...".format(ksp.shape))
    logging.info("trajectory Shape: {}...".format(coord.shape))
    logging.info("DCF Shape: {}....".format(dcf.shape))

    # 1.5)
    if ksp.shape[0] == 1:
        tr = 0.0052  # nicu
        xdgrasp_lambda = 0.05
        imoco_lambda = 0.02
        tv_device = 0
    else:
        tr = 0.0034  # ipf
        tv_device = -1
        xdgrasp_lambda = 0.05
        imoco_lambda = 0.02
    if dc_signal == 1:
        logging.info("Estimating Resp Waveform from DC signal...")
        logging.info("Using TR: {} seconds".format(tr))
        resp = estimate_resp(ksp[:, :, 0], tr)
        # resp = estimate_respSavitzkyGolay(ksp[:, :, 0], tr, 2.0, 2)
    plt.plot(resp)
    plt.title("Entire Waveform")
    plt.savefig(diagnosticsDir + '/respWaveformFull.png')
    plt.close()

    plt.plot(resp[int(60 / tr):int(60 / tr) + int(120 / tr)])
    plt.title("120 seconds of breathing")
    plt.savefig(diagnosticsDir + '/respWaveform120.png')
    plt.close()
    np.save(respPath, resp)

    # 2) AutoFOV to reduce matrix size
    logging.info("Running AutoFOV...")
    thresh = 0.05
    coord = autofov(ksp, coord, dcf**2, diagnosticsDir, num_ro=150, thresh=thresh, device=device)

    # 3) Bin Motion States
    logging.info("Running BinMotionStates...")
    ksp, coord, dcf = binMotionStates(ksp, coord, dcf, resp, nBins)
    del resp  # not needed anymore

    # 4) Low Res xdgrasp recon
    if os.path.exists(mrimgLPath) is False or ignoreExisting is True:
        logging.info("Running Low Res XDGrasp Reconstruction...")
        mrimg = xdgrasp(ksp, coord, dcf, res_scale=0.75, lambda_tv=xdgrasp_lambda, device=device, tv_device=tv_device)
        mrimgL = sp.resize(mrimg, (nBins, 192, 192, 192))
        mrimgL = np.moveaxis(np.abs(mrimgL), 0, -1)
        mrimgL = np.transpose(mrimgL, (2, 1, 0, 3))
        mrimgL = np.flip(mrimgL, (0, 1, 2))
        mrimgL = nib.Nifti1Image(mrimgL, np.eye(4))
        nib.save(mrimgL, mrimgLPath)
        del mrimgL

    # 5) iMoCo recon expir
    if os.path.exists(imgPath) is False or ignoreExisting is True:
        logging.info("Running iMoCo Reconstruction...")
        img = imoco(ksp, coord, dcf, mrimg, diagnosticsDir, res_scale=1.0, lambda_tv=imoco_lambda, inner_iter=15, outer_iter=20, device=device, nRef=nRef)
        img = sp.resize(np.abs(img), (256, 256, 256))
        img = nib.Nifti1Image(img, np.eye(4))
        nib.save(img, imgPath)
        del img

    # 6) iMoCo recon Insp
    if os.path.exists(imgInspPath) is False or ignoreExisting is True:
        logging.info("Running iMoCo Inspiratory Reconstruction...")
        imgInsp = imoco(ksp, coord, dcf, mrimg, diagnosticsDir, res_scale=1.0, lambda_tv=imoco_lambda, inner_iter=15, outer_iter=20, device=device, nRef=0)
        imgInsp = sp.resize(np.abs(imgInsp), (256, 256, 256))
        imgInsp = nib.Nifti1Image(imgInsp, np.eye(4))
        nib.save(imgInsp, imgInspPath)
        del imgInsp, mrimg

    # 7) Full Res xdgrasp recon
    if os.path.exists(mrimgPath) is False or ignoreExisting is True:
        logging.info("Running Full Res XDGrasp Reconstruction...")
        mrimg = xdgrasp(ksp, coord, dcf, res_scale=1.0, lambda_tv=xdgrasp_lambda, device=device, tv_device=tv_device)
        mrimg = sp.resize(mrimg, (nBins, 256, 256, 256))
        mrimg = np.moveaxis(np.abs(mrimg), 0, -1)
        mrimg = np.transpose(mrimg, (2, 1, 0, 3))
        mrimg = np.flip(mrimg, (0, 1, 2))
        mrimg = nib.Nifti1Image(mrimg, np.eye(4))
        nib.save(mrimg, mrimgPath)
        del ksp, coord, dcf, mrimg

    # 8) Full Res MoCo Expiratory
    if os.path.exists(imgMocoPath) is False or ignoreExisting is True:
        logging.info("Running Full Res MoCo Registration...")
        imgMoco = moco(mrimgPath, diagnosticsDir, nRef=nRef)
        imgMoco = nib.Nifti1Image(imgMoco, np.eye(4))
        nib.save(imgMoco, imgMocoPath)
        del imgMoco

    # 9) Full Res MoCo Inspiratory
    if os.path.exists(imgMocoInspPath) is False or ignoreExisting is True:
        logging.info("Running Full Res MoCo Insp Registration...")
        imgMocoInsp = moco(mrimgPath, diagnosticsDir, nRef=0)
        imgMocoInsp = nib.Nifti1Image(imgMocoInsp, np.eye(4))
        nib.save(imgMocoInsp, imgMocoInspPath)
        del imgMocoInsp

    timeF = (time.time() - timei) / 60
    logging.info("Finshed Subject {} in {} minutes".format(subject, timeF))
    # destroy data every loop
except KeyboardInterrupt:
    print("interrupted!")
