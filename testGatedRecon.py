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
from gatedRecon import gatedRecon
import logging
from pathlib import Path
import sigpy as sp
import numpy as np
import time
import matplotlib.pyplot as plt
import subprocess

logging.basicConfig(level=logging.INFO)
# subject = "103-005"
# visit = "20160114"
# subject = "103-016"
# visit = "20161121"
subject = "103-040"
visit = "20181005"
# subject = "103-041"
# visit = "20181011"
# subject = "103-042"
# visit = "20190221"
# rawDir = "/scratch/scratch_cnxx/ltorres/"+subject+"/mri/"+visit+"/"
# rawDir = "/export/home/ltorres/data/scratch_tmp/ltorres/"+subject+"/mri/"+visit+"/"
outDir = "/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf/" + subject + "/mri/" + visit + "/"
# rawDir = "/home/ltorres/data/ipf/ltorres/"+subject+"/mri/"+visit+"/"
# outDir = "/home/ltorres/data/ipf/ltorres/"+subject+"/mri/"+visit+"/"
# rawDir = "/home/ltorres/data/nicu"
# outDir = "/home/ltorres/data/nicu"
# set device
device = -1
nBins = 6
nCoils = 8
prefix = "Post"
# prefix = "Post"
ignoreExisting = True
register = 1
dc_signal = 1
nRef = -1  # Reference Frame (last ie expiratory)
xp = sp.Device(device).xp
try:
    timei = time.time()
    # print(visit)
    motionResolvedDir = os.path.join(outDir, prefix+"ContrastMotionResolved")
    iterativeMocoDir = os.path.join(outDir, prefix+"ContrastIterativeMoCo")
    iterativeMocoInspDir = os.path.join(outDir, prefix+"ContrastIterativeMoCoInsp")
    mocoDir = os.path.join(outDir, prefix+"ContrastMoCo")
    mocoInspDir = os.path.join(outDir, prefix + "ContrastMoCoInsp")
    noGateDir = os.path.join(outDir, prefix + "ContrastNoGate")
    hardGateDir = os.path.join(outDir, prefix + "ContrastHardGate")
    softGateDir = os.path.join(outDir, prefix + "ContrastSoftGate")

    rawDir = "/scratch/scratch_cnxx/ltorres/"+subject+"/mri/"+visit+"/"+prefix+"Contrast"
    Path(rawDir).mkdir(parents=True, exist_ok=True)
    print(rawDir)

    # Copy Raw Data
    fileList = os.listdir(rawDir)
    if "MRI_Raw.h5" in fileList:
        print("File Exists, Not Copying!")
    else:
        subprocess.call(["/export/home/ltorres/projects/xdgrasp/copyData.sh", "lat205", subject, visit, prefix])

    fileList = os.listdir(rawDir)
    if "MRI_Raw.h5" in fileList:
        print("File Exists, Begin!")
    else:
        pass
    # Set up data paths
    h5Path = os.path.join(rawDir, "MRI_Raw.h5")
    diagnosticsDir = os.path.join(motionResolvedDir, "diagnostics")
    mrimgPath = os.path.join(motionResolvedDir, "MotionResolved.nii.gz")
    mrimgLPath = os.path.join(motionResolvedDir, "MotionResolvedLowRes.nii.gz")
    imgMocoPath = os.path.join(mocoDir, "MoCo.nii.gz")
    imgPath = os.path.join(iterativeMocoDir, "iMoCo.nii.gz")
    respPath = os.path.join(diagnosticsDir, "resp.npy")
    imgInspPath = os.path.join(iterativeMocoInspDir, "iMoCo.nii.gz")
    imgMocoInspPath = os.path.join(mocoInspDir, "MoCo.nii.gz")
    imgNoGatePath = os.path.join(noGateDir, "noGate.nii.gz")
    imgHardGatePath = os.path.join(hardGateDir, "hardGate.nii.gz")
    imgSoftGatePath = os.path.join(softGateDir, "softGate.nii.gz")

    # Create outpath if doesn't exist.
    Path(motionResolvedDir).mkdir(parents=True, exist_ok=True)
    Path(iterativeMocoDir).mkdir(parents=True, exist_ok=True)
    Path(diagnosticsDir).mkdir(parents=True, exist_ok=True)
    Path(mocoDir).mkdir(parents=True, exist_ok=True)
    Path(iterativeMocoInspDir).mkdir(parents=True, exist_ok=True)
    Path(mocoInspDir).mkdir(parents=True, exist_ok=True)
    Path(noGateDir).mkdir(parents=True, exist_ok=True)
    Path(hardGateDir).mkdir(parents=True, exist_ok=True)
    Path(softGateDir).mkdir(parents=True, exist_ok=True)

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
        xdgrasp_lambda = 0.03
        imoco_lambda = 0.025
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
    coord = autofov(ksp, coord, dcf ** 2, diagnosticsDir, num_ro=100, thresh=thresh, device=device)

    # 4) noGating Recon
    if os.path.exists(imgNoGatePath) is False or ignoreExisting is True:
        imgNoGate = gatedRecon(ksp, coord, dcf, resp, gating_type="none", device=device)
        imgNoGate = sp.resize(np.abs(imgNoGate), (256, 256, 256))
        imgNoGate = nib.Nifti1Image(imgNoGate, np.eye(4))
        nib.save(imgNoGate, imgNoGatePath)
        del imgNoGate

    # 5) hardGating Recon
    if os.path.exists(imgHardGatePath) is False or ignoreExisting is True:
        imgHardGate = gatedRecon(ksp, coord, dcf, resp, gating_type="hard", gating_thresh=0.5, device=device)
        imgHardGate = sp.resize(np.abs(imgHardGate), (256, 256, 256))
        imgHardGate = nib.Nifti1Image(imgHardGate, np.eye(4))
        nib.save(imgHardGate, imgHardGatePath)
        del imgHardGate

    # 6) softGating Recon
    if os.path.exists(imgSoftGatePath) is False or ignoreExisting is True:
        imgSoftGate = gatedRecon(ksp, coord, dcf, resp, gating_type="soft", gating_thresh=0.25, gating_weight=3, device=device)
        imgSoftGate = sp.resize(np.abs(imgSoftGate), (256, 256, 256))
        imgSoftGate = nib.Nifti1Image(imgSoftGate, np.eye(4))
        nib.save(imgSoftGate, imgSoftGatePath)
        del imgSoftGate

    timeF = (time.time() - timei) / 60
    logging.info("Finshed Subject {} in {} minutes".format(subject, timeF))
    # destroy data every loop
except KeyboardInterrupt:
    print("interrupted!")
