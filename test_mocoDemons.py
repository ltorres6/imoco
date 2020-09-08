import os
import mocoDemons
import logging
from pathlib import Path
import sigpy as sp
import time

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
ignoreExisting = True
register = 1
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
    mocoDemonsDir = os.path.join(outDir, "PreContrastMoCoDemons")
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
    imgMocoDemonsPath = os.path.join(mocoDemonsDir, "MoCo.nii.gz")
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
    Path(mocoDemonsDir).mkdir(parents=True, exist_ok=True)

    imgMocoDemons = mocoDemons.mocoDemons(mrimgPath, diagnosticsDir, nRef=-1)

except KeyboardInterrupt:
    print("interrupted!")
