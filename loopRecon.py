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
from binMotionStates import binMotionStates
from xdgrasp import xdgrasp
import logging
from pathlib import Path
import cfl
import sigpy as sp
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
# loop_moco.py
codeDir = "/export/home/ltorres/projects/xdgrasp"
rootDir = "/scratch/scratch_cnxx/ltorres"
outDir = "/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf"
subjectList = os.listdir(rootDir)
subjectList.sort()
# set device
# device = 0
device = 3  # on CN cluster
nBins = 6
nCoils = 8
xp = sp.Device(device).xp
try:
    for ii in subjectList:
        subject = ii
        if int(subject[4:]) > 37:
            timei = time.time()
            # print(int(subject[4:]))
            visitList = os.listdir(os.path.join(rootDir, subject + "/mri/"))
            visit = visitList[0]
            # print(visit)
            subjectDir = os.path.join(rootDir, subject, "mri", visit)
            subjectOutDir = os.path.join(outDir, subject, "mri", visit, "PreContrastMotionResolved")
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

            # 2) AutoFOV to reduce matrix size
            logging.info("Running AutoFOV...")
            if int(subject[4:]) > 32:
                thresh = 0.02
            else:
                thresh = 0.05
            # Change threshold 2for specific subjects
            if int(subject[4:]):
                thresh = 0.03

            coord = autofov(
                ksp, coord, dcf, diagnosticsDir, num_ro=150, thresh=thresh, device=device
            )

            # # 3) Bin Motion States
            logging.info("Running BinMotionStates...")
            ksp, coord, dcf = binMotionStates(ksp, coord, dcf, resp, nBins)
            del resp

            # # 4) xdgrasp recon
            logging.info("Running Reconstruction...")
            img = xdgrasp(ksp, coord, dcf, res_scale=1.0, lambda_tv=0.05, device=device)
            del ksp, coord, dcf
            # cfl.write_cfl(mrimgPath, img)
            img = np.moveaxis(np.abs(img), 0, -1)
            # np.save(mrimgPathnpy,img)
            img = nib.Nifti1Image(img.astype("f"), np.eye(4))
            nib.save(img, mrimgPath)
            timeF = (timei - time.time()) / 60
            logging.info("Finshed Subject {} in {} minutes".format(subject, timeF))
            # destroy data every loop
            # del ksp, coord, dcf
except KeyboardInterrupt:
    print("interrupted!")
