import os

# import subprocess
from convertUTE import convertUTE
from autofov import autofov
from binMotionStates import binMotionStates
from xdgrasp import xdgrasp
import cfl

# loop_moco.py
codeDir = "/export/home/ltorres/projects/xdgrasp"
rootDir = "/scratch/scratch_cnxx/ltorres"
subjectList = os.listdir(rootDir)
subjectList.remove("ipf")
subjectList.sort()
# set device
device = 2
try:
    for ii in subjectList:
        subject = ii
        if int(subject[4:]) > 19:
            # print(int(subject[4:]))
            visitList = os.listdir(os.path.join(rootDir, subject + "/mri/"))
            visit = visitList[0]
            # print(visit)
            subjectDir = os.path.join(rootDir, subject, "mri", visit)
            print(subjectDir)
            fileList = os.listdir(subjectDir)
            if "MRI_Raw.h5" in fileList:
                "File Exists, Begin!"
            else:
                pass
            # Set up data paths
            h5_path = subjectDir + "/MRI_Raw.h5"
            ksp_path = subjectDir + "/ksp.npy"
            coord_path = subjectDir + "/coord.npy"
            dcf_path = subjectDir + "/dcf.npy"
            kspB_path = subjectDir + "/kspB.npy"
            coordB_path = subjectDir + "/coordB.npy"
            dcfB_path = subjectDir + "/dcfB.npy"
            mps_path = subjectDir + "/mps.npy"
            resp_path = subjectDir + "/resp.npy"
            mrimg_path = subjectDir + "/mrimg"  # no extention for cfl file writing
            diagnostics_path = subjectDir
            # 1) Convert MRI_Raw.h5 to cfl and read resp waveform.

            print("Running File Conversion...")
            ksp, coord, dcf, resp = convertUTE(h5_path)

            # 2) AutoFOV to reduce matrix size
            print("Running AutoFOV...")
            coord = autofov(ksp, coord, dcf, diagnostics_path, device=device)

            # 3) Bin Motion States
            print("Running BinMotionStates...")
            nBins = 6
            ksp, coord, dcf = binMotionStates(ksp, coord, dcf, resp, nBins)
            # 4) xdgrasp recon
            print("Running Reconstruction...")
            img = xdgrasp(ksp, coord, dcf, res_scale=0.5, lambda_tv=0.05, device=0)
            cfl.write_cfl(mrimg_path, img)
except KeyboardInterrupt:
    print("interrupted!")
