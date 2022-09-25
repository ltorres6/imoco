import os
import logging
from pathlib import Path
from writeDicoms import writeDicoms

os.environ["MATLAB_ROOT"] = "/usr/local/matlab2018"
os.environ["MATLAB_JAVA"] = "/usr/local/java/jre"
logging.basicConfig(level=logging.INFO)
rootDir = "/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf"
fileTypes = [
    "PreContrastNoGate",
    "PreContrastHardGate",
    "PreContrastSoftGate",
    "PreContrastMoCo",
    "PreContrastIterativeMoCo",
    "PreContrastMoCoInsp",
    "PreContrastIterativeMoCoInsp",
    "PostContrastNoGate",
    "PostContrastHardGate",
    "PostContrastSoftGate",
    "PostContrastMoCo",
    "PostContrastIterativeMoCo",
    "PostContrastMoCoInsp",
    "PostContrastIterativeMoCoInsp",

]
fileNames = [
    "noGate_gw.nii.gz",
    "hardGate_gw.nii.gz",
    "softGate_gw.nii.gz",
    "MoCo_gw.nii.gz",
    "iMoCo_gw.nii.gz",
    "MoCo_gw.nii.gz",
    "iMoCo_gw.nii.gz",
    "noGate_gw.nii.gz",
    "hardGate_gw.nii.gz",
    "softGate_gw.nii.gz",
    "MoCo_gw.nii.gz",
    "iMoCo_gw.nii.gz",
    "MoCo_gw.nii.gz",
    "iMoCo_gw.nii.gz",
]
subjectList = os.listdir(rootDir)
subjectList.sort()
try:
    for ii in subjectList:
        subject = ii
        if int(subject[4:]) < 43:
            print(int(subject[4:]))
            if os.path.exists(os.path.join(rootDir, subject + "/mri/")):
                visitList = os.listdir(os.path.join(rootDir, subject + "/mri/"))
            else:
                print("Subject Visit does not exist.")
                continue

            visit = visitList[0]
            # print(visit)
            subjectDir = os.path.join(rootDir, subject, "mri", visit)
            print(subjectDir)
            for jj in range(len(fileTypes)):
                # Set up data paths
                fileType = fileTypes[jj]
                fileName = fileNames[jj]
                dicomDir = os.path.join(subjectDir, fileType, "dicoms")
                imgDir = os.path.join(subjectDir, fileType)
                imgPath = os.path.join(subjectDir, fileType, fileName)

                if os.path.exists(imgPath):
                    print("File Exists, Begin!")
                else:
                    print("File does not exist.")
                    continue
                Path(dicomDir).mkdir(parents=True, exist_ok=True)

                # Write Dicoms
                if not os.path.exists(os.path.join(dicomDir, "0.dcm")):
                    logging.info("Writing Dicoms...")
                    writeDicoms(imgPath, dicomDir)
except KeyboardInterrupt:
    print("interrupted")
