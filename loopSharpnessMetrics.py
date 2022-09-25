import os
import subprocess
import logging
from pathlib import Path
from writeDicoms import writeDicoms

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

]
fileNames = [
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
        if int(subject[4:]) == 42:
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
                pathExist = os.path.isdir(dicomDir)
                if pathExist is False:
                os.makedirs(dicomDir)
                img = nib.load(imgPath).get_fdata()
                # Convert to uint16
                img = convert(img, 0, 65535, "uint16")
                # Orient Properly
                img = np.flip(np.flip(np.transpose(img, [2, 1, 0]), axis=1), axis=2)
                
                # Write Dicoms
                logging.info("Writing Dicoms...")
                writeDicoms(imgPath, dicomDir)
except KeyboardInterrupt:
    print("interrupted")
