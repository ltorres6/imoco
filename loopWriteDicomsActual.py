import os
import logging
from pathlib import Path
from writeDicoms import writeDicoms

os.environ["MATLAB_ROOT"] = "/usr/local/matlab2018"
os.environ["MATLAB_JAVA"] = "/usr/local/java/jre"
logging.basicConfig(level=logging.INFO)
rootDir = "/home/ltorres/data/recon/nicu/"
fileTypes = ["NoGate_run_cc", "HardGate_run_cc", "SoftGate_run_cc", "IterativeMoCo_run_cc", "MotionResolved_run_cc"]
fileNames = [
    "noGate.nii.gz",
    "hardGate50.nii.gz",
    "softGate0.8.nii.gz",
    "iMoCo0.05_frame0.nii.gz",
    "MotionResolved_exp0.050.nii.gz",
]
subjectList = os.listdir(rootDir)
subjectList.sort()
subjectList=["P006_Exam1", "P099_Exam1"]
try:
    for subject in subjectList:
        subjectDir = os.path.join(rootDir, subject)
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
