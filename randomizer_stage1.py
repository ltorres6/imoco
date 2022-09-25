import os
import logging
from pathlib import Path
from writeDicoms import writeDicoms
import time
import numpy as np
import csv


# Randomly Generate series numbers
os.environ["MATLAB_ROOT"] = "/usr/local/matlab2018"
os.environ["MATLAB_JAVA"] = "/usr/local/java/jre"
logging.basicConfig(level=logging.INFO)
rootDir = "/home/ltorres/data/recon/ipf/"
outDir = "/home/ltorres/data/recon/ipf_randomized_1/"
fileTypes = ["NoGate_final", "HardGate_final", "SoftGate_final", "IterativeMoCo_final", "MotionResolved_final"]
fileNames = [
    "noGate.nii.gz",
    "hardGate50.nii.gz",
    "softGate0.8.nii.gz",
    "iMoCo0.05_frame0.nii.gz",
    "MotionResolved_exp0.050.nii.gz",
]
subjectList = os.listdir(rootDir)
ignored = ["103-002", "103-009"]
subjectList = [x for x in subjectList if x not in ignored]
subjectList.sort()
# subjectList = ["P006_Exam1", "P099_Exam1"]
series_nums = np.random.permutation(5 * len(subjectList))
# print(series_nums)
mapping = {}
try:
    for subject in subjectList:
        visits = os.listdir(os.path.join(rootDir, subject + "/mri/"))
        visits.sort()
        visits = visits[0]
        visits = [visits] if isinstance(visits, str) else visits
        for visit in visits:
            subjectDir = os.path.join(rootDir, subject, "mri", visit, "pre_contrast")
            # print(subjectDir)
            for jj in range(len(fileTypes)):
                key, series_nums = series_nums[0], series_nums[1:]
                # Same UID base for each visit for dicoms
                modification_time = time.strftime("%H%M%S%f")
                modification_date = time.strftime("%Y%m%d")
                UID_base = "1.2.840.0.1.3680043.2.1125." + modification_date + ".1" + modification_time
                # Set up data paths
                fileType = fileTypes[jj]
                fileName = fileNames[jj]
                dicomDir = os.path.join(outDir, f"series_{key}")
                imgDir = os.path.join(subjectDir, fileType)
                imgPath = os.path.join(subjectDir, fileType, fileName)
                # Get first element and return view with -1 element
                mapping[key] = imgPath
                # print(mapping)
                # if os.path.exists(imgPath):
                #     print("File Exists, Begin!")
                # else:
                #     print("File does not exist.")
                #     continue
                Path(dicomDir).mkdir(parents=True, exist_ok=True)

                # Write Dicoms
                if not os.path.exists(os.path.join(dicomDir, "0.dcm")):
                    logging.info("Writing Dicoms...")
                    writeDicoms(imgPath, dicomDir, UID_base=UID_base, subject_id=key, series_num=key)

    with open(os.path.join(outDir, "mapping.csv"), "w") as f:
        w = csv.writer(f)
        # loop over dictionary keys and values
        for key, val in mapping.items():

            # write every key and value to file
            w.writerow([key, val])
except KeyboardInterrupt:
    print("interrupted")
    with open(os.path.join(outDir, "mapping.csv"), "w") as f:
        w = csv.writer(f)
        # loop over dictionary keys and values
        for key, val in mapping.items():

            # write every key and value to file
            w.writerow([key, val])
