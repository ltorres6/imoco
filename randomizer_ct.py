import os
import logging
from pathlib import Path
from writeDicoms import writeDicoms
import time
import numpy as np
import csv
import subprocess


# Randomly Generate series numbers
os.environ["MATLAB_ROOT"] = "/usr/local/matlab2018"
os.environ["MATLAB_JAVA"] = "/usr/local/java/jre"
logging.basicConfig(level=logging.INFO)
rootDir = "/home/ltorres/data/recon/ipf_ct/"
outDir = "/home/ltorres/data/recon/ipf_randomized_ct/"
subjectList = os.listdir(rootDir)
# subjectList = [x for x in subjectList if x not in ignored]
subjectList.sort()
# subjectList = ["P006_Exam1", "P099_Exam1"]
series_nums = np.random.permutation(len(subjectList))
# print(series_nums)
mapping = {}
try:
    for subject in subjectList:
        visits = os.listdir(os.path.join(rootDir, subject))
        visits.sort()
        visits = visits[0]
        visits = [visits] if isinstance(visits, str) else visits
        for visit in visits:
            subjectDir = os.path.join(rootDir, subject, visit)
            key, series_nums = series_nums[0], series_nums[1:]
            imgPath = os.path.join(subjectDir)
            # print(imgPath)
            # Get first element and return view with -1 element
            mapping[key] = imgPath
            os.chdir(imgPath)
            p = subprocess.Popen(
                
                    'dcmodify -ie -gin -nb -ea "(0010,0010)" -ea "(0010,0020)" -ea "(0010,0030)" -ea "(0020,000E)" -ea "(0020,000D)" -ea "(0008,0080)" -ea "(0008,0081)" -ea "(0008,0050)" -ea "(0008,0090)" -ea "(0008,1070)" -ea "(0008,1155)" -ea "(0010,1000)" -ea "(0020,0010)" -ea "(0020,4000)"'.split(),
                    "*",
              
            )
            # dcmodify -ie -gin -nb -ea "(0010,0010)" -ea "(0010,0020)" -ea "(0010,0030)" -ea "(0020,000E)" -ea "(0020,000D)" -ea "(0008,0080)" -ea "(0008,0081)" -ea "(0008,0050)" -ea "(0008,0090)" -ea "(0008,1070)" -ea "(0008,1155)" -ea "(0010,1000)" -ea "(0020,0010)" -ea "(0020,4000)" *.dcm
            p.wait()
except KeyboardInterrupt:
    print("interrupted")
