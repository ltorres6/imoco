import os

cores = "1"
os.environ["OMP_NUM_THREADS"] = cores  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = cores  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = cores  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = cores  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = cores  # export NUMEXPR_NUM_THREADS=6
import numpy as np
import time
from runRecon import runRecon
import logging as logging

logging.basicConfig(level=logging.INFO)
subDir = "/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf/"
subjectList = os.listdir(subDir)
subjectList.sort()
imoco_lambda = 0.05
imoco_lambdas = np.array([0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
device = 0
try:
    for ii in subjectList:
        subject = ii
        if int(subject[4:]) == 40:
            timei = time.time()
            # print(int(subject[4:]))
            visitList = os.listdir(os.path.join(subDir, subject + "/mri/"))
            visit = visitList[0]
            for qq in imoco_lambdas:
                runRecon(subject, visit, qq, postfix=qq, device=device)

            timeF = (time.time() - timei) / 60
            logging.info("Finshed Subject {} in {} minutes".format(subject, timeF))
            # destroy data every loop
except KeyboardInterrupt:
    print("interrupted!")
