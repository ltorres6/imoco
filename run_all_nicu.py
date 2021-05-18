from run_main import run
import logging
import sys
from pathlib import Path
import numpy as np

# import sigpy.plot as plt
# import nibabel as nib
# import os
# from normalize import normalize

subjects = ["P24", "P62", "P67", "P84", "P89_Exam1", "P95", "P117", "P118_Exam2", "P147", "P150", "P153", "P154"]
# Subset for regularization parameter tests
select_subjects = []
# select_subjects = ["P24", "P89_Exam1", "P118_Exam2", "P153"]

postfix = "_mybinning"
try:
    for subject in subjects:
        raw_dir = "/home/ltorres/data/rawdata/nicu/{}/".format(subject)
        out_dir = "/home/ltorres/data/recon/nicu/{}/".format(subject)
        Path(out_dir + f"/diagnostics{postfix}/").mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(out_dir + f"/diagnostics{postfix}/recon_log.txt", mode="a"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.info("Running Subject {}".format(subject))
        if any(s in subject for s in select_subjects):
            softgating_decays = np.arange(0.5, 5, 0.5).tolist()  # To test sharpness metric
            imoco_lambdas = [
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.1,
            ]  # To test regularization parameter
            xdgrasp_lambdas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        else:
            softgating_decays = 2.0
            imoco_lambdas = 0.05
            xdgrasp_lambdas = 0.02
        run(
            raw_dir,
            out_dir,
            softgating_decays=softgating_decays,
            imoco_lambdas=imoco_lambdas,
            xdgrasp_lambdas=xdgrasp_lambdas,
            postfix=postfix,
        )
except KeyboardInterrupt:
    pass
