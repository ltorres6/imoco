import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

import sendText as send_text
from run_main import run

# import sigpy.plot as plt
# import nibabel as nib
# import os
# from normalize import normalize
sub_dir = "/home/ltorres/data/rawdata/nicu/"
ignored = ["Original Subjects"]
subjects = [x for x in os.listdir(sub_dir) if x not in ignored]
subjects.sort()
# subjects = subjects[:2]
study = "nicu"
# Subset for regularization parameter tests
# select_subjects = ["P099_Exam1", "P122_Exam1", "P173_Exam3", "P179_Exam1"]
# subjects = select_subjects
select_subjects = []
prefix = ""
postfix = "_final"
try:
    for subject_id, subject in enumerate(subjects):
        # Check if directory exists
        if not os.path.isdir(os.path.join(sub_dir, subject)):
            continue
        raw_dir = "/home/ltorres/data/rawdata/nicu/{}/{}".format(subject, prefix)
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
            softgating_decays = np.arange(0.0, 2.0, 0.2).tolist()  # To test sharpness metric
            hardgating_weights = np.arange(0, 100 + 0.001, 5).tolist()  # To test sharpness metric
            imoco_lambdas = np.arange(0.01, 0.07, 0.01).tolist()  # To test regularization parameter
            xdgrasp_lambdas = [0.01, 0.03, 0.05]
            lowRes_xdgrasp_lambda = 0.08
        else:
            hardgating_weights = 50.0
            softgating_decays = 0.8
            imoco_lambdas = [0.05]
            xdgrasp_lambdas = [0.1]
            lowRes_xdgrasp_lambda = 0.08

        # Same UID base for each visit for dicoms
        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")
        UID_base = "1.2.840.0.1.3680043.2.1125." + modification_date + ".1" + modification_time

        n_bins = 8
        time_start = time.time()
        run(
            raw_dir,
            out_dir,
            hardgating_weights=hardgating_weights,
            softgating_decays=softgating_decays,
            imoco_lambdas=imoco_lambdas,
            xdgrasp_lambdas=xdgrasp_lambdas,
            lowRes_xdgrasp_lambda=lowRes_xdgrasp_lambda,
            prefix=prefix,
            postfix=postfix,
            subject=subject,
            subject_id=subject_id,
            study=study,
            reference_frames=[0],
            n_bins=n_bins,
            do_noGating=True,
            do_HardGating=True,
            do_SoftGating=True,
            do_LowRes=True,
            do_iMoCoExp=True,
            do_HighRes=True,
            do_MoCoExp=False,
            do_gridded_motion_resolved=True,
            overwrite_raw=False,
            overwrite_recons=False,
            UID_base=UID_base,
        )
        time_finish = (time.time() - time_start) / 3600
        try:
            send_text.send(f"Finished Subject: {subject} in {time_finish} hours")
        except:
            pass


except (KeyboardInterrupt, MemoryError):
    pass
