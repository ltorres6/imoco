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
sub_dir = "/scratch_cn1/ltorres/"
subjects = os.listdir(sub_dir)
subjects.sort()
# subjects = subjects[: len(subjects) // 2]
# subjects = subjects[len(subjects) // 2:]
device = 0
print(subjects)
# subjects = ["103-042"]
# visits = ['20151222']
study = "fe_ute"
# Subset for regularization parameter tests
# select_subjects = ["103-005", "103-010", "103-038", "103-039", "103-042"]
select_subjects = []

prefix = ""
postfix = "_final"
contrast_type = ""
try:
    for subject_id, subject in enumerate(subjects):
        visits = os.listdir(f"/scratch_cn1/ltorres/{subject}/")
        visits.sort()
        visit = visits[0]
        raw_dir = f"/scratch_cn1/ltorres/{subject}/{visit}/raw_data/"
        out_dir = f"/scratch_cn1/ltorres/{subject}/{visit}/processed_data/"
        # tr = 0.00364
        # np.save(f"/scratch_cn1/ltorres/{subject}/{visit}/raw_data/tr.npy", tr)
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
            softgating_decays = np.arange(0.0, 1.5, 0.2).tolist()  # To test sharpness metric
            hardgating_weights = np.arange(0, 100 + 0.001, 5).tolist()  # To test sharpness metric
            imoco_lambdas = np.arange(0.01, 0.1, 0.01).tolist()  # To test regularization parameter
            xdgrasp_lambdas = [0.01, 0.03, 0.05]
            lowRes_xdgrasp_lambda = 0.04

        else:
            hardgating_weights = 50.0
            softgating_decays = 0.8
            imoco_lambdas = [0.05]
            xdgrasp_lambdas = 0.05
            lowRes_xdgrasp_lambda = 0.04

        n_bins = 6

        # Same UID base for each visit for dicoms
        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")
        UID_base = "1.2.840.0.1.3680043.2.1125." + modification_date + ".1" + modification_time
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
            overwrite_raw=True,
            overwrite_recons=False,
            flip_resp=True,
            UID_base=UID_base,
            device=device,
            max_coils=20,
        )
        time_finish = (time.time() - time_start) / 3600
        try:
            send_text.send(f"Finished Subject: {subject} in {time_finish} hours")
        except:
            pass

except KeyboardInterrupt:
    pass
