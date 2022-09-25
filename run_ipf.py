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
sub_dir = "/home/ltorres/data/rawdata/ipf/"
subjects = os.listdir(sub_dir)
subjects.sort()
# subjects = ["103-042"]
# visits = ['20151222']
study = "ipf"
# Subset for regularization parameter tests
# select_subjects = ["103-005", "103-010", "103-038", "103-039", "103-042"]
select_subjects = []
ipf_subjects = [
    "103-003",
    "103-005",
    "103-009",
    "103-012",
    "103-013",
    "103-015",
    "103-016",
    "103-018",
    "103-020",
    "103-021",
    "103-023",
    "103-025",
    "103-026",
    "103-030",
    "103-031",
    "103-035",
    "103-037",
    "103-038",
    "103-039",
    "103-040",
    "103-041",
    "103-042",
]
subjects = ipf_subjects
resp_flips = {
    "103-001": False,
    "103-002": False,
    "103-003": False,
    "103-004": True,
    "103-005": True,
    "103-006": True,
    "103-007": False,
    "103-008": False,
    "103-009": False,
    "103-010": True,
    "103-011": False,
    "103-012": False,
    "103-013": False,
    "103-014": False,
    "103-015": True,
    "103-016": False,
    "103-017": True,
    "103-018": False,
    "103-019": True,
    "103-020": True,
    "103-021": True,
    "103-022": False,
    "103-023": False,
    "103-024": False,
    "103-025": False,
    "103-026": True,
    "103-027": True,
    "103-028": True,
    "103-029": True,
    "103-030": False,
    "103-031": True,
    "103-032": False,
    "103-033": True,
    "103-034": True,
    "103-035": False,
    "103-036": False,
    "103-037": True,
    "103-038": True,
    "103-039": True,
    "103-040": False,
    "103-041": True,
    "103-042": True,
}
prefix = ""
postfix = "_final"
contrast_type = "pre_contrast"
try:
    for subject_id, subject in enumerate(subjects):
        # Check if directory exists
        if not os.path.isdir(os.path.join(sub_dir, subject + "/mri/")):
            continue
        # Search for visits here
        visits = os.listdir(os.path.join(sub_dir, subject + "/mri/"))
        visits.sort()
        # print(visits)
        visits = visits[0]
        visits = [visits] if isinstance(visits, str) else visits
        for visit in visits:
            raw_dir = f"/home/ltorres/data/rawdata/ipf/{subject}/mri/{visit}/{contrast_type}"
            out_dir = f"/home/ltorres/data/recon/ipf/{subject}/mri/{visit}/{contrast_type}"
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
                visit=visit,
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
                do_gridded_motion_resolved=False,
                overwrite_raw=False,
                overwrite_recons=False,
                flip_resp=resp_flips[subject],
                UID_base=UID_base,
            )
            time_finish = (time.time() - time_start) / 3600
            try:
                if time_finish > 0.1:
                    send_text.send(f"Finished Subject: {subject} in {time_finish} hours")
            except:
                pass

except KeyboardInterrupt:
    pass
