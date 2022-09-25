from run_main import run
import logging
import sys, os
from pathlib import Path
import numpy as np
import sendText as send_text
import subprocess

# import sigpy.plot as plt
# import nibabel as nib
# import os
# from normalize import normalize
sub_dir = "/media/ltorres/Seagate Expansion Drive/rawdata/ipf/"
subjects = os.listdir(sub_dir)
subjects.sort()
# subjects = ["103-002", "103-042"]
# subjects = ["103-042"]
# visits = ['20151222']
study = "ipf"
# Subset for regularization parameter tests
# select_subjects = ["103-005", "103-010", "103-038", "103-039", "103-042"]
select_subjects = []

prefix = "pre_contrast"
postfix = "_run_final_test"
contrast_type = "pre_contrast"
try:
    for subject in subjects:
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
            raw_dir = f"/media/ltorres/Seagate Expansion Drive/rawdata/ipf/{subject}/mri/{visit}/"
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
            print(visit)
            subprocess.call(
                ["/home/ltorres/projects/motion_compensation_ipf/copy_from_drive.sh", subject, visit, "Pre", study]
            )
            try:
                send_text.send(f"Copied Subject: {subject}")

            except:
                pass

except KeyboardInterrupt:
    pass

