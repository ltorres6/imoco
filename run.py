import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

import sendText as send_text
from run_main import run


study = "iowa"
prefix = ""
postfix = "cc_not_whitened"
contrast_type = ""

raw_dir = "/home/ltorres/data/new_oe_data/"
out_dir = "/home/ltorres/data/new_oe_data/"
# tr = 0.0037
# np.save(f"/home/ltorres/data/rawdata/iowa_fidall/{subject}/tr.npy", tr)
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
logging.info("Running Subject...")
hardgating_weights = 50.0
softgating_decays = 0.8
imoco_lambdas = [0.05]
xdgrasp_lambdas = 0.05
lowRes_xdgrasp_lambda = 0.04

n_bins = 6
n_coils = 40

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
    flip_resp=False,
    UID_base=UID_base,
    fovNReadout=75,
    fovthresh=0.05,
    sigma=0.2,
    tau=0.2,
    max_coils=n_coils,
    final_matrix_size=(320, 320, 320),
    resolution=[1.00, 1.00, 1.00],
    recalc_dcf=False,
    pre_whiten=False,
    clean_by_resp=True,
    load_registration=False,
    iterations=30,
)
time_finish = (time.time() - time_start) / 3600
try:
    send_text.send(f"Finished Subject in {time_finish} hours")
except:
    pass
