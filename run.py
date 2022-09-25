from run_main import run
import logging
import sys
from pathlib import Path
import numpy as np

select_subjects = []

postfix = ""
raw_dir = "/home/ltorres/data/new_oe_data/"
out_dir = "/home/ltorres/data/new_oe_data/"
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
logging.info("Running...")
softgating_decays = 0.8
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
