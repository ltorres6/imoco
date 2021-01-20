from run_main import run
import logging
import sys
from pathlib import Path

subjects = ["P24", "P62", "P67", "P84", "P89_Exam1", "P95", "P117", "P118_Exam2", "P147", "P150", "P153", "P154"]

for subject in subjects:
    raw_dir = "/home/ltorres/data/rawdata/nicu/{}/".format(subject)
    out_dir = "/home/ltorres/data/recon/nicu/{}/".format(subject)
    Path(out_dir + "/diagnostics/").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        filemode="a",
        handlers=[logging.FileHandler(out_dir + "/diagnostics/recon_log.txt"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Running Subject {}".format(subject))
    run(raw_dir, out_dir)
