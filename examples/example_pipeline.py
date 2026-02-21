#!/usr/bin/env python
"""Example script showing how to run the iMoCo pipeline.

Usage:
    # Using the CLI entry point:
    imoco examples/example_config.yaml

    # With path overrides:
    imoco examples/example_config.yaml --raw_dir /data/subject01 --out_dir /output/subject01

    # Using this script directly:
    python examples/example_pipeline.py config.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from imoco.pipeline import load_config, run


def main():
    parser = argparse.ArgumentParser(description="Run the iMoCo reconstruction pipeline")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--raw_dir", type=str, default=None, help="Override raw data directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    if args.raw_dir:
        cfg["raw_dir"] = args.raw_dir
    if args.out_dir:
        cfg["out_dir"] = args.out_dir

    # Set up logging
    postfix = cfg.get("postfix", "")
    diagnostics_dir = Path(cfg["out_dir"]) / f"diagnostics{postfix}"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(diagnostics_dir / "recon_log.txt", mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Run pipeline
    time_start = time.time()
    run(cfg)
    time_finish = (time.time() - time_start) / 3600
    logging.info(f"Total pipeline time: {time_finish:.2f} hours")


if __name__ == "__main__":
    main()
