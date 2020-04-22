#! /usr/bin/env python
import logging
import sigpy.mri as mr
import h5py
import numpy as np
import argparse
import os

# import sigpy.plot as plt

parser = argparse.ArgumentParser(
    description="Converts UWUTE h5 files to npy arrays in natural time ordering."
)
parser.add_argument("h5_file", type=str)
parser.add_argument("ksp_file", type=str)
parser.add_argument("coord_file", type=str)
parser.add_argument("dcf_file", type=str)
parser.add_argument("resp_file", type=str)
parser.add_argument("--dsfSpokes", type=float, default=1.0)
args = parser.parse_args()


logging.basicConfig(level=logging.INFO)

with h5py.File(args.h5_file, "r") as hf:

    try:
        time = np.squeeze(hf["Gating"]["time"])
        order = np.argsort(time)
    except Exception:
        time = np.squeeze(hf["Gating"]["TIME_E0"])
        order = np.argsort(time)

    try:
        resp = np.squeeze(hf["Gating"]["resp"])
        resp = resp[order]
    except Exception:
        resp = np.squeeze(hf["Gating"]["RESP_E0"])
        resp = resp[order]

    coord = []
    for i in ["Z", "Y", "X"]:
        logging.info(f"Loading {i} coord.")

        coord.append(hf["Kdata"][f"K{i}_E0"][0][order])

    coord = np.stack(coord, axis=-1)

    logging.info("Loading dcf")
    dcf = hf["Kdata"]["KW_E0"][0][order]

    num_coils = 0
    while f"KData_E0_C{num_coils}" in hf["Kdata"]:
        num_coils += 1
    logging.info(f"Number of coils: {num_coils}")

    ksp = []
    for c in range(num_coils):
        logging.info(f"Loading kspace, coil {c + 1} / {num_coils}.")

        k = hf["Kdata"][f"KData_E0_C{c}"]
        ksp.append(k["real"][0][order] + 1j * k["imag"][0][order])
    ksp = np.stack(ksp, axis=0)

    try:
        noise = hf["Kdata"]["Noise"]["real"] + 1j * hf["Kdata"]["Noise"]["imag"]
        logging.info("Whitening ksp.")
        cov = mr.util.get_cov(noise)
        ksp = mr.util.whiten(ksp, cov)
    except Exception:
        ksp /= np.abs(ksp).max()
        logging.info("No noise data.")
        pass

    totalSpokes = ksp.shape[1]
    nSpokes = int(totalSpokes // args.dsfSpokes)
    print("Total Number of Spokes: {}, Requested Number of Spokes: {}".format(totalSpokes, nSpokes))
    ksp = ksp[:, :nSpokes, :]
    coord = coord[:nSpokes, :, :]
    dcf = dcf[:nSpokes, :]
    logging.info("Saving data.")
    if os.path.isfile(args.ksp_file):
        os.remove(args.ksp_file)
        os.remove(args.coord_file)
        os.remove(args.dcf_file)
        os.remove(args.resp_file)
    np.save(args.ksp_file, ksp)
    np.save(args.coord_file, coord)
    np.save(args.dcf_file, dcf)
    np.save(args.resp_file, resp)
