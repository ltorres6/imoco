import argparse
import sigpy as sp
import numpy as np
import nibabel as nib
from tqdm import trange
import time
import logging
import copy
import sigpy.plot as plt


def griddedRecon(ksp_in, coord_in, dcf_in, n_bins, device=0):
    timeStart = time.time()
    sp.Device(device).use()
    xp = sp.Device(device).xp
    if device >= 0:
        logging.info("Using GPU...")
    else:
        logging.info("Using CPU...")

    # Copy input data
    ksp = copy.deepcopy(ksp_in)
    coord = copy.deepcopy(coord_in)
    dcf = copy.deepcopy(dcf_in)
    nCoils, nSpokes, nReadouts = ksp[0].shape

    logging.info(f"Running Gridding Recon For {n_bins} Motion States")
    img_shape = sp.estimate_shape(coord[0])

    # Reconstruction
    pbarOuter = trange(nCoils, leave=True, ncols=80)
    img_final = []
    for b in range(n_bins):
        coordB = sp.to_device(coord[b], device)
        kspB = ksp[b] * (dcf[b] ** 2)
        with sp.Device(device):
            img = 0
            for c in pbarOuter:
                timeI = time.time()
                pbarOuter.set_description(f"Gridding Recon - Coil: {c}")
                ksp_c = sp.to_device(kspB[c], device)
                img_c = sp.nufft_adjoint(ksp_c, coordB, oshape=img_shape)
                img = img + sp.to_device(xp.abs(img_c ** 2), -1)
                pbarOuter.set_postfix(time=(time.time() - timeI) / 60)
            img = img ** 0.5
        img_final.append(img)
    timeFinish = time.time()
    logging.info("Recon Finished in: {} min...".format((timeFinish - timeStart) / 60))
    del img_c, ksp_c, dcf, coord, ksp
    img_final = sp.to_device(img_final)
    # plt.ImagePlot(np.stack(img_final))
    img_final = np.stack(img_final)
    return img_final


if __name__ == "__main__":
    # IO parameters
    parser = argparse.ArgumentParser(description="Gated recon.")
    parser.add_argument("ksp_file", type=str, help="k-space file.")
    parser.add_argument("coord_file", type=str, help="trajectory file.")
    parser.add_argument("dcf_file", type=str, help="dcf file.")
    parser.add_argument("resp_file", type=str, help="resp. waveform file.")
    parser.add_argument("img_file", type=str, help="img out filepath.")
    parser.add_argument("--device", type=int, default=-1, help="Computing device.")
    parser.add_argument(
        "--gating_type", type=str, default="none", help="Gating Type. Options are 'none', 'hard','soft'",
    )
    parser.add_argument(
        "--gating_thresh", type=float, default=50, help="Gating Threshold. Options range from 0.0 to 1.0",
    )
    parser.add_argument("--gating_weight", type=float, default=1.0, help="Gating weight decay for soft threshold.")
    args = parser.parse_args()

    # Read in data
    ksp = np.load(args.ksp_file)
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)
    resp = np.load(args.resp_file)
    img = gatedRecon(
        ksp,
        coord,
        dcf,
        resp,
        gating_type=args.gating_type,
        gating_thresh=args.gating_thresh,
        gating_weight=args.gating_weight,
        device=args.device,
    )
    print("writing data...")
    img = sp.resize(np.abs(img), (256, 256, 256))
    img = nib.Nifti1Image(img, np.eye(4))
    nib.save(img, args.img_file)
