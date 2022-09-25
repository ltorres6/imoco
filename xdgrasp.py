import argparse
import sigpy as sp
import numpy as np
import sigpy.mri as mr
from imoco_e import cfl, ext
from imoco_e.linop_e import NFTs
from tqdm import trange
import time
import logging
import os
from pathlib import Path
import copy


def save_slice(img, save_dir):
    img_save = np.squeeze(np.abs(sp.to_device(img[:, img.shape[1] // 2, :])))
    imgShapeString = "_".join(map(str, img_save.shape[::-1])) + "Shape.dat"
    with open(save_dir + "iter_slices" + imgShapeString, "ab") as f:
        f.write(img_save.tobytes())
    del img_save


def xdgrasp(
    ksp_in,
    coord_in,
    dcf_in,
    diagnostics_base_path,
    res_scale=1.0,
    lambda_tv=0.05,
    inner_iter=10,
    outer_iter=25,
    device=0,
    tv_device=-1,
):
    timeStart = time.time()
    sp.Device(device).use()
    if device >= 0:
        logging.debug("Using GPU...")
    else:
        logging.debug("Using CPU...")
    save_iter_slice = True

    # Copy input data
    ksp = copy.deepcopy(ksp_in)
    coord = copy.deepcopy(coord_in)
    dcf = copy.deepcopy(dcf_in)

    # As lists, list has len of n_bins
    # (n_bins, n_coils, n_projections, n_readouts)
    logging.debug("Kspace Shape: {}...".format(ksp[0].shape))
    # (n_bins, n_projections, n_readouts, n_dim)
    logging.debug("trajectory Shape: {}...".format(coord[0].shape))
    # (n_bins, n_projections, n_readouts)
    logging.debug("DCF Shape: {}....".format(dcf[0].shape))

    nf_arr = np.sqrt(np.sum(coord[0][0, :, :] ** 2, axis=1))
    nReadouts = np.sum(nf_arr < np.max(nf_arr) * res_scale)
    del nf_arr

    ksp = [bin_data[..., :nReadouts] for bin_data in ksp]
    coord = [bin_data[:, :nReadouts, :] for bin_data in coord]
    dcf = [bin_data[..., :nReadouts] for bin_data in dcf]

    logging.debug("Image Shape Estimate: {}".format(sp.estimate_shape(coord[0])))
    nPhases = len(ksp)
    nCoils, nSpokes, nReadouts = ksp[0].shape

    # Make a dedicated diagnostics directory
    diagnostics_dir = "{}/diagnostics/".format(Path(diagnostics_base_path))
    Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)

    # calibration
    logging.info("Running Jsense calibration...")
    # Map for each Motion Phase?
    mps = mr.app.JsenseRecon(
        ksp[0],
        coord=coord[0],
        weights=dcf[0] ** 2,
        mps_ker_width=12,
        ksp_calib_width=32,
        lamda=0,
        device=device,
        max_iter=10,
        max_inner_iter=10,
        show_pbar=False,
    ).run()
    mps = sp.to_device(mps)
    if nCoils <= 1:
        mps = np.ones_like(mps)
    tshape = mps.shape[1:]
    S = sp.linop.Multiply(tshape, mps)
    del mps

    logging.debug("Image Shape: {}....".format(tshape))
    logging.debug("Computing Linops...")
    PFTSs = []
    for ii in range(nPhases):
        FTs = NFTs((nCoils,) + tshape, coord[ii], device=sp.Device(device))
        W = sp.linop.Multiply((nCoils, dcf[ii].shape[0], nReadouts,), dcf[ii],)
        FTSs = W * FTs * S
        PFTSs.append(FTSs)

    logging.debug("Computing Preconditioner...")
    timeI = time.time()
    L = 0
    for p in range(nPhases):
        L += np.sum(np.abs(PFTSs[p].H * PFTSs[p] * np.complex64(np.ones(tshape))))
        # L = np.mean(np.abs(L))
    L = L / (np.prod(tshape) * nPhases)
    timeF = time.time()
    logging.debug("Preconditioner Value: {}".format(L))
    logging.debug("Time for preconditioner: {} seconds.".format(timeF - timeI))

    # reconstruction
    # Apply density compensation
    for p in range(nPhases):
        for c in range(nCoils):
            ksp[p][c] = ksp[p][c] * dcf[p]
    # dcf = dcf[:, xp.newaxis, ...]
    # plt.ImagePlot(ksp)
    # ksp = ksp * dcf
    # plt.ImagePlot(ksp)
    img = np.zeros((nPhases,) + tshape, dtype=np.complex64)
    Y = [np.zeros_like(k) for k in ksp]
    img_0 = np.zeros_like(img)
    tau = 0.4
    sigma = 0.4
    logging.info("Running XD-Grasp")
    pbarOuter = trange(outer_iter, leave=True, ncols=80)
    cost_loss = []
    for ii in pbarOuter:
        timeI = time.time()
        pbarOuter.set_description(f"XD-Grasp Iter {ii}")
        # Save a slice for diagnostics
        if save_iter_slice:
            save_slice(img[0], diagnostics_dir)
        for p in range(nPhases):
            Y[p] = (Y[p] + sigma * (1 / L * PFTSs[p] * img[p] - ksp[p])) / (1 + sigma)
            img[p] = img[p] - tau * PFTSs[p].H * Y[p]
        img = np.complex64(ext.TVt_prox(img, lambda_tv))
        timeF = time.time()
        pbarOuter.set_postfix(loss=np.linalg.norm(img - img_0) / np.linalg.norm(img), time=timeF - timeI)
        cost_loss.append(np.linalg.norm(img - img_0) / np.linalg.norm(img))
        img_0 = img.copy()

    timeFinish = time.time()
    cost_loss = np.array(cost_loss)
    np.savetxt(
        os.path.join(diagnostics_dir, "xdgrasp_loss_lambda" + str(lambda_tv) + "_res" + str(res_scale) + ".txt",), cost_loss,
    )
    logging.info(f"XDGrasp Recon Finished in: {(timeFinish - timeStart) / 60} min...")
    img = sp.to_device(img)
    return img


if __name__ == "__main__":
    # IO parameters
    parser = argparse.ArgumentParser(description="XD-GRASP recon.")
    parser.add_argument("ksp_file", type=str, help="k-space file.")
    parser.add_argument("coord_file", type=str, help="trajectory file.")
    parser.add_argument("dcf_file", type=str, help="dcf file.")
    parser.add_argument("img_file", type=str, help="img out file.")
    parser.add_argument("--res_scale", type=float, default=1.0, help="scale of resolution 0-1")
    parser.add_argument("--lambda_tv", type=float, default=2e-2, help="TV regularization, 0.05")
    parser.add_argument("--inner_iter", type=int, default=10, help="Num of inner Iterations.")
    parser.add_argument("--outer_iter", type=int, default=20, help="Num of outer Iterations.")
    parser.add_argument("--device", type=int, default=0, help="Computing device.")
    args = parser.parse_args()

    # Read in data
    ksp = np.load(args.ksp_file)
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)

    img = xdgrasp(ksp, coord, dcf, args.res_scale, args.lambda_tv, args.inner_iter, args.outer_iter, args.device,)
    print("writing data...")
    # plt.ImagePlot(img)
    cfl.write_cfl(args.img_file, img)
