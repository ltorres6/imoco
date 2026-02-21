import copy
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import sigpy.mri as mr
from tqdm import trange

from imoco.utils.linops import DLD, NFTs, TVt_prox


def _plot_losses(loss_file, diagnostics_dir, name):
    loss = []
    with open(loss_file, "r") as f:
        for row in f:
            loss.append(float(row))
    plt.plot(loss, color="g", label="File Data")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss", fontsize=20)
    plt.legend()
    plt.savefig(os.path.join(diagnostics_dir, f"{name}.png"))
    plt.close()


def _save_slice(img, save_dir):
    img_save = np.squeeze(np.abs(sp.to_device(img[:, img.shape[1] // 2, :])))
    imgShapeString = "_".join(map(str, img_save.shape[::-1])) + "Shape.dat"
    with open(os.path.join(save_dir, "iter_slices" + imgShapeString), "ab") as f:
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
    outer_iter=20,
    device=0,
    tv_device=-1,
    sigma=0.4,
    tau=0.4,
):
    """XD-GRASP reconstruction (eXtra-Dimensional Golden-angle RAdial Sparse Parallel MRI).

    Performs motion-resolved compressed sensing reconstruction with temporal
    total variation regularization.

    Reference: Section II-F of the JMRI paper.

    Args:
        ksp_in (list): Binned k-space data, list of length n_bins.
        coord_in (list): Binned coordinates, list of length n_bins.
        dcf_in (list): Binned density compensation, list of length n_bins.
        diagnostics_base_path (str): Base path for diagnostics directory.
        res_scale (float): Resolution scale factor (0-1).
        lambda_tv (float): Temporal total variation regularization weight.
        inner_iter (int): Number of inner iterations (unused, kept for API compat).
        outer_iter (int): Number of outer iterations.
        device (int): Computing device (-1 for CPU, >=0 for GPU).
        tv_device (int): Device for TV proximal operator.
        sigma (float): Primal-dual step size (primal).
        tau (float): Primal-dual step size (dual).

    Returns:
        ndarray: Motion-resolved images of shape (n_bins, Nx, Ny, Nz).
    """
    timeStart = time.time()
    sp.Device(device).use()
    if device >= 0:
        logging.debug("Using GPU...")
    else:
        logging.debug("Using CPU...")

    ksp = copy.deepcopy(ksp_in)
    coord = copy.deepcopy(coord_in)
    dcf = copy.deepcopy(dcf_in)

    logging.debug("Kspace Shape: {}...".format(ksp[0].shape))
    logging.debug("trajectory Shape: {}...".format(coord[0].shape))
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

    diagnostics_dir = os.path.join(str(Path(diagnostics_base_path)), "diagnostics")
    Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)

    logging.info("Running Jsense calibration...")
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
        W = sp.linop.Multiply(
            (nCoils, dcf[ii].shape[0], nReadouts),
            dcf[ii],
        )
        FTSs = W * FTs * S
        PFTSs.append(FTSs)

    logging.debug("Computing Preconditioner...")
    timeI = time.time()
    L = 0
    for p in range(nPhases):
        L += np.sum(np.abs(PFTSs[p].H * PFTSs[p] * np.complex64(np.ones(tshape))))
    L = L / (np.prod(tshape) * nPhases)
    timeF = time.time()
    logging.debug("Preconditioner Value: {}".format(L))
    logging.debug("Time for preconditioner: {} seconds.".format(timeF - timeI))

    for p in range(nPhases):
        for c in range(nCoils):
            ksp[p][c] = ksp[p][c] * dcf[p]

    img = np.zeros((nPhases,) + tshape, dtype=np.complex64)
    Y = [np.zeros_like(k) for k in ksp]
    img_0 = np.zeros_like(img)
    logging.info("Running XD-Grasp")
    pbarOuter = trange(outer_iter, leave=True, ncols=80)
    cost_loss = []
    for ii in pbarOuter:
        timeI = time.time()
        pbarOuter.set_description(f"XD-Grasp Iter {ii}")
        _save_slice(img[0], diagnostics_dir)
        for p in range(nPhases):
            Y[p] = (Y[p] + sigma * (1 / L * PFTSs[p] * img[p] - ksp[p])) / (1 + sigma)
            img[p] = img[p] - tau * PFTSs[p].H * Y[p]
        img = np.complex64(TVt_prox(img, lambda_tv))
        timeF = time.time()
        pbarOuter.set_postfix(
            loss=np.linalg.norm(img - img_0) / np.linalg.norm(img), time=timeF - timeI
        )
        cost_loss.append(np.linalg.norm(img - img_0) / np.linalg.norm(img))
        img_0 = img.copy()

    timeFinish = time.time()
    cost_loss = np.array(cost_loss)
    loss_name = f"xdgrasp_loss_lambda{lambda_tv}_res{res_scale}"
    np.savetxt(os.path.join(diagnostics_dir, loss_name + ".txt"), cost_loss)
    _plot_losses(os.path.join(diagnostics_dir, loss_name + ".txt"), diagnostics_dir, loss_name)
    logging.info(f"XDGrasp Recon Finished in: {(timeFinish - timeStart) / 60} min...")
    img = sp.to_device(img)
    return img
