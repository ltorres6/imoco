"""Gridded motion-resolved reconstruction (experimental).

This module provides a simple gridded (non-iterative) motion-resolved
reconstruction. It was not used in the published paper but can be
useful for quick visualization of motion states.
"""

import copy
import logging
import time

import numpy as np
import sigpy as sp
from tqdm import trange


def griddedRecon(ksp_in, coord_in, dcf_in, n_bins, device=0):
    """Gridded (non-iterative) motion-resolved reconstruction.

    Performs a coil-by-coil NUFFT adjoint for each motion bin.

    Args:
        ksp_in (list): Binned k-space data, list of length n_bins.
        coord_in (list): Binned coordinates, list of length n_bins.
        dcf_in (list): Binned density compensation, list of length n_bins.
        n_bins (int): Number of motion bins.
        device (int): Computing device (-1 for CPU, >=0 for GPU).

    Returns:
        ndarray: Motion-resolved images of shape (n_bins, Nx, Ny, Nz).
    """
    timeStart = time.time()
    sp.Device(device).use()
    xp = sp.Device(device).xp
    if device >= 0:
        logging.info("Using GPU...")
    else:
        logging.info("Using CPU...")

    ksp = copy.deepcopy(ksp_in)
    coord = copy.deepcopy(coord_in)
    dcf = copy.deepcopy(dcf_in)
    nCoils, nSpokes, nReadouts = ksp[0].shape

    logging.info(f"Running Gridding Recon For {n_bins} Motion States")
    img_shape = sp.estimate_shape(coord[0])

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
    img_final = np.stack(img_final)
    return img_final
