import copy
import logging
import time

import numpy as np
import sigpy as sp
from tqdm import trange


def gatingWeights(resp, gating_type="hard", percentile=25, decay=1, flip=False):
    """Compute respiratory gating weights.

    Reference: Section II-E of the JMRI paper.

    Args:
        resp (ndarray): Respiratory signal of length num_spokes.
        gating_type (str): "hard" for binary gating, "soft" for exponential weighting.
        percentile (float): Percentile threshold for gating (0-100).
        decay (float): Exponential decay constant for soft gating.
        flip (bool): If True, invert the respiratory signal.

    Returns:
        ndarray: Gating weights of length num_spokes. For hard gating, values
            are 0 or 1. For soft gating, values are in (0, 1].
    """
    margin = 5
    sigma = 1.4628 * np.median(np.abs(resp - np.median(resp)))
    resp = -1 * (resp - np.median(resp)) / sigma
    thresh_extreme = [np.percentile(resp, margin), np.percentile(resp, 100 - margin)]
    idx = (resp >= thresh_extreme[0]) & (resp < thresh_extreme[1])
    idx_exclude = (resp < thresh_extreme[0]) & (resp >= thresh_extreme[1])
    resp_temp = resp[idx]
    thresh = np.percentile(resp_temp, percentile)
    if flip:
        resp *= -1
    if gating_type == "hard":
        W = np.where(resp < thresh, 1, 0)
        W[idx_exclude] = 0
        return W
    elif gating_type == "soft":
        W = np.exp(-decay * np.maximum((resp - thresh), 0))
        W[idx_exclude] = 0
        return W


def gatedRecon(
    ksp_in,
    coord_in,
    dcf_in,
    resp_in,
    gating_type="none",
    gating_thresh=50,
    gating_weight=1.0,
    device=0,
    flip=False,
):
    """Gated NUFFT reconstruction (no-gating, hard-gating, or soft-gating).

    Performs a coil-by-coil NUFFT adjoint reconstruction with optional
    respiratory gating.

    Reference: Section II-E of the JMRI paper.

    Args:
        ksp_in (ndarray): K-space data of shape (C, num_spokes, num_ro).
        coord_in (ndarray): Coordinates of shape (num_spokes, num_ro, D).
        dcf_in (ndarray): Density compensation of shape (num_spokes, num_ro).
        resp_in (ndarray): Respiratory signal of length num_spokes.
        gating_type (str): "none", "hard", or "soft".
        gating_thresh (float): Gating threshold percentile.
        gating_weight (float): Exponential decay for soft gating.
        device (int): Computing device (-1 for CPU, >=0 for GPU).
        flip (bool): If True, invert the respiratory signal before gating.

    Returns:
        ndarray: Reconstructed 3D image.
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
    dcf = copy.deepcopy(dcf_in**2)
    resp = copy.deepcopy(resp_in)

    logging.info("Kspace Shape: {}...".format(ksp.shape))
    logging.info("trajectory Shape: {}...".format(coord.shape))
    logging.info("DCF Shape: {}....".format(dcf.shape))
    logging.info("Image Shape Estimate: {}".format(sp.estimate_shape(coord)))
    nCoils, nSpokes, nReadouts = ksp.shape

    img_shape = sp.estimate_shape(coord)
    logging.info(
        "(Complex) Image Size Estimate: {}MB....".format(
            np.prod(img_shape) * ksp.itemsize // (1024 * 1024)
        )
    )
    logging.info("Running Gated Recon Type:{}".format(gating_type))

    if gating_type == "none":
        pass
    elif gating_type == "hard":
        W = gatingWeights(
            resp, gating_type="hard", percentile=gating_thresh, decay=gating_weight, flip=flip,
        )
        idx = W == 1
        ksp = ksp[:, idx]
        coord = coord[idx]
        dcf = dcf[idx]
        del W, idx
    elif gating_type == "soft":
        W = gatingWeights(
            resp, gating_type="soft", percentile=gating_thresh, decay=gating_weight, flip=flip,
        )
        W_correct = np.broadcast_to(W[..., None], W.shape + (ksp.shape[2],))
        dcf = dcf * W_correct
        del W, W_correct
    else:
        raise ValueError("Unknown Gating Type.")

    pbarOuter = trange(nCoils, leave=True, ncols=80)
    coord = sp.to_device(coord, device)
    ksp = ksp * dcf
    with sp.Device(device):
        img = 0
        for c in pbarOuter:
            timeI = time.time()
            pbarOuter.set_description(f"{gating_type}Recon - Coil: {c}")
            ksp_c = sp.to_device(ksp[c], device)
            img_c = sp.nufft_adjoint(ksp_c, coord, oshape=img_shape)
            img = img + sp.to_device(img_c * xp.conj(img_c), -1)
            pbarOuter.set_postfix(time=(time.time() - timeI) / 60)
        img = np.abs(np.sqrt(img))

    timeFinish = time.time()
    logging.info("Recon Finished in: {} min...".format((timeFinish - timeStart) / 60))
    del img_c, ksp_c, dcf, coord, ksp
    img = np.transpose(img, (2, 1, 0))
    img = np.flip(img, (0, 1, 2))
    return img
