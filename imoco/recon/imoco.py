import copy
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import sigpy as sp
import sigpy.mri as mr
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import transform
from skimage.morphology import ball
from skimage.segmentation import (
    inverse_gaussian_gradient,
    morphological_geodesic_active_contour,
)
from scipy import ndimage
from tqdm import trange

from imoco.registration.ants import ANTsReg
from imoco.registration.interpolation import M_scale, interp_op
from imoco.utils.linops import DLD, NFTs
from imoco.utils.normalize import normalize


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


def _estimate_mask(img_in):
    img = gaussian_filter(img_in, [1] * 3)
    gradient = inverse_gaussian_gradient(img)
    bg_mask = morphological_geodesic_active_contour(
        gradient, iterations=60, init_level_set=np.ones_like(img), smoothing=1, balloon=-1,
    )
    strel = ball(2, dtype=np.uint8)
    bg_mask = ndimage.binary_closing(bg_mask, structure=strel, iterations=10)
    bg_mask = ndimage.binary_fill_holes(bg_mask, structure=strel)
    all_zeros = not np.any(bg_mask)
    if all_zeros:
        logging.info("Setting mask to ones!")
        bg_mask = np.ones_like(img)
    return bg_mask


def _save_slice(img, save_dir):
    img_save = np.squeeze(np.abs(sp.to_device(img[:, img.shape[1] // 2, :])))
    imgShapeString = "_".join(map(str, img_save.shape[::-1])) + "Shape.dat"
    with open(os.path.join(save_dir, "iter_slices" + imgShapeString), "ab") as f:
        f.write(img_save.tobytes())
    del img_save


def _save_mask_diagnostic(img, save_dir, save_name):
    imc = normalize(img[:, img.shape[1] // 2, :], 0, 255)
    imc = Image.fromarray(transform.resize(imc, (256, 256)))
    imc = imc.convert("1")
    imc.save(os.path.join(save_dir, save_name + "_coronal.jpg"))

    ims = normalize(img[:, :, img.shape[2] // 2], 0, 255)
    ims = Image.fromarray(transform.resize(ims, (256, 256)))
    ims = ims.convert("1")
    ims.save(os.path.join(save_dir, save_name + "_saggital.jpg"))

    ima = normalize(img[img.shape[0] // 2, :, :], 0, 255)
    ima = Image.fromarray(transform.resize(ima, (256, 256)))
    ima = ima.convert("1")
    ima.save(os.path.join(save_dir, save_name + "_axial.jpg"))


def imoco(
    ksp_in,
    coord_in,
    dcf_in,
    mrimg,
    diagnostics_base_path,
    res_scale=1.0,
    mr_scale=0.75,
    lambda_tv=0.05,
    inner_iter=15,
    outer_iter=25,
    device=-1,
    nRef=-1,
    reg_flag=0,
    diffusion_reg=0.1,
    sigma=0.4,
    tau=0.4,
    resolution=[1.25, 1.25, 1.25],
):
    """Iterative Motion Compensation (iMoCo) reconstruction.

    Jointly estimates a motion-compensated image by alternating between
    NUFFT-based reconstruction and ANTs-based deformable registration.

    Reference: Section II-G of the JMRI paper.

    Args:
        ksp_in (list): Binned k-space data, list of length n_bins.
        coord_in (list): Binned coordinates, list of length n_bins.
        dcf_in (list): Binned density compensation, list of length n_bins.
        mrimg (ndarray): Low-resolution motion-resolved images for
            initial registration, shape (n_bins, Nx, Ny, Nz).
        diagnostics_base_path (str): Base path for diagnostics directory.
        res_scale (float): Resolution scale factor (0-1).
        mr_scale (float): Resolution scale of the low-res input images.
        lambda_tv (float): Total variation regularization weight.
        inner_iter (int): Number of inner iterations (unused, kept for API compat).
        outer_iter (int): Number of outer primal-dual iterations.
        device (int): Computing device (-1 for CPU, >=0 for GPU).
        nRef (int): Reference frame index for registration.
        reg_flag (int): 1 to compute registration, 0 to load from disk.
        diffusion_reg (float): Diffusion regularization for ANTs SyN.
        sigma (float): Primal-dual step size (primal).
        tau (float): Primal-dual step size (dual).
        resolution (list): Voxel resolution in mm, length 3.

    Returns:
        ndarray: Motion-compensated 3D image.
    """
    timeStart = time.time()
    sp.Device(device).use()
    if device >= 0:
        logging.info("Using GPU...")
    else:
        logging.info("Using CPU...")

    ksp = copy.deepcopy(ksp_in)
    coord = copy.deepcopy(coord_in)
    dcf = copy.deepcopy(dcf_in)

    logging.info("Kspace Shape: {}...".format(ksp[0].shape))
    logging.info("trajectory Shape: {}...".format(coord[0].shape))
    logging.info("DCF Shape: {}....".format(dcf[0].shape))

    nf_arr = np.sqrt(np.sum(coord[0][0, :, :] ** 2, axis=1))
    nReadouts = np.sum(nf_arr < np.max(nf_arr) * res_scale)
    del nf_arr

    ksp = [bin_data[..., :nReadouts] for bin_data in ksp]
    coord = [bin_data[:, :nReadouts, :] for bin_data in coord]
    dcf = [bin_data[..., :nReadouts] for bin_data in dcf]

    logging.info("Image Shape Estimate: {}".format(sp.estimate_shape(coord[0])))
    nPhases = len(ksp)
    nCoils, nSpokes, nReadouts = ksp[0].shape

    diagnostics_dir = os.path.join(str(Path(diagnostics_base_path)), "diagnostics")
    Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)

    logging.info("Running calibration...")
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

    logging.info("Registration...")
    if reg_flag == 1:
        timei = time.time()
        vox_res = [r / mr_scale for r in resolution]
        M_fields = []
        iM_fields = []
        fixed_mask = _estimate_mask(normalize(np.abs(mrimg[nRef]), 0, 1))
        _save_mask_diagnostic(fixed_mask.astype(float), diagnostics_dir, "fixed_mask")
        pbar = trange(nPhases, leave=True, ncols=80)
        for ii in pbar:
            pbar.set_description(f"Registering Frame: {ii}...")
            if ii == nRef:
                M_field = np.zeros(mrimg.shape[1:] + (3,))
                iM_field = np.zeros(mrimg.shape[1:] + (3,))
            else:
                moving_mask = _estimate_mask(normalize(np.abs(mrimg[ii]), 0, 1))
                _save_mask_diagnostic(moving_mask.astype(float), diagnostics_dir, f"moving_mask_{ii}")
                M_field, iM_field = ANTsReg(
                    normalize(gaussian_filter(np.abs(mrimg[nRef]), 0.5), 0, 1),
                    normalize(gaussian_filter(np.abs(mrimg[ii]), 0.5), 0, 1),
                    fixed_mask,
                    moving_mask,
                    vox_res=vox_res,
                    frame=ii,
                    fluid=0.0,
                    diffusion=diffusion_reg,
                    diagnostics_dir=diagnostics_dir,
                )

            M_fields.append(M_field)
            iM_fields.append(iM_field)

        logging.info("Motion Field scaling...")
        M_fields = [M_scale(M, tshape) for M in M_fields]
        iM_fields = [M_scale(iM, tshape) for iM in iM_fields]

        logging.info("Saving Motion Fields as nii...")
        tmp = np.asarray(M_fields)
        tmp = np.moveaxis(tmp, 0, -1)
        tmp = np.transpose(tmp, (2, 1, 0, 3, 4))
        tmp = np.flip(tmp, (0, 1, 2))
        tmp = nib.Nifti1Image(tmp, np.eye(4))
        nib.save(tmp, os.path.join(diagnostics_dir, "M_mr.nii.gz"))

        tmp = np.asarray(iM_fields)
        tmp = np.moveaxis(tmp, 0, -1)
        tmp = np.transpose(tmp, (2, 1, 0, 3, 4))
        tmp = np.flip(tmp, (0, 1, 2))
        tmp = nib.Nifti1Image(tmp, np.eye(4))
        nib.save(tmp, os.path.join(diagnostics_dir, "iM_mr.nii.gz"))
        timeF = (time.time() - timei) / 60
        logging.info("Finished Registration in {} hours".format(timeF / 60))
        del tmp
    else:
        logging.info("Reading Motion Fields from disk...")
        M_fields = nib.load(os.path.join(diagnostics_dir, "M_mr.nii.gz")).get_fdata()
        M_fields = np.flip(M_fields, (0, 1, 2))
        M_fields = np.transpose(M_fields, (2, 1, 0, 3, 4))
        M_fields = np.moveaxis(M_fields, -1, 0)

        iM_fields = nib.load(os.path.join(diagnostics_dir, "iM_mr.nii.gz")).get_fdata()
        iM_fields = np.flip(iM_fields, (0, 1, 2))
        iM_fields = np.transpose(iM_fields, (2, 1, 0, 3, 4))
        iM_fields = np.moveaxis(iM_fields, -1, 0)
        M_fields = [M_fields[p] for p in range(M_fields.shape[0])]
        iM_fields = [iM_fields[p] for p in range(iM_fields.shape[0])]

    logging.info("Prep...")
    PFTSMs = []
    for p in range(nPhases):
        FT = NFTs((nCoils,) + tshape, coord[p], device=sp.Device(device))
        M = interp_op(tshape, iM_fields[p], M_fields[p])
        M = DLD(M, device=sp.Device(device))
        W = sp.linop.Multiply(
            (nCoils, dcf[p].shape[0], nReadouts),
            dcf[p],
        )
        FTSM = W * FT * S * M
        PFTSMs.append(FTSM)

    logging.info("Computing Preconditioner...")
    L = 0
    for p in range(nPhases):
        L += PFTSMs[p].H * PFTSMs[p] * np.complex64(np.ones(tshape))
    L = np.sum(np.abs(L))
    L = L / (np.prod(tshape) * nPhases)
    logging.info("Preconditioner Value: {}".format(L))

    for p in range(nPhases):
        for c in range(nCoils):
            ksp[p][c] = ksp[p][c] * dcf[p]

    TV = sp.linop.FiniteDifference(tshape, axes=(0, 1, 2))

    logging.info("Running iMoCo Recon...")
    alpha = 0
    for p in range(nPhases):
        alpha += PFTSMs[p].H * ksp[p]
    alpha = np.max(np.abs(alpha))
    logging.info("alpha:{}".format(alpha))

    img = np.zeros(tshape, dtype=np.complex64)
    Y = [np.zeros_like(k) for k in ksp]
    img_0 = np.zeros_like(img)
    q = np.zeros((3,) + tshape, dtype=np.complex64)
    pbarOuter = trange(outer_iter, leave=True, ncols=80)
    loss = []
    for ii in pbarOuter:
        timeI = time.time()
        pbarOuter.set_description(f"iMoco Iter: {ii}")
        _save_slice(img, diagnostics_dir)
        accum = 0
        for p in range(nPhases):
            Y[p] = (Y[p] + sigma * (PFTSMs[p] * img - ksp[p])) / (1 + sigma)
            accum += PFTSMs[p].H * Y[p]
        q = q + sigma * TV * img
        q = q / (np.maximum(np.abs(q), alpha) / alpha)
        img = img - tau * (1 / L * accum + lambda_tv * TV.H * q)
        timeF = time.time()
        pbarOuter.set_postfix(
            loss=np.linalg.norm(img - img_0) / np.linalg.norm(img), time=timeF - timeI
        )
        loss.append(np.linalg.norm(img - img_0) / np.linalg.norm(img))
        img_0 = img.copy()

    img = np.transpose(img, (2, 1, 0))
    img = np.flip(img, (0, 1, 2))
    loss = np.array(loss)
    loss_name = f"imoco_loss_refFrame{nRef}_lambda{lambda_tv}_res{res_scale}"
    np.savetxt(os.path.join(diagnostics_dir, loss_name + ".txt"), loss)
    _plot_losses(os.path.join(diagnostics_dir, loss_name + ".txt"), diagnostics_dir, loss_name)
    timeFinish = time.time()
    logging.info("iMoco Recon Finished in: {} hrs...".format((timeFinish - timeStart) / 3600))
    return img
