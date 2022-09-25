import argparse
import copy
import logging
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as plt2
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import transform
from skimage.morphology import ball
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour
from tqdm import trange

from imoco_e import cfl, reg
from imoco_e.linop_e import DLD, NFTs
from normalize import normalize


def estimate_mask(img_in):
    img = gaussian_filter(img_in, [1] * 3)
    gradient = inverse_gaussian_gradient(img)
    bg_mask = morphological_geodesic_active_contour(
        gradient, iterations=60, init_level_set=np.ones_like(img), smoothing=1, balloon=-1
    )
    strel = ball(2, dtype=np.uint8)
    bg_mask = ndimage.binary_closing(bg_mask, structure=strel, iterations=10)
    bg_mask = ndimage.binary_fill_holes(bg_mask, structure=strel)
    bg_mask = bg_mask
    # plt2.ImagePlot(bg_mask)
    all_zeros = not np.any(bg_mask)
    if all_zeros:
        logging.info("Setting mask to ones!")
        bg_mask = np.ones_like(img)
        bg_mask
    # plt2.ImagePlot(bg_mask)
    return bg_mask


def save_mask_diagnostic(img, save_dir, save_name):
    # add a zero in the middle in case it's all ones so there is a display range
    # img[img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2] = 0
    imc = normalize(img[:, img.shape[1] // 2, :], 0, 255)
    imc = Image.fromarray(transform.resize(imc, (256, 256)))
    imc = imc.convert("1")
    imc.save(save_dir + save_name + "_coronal.jpg")

    ims = normalize(img[:, :, img.shape[2] // 2], 0, 255)
    ims = Image.fromarray(transform.resize(ims, (256, 256)))
    ims = ims.convert("1")
    ims.save(save_dir + save_name + "_saggital.jpg")

    ima = normalize(img[img.shape[0] // 2, :, :], 0, 255)
    ima = Image.fromarray(transform.resize(ima, (256, 256)))
    ima = ima.convert("1")
    ima.save(save_dir + save_name + "_axial.jpg")


def save_slice(img, save_dir):
    img_save = np.squeeze(np.abs(sp.to_device(img[:, img.shape[1] // 2, :])))
    imgShapeString = "_".join(map(str, img_save.shape[::-1])) + "Shape.dat"
    with open(save_dir + "iter_slices" + imgShapeString, "ab") as f:
        f.write(img_save.tobytes())
    del img_save


def imoco(
    ksp_in,
    coord_in,
    dcf_in,
    mrimg,
    diagnostics_base_path,
    res_scale=1.0,
    mr_scale=0.75,
    lambda_tv=0.05,
    inner_iter=10,
    outer_iter=20,
    device=-1,
    nRef=0,
    reg_flag=0,
    diffusion_reg=0.1
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
    # print(diagnostics_dir)
    Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)

    # calibration
    logging.info("Running calibration...")
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

    logging.info("Registration...")
    vox_res = [0.7 / mr_scale] * 3
    M_fields = []
    iM_fields = []
    fixed_mask = estimate_mask(normalize(np.abs(mrimg[nRef]), 0, 1))
    save_mask_diagnostic(fixed_mask.astype(float), diagnostics_dir, "fixed_mask")
    if reg_flag == 1:
        pbar = trange(nPhases, leave=True, ncols=80)
        for ii in pbar:
            # for ii in [3]:
            pbar.set_description(f"Registering Frame: {ii}...")
            if ii == nRef:
                M_field = np.zeros(mrimg.shape[1:] + (3,))
                iM_field = np.zeros(mrimg.shape[1:] + (3,))
            else:
                moving_mask = estimate_mask(normalize(np.abs(mrimg[ii]), 0, 1))
                save_mask_diagnostic(moving_mask.astype(float), diagnostics_dir, f"moving_mask_{ii}")
                M_field, iM_field = reg.ANTsReg(
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

        # M_fields = np.asarray(M_fields)
        # iM_fields = np.asarray(iM_fields)
        # np.save(diagnostics_dir + "/M_mr.npy", M_fields)
        # np.save(diagnostics_dir + "/iM_mr.npy", iM_fields)
        # M_fields = np.load(diagnostics_dir + "/M_mr.npy")
        # iM_fields = np.load(diagnostics_dir + "/iM_mr.npy")

        # iM_fields = [iM_fields[i] for i in range(iM_fields.shape[0])]
        # M_fields = [M_fields[i] for i in range(M_fields.shape[0])]

        # Scale Motion field (multply values by scale and expand by scale)
        logging.debug("Motion Field scaling...")
        M_fields = [reg.M_scale(M, tshape) for M in M_fields]
        iM_fields = [reg.M_scale(M, tshape) for M in iM_fields]

        logging.debug("Saving Motion Fields as nii...")
        tmp = np.asarray(M_fields)
        # print(tmp.shape)
        tmp = np.moveaxis(tmp, 0, -1)
        tmp = np.transpose(tmp, (2, 1, 0, 3, 4))
        tmp = np.flip(tmp, (0, 1, 2))
        tmp = nib.Nifti1Image(tmp, np.eye(4))
        nib.save(tmp, diagnostics_dir + "/M_mr.nii.gz")

        tmp = np.asarray(iM_fields)
        # print(tmp.shape)
        tmp = np.moveaxis(tmp, 0, -1)
        tmp = np.transpose(tmp, (2, 1, 0, 3, 4))
        tmp = np.flip(tmp, (0, 1, 2))
        tmp = nib.Nifti1Image(tmp, np.eye(4))
        nib.save(tmp, diagnostics_dir + "/iM_mr.nii.gz")
        del tmp
    else:
        logging.debug("Reading Motion Fields from disk...")
        M_fields = nib.load(diagnostics_dir + "/M_mr.nii.gz").get_fdata()
        M_fields = np.flip(M_fields, (0, 1, 2))
        M_fields = np.transpose(M_fields, (2, 1, 0, 3, 4))
        M_fields = np.moveaxis(M_fields, -1, 0)

        iM_fields = nib.load(diagnostics_dir + "/iM_mr.nii.gz").get_fdata()
        iM_fields = np.flip(iM_fields, (0, 1, 2))
        iM_fields = np.transpose(iM_fields, (2, 1, 0, 3, 4))
        iM_fields = np.moveaxis(iM_fields, -1, 0)
    # Recon
    logging.debug("Prep...")
    PFTSMs = []
    for p in range(nPhases):
        # Is.append(sp.linop.Identity(tshape))
        FT = NFTs((nCoils,) + tshape, coord[p], device=sp.Device(device))
        M = reg.interp_op(tshape, iM_fields[p])
        M = DLD(M, device=sp.Device(device))
        W = sp.linop.Multiply((nCoils, dcf[p].shape[0], nReadouts,), dcf[p],)
        FTSM = W * FT * S * M
        PFTSMs.append(FTSM)
        # FTs.append(FT)
        # Ms.append(M)
        # Ws.append(W)
    # PFTSMs = (
    #     Diags(
    #         PFTSMs,
    #         oshape=(
    #             nPhases,
    #             nCoils,
    #             nSpokes,
    #             nReadouts,
    #         ),
    #         ishape=(nPhases,) + tshape,
    #     )
    #     * Vstacks(Is, ishape=tshape, oshape=(nPhases,) + tshape)
    # )

    logging.debug("Computing Preconditioner...")
    timeI = time.time()
    L = 0
    for p in range(nPhases):
        L += PFTSMs[p].H * PFTSMs[p] * np.complex64(np.ones(tshape))
    L = np.sum(np.abs(L))
    L = L / (np.prod(tshape) * nPhases)
    logging.debug("Preconditioner Value: {}".format(L))

    # Apply density compensation
    for p in range(nPhases):
        for c in range(nCoils):
            ksp[p][c] = ksp[p][c] * dcf[p]

    TV = sp.linop.FiniteDifference(tshape, axes=(0, 1, 2))

    # ADMM
    logging.info("Running iMoCo Recon...")
    # Compute alpha
    alpha = 0
    for p in range(nPhases):
        alpha += PFTSMs[p].H * ksp[p]
    alpha = np.max(np.abs(alpha))

    logging.debug("alpha:{}".format(alpha))
    sigma = 0.4
    tau = 0.4
    img = np.zeros(tshape, dtype=np.complex64)
    Y = [np.zeros_like(k) for k in ksp]
    img_0 = np.zeros_like(img)
    q = np.zeros((3,) + tshape, dtype=np.complex64)
    pbarOuter = trange(outer_iter, leave=True, ncols=80)
    cost_loss = []
    for ii in pbarOuter:
        timeI = time.time()
        pbarOuter.set_description(f"iMoco Iter: {ii}")
        # Save a slice for diagnostics
        if save_iter_slice:
            save_slice(img, diagnostics_dir)
        accum = 0
        for p in range(nPhases):
            Y[p] = (Y[p] + sigma * (PFTSMs[p] * img - ksp[p])) / (1 + sigma)
            accum += PFTSMs[p].H * Y[p]
            timeF = time.time()
        # Prox?
        q = q + sigma * TV * img
        q = q / (np.maximum(np.abs(q), alpha) / alpha)
        img = img - tau * (1 / L * accum + lambda_tv * TV.H * q)
        pbarOuter.set_postfix(loss=np.linalg.norm(img - img_0) / np.linalg.norm(img), time=timeF - timeI)
        cost_loss.append(np.linalg.norm(img - img_0) / np.linalg.norm(img))
        img_0 = img.copy()
    img = np.transpose(img, (2, 1, 0))
    img = np.flip(img, (0, 1, 2))
    cost_loss = np.array(cost_loss)
    np.savetxt(
        os.path.join(
            diagnostics_dir,
            "imoco_loss_refFrame" + str(nRef) + "_lambda" + str(lambda_tv) + "_res" + str(res_scale) + ".txt",
        ),
        cost_loss,
    )
    timeFinish = time.time()
    logging.info("iMoco Recon Finished in: {} hrs...".format((timeFinish - timeStart) / 3600))
    return img


if __name__ == "__main__":
    # IO parameters
    parser = argparse.ArgumentParser(description="imoco recon.")
    parser.add_argument("ksp_file", type=str, help="k-space file.")
    parser.add_argument("coord_file", type=str, help="coordectory file.")
    parser.add_argument("dcf_file", type=str, help="dcf file.")
    parser.add_argument("img_file", type=str, help="img out file.")
    parser.add_argument("--res_scale", type=float, default=1.0, help="scale of resolution 0-1")
    parser.add_argument("--lambda_tv", type=float, default=2e-2, help="TV regularization, 0.05")
    parser.add_argument("--inner_iter", type=int, default=10, help="Num of inner Iterations.")
    parser.add_argument("--outer_iter", type=int, default=20, help="Num of outer Iterations.")
    parser.add_argument("--device", type=int, default=0, help="Computing device.")
    args = parser.parse_args()

    # Read in ksp
    ksp = np.load(args.ksp_file)
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)

    img = imoco(ksp, coord, dcf, args.res_scale, args.lambda_tv, args.inner_iter, args.outer_iter, args.device,)
    print("writing ksp...")
    # plt.ImagePlot(img)
    cfl.write_cfl(args.img_file, img)
