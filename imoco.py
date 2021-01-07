import argparse
import sigpy as sp
import sigpy.mri as mr
import numpy as np
from imoco_e import cfl, reg
from imoco_e.linop_e import NFTs, Diags, DLD, Vstacks
from tqdm import trange
import logging
import time
import os
from scipy.ndimage import median_filter
import nibabel as nib


def imoco(
    ksp,
    coord,
    dcf,
    mrimg,
    diagnosticsPath,
    res_scale=1.0,
    lambda_tv=0.05,
    inner_iter=15,
    outer_iter=20,
    device=-1,
    nRef=-1,
    reg_flag=0,
):
    timeStart = time.time()
    sp.Device(device).use()
    xp = sp.Device(device).xp
    if device >= 0:
        logging.info("Using GPU...")
    else:
        logging.info("Using CPU...")

    logging.info("Kspace Shape: {}...".format(ksp.shape))
    logging.info("trajectory Shape: {}...".format(coord.shape))
    logging.info("DCF Shape: {}....".format(dcf.shape))

    nf_arr = np.sqrt(np.sum(coord[0, 0, :, :] ** 2, axis=1))
    nReadouts = np.sum(nf_arr < np.max(nf_arr) * res_scale)
    del nf_arr

    coord = coord[..., :nReadouts, :]
    ksp = ksp[..., :nReadouts]
    dcf = dcf[..., :nReadouts]

    logging.info("Image Shape Estimate: {}".format(sp.estimate_shape(coord)))
    # nPhases, nEcalib, nCoils, nSpokes, nReadouts, _ = data.shape
    nPhases, nCoils, nSpokes, nReadouts = ksp.shape

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
    ).run()
    mps = sp.to_device(mps)
    if nCoils <= 1:
        mps = np.ones_like(mps)
    tshape = mps.shape[1:]
    S = sp.linop.Multiply(tshape, mps)

    # registration
    # print("Registration...")
    # # Options
    # # Compute Device: -1=CPU or 0=GPU
    # # Demons Force Variation - "passive" , "active", "inverseConsistent" https://arxiv.org/pdf/0909.0928.pdf
    # variant = "active"
    # diffeomorphic = False
    # compositionType = "A"
    # nLevels = 3
    # cThresh = 1e-6
    # max_iter = 4000
    # alpha = 2.0

    # # Gaussian Smoothing Sigmas.
    # diffusionSigmas = 2.0
    # fluidSigmas = 2.0

    # M_fields = []
    # iM_fields = []
    # for ii in range(nPhases):
    #     W, invW = Demons(
    #         mrimg[nRef],
    #         mrimg[ii],
    #         nLevels=nLevels,
    #         diffusionSigmas=diffusionSigmas,
    #         fluidSigmas=fluidSigmas,
    #         max_iter=max_iter,
    #         alpha=alpha,
    #         cThresh=cThresh,
    #         variant=variant,
    #         diffeomorphic=diffeomorphic,
    #         compositionType=compositionType,
    #         device=device,
    #     ).run()
    #     M_fields.append(W)
    #     iM_fields.append(invW)
    # M_fields = np.asarray(M_fields)
    # iM_fields = np.asarray(iM_fields)
    # registration

    logging.info("Registration...")
    M_fields = []
    iM_fields = []
    if reg_flag is 1:
        pbar = trange(nPhases, leave=True)
        for ii in pbar:
            pbar.set_description("Registering Frame # {}...".format(ii))
            M_field, iM_field = reg.ANTsReg(
                median_filter(np.abs(mrimg[nRef]), 3), median_filter(np.abs(mrimg[ii]), 3)
            )
            M_fields.append(M_field)
            iM_fields.append(iM_field)
        M_fields = np.asarray(M_fields)
        iM_fields = np.asarray(iM_fields)
        # np.save(diagnosticsPath + "/M_mr.npy", M_fields)
        # np.save(diagnosticsPath + "/iM_mr.npy", iM_fields)
        # M_fields = np.load(diagnosticsPath + "/M_mr.npy")
        # iM_fields = np.load(diagnosticsPath + "/iM_mr.npy")

        iM_fields = [iM_fields[i] for i in range(iM_fields.shape[0])]
        M_fields = [M_fields[i] for i in range(M_fields.shape[0])]

        # Scale Motion field (multply values by scale and expand by scale)
        logging.info("Motion Field scaling...")
        M_fields = [reg.M_scale(M, tshape) for M in M_fields]
        iM_fields = [reg.M_scale(M, tshape) for M in iM_fields]

        logging.info("Saving Motion Fields as nii...")
        tmp = np.asarray(M_fields)
        print(tmp.shape)
        tmp = np.moveaxis(tmp, 0, -1)
        tmp = np.transpose(tmp, (2, 1, 0, 3, 4))
        tmp = np.flip(tmp, (0, 1, 2))
        tmp = nib.Nifti1Image(tmp, np.eye(4))
        nib.save(tmp, diagnosticsPath + "/M_mr.nii.gz")

        tmp = np.asarray(iM_fields)
        print(tmp.shape)
        tmp = np.moveaxis(tmp, 0, -1)
        tmp = np.transpose(tmp, (2, 1, 0, 3, 4))
        tmp = np.flip(tmp, (0, 1, 2))
        tmp = nib.Nifti1Image(tmp, np.eye(4))
        nib.save(tmp, diagnosticsPath + "/iM_mr.nii.gz")
        del tmp
    else:
        logging.info("Reading Motion Fields from disk...")
        M_fields = nib.load(diagnosticsPath + "/M_mr.nii.gz").get_fdata()
        M_fields = np.flip(M_fields, (0, 1, 2))
        M_fields = np.transpose(M_fields, (2, 1, 0, 3, 4))

        iM_fields = nib.load(diagnosticsPath + "/iM_mr.nii.gz").get_fdata()
        iM_fields = np.flip(iM_fields, (0, 1, 2))
        iM_fields = np.transpose(iM_fields, (2, 1, 0, 3, 4))

    # Recon
    logging.info("Prep...")
    Ms = []
    M0s = []
    for i in range(nPhases):
        M = reg.interp_op(tshape, M_fields[i])
        M0 = reg.interp_op(tshape, np.zeros(tshape + (3,)))
        M = DLD(M, device=sp.Device(device))
        M0 = DLD(M0, device=sp.Device(device))
        Ms.append(M)
        M0s.append(M0)
    Ms = Diags(Ms, oshape=(nPhases,) + tshape, ishape=(nPhases,) + tshape)
    M0s = Diags(M0s, oshape=(nPhases,) + tshape, ishape=(nPhases,) + tshape)

    PFTSMs = []
    Is = []
    for i in range(nPhases):
        Is.append(sp.linop.Identity(tshape))
        FTs = NFTs((nCoils,) + tshape, coord[i], device=sp.Device(device))
        M = reg.interp_op(tshape, iM_fields[i])
        M = DLD(M, device=sp.Device(device))
        W = sp.linop.Multiply(
            (
                nCoils,
                nSpokes,
                nReadouts,
            ),
            dcf[i],
        )
        FTSM = W * FTs * S * M
        PFTSMs.append(FTSM)
    PFTSMs = (
        Diags(
            PFTSMs,
            oshape=(
                nPhases,
                nCoils,
                nSpokes,
                nReadouts,
            ),
            ishape=(nPhases,) + tshape,
        )
        * Vstacks(Is, ishape=tshape, oshape=(nPhases,) + tshape)
    )

    # precondition
    logging.info("Preconditioner calculation...")
    tmp = PFTSMs.H * PFTSMs * np.complex64(np.ones(tshape))
    L = np.mean(np.abs(tmp))
    logging.info("Preconditioner Value: {}".format(L))

    dcf = dcf[:, xp.newaxis, ...]
    ksp = ksp * dcf
    TV = sp.linop.FiniteDifference(PFTSMs.ishape, axes=(0, 1, 2))
    # ####### debug
    # print("TV dim:{}".format(TV.oshape))
    # proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(TV.oshape, lambda_tv), TV)

    # ADMM
    logging.info("Running iMoCo Recon...")
    alpha = np.max(np.abs(PFTSMs.H * ksp))
    # debug
    logging.debug("alpha:{}".format(alpha))
    sigma = 0.4
    tau = 0.4
    X = np.zeros(tshape, dtype=np.complex64)
    p = np.zeros_like(ksp)
    X0 = np.zeros_like(X)
    q = np.zeros((3,) + tshape, dtype=np.complex64)
    pbarOuter = trange(outer_iter, leave=True)
    lossVal = []
    for ii in pbarOuter:
        timeI = time.time()
        pbarOuter.set_description("iMoco Outer Iter {}".format(ii))
        p = (p + sigma * (PFTSMs * X - ksp)) / (1 + sigma)
        q = q + sigma * TV * X
        q = q / (np.maximum(np.abs(q), alpha) / alpha)
        X = X - tau * (1 / L * PFTSMs.H * p + lambda_tv * TV.H * q)
        timeF = time.time()
        pbarOuter.set_postfix(loss=np.linalg.norm(X - X0) / np.linalg.norm(X), time=timeF - timeI)
        lossVal.append(np.linalg.norm(X - X0) / np.linalg.norm(X))
        X0 = X
    X = np.transpose(X, (2, 1, 0))
    X = np.flip(X, (0, 1, 2))
    lossVal = np.array(lossVal)
    np.save(
        os.path.join(
            diagnosticsPath,
            "loss_frm" + str(nRef) + "_lambda" + str(lambda_tv) + "_res" + str(res_scale),
        ),
        lossVal,
    )
    timeFinish = time.time()
    logging.info("iMoco Recon Finished in: {} min...".format((timeFinish - timeStart) / 60))
    return X


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

    img = imoco(
        ksp,
        coord,
        dcf,
        args.res_scale,
        args.lambda_tv,
        args.inner_iter,
        args.outer_iter,
        args.device,
    )
    print("writing ksp...")
    # plt.ImagePlot(img)
    cfl.write_cfl(args.img_file, img)
