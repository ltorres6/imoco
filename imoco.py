import argparse
import sigpy as sp
import sigpy.mri as mr
import numpy as np
from imoco_e import cfl, ext, reg
from imoco_e.linop_e import NFTs, Diags, DLD, Vstacks


def imoco(
    ksp,
    coord,
    dcf,
    mrimg,
    fname,
    res_scale=1.0,
    lambda_tv=0.05,
    inner_iter=15,
    outer_iter=20,
    device=-1,
    nRef=-1,
    reg_flag=0,
):
    sp.Device(device).use()
    xp = sp.Device(device).xp
    if device >= 0:
        print("Using GPU...")
    else:
        print("Using CPU...")

    print("Kspace Shape: {}...".format(ksp.shape))
    print("trajectory Shape: {}...".format(coord.shape))
    print("DCF Shape: {}....".format(dcf.shape))

    nf_arr = np.sqrt(np.sum(coord[0, 0, :, :] ** 2, axis=1))
    nReadouts = np.sum(nf_arr < np.max(nf_arr) * res_scale)
    del nf_arr

    coord = coord[..., :nReadouts, :]
    ksp = ksp[..., :nReadouts]
    dcf = dcf[..., :nReadouts]

    print("Image Shape Estimate: {}".format(sp.estimate_shape(coord)))
    # nPhases, nEcalib, nCoils, nSpokes, nReadouts, _ = data.shape
    nPhases, nCoils, nSpokes, nReadouts = ksp.shape

    # calibration
    print("Running calibration...")
    # Map for each Motion Phase?
    mps = mr.app.JsenseRecon(
        ksp[0],
        coord=coord[0],
        weights=dcf[0],
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
    print(tshape)
    S = sp.linop.Multiply(tshape, mps)

    # registration
    # print("Registration...")
    # # Options
    # # Compute Device: -1=CPU or 0=GPUmrimg
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
    ## registration

    print("Registration...")
    M_fields = []
    iM_fields = []
    if reg_flag is 1:
        for i in range(nPhases):
            M_field, iM_field = reg.ANTsReg(np.abs(mrimg[nRef]), np.abs(mrimg[i]))
            M_fields.append(M_field)
            iM_fields.append(iM_field)
        M_fields = np.asarray(M_fields)
        iM_fields = np.asarray(iM_fields)
        np.save(fname + "/M_mr.npy", M_fields)
        np.save(fname + "/iM_mr.npy", iM_fields)
    else:
        M_fields = np.load(fname + "/M_mr.npy")
        iM_fields = np.load(fname + "/iM_mr.npy")

    iM_fields = [iM_fields[i] for i in range(iM_fields.shape[0])]
    M_fields = [M_fields[i] for i in range(M_fields.shape[0])]

    ######## TODO scale M_field
    print("Motion Field scaling...")
    M_fields = [reg.M_scale(M, tshape) for M in M_fields]
    iM_fields = [reg.M_scale(M, tshape) for M in iM_fields]

    # Recon
    print("Prep...")
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
        W = sp.linop.Multiply((nCoils, nSpokes, nReadouts,), dcf[i])
        FTSM = W * FTs * S * M
        PFTSMs.append(FTSM)
    PFTSMs = Diags(
        PFTSMs, oshape=(nPhases, nCoils, nSpokes, nReadouts,), ishape=(nPhases,) + tshape
    ) * Vstacks(Is, ishape=tshape, oshape=(nPhases,) + tshape)

    # precondition
    print("Preconditioner calculation...")
    tmp = PFTSMs.H * PFTSMs * np.complex64(np.ones(tshape))
    L = np.mean(np.abs(tmp))
    wksp = ksp * np.expand_dims(dcf, axis=1)
    TV = sp.linop.FiniteDifference(PFTSMs.ishape, axes=(0, 1, 2))
    # ####### debug
    # print("TV dim:{}".format(TV.oshape))
    # proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(TV.oshape, lambda_tv), TV)

    # ADMM
    print("Recon...")
    alpha = np.max(np.abs(PFTSMs.H * wksp))
    ###### debug
    print("alpha:{}".format(alpha))
    sigma = 0.4
    tau = 0.4
    X = np.zeros(tshape, dtype=np.complex64)
    p = np.zeros_like(wksp)
    X0 = np.zeros_like(X)
    q = np.zeros((3,) + tshape, dtype=np.complex64)
    for i in range(outer_iter):
        p = (p + sigma * (PFTSMs * X - wksp)) / (1 + sigma)
        q = q + sigma * TV * X
        q = q / (np.maximum(np.abs(q), alpha) / alpha)

        X0 = X
        X = X - tau * (1 / L * PFTSMs.H * p + lambda_tv * TV.H * q)
        print("outer iter:{}, res:{}".format(i, np.linalg.norm(X - X0) / np.linalg.norm(X0 + 1e-9)))

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
