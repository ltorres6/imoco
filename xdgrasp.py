import argparse
import sigpy as sp
import numpy as np
from coilCompression import pcaCoilCompression
import sigpy.mri as mr
import linops
from tqdm import trange
import cfl
import time
import sigpy.plot as plt


def xdgrasp(
    ksp, coord, dcf, res_scale=1.0, lambda_tv=0.05, inner_iter=10, outer_iter=20, device=0,
):
    xp = sp.Device(device).xp
    if device == 0:
        print("Using GPU...")
    else:
        print("Using CPU...")

    nf_arr = np.sqrt(np.sum(coord[0, 0, :, :] ** 2, axis=1))
    nReadouts = np.sum(nf_arr < np.max(nf_arr) * res_scale)
    print("Kspace Shape: {}...".format(ksp.shape))
    print("trajectory Shape: {}...".format(coord.shape))
    print("DCF Shape: {}....".format(dcf.shape))

    # Coil Compression
    if ksp.shape[1] > 6:
        print("Running Coil Compression...")
        ksp = pcaCoilCompression(kdata=ksp, axis=1, target_channels=6)
        print("Coil Compressed kspace shape: {} ...".format(ksp.shape))
    coord = coord[..., :nReadouts, :]
    ksp = ksp[..., :nReadouts]
    dcf = dcf[..., :nReadouts] ** 0.5

    print("Image Shape Estimate: {}".format(sp.estimate_shape(coord)))
    # nPhases, nEcalib, nCoils, nSpokes, nReadouts, _ = data.shape
    nPhases, nCoils, nSpokes, nReadouts = ksp.shape
    tshape = (
        np.int(np.max(coord[..., 0]) - np.min(coord[..., 0])),
        np.int(np.max(coord[..., 1]) - np.min(coord[..., 1])),
        np.int(np.max(coord[..., 2]) - np.min(coord[..., 2])),
    )

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
    S = []
    for ii in range(nPhases):
        S.append(sp.linop.Multiply(tshape, mps))
        # for jj in range(nCoils):
        #     S.append(sp.linop.Multiply(tshape, mps[jj]))
    # del mps

    L = np.zeros((nPhases,) + tshape, dtype=np.complex64)
    img_ones = xp.ones(tshape, dtype=xp.complex64)
    print("Computing Preconditioner")
    timeI = time.time()
    for ii in trange(nPhases, desc="Motion Phases"):
        coord_t = sp.to_device(coord[ii], device)
        dcf_t = sp.to_device(dcf[ii], device)
        img_t = 0
        for jj in range(nCoils):
            # idx = ii * nCoils + jj
            mps_c = sp.to_device(mps[jj], device)
            # plt.ImagePlot(mps_c)
            # plt.ImagePlot(img_ones * mps_c)
            img_tc = xp.squeeze(sp.nufft(img_ones * mps_c, coord_t))
            # plt.ImagePlot(img_tc)
            img_tc *= dcf_t ** 2
            # plt.ImagePlot(img_tc)
            img_tc = sp.nufft_adjoint(img_tc, coord_t, oshape=tshape)
            # plt.ImagePlot(img_tc)
            img_tc *= xp.conj(mps_c)
            # plt.ImagePlot(img_tc)
            img_t += img_tc
            # plt.ImagePlot(img_t)
        L[ii, ...] = sp.to_device(img_t)
    L = np.mean(np.abs(L))
    print(L)
    timeF = time.time()
    print("Time for preconditioner: {} seconds.".format(timeF - timeI))

    # reconstruction
    print("running reconstruction")
    dcf = dcf[:, xp.newaxis, ...]
    ksp *= dcf
    print(ksp.shape)
    img = np.zeros((nPhases,) + tshape, dtype=np.complex64)
    Y = np.zeros_like(ksp)
    img_0 = np.zeros_like(img)
    tau = 0.4
    sigma = 0.4
    # memory_size = 2 * 8 * np.prod(tshape) / (1024 ** 3)
    # if memory_size > 0.86:
    #     print("DOING TV ON CPU!!")
    #     tv_device = -1
    # else:
    #     print("DOING TV ON GPU...")
    #     tv_device = 0
    tv_device = -1
    # Do Phase by Phase
    pbarOuter = trange(outer_iter, leave=True)
    for ii in pbarOuter:
        timeI = time.time()
        pbarOuter.set_description("Processing Outer Iteration {}".format(ii))
        # pbarMotion = trange(nPhases, desc="Motion Phases")
        for jj in range(nPhases):
            # pbarMotion.set_description("Processing Motion Phase {}".format(jj))
            img_p = sp.to_device(img[jj], device)
            coord_t = xp.squeeze(sp.to_device(coord[jj], device))
            dcf_t = xp.squeeze(sp.to_device(dcf[jj], device))
            img_t = 0
            # Do coil by coil
            for kk in range(nCoils):
                # idx = jj * nCoils + kk
                # WFS = WF * S[idx]
                # kspace to memory
                mps_c = xp.squeeze(sp.to_device(mps[kk], device))
                ksp_p = sp.to_device(ksp[jj, kk], device)
                Yt = sp.to_device(Y[jj, kk], device)
                # update Yt
                Yt += sigma * (1 / L * dcf_t * xp.squeeze(sp.nufft(img_p * mps_c, coord_t)) - ksp_p)
                Yt /= 1 + sigma
                # Accumulate over coils
                img_p -= tau * xp.conj(mps_c) * sp.nufft_adjoint(dcf_t * Yt, coord_t, oshape=tshape)
                Y[jj, kk] = sp.to_device(Yt.copy())
                del Yt, ksp_p
                img_t += img_p
                # plt.ImagePlot(img_p)
            # img_p = xp.stack((img_t.copy(), img_p2))
            img[jj] = sp.to_device(img_t)
        del img_p, img_t, coord_t, dcf_t  # Try to clear GPU memory.
        img = sp.to_device(linops.TVt_prox(img, lambda_tv, iter_max=inner_iter, device=tv_device))
        # plt.ImagePlot(img_p)
        # img[jj] = sp.to_device(img_p[0])
        timeF = time.time()
        pbarOuter.set_postfix(
            loss=np.linalg.norm(img - img_0) / np.linalg.norm(img), time=timeF - timeI
        )
        img_0 = img
    print("done...")
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

    img = xdgrasp(
        ksp,
        coord,
        dcf,
        args.res_scale,
        args.lambda_tv,
        args.inner_iter,
        args.outer_iter,
        args.device,
    )
    print("writing data...")
    plt.ImagePlot(img)
    cfl.write_cfl(args.img_file, img)
    # np.save(args.img, img)
