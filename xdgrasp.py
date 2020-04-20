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
    ksp, coord, dcf, res_scale, lambda_tv, inner_iter, outer_iter, device,
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
    dcf = dcf[..., :nReadouts]

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
        for jj in range(nCoils):
            S.append(sp.linop.Multiply(tshape, mps[jj]))
    del mps

    L = np.zeros((nPhases,) + tshape, dtype=np.complex64)
    Ones_t = xp.ones(tshape, dtype=xp.complex64)
    print("Computing Preconditioner")
    timeI = time.time()
    for ii in trange(nPhases, desc="Motion Phases"):
        # Generate DCF and NUFFT Combined Operator
        WF = sp.linop.Multiply(
            (nSpokes, nReadouts), sp.to_device(np.squeeze(dcf[ii]), device)
        ) * sp.linop.NUFFT(tshape, coord=sp.to_device(np.squeeze(coord[ii]), device))
        Lt = 0
        for jj in range(nCoils):
            idx = ii * nCoils + jj
            WFS = WF * S[idx]
            Lt += sp.to_device(WFS.H * WFS * Ones_t)
        L[ii] = sp.to_device(Lt)
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
    memory_size = 2 * 8 * np.prod(tshape) / (1024 ** 3)
    if memory_size > 0.86:
        print("DOING TV ON CPU!!")
        tv_device = -1
    else:
        print("DOING TV ON GPU...")
        tv_device = 0
    # Do Phase by Phase
    for ii in trange(outer_iter, desc="Outer Iterations"):
        timeI = time.time()
        # plt.ImagePlot(img_0)
        for jj in trange(nPhases, desc="Motion Phases"):
            img_p = sp.to_device(img[jj], device)
            if jj > 0:
                img_p2 = sp.to_device(img[jj - 1], device)
            else:
                # img_p2 = sp.to_device(img[jj + 1], device)
                img_p2 = xp.zeros(img_p.shape, dtype=xp.complex64)
            # Generate DCF and NUFFT Combined Operator
            WF = sp.linop.Multiply(
                (nSpokes, nReadouts), sp.to_device(np.squeeze(dcf[jj]), device)
            ) * sp.linop.NUFFT(tshape, coord=sp.to_device(np.squeeze(coord[jj]), device))
            img_t = 0
            for kk in range(nCoils):
                idx = jj * nCoils + kk
                WFS = WF * S[idx]
                # kspace to memory
                ksp_p = sp.to_device(ksp[jj, kk], device)
                Yt = sp.to_device(Y[jj, kk], device)
                # update Yt
                Yt += sigma * (1 / L * WFS * img_p - ksp_p)
                Yt /= 1 + sigma
                # Accumulate over coils
                img_p -= tau * WFS.H(Yt)
                Y[jj, kk] = sp.to_device(Yt.copy())
                del Yt, ksp_p
                img_t += img_p
                # plt.ImagePlot(img_p)
            img_p = xp.stack((img_t.copy(), img_p2))
            del img_t, WF, WFS
            img_p = linops.TVt_prox(img_p, lambda_tv, iter_max=inner_iter, device=tv_device)
            # plt.ImagePlot(img_p)
            img[jj] = sp.to_device(img_p[0])
        timeF = time.time()
        print(
            "outer iter:{}, res:{}, time:{} seconds".format(
                ii, np.linalg.norm(img - img_0) / np.linalg.norm(img), timeF - timeI
            )
        )
        img_0 = img.copy()

        # print("outer iter:{}, time:{} seconds".format(ii, timeF - timeI))
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
    parser.add_argument("--lambda_tv", type=float, default=5e-2, help="TV regularization, 0.05")
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
