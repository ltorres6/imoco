import argparse
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import sigpy as sp
from convert_ute import convert_ute

from autofov import autofov
from bin_motion_states import bin_motion_states
from estimate_respSavitzkyGolay import estimate_respSavitzkyGolay
from filter_bulk import filter_bulk
from gatedRecon import gatedRecon
from imoco import imoco
from moco import moco
from normalize import normalize
from xdgrasp import xdgrasp


def run(
    raw_dir,
    out_dir,
    softgating_decays=[2.0],
    imoco_lambdas=[0.05],
    xdgrasp_lambdas=[0.02],
    reference_frames=[-1],
    postfix="",
):

    use_external = True
    if use_external:
        # Set to false if using external....
        remove_bulk_motion = False
    else:
        remove_bulk_motion = True
    compress_coils = False
    flip_resp = False

    do_noGating = False
    do_HardGating = False
    do_SoftGating = False
    do_LowRes = False
    do_iMoCoExp = True
    do_HighRes = False
    do_MoCoExp = False
    do_iMoCoInsp = False
    do_MoCoInsp = False
    overwrite_raw = True

    # set device
    device = 0
    nBins = 6
    max_coils = 8
    dc_signal = 1
    spokesDSF = 1.0
    fovthresh = 0.08
    fovNReadout = 70
    softgating_decays = (
        [softgating_decays]
        if isinstance(softgating_decays, float)
        else softgating_decays
    )
    imoco_lambdas = (
        [imoco_lambdas] if isinstance(imoco_lambdas, float) else imoco_lambdas
    )
    xdgrasp_lambdas = (
        [xdgrasp_lambdas] if isinstance(xdgrasp_lambdas, float) else xdgrasp_lambdas
    )
    reference_frames = (
        [reference_frames] if isinstance(reference_frames, int) else reference_frames
    )

    lowRes_xdgrasp_lambda = 0.0075
    logging.info("Low Res XDGRASP Lambda: {}".format(lowRes_xdgrasp_lambda))
    tv_device = 0

    timei = time.time()
    # print(visit)
    motionResolvedDir = os.path.join(out_dir, "MotionResolved" + postfix)
    iterativeMocoDir = os.path.join(out_dir, "IterativeMoCo" + postfix)
    iterativeMocoInspDir = os.path.join(out_dir, "IterativeMoCoInsp" + postfix)
    mocoDir = os.path.join(out_dir, "MoCo" + postfix)
    mocoInspDir = os.path.join(out_dir, "MoCoInsp" + postfix)
    noGateDir = os.path.join(out_dir, "NoGate" + postfix)
    hardGateDir = os.path.join(out_dir, "HardGate" + postfix)
    softGateDir = os.path.join(out_dir, "SoftGate" + postfix)
    diagnostics_dir = os.path.join(out_dir, "diagnostics{}/".format(postfix))

    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    # Copy Raw Data
    fileList = os.listdir(raw_dir)
    if "MRI_Raw.h5" in fileList:
        logging.info("File Exists, Not Copying!")
    else:
        logging.error("need to fix this.")
        # subprocess.call(["pcvipr_recon_binary", "-f ", raw_dir + "P\*", "-export_kdata"], cwd=raw_dir)

    fileList = os.listdir(raw_dir)
    if "MRI_Raw.h5" in fileList:
        logging.info("File Exists, Begin!")
    else:
        pass
    # Set up data paths
    h5Path = os.path.join(raw_dir, "MRI_Raw.h5")
    mrimgPath = os.path.join(motionResolvedDir, "MotionResolved.nii.gz")
    mrimg_expPath = os.path.join(motionResolvedDir, "MotionResolved_exp.nii.gz")
    mrimgLPath = os.path.join(motionResolvedDir, "MotionResolvedLowRes.npy")
    imgMocoPath = os.path.join(mocoDir, "MoCo.nii.gz")
    imgPath = os.path.join(iterativeMocoDir, "iMoCo.nii.gz")
    respPath = os.path.join(diagnostics_dir, "resp.npy")
    imgInspPath = os.path.join(iterativeMocoInspDir, "iMoCo.nii.gz")
    imgMocoInspPath = os.path.join(mocoInspDir, "MoCo.nii.gz")
    imgNoGatePath = os.path.join(noGateDir, "noGate.nii.gz")
    imgHardGatePath = os.path.join(hardGateDir, "hardGate.nii.gz")
    imgSoftGatePath = os.path.join(softGateDir, "softGate.nii.gz")
    ksp_file = os.path.join(raw_dir, "ksp.npy")
    coord_file = os.path.join(raw_dir, "coord.npy")
    dcf_file = os.path.join(raw_dir, "dcf.npy")
    resp_file = os.path.join(raw_dir, "resp.npy")
    tr_file = os.path.join(raw_dir, "tr.npy")

    # Create diagnostics directory  if doesn't exist.
    Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)

    # 1) Convert MRI_Raw.h5 to cfl and read resp waveform.
    if os.path.isfile(ksp_file) is False or overwrite_raw is True:
        logging.info("Loading and Saving.....")
        logging.info("Running File Conversion...")
        # ksp, coord, dcf, resp, tr = convertUTE(h5Path, max_coils, dsfSpokes=spokesDSF)
        ksp, coord, dcf, resp, tr = convert_ute(
            h5Path,
            max_coils=max_coils,
            dsfSpokes=spokesDSF,
            compress_coils=compress_coils,
        )
        try:
            os.remove(ksp_file)
            os.remove(coord_file)
            os.remove(dcf_file)
            os.remove(resp_file)
            os.remove(tr_file)
        except OSError:
            pass
        np.save(ksp_file, ksp)
        np.save(coord_file, coord)
        np.save(dcf_file, dcf)
        np.save(resp_file, resp)
        np.save(tr_file, tr)
    else:
        ksp = np.load(ksp_file)
        coord = np.load(coord_file)
        dcf = np.load(dcf_file)
        resp = np.load(resp_file)
        tr = np.load(tr_file)
    # Scale DCF for improved convergence
    dcf **= 0.5

    # Read Affine Transformation
    # header = read_pcvipr_header(raw_dir)
    # affine_t = np.array(
    #     [
    #         [header["ix"], header["iy"], header["iz"], 0],
    #         [header["jx"], header["jy"], header["jz"], 0],
    #         [header["kx"], header["ky"], header["kz"], 0],
    #         [header["sx"], header["sy"], header["sz"], 1],
    #     ]
    # )
    affine_t = np.eye(4)

    logging.info("Kspace Shape: {}...".format(ksp.shape))
    logging.info("trajectory Shape: {}...".format(coord.shape))
    logging.info("DCF Shape: {}....".format(dcf.shape))
    logging.info(f"Repetition Time: {tr} seconds")

    if dc_signal == 1:
        logging.info("Estimating Resp Waveform from DC signal...")
        logging.info("Using TR: {} seconds".format(tr))
        # resp = estimate_resp(ksp[:, :, 0], tr, fl=0.1, fh=1.5, fw=0.01, usePhase=False)
        resp = estimate_respSavitzkyGolay(
            ksp[:, :, 0],
            tr,
            window=1.0,
            order=2,
            detrend_window=10.0,
            usePhase=False,
            useDetrend=True,
        )
    if flip_resp:
        resp *= -1

    plt.plot(resp)
    plt.title("Entire Waveform")
    plt.savefig(diagnostics_dir + "respWaveformFull.png")
    plt.close()

    plt.plot(resp[int(60 / tr) : int(60 / tr) + int(120 / tr)])
    plt.title("120 seconds of breathing")
    plt.savefig(diagnostics_dir + "respWaveform120.png")
    plt.close()

    plt.plot(resp[int(60 / tr) : int(60 / tr) + int(60 / tr)])
    plt.title("60 seconds of breathing")
    plt.savefig(diagnostics_dir + "respWaveform60.png")
    plt.close()
    np.save(respPath, resp)

    # 2) AutoFOV to reduce matrix size
    logging.info("Running AutoFOV...")
    coord = autofov(
        ksp,
        coord,
        dcf**2,
        diagnostics_dir,
        num_ro=fovNReadout,
        thresh=fovthresh,
        device=device,
        radial=False,
    )

    # 6) Bulk filter
    if remove_bulk_motion:
        logging.info("Filtering Resp...")
        ksp, coord, dcf, resp = filter_bulk(ksp, coord, dcf, resp, tr, diagnostics_dir)

    # 3) noGating Recon
    if do_noGating:
        Path(noGateDir).mkdir(parents=True, exist_ok=True)
        imgNoGate = gatedRecon(
            ksp, coord, dcf, resp, gating_type="none", device=device, flip=flip_resp
        )
        imgNoGate = normalize(sp.resize(np.abs(imgNoGate), (256, 256, 256)), 0, 255)
        imgNoGate = nib.Nifti1Image(imgNoGate, affine_t)
        nib.save(imgNoGate, imgNoGatePath)
        del imgNoGate

    # 4) hardGating Recon
    if do_HardGating:
        Path(hardGateDir).mkdir(parents=True, exist_ok=True)
        imgHardGate = gatedRecon(
            ksp,
            coord,
            dcf,
            resp,
            gating_type="hard",
            gating_thresh=50,
            device=device,
            flip=flip_resp,
        )
        imgHardGate = normalize(sp.resize(np.abs(imgHardGate), (256, 256, 256)), 0, 255)
        imgHardGate = nib.Nifti1Image(imgHardGate, affine_t)
        nib.save(imgHardGate, imgHardGatePath)
        del imgHardGate

    # 5) softGating Recon
    if do_SoftGating:
        Path(softGateDir).mkdir(parents=True, exist_ok=True)
        for softgating_decay in softgating_decays[::-1]:
            if len(softgating_decays) != 1:
                imgSoftGatePath = os.path.join(
                    softGateDir, "softGate{}.nii.gz".format(softgating_decay)
                )
            imgSoftGate = gatedRecon(
                ksp,
                coord,
                dcf,
                resp,
                gating_type="soft",
                gating_thresh=25,
                gating_weight=softgating_decay,
                device=-device,
                flip=flip_resp,
            )
            imgSoftGate = normalize(
                sp.resize(np.abs(imgSoftGate), (256, 256, 256)), 0, 255
            )
            imgSoftGate = nib.Nifti1Image(imgSoftGate, affine_t)
            nib.save(imgSoftGate, imgSoftGatePath)
            del imgSoftGate

    # 6) Bin Motion States
    logging.info("Running bin_motion_states...")
    ksp, coord, dcf = bin_motion_states(
        ksp,
        coord,
        dcf,
        resp,
        nBins,
        diagnostics_dir,
        filter_bulk=False,
        filter_extremes=True,
        external=use_external,
        external_path=raw_dir,
    )
    del resp

    # 7) Low Res xdgrasp recon
    if do_LowRes:
        logging.info("Running Low Res XDGrasp Reconstruction...")
        Path(motionResolvedDir).mkdir(parents=True, exist_ok=True)
        mrimg = xdgrasp(
            ksp,
            coord,
            dcf,
            motionResolvedDir,
            res_scale=0.75,
            lambda_tv=lowRes_xdgrasp_lambda,
            device=device,
            tv_device=tv_device,
        )
        # mrimgL = sp.resize(mrimg, (nBins, 192, 192, 192))
        # mrimgL = normalize(np.moveaxis(np.abs(mrimgL), 0, -1), 0, 255)
        # mrimgL = np.transpose(mrimgL, (2, 1, 0, 3))
        # mrimgL = np.flip(mrimgL, (0, 1, 2))
        # mrimgL = nib.Nifti1Image(mrimgL, affine_t)
        np.save(mrimgLPath, mrimg)
        del mrimg

    # 8) iMoCo recon
    if do_iMoCoExp:
        logging.info("Running iMoCo Reconstruction...")
        Path(iterativeMocoDir).mkdir(parents=True, exist_ok=True)
        for reference_frame in reference_frames[::-1]:
            logging.info(f"Using Reference Frame {reference_frame}")
            counter = 0
            for imoco_lambda in imoco_lambdas[::-1]:
                imgPath = os.path.join(
                    iterativeMocoDir,
                    f"iMoCo{imoco_lambda}_frame{reference_frame}.nii.gz",
                )
                if counter != 0:
                    register_imoco = 0
                else:
                    register_imoco = 1
                try:
                    mrimg = np.load(mrimgLPath)
                except Exception:
                    logging.error("Could not read low res xd-grasp reconstruction")

                img = imoco(
                    ksp,
                    coord,
                    dcf,
                    mrimg,
                    iterativeMocoDir,
                    res_scale=1.0,
                    lambda_tv=imoco_lambda,
                    inner_iter=15,
                    outer_iter=20,
                    device=device,
                    nRef=reference_frame,
                    reg_flag=register_imoco,
                )
                img = normalize(sp.resize(np.abs(img), (256, 256, 256)), 0, 255)
                img = nib.Nifti1Image(img, np.eye(4))
                nib.save(img, imgPath)
                counter += 1
                del img, mrimg

    ## ---- Deprecating in favor of list of reference frames
    # # 9) iMoCo recon Insp
    # if do_iMoCoInsp:
    #     logging.info("Running iMoCo Inspiratory Reconstruction...")
    #     Path(iterativeMocoInspDir).mkdir(parents=True, exist_ok=True)
    #     try:
    #         mrimg = np.load(mrimgLPath)
    #     except Exception:
    #         logging.error("Could not read low res xd-grasp reconstruction")

    #     imgInsp = imoco(
    #         ksp,
    #         coord,
    #         dcf,
    #         mrimg,
    #         iterativeMocoInspDir,
    #         res_scale=1.0,
    #         lambda_tv=imoco_lambda,
    #         inner_iter=15,
    #         outer_iter=20,
    #         device=device,
    #         nRef=0,
    #         reg_flag=register,
    #     )
    #     imgInsp = sp.resize(np.abs(imgInsp), (256, 256, 256))
    #     imgInsp = nib.Nifti1Image(imgInsp, np.eye(4))
    #     nib.save(imgInsp, imgInspPath)
    #     del imgInsp, mrimg

    # 10) Full Res xdgrasp recon
    if do_HighRes:
        Path(motionResolvedDir).mkdir(parents=True, exist_ok=True)
        for xdgrasp_lambda in xdgrasp_lambdas[::-1]:
            if len(xdgrasp_lambdas) != 1:
                mrimgPath = os.path.join(
                    motionResolvedDir, "MotionResolved{}.nii.gz".format(xdgrasp_lambda)
                )
                mrimg_expPath = os.path.join(
                    motionResolvedDir,
                    "MotionResolved_exp{}.nii.gz".format(xdgrasp_lambda),
                )
            logging.info("Running Full Res XDGrasp Reconstruction...")
            mrimg = xdgrasp(
                ksp,
                coord,
                dcf,
                motionResolvedDir,
                res_scale=1.0,
                lambda_tv=xdgrasp_lambda,
                device=device,
                tv_device=tv_device,
            )
            mrimg = sp.resize(mrimg, (nBins, 256, 256, 256))
            mrimg = normalize(np.moveaxis(np.abs(mrimg), 0, -1), 0, 255)
            mrimg = np.transpose(mrimg, (2, 1, 0, 3))
            mrimg = np.flip(mrimg, (0, 1, 2))
            mrimg = nib.Nifti1Image(mrimg, affine_t)
            mrimg_exp = nib.Nifti1Image(mrimg.get_fdata()[..., 0], affine_t)
            nib.save(mrimg, mrimgPath)
            nib.save(mrimg_exp, mrimg_expPath)
            del mrimg, mrimg_exp
        del ksp, coord, dcf

    # 11) Full Res MoCo Expiratory
    if do_MoCoExp:
        Path(mocoDir).mkdir(parents=True, exist_ok=True)
        imgMoco = moco(mrimgPath, diagnostics_dir, nRef=nRef)
        imgMoco = nib.Nifti1Image(normalize(imgMoco, 0, 255), affine_t)
        nib.save(imgMoco, imgMocoPath)
        del imgMoco

    # 12) Full Res MoCo Inspiratory
    if do_MoCoInsp:
        Path(mocoInspDir).mkdir(parents=True, exist_ok=True)
        imgMocoInsp = moco(mrimgPath, diagnostics_dir, nRef=0)
        imgMocoInsp = nib.Nifti1Image(normalize(imgMocoInsp, 0, 255), affine_t)
        nib.save(imgMocoInsp, imgMocoInspPath)
        del imgMocoInsp

    timeF = (time.time() - timei) / 60
    logging.info("Finshed Recon in {} minutes".format(timeF))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="multi recons")
    parser.add_argument("raw_dir", type=str, help="raw data directory")
    parser.add_argument("out_dir", type=str, help="desired output directory")
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="add a string to directories. Useful for different runs",
    )
    parser.add_argument(
        "--softgating_decay",
        type=float,
        default=1.5,
        help="Softgating exponential decay constant",
    )
    parser.add_argument(
        "--imoco_lambda",
        type=float,
        default=0.05,
        help="iMoCo TGV regularization parameter",
    )
    parser.add_argument(
        "--xdgrasp_lambda",
        type=float,
        default=0.025,
        help="XD-GRASP TV regularization parameter",
    )
    parser.add_argument(
        "--reference_frames",
        type=int,
        nargs="+",
        default=-1,
        help="Registration Reference Frame",
    )
    args = parser.parse_args()
    Path(args.out_dir + f"/diagnostics{args.postfix}/").mkdir(
        parents=True, exist_ok=True
    )
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                args.out_dir + f"/diagnostics{args.postfix}/recon_log.txt", mode="a"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Reference Frames: {args.reference_frames}")
    logging.info(f"Running: {args.raw_dir}")
    run(
        args.raw_dir,
        args.out_dir,
        softgating_decays=args.softgating_decay,
        imoco_lambdas=args.imoco_lambda,
        xdgrasp_lambdas=args.xdgrasp_lambda,
        reference_frames=args.reference_frames,
        postfix=args.postfix,
    )
