import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import sigpy as sp
import torch as th

import ute_recon_tools.convert_ute as convert_ute
from autofov import autofov
from bin_motion_states import bin_motion_states, bin_periodically, clean_resp
from estimate_resp import estimate_resp
from estimate_respSavitzkyGolay import estimate_respSavitzkyGolay
from filter_bulk import filter_bulk
from gatedRecon import gatedRecon
from gridded_motion_resolved_recon import griddedRecon
from imoco import imoco
from moco import moco
from normalize import normalize
from writeDicoms import writeDicoms
from xdgrasp import xdgrasp


def run(
    raw_dir,
    out_dir,
    hardgating_weights=[50],
    softgating_decays=[2.0],
    imoco_lambdas=[0.05],
    xdgrasp_lambdas=[0.02],
    lowRes_xdgrasp_lambda=0.0075,
    reference_frames=[0],
    n_bins=8,
    prefix=None,
    postfix=None,
    subject=None,
    subject_id=None,
    visit=None,
    study=None,
    flip_resp=False,
    use_external=False,
    do_noGating=False,
    do_HardGating=False,
    do_SoftGating=False,
    do_LowRes=False,
    do_iMoCoExp=False,
    do_HighRes=False,
    do_MoCoExp=False,
    do_gridded_motion_resolved=True,
    overwrite_raw=True,
    overwrite_recons=True,
    UID_base=None,
):
    if postfix is None:
        postfix = ""
    if prefix is None:
        prefix = ""

    use_external = False
    remove_bulk_motion = False
    if use_external:
        # Set to false if using external....
        remove_bulk_motion = False
    compress_coils = True
    logging.debug(f"Resp Flip {flip_resp}")
    # flip_resp = True
    # use_detrend = False
    # set device
    device = 0

    # Randomly Generate series numbers
    series_nums = np.random.permutation(5)

    max_coils = 8
    dc_signal = 1
    spokesDSF = 1.0
    fovthresh = 0.05
    fovNReadout = 140
    hardgating_weights = [hardgating_weights] if isinstance(hardgating_weights, float) else hardgating_weights
    softgating_decays = [softgating_decays] if isinstance(softgating_decays, float) else softgating_decays
    imoco_lambdas = [imoco_lambdas] if isinstance(imoco_lambdas, float) else imoco_lambdas
    xdgrasp_lambdas = [xdgrasp_lambdas] if isinstance(xdgrasp_lambdas, float) else xdgrasp_lambdas
    reference_frames = [reference_frames] if isinstance(reference_frames, int) else reference_frames

    logging.debug("Low Res XDGRASP Lambda: {}".format(lowRes_xdgrasp_lambda))

    timei = time.time()
    # print(visit)
    motionResolvedDir = os.path.join(out_dir, prefix + "MotionResolved" + postfix)
    iterativeMocoDir = os.path.join(out_dir, prefix + "IterativeMoCo" + postfix)
    mocoDir = os.path.join(out_dir, prefix + "MoCo" + postfix)
    noGateDir = os.path.join(out_dir, prefix + "NoGate" + postfix)
    hardGateDir = os.path.join(out_dir, prefix + "HardGate" + postfix)
    softGateDir = os.path.join(out_dir, prefix + "SoftGate" + postfix)
    diagnostics_dir = os.path.join(out_dir, "diagnostics{}/".format(postfix))

    if overwrite_recons:
        if os.path.exists(motionResolvedDir):
            shutil.rmtree(motionResolvedDir)
        if os.path.exists(iterativeMocoDir):
            shutil.rmtree(iterativeMocoDir)
        if os.path.exists(mocoDir):
            shutil.rmtree(mocoDir)
        if os.path.exists(noGateDir):
            shutil.rmtree(noGateDir)
        if os.path.exists(hardGateDir):
            shutil.rmtree(hardGateDir)
        if os.path.exists(softGateDir):
            shutil.rmtree(softGateDir)
        if os.path.exists(diagnostics_dir):
            shutil.rmtree(diagnostics_dir)
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    fileList = os.listdir(raw_dir)
    if "MRI_Raw.h5" in fileList:
        logging.debug("File Exists, Begin!")
    else:
        if all(v is not None for v in [subject, visit, study, prefix]):
            logging.debug("File Does Not Exist, Trying to Pull!")
            if prefix.lower().startswith("Pre"):
                prefix_short = "Pre"
            elif prefix.lower().startswith("Post"):
                prefix_short = "Post"
            else:
                pass

            subprocess.call(
                [
                    "/export/home/ltorres/projects/motion_compensation_ipf/copyData.sh",
                    "lat205",
                    subject,
                    visit,
                    prefix_short,
                    study,
                ]
            )
        else:
            pass

    # Skip if file still doesn't exist.
    fileList = os.listdir(raw_dir)
    if "MRI_Raw.h5" not in fileList:
        logging.warn("File Does Not Exist, Skipping!")
        return

    # Set up data paths
    h5Path = os.path.join(raw_dir, "MRI_Raw.h5")
    mrimgLPath = os.path.join(motionResolvedDir, "MotionResolvedLowRes.npy")
    griddedmrPath = os.path.join(motionResolvedDir, "GriddedMotionResolved.nii.gz")
    imgNoGatePath = os.path.join(noGateDir, "noGate.nii.gz")
    imgHardGatePath = os.path.join(hardGateDir, "hardGate.nii.gz")
    ksp_file = os.path.join(raw_dir, "ksp.npy")
    coord_file = os.path.join(raw_dir, "coord.npy")
    dcf_file = os.path.join(raw_dir, "dcf.npy")
    resp_file = os.path.join(raw_dir, "resp.npy")
    tr_file = os.path.join(raw_dir, "tr.npy")
    resp_txt_file = os.path.join(diagnostics_dir, "resp.txt")
    dc_txt_file = os.path.join(diagnostics_dir, "dc.txt")
    # Create diagnostics directory  if doesn't exist.
    Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)

    # 1) Convert MRI_Raw.h5 to cfl and read resp waveform.
    if os.path.isfile(ksp_file) is False or overwrite_raw is True:
        logging.info("Loading and Saving.....")
        # ksp, coord, dcf, resp, tr = convertUTE(h5Path, max_coils, dsfSpokes=spokesDSF)
        ksp, coord, dcf, resp, tr = convert_ute.convert_ute(
            h5Path, max_coils=max_coils, dsfSpokes=spokesDSF, compress_coils=compress_coils
        )
        try:
            os.remove(ksp_file)
            os.remove(coord_file)
            os.remove(dcf_file)
            os.remove(resp_file)
            os.remove(resp_txt_file)
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
        logging.debug("Loading Data")
        ksp = np.load(ksp_file)
        coord = np.load(coord_file)
        dcf = np.load(dcf_file)
        resp = np.load(resp_file)
        tr = np.load(tr_file)
    # Scale DCF for improved convergence
    dcf **= 0.5
    if subject in ["P006_Exam1"]:
        for i in range(3):
            coord[:, :, i] = -coord[:, :, i]
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

    logging.debug("Kspace Shape: {}...".format(ksp.shape))
    logging.debug("trajectory Shape: {}...".format(coord.shape))
    logging.debug("DCF Shape: {}....".format(dcf.shape))
    logging.debug(f"Repetition Time: {tr} seconds")

    if dc_signal == 1:
        logging.info("Estimating Resp Waveform from DC signal...")
        logging.info("Using TR: {} seconds".format(tr))
        [resp, dc] = estimate_resp(ksp[:, :, 0], tr, fl=0.1, fh=1.2, fw=0.01, usePhase=False)
        if flip_resp:
            dc *= -1
        # [resp, dc] = estimate_respSavitzkyGolay(
        #     ksp[:, :, 0], tr, window=1.0, order=2, detrend_window=10.0, usePhase=False, useDetrend=use_detrend,
        # )
        plt.hist(dc, 100, orientation="horizontal")
        plt.title("DC signal histogram")
        plt.savefig(diagnostics_dir + "DC_histogram.png")
        plt.close()

        plt.plot(dc)
        plt.title("Original DC signal")
        plt.savefig(diagnostics_dir + "dc_signal.png")
        plt.close()

        plt.plot(dc[int(60 / tr) : int(60 / tr) + int(60 / tr)])
        plt.title("60 seconds DC signal")
        plt.savefig(diagnostics_dir + "dc_signal_60.png")
        plt.close()

        np.savetxt(dc_txt_file, dc)
        del dc
    if flip_resp:
        resp *= -1

    # Clean data based on k-space signal
    ksp, coord, dcf, resp = clean_resp(ksp, coord, dcf, resp, diagnostics_dir)

    np.save(resp_file, resp)
    np.savetxt(resp_txt_file, resp)

    plt.hist(resp, 100, orientation="horizontal")
    plt.title("resp signal histogram")
    plt.savefig(diagnostics_dir + "resp_histogram.png")
    plt.close()

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

    # 2) AutoFOV to reduce matrix size
    logging.info("Running AutoFOV...")
    coord = autofov(
        ksp, coord, dcf ** 2, diagnostics_dir, num_ro=fovNReadout, thresh=fovthresh, device=device, radial=False,
    )

    # 3) noGating Recon
    if do_noGating:
        if os.path.isfile(imgNoGatePath) is False or overwrite_recons is True:
            Path(noGateDir).mkdir(parents=True, exist_ok=True)
            dicomDir = os.path.join(noGateDir, f"series_{series_nums[0]}")
            imgNoGate = gatedRecon(ksp, coord, dcf, resp, gating_type="none", device=device, flip=False)
            imgNoGate = normalize(sp.resize(np.abs(imgNoGate), (256, 256, 256)), 0, 255)
            imgNoGate = nib.Nifti1Image(imgNoGate, affine_t)
            nib.save(imgNoGate, imgNoGatePath)
            writeDicoms(imgNoGatePath, dicomDir, UID_base=UID_base, subject_id=subject_id, series_num=series_nums[0])
            del imgNoGate

    # 4) hardGating Recon
    if do_HardGating:
        Path(hardGateDir).mkdir(parents=True, exist_ok=True)
        for hardgating_weight in hardgating_weights[::-1]:
            imgHardGatePath = os.path.join(hardGateDir, f"hardGate{hardgating_weight:.0f}.nii.gz")
            dicomDir = os.path.join(hardGateDir, f"series_{series_nums[1]}")
            if os.path.isfile(imgHardGatePath) is False or overwrite_recons is True:
                Path(hardGateDir).mkdir(parents=True, exist_ok=True)
                imgHardGate = gatedRecon(
                    ksp, coord, dcf, resp, gating_type="hard", gating_thresh=hardgating_weight, device=device, flip=False,
                )
                imgHardGate = normalize(sp.resize(np.abs(imgHardGate), (256, 256, 256)), 0, 255)
                imgHardGate = nib.Nifti1Image(imgHardGate, affine_t)
                nib.save(imgHardGate, imgHardGatePath)
                writeDicoms(imgHardGatePath, dicomDir, UID_base=UID_base, subject_id=subject_id, series_num=series_nums[1])
                del imgHardGate

    # 5) softGating Recon
    if do_SoftGating:
        Path(softGateDir).mkdir(parents=True, exist_ok=True)
        for softgating_decay in softgating_decays[::-1]:
            imgSoftGatePath = os.path.join(softGateDir, f"softGate{softgating_decay:.1f}.nii.gz")
            dicomDir = os.path.join(softGateDir, f"series_{series_nums[2]}")
            if os.path.isfile(imgSoftGatePath) is False or overwrite_recons is True:
                try:
                    imgSoftGate = gatedRecon(
                        ksp,
                        coord,
                        dcf,
                        resp,
                        gating_type="soft",
                        gating_thresh=20,
                        gating_weight=softgating_decay,
                        device=device,
                        flip=False,
                    )
                except:
                    logging.warn("GPU memory exceeded or otherwise failed on GPU. Trying CPU.")
                    imgSoftGate = gatedRecon(
                        ksp,
                        coord,
                        dcf,
                        resp,
                        gating_type="soft",
                        gating_thresh=20,
                        gating_weight=softgating_decay,
                        device=-1,
                        flip=False,
                    )
                imgSoftGate = normalize(sp.resize(np.abs(imgSoftGate), (256, 256, 256)), 0, 255)
                imgSoftGate = nib.Nifti1Image(imgSoftGate, affine_t)
                nib.save(imgSoftGate, imgSoftGatePath)
                writeDicoms(imgSoftGatePath, dicomDir, UID_base=UID_base, subject_id=subject_id, series_num=series_nums[2])
                del imgSoftGate

    # Bin Motion States
    ksp, coord, dcf = bin_periodically(ksp, coord, dcf, resp, n_bins, diagnostics_dir)
    # logging.info("Running bin_motion_states...")
    # ksp, coord, dcf = bin_motion_states(
    #     ksp,
    #     coord,
    #     dcf,
    #     resp,
    #     n_bins,
    #     diagnostics_dir,
    #     filter_bulk=False,
    #     filter_extremes=True,
    #     external=use_external,
    #     external_path=raw_dir,
    # )
    del resp

    if do_gridded_motion_resolved:
        if os.path.isfile(griddedmrPath) is False or overwrite_recons is True:
            logging.info("Running Gridded Motion Resolved Reconstruction...")
            Path(motionResolvedDir).mkdir(parents=True, exist_ok=True)
            gmrimg = griddedRecon(ksp, coord, dcf, n_bins, device=0)
            gmrimg = normalize(np.moveaxis(np.abs(gmrimg), 0, -1), 0, 255)
            gmrimg = np.transpose(gmrimg, (2, 1, 0, 3))
            gmrimg = np.flip(gmrimg, (0, 1, 2))
            gmrimg = sp.resize(gmrimg, (256, 256, 256, n_bins))
            gmrimg = nib.Nifti1Image(gmrimg, affine_t)
            nib.save(gmrimg, griddedmrPath)
            del gmrimg

    # 7) Low Res xdgrasp recon
    if do_LowRes:
        if os.path.isfile(mrimgLPath) is False or overwrite_recons is True:
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
                tv_device=0,
            )
            # mrimgL = sp.resize(mrimg, (n_bins, 192, 192, 192))
            # mrimgL = normalize(np.moveaxis(np.abs(mrimgL), 0, -1), 0, 255)
            # mrimgL = np.transpose(mrimgL, (2, 1, 0, 3))
            # mrimgL = np.flip(mrimgL, (0, 1, 2))
            # mrimgL = nib.Nifti1Image(mrimgL, affine_t)
            np.save(mrimgLPath, mrimg)
            del mrimg

    # 8) iMoCo recon
    if do_iMoCoExp:
        Path(iterativeMocoDir).mkdir(parents=True, exist_ok=True)
        for reference_frame in reference_frames[::-1]:
            for imoco_lambda in imoco_lambdas[::-1]:
                imgPath = os.path.join(iterativeMocoDir, f"iMoCo{imoco_lambda:.2f}_frame{reference_frame}.nii.gz")
                dicomDir = os.path.join(iterativeMocoDir, f"series_{series_nums[3]}")
                if os.path.isfile(imgPath) is False or overwrite_recons is True:
                    logging.info("Running iMoCo Reconstruction...")
                    logging.debug(f"Using Reference Frame {reference_frame}")
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
                        mr_scale=0.75,
                        lambda_tv=imoco_lambda,
                        inner_iter=10,
                        outer_iter=20,
                        device=device,
                        nRef=reference_frame,
                        reg_flag=register_imoco,
                        diffusion_reg=0.5,
                    )
                    img = normalize(sp.resize(np.abs(img), (256, 256, 256)), 0, 255)
                    img = nib.Nifti1Image(img, np.eye(4))
                    nib.save(img, imgPath)
                    writeDicoms(imgPath, dicomDir, UID_base=UID_base, subject_id=subject_id, series_num=series_nums[3])
                    del img, mrimg
                    th.cuda.empty_cache()

    # 10) Full Res xdgrasp recon
    if do_HighRes:
        Path(motionResolvedDir).mkdir(parents=True, exist_ok=True)
        for xdgrasp_lambda in xdgrasp_lambdas[::-1]:
            mrimgPath = os.path.join(motionResolvedDir, f"MotionResolved{xdgrasp_lambda:.3f}.nii.gz")
            mrimg_expPath = os.path.join(motionResolvedDir, f"MotionResolved_exp{xdgrasp_lambda:.3f}.nii.gz")
            dicomDir = os.path.join(motionResolvedDir, f"series_{series_nums[4]}")
            # print(mrimgPath)
            if os.path.isfile(mrimgPath) is False or overwrite_recons is True:
                logging.info("Running Full Res XDGrasp Reconstruction...")
                mrimg = xdgrasp(
                    ksp, coord, dcf, motionResolvedDir, res_scale=1.0, lambda_tv=xdgrasp_lambda, device=device, tv_device=0,
                )
                mrimg = normalize(np.moveaxis(np.abs(mrimg), 0, -1), 0, 255)
                mrimg = np.transpose(mrimg, (2, 1, 0, 3))
                mrimg = np.flip(mrimg, (0, 1, 2))
                mrimg = sp.resize(mrimg, (256, 256, 256, n_bins))
                mrimg = nib.Nifti1Image(mrimg, affine_t)
                mrimg_exp = nib.Nifti1Image(mrimg.get_fdata()[..., -1], affine_t)
                nib.save(mrimg, mrimgPath)
                nib.save(mrimg_exp, mrimg_expPath)
                writeDicoms(mrimg_expPath, dicomDir, UID_base=UID_base, subject_id=subject_id, series_num=series_nums[4])
                del mrimg, mrimg_exp
                cp._default_memory_pool.free_all_blocks()

    del ksp, coord, dcf

    # 11) Full Res MoCo Expiratory
    if do_MoCoExp:
        Path(mocoDir).mkdir(parents=True, exist_ok=True)
        for xdgrasp_lambda in xdgrasp_lambdas[::-1]:
            mrimgPath = os.path.join(motionResolvedDir, f"MotionResolved{xdgrasp_lambda:.3f}.nii.gz")
            dicomDir = os.path.join(motionResolvedDir, f"series_{series_nums[5]}")
            for reference_frame in reference_frames[::-1]:
                imgPath = os.path.join(mocoDir, f"MoCo{xdgrasp_lambda:.3f}_frame{reference_frame}.nii.gz")
                if os.path.isfile(imgPath) is False or overwrite_recons is True:
                    logging.info(f"Running MoCo Registrations")
                    logging.debug(f"Using Reference Frame {reference_frame}")
                    imgMoco = moco(mrimgPath, diagnostics_dir, nRef=reference_frame)
                    imgMoco = nib.Nifti1Image(normalize(imgMoco, 0, 255), affine_t)
                    nib.save(imgMoco, imgPath)
                    writeDicoms(imgPath, dicomDir, UID_base=UID_base, subject_id=subject_id, series_num=series_nums[5])
                    del imgMoco
                    th.cuda.empty_cache()

    timeF = (time.time() - timei) / 60
    logging.info("Finshed Recon in {} minutes".format(timeF))
    th.cuda.empty_cache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="multi recons")
    parser.add_argument("raw_dir", type=str, help="raw data directory")
    parser.add_argument("out_dir", type=str, help="desired output directory")
    parser.add_argument("--postfix", type=str, default="", help="add a string to directories. Useful for different runs")
    parser.add_argument("--softgating_decay", type=float, default=1.5, help="Softgating exponential decay constant")
    parser.add_argument("--imoco_lambda", type=float, default=0.05, help="iMoCo TGV regularization parameter")
    parser.add_argument("--xdgrasp_lambda", type=float, default=0.025, help="XD-GRASP TV regularization parameter")
    parser.add_argument("--reference_frames", type=int, nargs="+", default=-1, help="Registration Reference Frame")
    args = parser.parse_args()
    Path(args.out_dir + f"/diagnostics{args.postfix}/").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.out_dir + f"/diagnostics{args.postfix}/recon_log.txt", mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.debug(f"Reference Frames: {args.reference_frames}")
    logging.debug(f"Running: {args.raw_dir}")
    run(
        args.raw_dir,
        args.out_dir,
        softgating_decays=args.softgating_decay,
        imoco_lambdas=args.imoco_lambda,
        xdgrasp_lambdas=args.xdgrasp_lambda,
        reference_frames=args.reference_frames,
        postfix=args.postfix,
    )
