import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import sigpy as sp
import yaml
from sigpy.mri.dcf import pipe_menon_dcf
from tqdm import tqdm

import ute_recon_tools.convert_ute as convert_ute
from imoco.motion.binning import bin_motion_states, bin_periodically, clean_resp
from imoco.motion.estimate_resp import estimate_resp, estimate_respSavitzkyGolay
from imoco.recon.gated import gatedRecon, gatingWeights
from imoco.recon.gridded import griddedRecon
from imoco.recon.imoco import imoco
from imoco.recon.moco import moco
from imoco.recon.xdgrasp import xdgrasp
from imoco.utils.autofov import autofov
from imoco.utils.dicom import writeDicoms
from imoco.utils.normalize import normalize

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Paths
    "raw_dir": None,
    "out_dir": None,
    # Pipeline flags
    "do_noGating": True,
    "do_HardGating": True,
    "do_SoftGating": True,
    "do_LowRes": True,
    "do_HighRes": True,
    "do_iMoCoExp": True,
    "do_MoCoExp": False,
    "do_gridded_motion_resolved": True,
    # Overwrite flags
    "overwrite_raw": True,
    "overwrite_recons": False,
    # Reconstruction parameters
    "n_bins": 6,
    "max_coils": 8,
    "device": 0,
    "iterations": 20,
    "final_matrix_size": [256, 256, 256],
    "resolution": [1.25, 1.25, 1.25],
    # Algorithm parameters
    "hardgating_weights": [50],
    "softgating_decays": [2.0],
    "imoco_lambdas": [0.05],
    "xdgrasp_lambdas": [0.02],
    "lowRes_xdgrasp_lambda": 0.0075,
    "sigma": 0.4,
    "tau": 0.4,
    "reference_frames": [-1],
    # Data preprocessing
    "flip_resp": False,
    "recalc_dcf": False,
    "pre_whiten": False,
    "clean_by_resp": True,
    "load_registration": False,
    # Autofov
    "fovNReadout": 75,
    "fovthresh": 0.03,
    # Optional metadata
    "prefix": "",
    "postfix": "",
    "subject_id": None,
    "UID_base": None,
}


def load_config(config_path):
    """Load pipeline configuration from a YAML file.

    Any keys not specified in the file will use their default values.

    Args:
        config_path (str): Path to YAML configuration file.

    Returns:
        dict: Merged configuration dictionary.
    """
    cfg = dict(DEFAULT_CONFIG)
    with open(config_path, "r") as f:
        user_cfg = yaml.safe_load(f)
    if user_cfg:
        cfg.update(user_cfg)
    # Ensure list types
    for key in ["hardgating_weights", "softgating_decays", "imoco_lambdas",
                 "xdgrasp_lambdas", "reference_frames", "final_matrix_size", "resolution"]:
        if isinstance(cfg[key], (int, float)):
            cfg[key] = [cfg[key]]
    # Convert final_matrix_size to tuple
    cfg["final_matrix_size"] = tuple(cfg["final_matrix_size"])
    return cfg


def _plot_losses(loss_file, diagnostics_dir):
    loss = []
    with open(loss_file, "r") as f:
        for row in f:
            loss.append(float(row))
    plt.plot(loss, color="g", label="File Data")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss", fontsize=20)
    plt.legend()
    plt.savefig(os.path.join(diagnostics_dir, "Loss.png"))
    plt.close()


def data_usage_report(
    total_acquired,
    after_cleaning,
    hard_gating_count=None,
    soft_gating_effective=None,
    bin_counts=None,
    n_bins=6,
    diagnostics_dir=None,
):
    """Generate and log a data usage report.

    Reports the percentage of acquired spokes used by each reconstruction
    method, which is useful for answering reviewer questions about how much
    data each technique uses.

    Args:
        total_acquired (int): Total number of acquired spokes.
        after_cleaning (int): Number of spokes after respiratory cleaning.
        hard_gating_count (dict, optional): {threshold: count} for hard gating.
        soft_gating_effective (dict, optional): {decay: effective_count} for soft gating.
        bin_counts (list, optional): List of spoke counts per bin.
        n_bins (int): Number of motion bins.
        diagnostics_dir (str, optional): Directory to write report file.

    Returns:
        str: Formatted report string.
    """
    lines = []
    lines.append("=== Data Usage Report ===")
    lines.append(f"Total acquired spokes:              {total_acquired:>10,d}")
    lines.append(
        f"After respiratory cleaning:         {after_cleaning:>10,d}  "
        f"({100 * after_cleaning / total_acquired:.1f}%)"
    )
    lines.append(
        f"  No Gating:                        {after_cleaning:>10,d}  "
        f"({100 * after_cleaning / total_acquired:.1f}% of acquired)"
    )
    if hard_gating_count:
        for thresh, count in hard_gating_count.items():
            lines.append(
                f"  Hard Gating ({thresh}th percentile):  {count:>10,d}  "
                f"({100 * count / total_acquired:.1f}% of acquired)"
            )
    if soft_gating_effective:
        for decay, eff in soft_gating_effective.items():
            lines.append(
                f"  Soft Gating (decay={decay}) effective: {eff:>10,.0f}  "
                f"({100 * eff / total_acquired:.1f}% of acquired)"
            )
    if bin_counts is not None:
        total_binned = sum(bin_counts)
        lines.append(
            f"After periodic binning:             {total_binned:>10,d}  "
            f"({100 * total_binned / total_acquired:.1f}% of acquired)"
        )
        for b, count in enumerate(bin_counts):
            label = "end-expiration" if b == 0 else ("end-inspiration" if b == n_bins - 1 else "")
            label_str = f" ({label})" if label else ""
            lines.append(
                f"  Bin {b}{label_str}:{' ' * (27 - len(f'Bin {b}{label_str}'))}{count:>10,d}  "
                f"({100 * count / total_acquired:.1f}%)"
            )
        excluded = after_cleaning - total_binned
        lines.append(
            f"  Excluded (invalid cycles):        {excluded:>10,d}  "
            f"({100 * excluded / total_acquired:.1f}%)"
        )
        lines.append(
            f"  XD-GRASP / iMoCo:                {total_binned:>10,d}  "
            f"({100 * total_binned / total_acquired:.1f}% of acquired, all valid bins)"
        )

    report = "\n".join(lines)
    log.info("\n" + report)
    if diagnostics_dir:
        with open(os.path.join(diagnostics_dir, "data_usage.txt"), "w") as f:
            f.write(report + "\n")
    return report


def run(cfg):
    """Run the full iMoCo reconstruction pipeline.

    Args:
        cfg (dict): Configuration dictionary (see ``DEFAULT_CONFIG`` for keys).
    """
    raw_dir = cfg["raw_dir"]
    out_dir = cfg["out_dir"]
    postfix = cfg["postfix"]
    prefix = cfg["prefix"]
    n_bins = cfg["n_bins"]
    device = cfg["device"]
    final_matrix_size = cfg["final_matrix_size"]
    resolution = cfg["resolution"]
    iterations = cfg["iterations"]

    hardgating_weights = cfg["hardgating_weights"]
    softgating_decays = cfg["softgating_decays"]
    imoco_lambdas = cfg["imoco_lambdas"]
    xdgrasp_lambdas = cfg["xdgrasp_lambdas"]
    lowRes_xdgrasp_lambda = cfg["lowRes_xdgrasp_lambda"]
    reference_frames = cfg["reference_frames"]
    sigma = cfg["sigma"]
    tau = cfg["tau"]

    UID_base = cfg["UID_base"]
    subject_id = cfg["subject_id"]

    compress_coils = True
    dc_signal = 1
    spokesDSF = 1.0

    log.info(f"Resp Flip {cfg['flip_resp']}")

    series_nums = np.random.permutation(5)
    tv_device = 0

    timei = time.time()
    motionResolvedDir = os.path.join(out_dir, prefix + "MotionResolved" + postfix)
    iterativeMocoDir = os.path.join(out_dir, prefix + "IterativeMoCo" + postfix)
    mocoDir = os.path.join(out_dir, prefix + "MoCo" + postfix)
    noGateDir = os.path.join(out_dir, prefix + "NoGate" + postfix)
    hardGateDir = os.path.join(out_dir, prefix + "HardGate" + postfix)
    softGateDir = os.path.join(out_dir, prefix + "SoftGate" + postfix)
    diagnostics_dir = os.path.join(out_dir, f"diagnostics{postfix}")

    if cfg["overwrite_recons"]:
        for d in [motionResolvedDir, iterativeMocoDir, mocoDir, noGateDir, hardGateDir, softGateDir, diagnostics_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    fileList = os.listdir(raw_dir)
    if "MRI_Raw.h5" not in fileList and "ksp.npy" not in fileList:
        log.info("File Does Not Exist, Skipping!")
        return
    log.info("File Exists, Begin!")

    # File paths
    h5Path = os.path.join(raw_dir, "MRI_Raw.h5")
    mrimgLPath = os.path.join(motionResolvedDir, "MotionResolvedLowRes.npy")
    mrimgLniiPath = os.path.join(motionResolvedDir, "MotionResolvedLowRes.nii.gz")
    griddedmrPath = os.path.join(motionResolvedDir, "GriddedMotionResolved.nii.gz")
    imgNoGatePath = os.path.join(noGateDir, "noGate.nii.gz")
    ksp_file = os.path.join(raw_dir, "ksp.npy")
    coord_file = os.path.join(raw_dir, "coord.npy")
    dcf_file = os.path.join(raw_dir, "dcf.npy")
    resp_file = os.path.join(raw_dir, "resp.npy")
    tr_file = os.path.join(raw_dir, "tr.npy")
    resp_txt_file = os.path.join(diagnostics_dir, "resp.txt")
    dc_txt_file = os.path.join(diagnostics_dir, "dc.txt")
    noise_file = os.path.join(raw_dir, "noise.npy")

    Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)

    # 1) Convert MRI_Raw.h5 or load existing data
    if not os.path.isfile(ksp_file) or cfg["overwrite_raw"]:
        log.info("Loading and Saving.....")
        log.info("Running File Conversion...")
        ksp, coord, dcf, resp, tr, noise = convert_ute.convert_ute(
            h5Path,
            max_coils=cfg["max_coils"],
            dsfSpokes=spokesDSF,
            compress_coils=compress_coils,
            pre_whiten=cfg["pre_whiten"],
        )
        for f in [ksp_file, coord_file, dcf_file, resp_file, resp_txt_file, tr_file, noise_file]:
            if os.path.isfile(f):
                os.remove(f)
        np.save(ksp_file, ksp)
        np.save(coord_file, coord)
        np.save(dcf_file, dcf)
        np.save(resp_file, resp)
        np.save(tr_file, tr)
        np.save(noise_file, noise)
        del noise
    else:
        log.info("Loading Data")
        ksp = np.load(ksp_file)
        coord = np.load(coord_file)
        dcf = np.load(dcf_file)
        try:
            resp = np.load(resp_file)
        except FileNotFoundError:
            log.info("resp doesn't exist, setting dc_flag=1")
            dc_signal = 1
        tr = np.load(tr_file)

    if cfg["recalc_dcf"]:
        dcf = sp.to_device(pipe_menon_dcf(coord, device=device))

    dcf **= 0.5
    ksp /= np.abs(ksp).max()
    affine_t = np.eye(4)

    log.info("Kspace Shape: {}...".format(ksp.shape))
    log.info("trajectory Shape: {}...".format(coord.shape))
    log.info("DCF Shape: {}....".format(dcf.shape))
    log.info(f"Repetition Time: {tr} seconds")

    total_acquired_spokes = ksp.shape[1]

    if dc_signal == 1:
        log.info("Estimating Resp Waveform from DC signal...")
        log.info("Using TR: {} seconds".format(tr))
        [resp, dc] = estimate_resp(ksp[:, :, 0], tr, fl=0.02, fh=0.4, fw=0.01, usePhase=False)

        if cfg["flip_resp"]:
            dc *= -1

        plt.hist(dc, 100, orientation="horizontal")
        plt.title("DC signal histogram")
        plt.savefig(os.path.join(diagnostics_dir, "DC_histogram.png"))
        plt.close()

        plt.plot(dc)
        plt.title("Original DC signal")
        plt.savefig(os.path.join(diagnostics_dir, "dc_signal.png"))
        plt.close()

        plt.plot(dc[int(60 / tr) : int(60 / tr) + int(60 / tr)])
        plt.title("60 seconds DC signal")
        plt.savefig(os.path.join(diagnostics_dir, "dc_signal_60.png"))
        plt.close()

        np.savetxt(dc_txt_file, dc)
        del dc
    if cfg["flip_resp"]:
        resp *= -1

    # Clean data based on respiratory signal
    if cfg["clean_by_resp"]:
        ksp, coord, dcf, resp = clean_resp(ksp, coord, dcf, resp, diagnostics_dir)
    after_cleaning_spokes = ksp.shape[1]

    np.save(resp_file, resp)
    np.savetxt(resp_txt_file, resp)

    plt.hist(resp, 100, orientation="horizontal")
    plt.title("resp signal histogram")
    plt.savefig(os.path.join(diagnostics_dir, "resp_histogram.png"))
    plt.close()

    plt.plot(resp)
    plt.title("Entire Waveform")
    plt.savefig(os.path.join(diagnostics_dir, "respWaveformFull.png"))
    plt.close()

    plt.plot(resp[int(60 / tr) : int(60 / tr) + int(120 / tr)])
    plt.title("120 seconds of breathing")
    plt.savefig(os.path.join(diagnostics_dir, "respWaveform120.png"))
    plt.close()

    plt.plot(resp[int(60 / tr) : int(60 / tr) + int(60 / tr)])
    plt.title("60 seconds of breathing")
    plt.savefig(os.path.join(diagnostics_dir, "respWaveform60.png"))
    plt.close()

    # 2) AutoFOV
    log.info("Running AutoFOV...")
    coord = autofov(
        ksp, coord, dcf**2, diagnostics_dir,
        num_ro=cfg["fovNReadout"], thresh=cfg["fovthresh"], device=device, radial=False,
    )

    # --- Data usage tracking ---
    hard_gating_counts = {}
    soft_gating_effectives = {}

    # 3) No Gating Recon
    if cfg["do_noGating"]:
        imgNoGatePath = os.path.join(noGateDir, "noGate.nii.gz")
        if not os.path.isfile(imgNoGatePath) or cfg["overwrite_recons"]:
            Path(noGateDir).mkdir(parents=True, exist_ok=True)
            dicomDir = os.path.join(noGateDir, f"series_{series_nums[0]}")
            imgNoGate = gatedRecon(ksp, coord, dcf, resp, gating_type="none", device=device, flip=False)
            imgNoGate = normalize(sp.resize(np.abs(imgNoGate), final_matrix_size), 0, 255)
            imgNoGate = nib.Nifti1Image(imgNoGate, affine_t)
            nib.save(imgNoGate, imgNoGatePath)
            if UID_base:
                writeDicoms(imgNoGatePath, dicomDir, UID_base=UID_base, subject_id=subject_id, series_num=series_nums[0])
            del imgNoGate

    # 4) Hard Gating Recon
    if cfg["do_HardGating"]:
        Path(hardGateDir).mkdir(parents=True, exist_ok=True)
        for hardgating_weight in hardgating_weights[::-1]:
            imgHardGatePath = os.path.join(hardGateDir, f"hardGate{hardgating_weight:.0f}.nii.gz")
            dicomDir = os.path.join(hardGateDir, f"series_{series_nums[1]}")
            if not os.path.isfile(imgHardGatePath) or cfg["overwrite_recons"]:
                imgHardGate = gatedRecon(
                    ksp, coord, dcf, resp, gating_type="hard",
                    gating_thresh=hardgating_weight, device=device, flip=False,
                )
                imgHardGate = normalize(sp.resize(np.abs(imgHardGate), final_matrix_size), 0, 255)
                imgHardGate = nib.Nifti1Image(imgHardGate, affine_t)
                nib.save(imgHardGate, imgHardGatePath)
                if UID_base:
                    writeDicoms(imgHardGatePath, dicomDir, UID_base=UID_base, subject_id=subject_id, series_num=series_nums[1])
                del imgHardGate
            # Track data usage
            W = gatingWeights(resp, gating_type="hard", percentile=hardgating_weight)
            hard_gating_counts[hardgating_weight] = int(np.sum(W == 1))

    # 5) Soft Gating Recon
    if cfg["do_SoftGating"]:
        Path(softGateDir).mkdir(parents=True, exist_ok=True)
        for softgating_decay in softgating_decays[::-1]:
            imgSoftGatePath = os.path.join(softGateDir, f"softGate{softgating_decay:.1f}.nii.gz")
            dicomDir = os.path.join(softGateDir, f"series_{series_nums[2]}")
            if not os.path.isfile(imgSoftGatePath) or cfg["overwrite_recons"]:
                try:
                    imgSoftGate = gatedRecon(
                        ksp, coord, dcf, resp, gating_type="soft",
                        gating_thresh=20, gating_weight=softgating_decay, device=device, flip=False,
                    )
                except Exception:
                    log.info("GPU memory exceeded or otherwise failed on GPU. Trying CPU.")
                    imgSoftGate = gatedRecon(
                        ksp, coord, dcf, resp, gating_type="soft",
                        gating_thresh=20, gating_weight=softgating_decay, device=-1, flip=False,
                    )
                imgSoftGate = normalize(sp.resize(np.abs(imgSoftGate), final_matrix_size), 0, 255)
                imgSoftGate = nib.Nifti1Image(imgSoftGate, affine_t)
                nib.save(imgSoftGate, imgSoftGatePath)
                if UID_base:
                    writeDicoms(imgSoftGatePath, dicomDir, UID_base=UID_base, subject_id=subject_id, series_num=series_nums[2])
                del imgSoftGate
            # Track data usage
            W = gatingWeights(resp, gating_type="soft", percentile=20, decay=softgating_decay)
            soft_gating_effectives[softgating_decay] = float(np.sum(W))

    # Bin Motion States
    ksp, coord, dcf = bin_periodically(ksp, coord, dcf, resp, n_bins, diagnostics_dir)

    # Track binned data usage
    bin_counts = [ksp_b.shape[1] for ksp_b in ksp]

    del resp

    # 6) Gridded Motion Resolved
    if cfg["do_gridded_motion_resolved"]:
        if not os.path.isfile(griddedmrPath) or cfg["overwrite_recons"]:
            log.info("Running Gridded Motion Resolved Reconstruction...")
            Path(motionResolvedDir).mkdir(parents=True, exist_ok=True)
            gmrimg = griddedRecon(ksp, coord, dcf, n_bins, device=0)
            gmrimg = normalize(np.moveaxis(np.abs(gmrimg), 0, -1), 0, 255)
            gmrimg = np.transpose(gmrimg, (2, 1, 0, 3))
            gmrimg = np.flip(gmrimg, (0, 1, 2))
            gmrimg = sp.resize(gmrimg, final_matrix_size + (n_bins,))
            gmrimg = nib.Nifti1Image(gmrimg, affine_t)
            nib.save(gmrimg, griddedmrPath)
            del gmrimg

    # 7) Low Res XD-GRASP
    if cfg["do_LowRes"]:
        if not os.path.isfile(mrimgLPath) or cfg["overwrite_recons"]:
            log.info("Running Low Res XDGrasp Reconstruction...")
            Path(motionResolvedDir).mkdir(parents=True, exist_ok=True)
            mrimg = xdgrasp(
                ksp, coord, dcf, motionResolvedDir, res_scale=0.75,
                lambda_tv=lowRes_xdgrasp_lambda, device=device, tv_device=0,
                sigma=sigma, tau=tau, outer_iter=iterations,
            )
            mrimg2 = normalize(np.moveaxis(np.abs(mrimg), 0, -1), 0, 255)
            mrimg2 = np.transpose(mrimg2, (2, 1, 0, 3))
            mrimg2 = np.flip(mrimg2, (0, 1, 2))
            mrimg2 = nib.Nifti1Image(mrimg2, affine_t)
            nib.save(mrimg2, mrimgLniiPath)
            np.save(mrimgLPath, mrimg)
            del mrimg, mrimg2

    # 8) iMoCo Recon
    if cfg["do_iMoCoExp"]:
        Path(iterativeMocoDir).mkdir(parents=True, exist_ok=True)
        for reference_frame in reference_frames[::-1]:
            for imoco_lambda in imoco_lambdas[::-1]:
                imgPath = os.path.join(iterativeMocoDir, f"iMoCo{imoco_lambda:.2f}_frame{reference_frame}.nii.gz")
                dicomDir = os.path.join(iterativeMocoDir, f"series_{series_nums[3]}")
                if not os.path.isfile(imgPath) or cfg["overwrite_recons"]:
                    log.info("Running iMoCo Reconstruction...")
                    log.info(f"Using Reference Frame {reference_frame}")
                    try:
                        mrimg = np.load(mrimgLPath)
                    except Exception:
                        log.error("Could not read low res xd-grasp reconstruction")
                        continue
                    register_imoco = 0 if cfg["load_registration"] else 1
                    img = imoco(
                        ksp, coord, dcf, mrimg, iterativeMocoDir,
                        res_scale=1.0, mr_scale=0.75, lambda_tv=imoco_lambda,
                        inner_iter=10, outer_iter=iterations, device=device,
                        nRef=reference_frame, reg_flag=register_imoco,
                        diffusion_reg=0.0, sigma=sigma * 0.8, tau=tau * 0.8,
                        resolution=resolution,
                    )
                    img = normalize(sp.resize(np.abs(img), final_matrix_size), 0, 255)
                    img = nib.Nifti1Image(img, np.eye(4))
                    nib.save(img, imgPath)
                    if UID_base:
                        writeDicoms(imgPath, dicomDir, UID_base=UID_base, subject_id=subject_id, series_num=series_nums[3])
                    del img, mrimg

    # 9) Full Res XD-GRASP
    if cfg["do_HighRes"]:
        Path(motionResolvedDir).mkdir(parents=True, exist_ok=True)
        for xdgrasp_lambda in xdgrasp_lambdas[::-1]:
            mrimgPath = os.path.join(motionResolvedDir, f"MotionResolved{xdgrasp_lambda:.3f}.nii.gz")
            mrimg_expPath = os.path.join(motionResolvedDir, f"MotionResolved_exp{xdgrasp_lambda:.3f}.nii.gz")
            dicomDir = os.path.join(motionResolvedDir, f"series_{series_nums[4]}")
            if not os.path.isfile(mrimgPath) or cfg["overwrite_recons"]:
                log.info("Running Full Res XDGrasp Reconstruction...")
                mrimg = xdgrasp(
                    ksp, coord, dcf, motionResolvedDir, res_scale=1.0,
                    lambda_tv=xdgrasp_lambda, device=device, tv_device=tv_device,
                    sigma=sigma, tau=tau, outer_iter=iterations,
                )
                mrimg = normalize(np.moveaxis(np.abs(mrimg), 0, -1), 0, 255)
                mrimg = np.transpose(mrimg, (2, 1, 0, 3))
                mrimg = np.flip(mrimg, (0, 1, 2))
                mrimg = sp.resize(mrimg, final_matrix_size + (n_bins,))
                mrimg = nib.Nifti1Image(mrimg, affine_t)
                mrimg_exp = nib.Nifti1Image(mrimg.get_fdata()[..., 0], affine_t)
                nib.save(mrimg, mrimgPath)
                nib.save(mrimg_exp, mrimg_expPath)
                if UID_base:
                    writeDicoms(mrimg_expPath, dicomDir, UID_base=UID_base, subject_id=subject_id, series_num=series_nums[4])
                del mrimg, mrimg_exp
                cp._default_memory_pool.free_all_blocks()

        del ksp, coord, dcf

    # 10) Full Res MoCo
    if cfg["do_MoCoExp"]:
        Path(mocoDir).mkdir(parents=True, exist_ok=True)
        for xdgrasp_lambda in xdgrasp_lambdas[::-1]:
            mrimgPath = os.path.join(motionResolvedDir, f"MotionResolved{xdgrasp_lambda:.3f}.nii.gz")
            for reference_frame in reference_frames[::-1]:
                imgPath = os.path.join(mocoDir, f"MoCo{xdgrasp_lambda:.3f}_frame{reference_frame}.nii.gz")
                if not os.path.isfile(imgPath) or cfg["overwrite_recons"]:
                    log.info(f"Running MoCo Registrations")
                    log.info(f"Using Reference Frame {reference_frame}")
                    imgMoco = moco(
                        mrimgPath, mf_dir=os.path.join(motionResolvedDir, "diagnostics"),
                        nRef=reference_frame, reg_flag=1, res_scale=1.0, resolution=resolution,
                    )
                    imgMoco = nib.Nifti1Image(normalize(imgMoco, 0, 255), affine_t)
                    nib.save(imgMoco, imgPath)
                    if UID_base:
                        writeDicoms(imgPath, os.path.join(motionResolvedDir, f"series_{series_nums[4]}"),
                                    UID_base=UID_base, subject_id=subject_id, series_num=series_nums[4])
                    del imgMoco

    # Data usage report
    data_usage_report(
        total_acquired=total_acquired_spokes,
        after_cleaning=after_cleaning_spokes,
        hard_gating_count=hard_gating_counts if hard_gating_counts else None,
        soft_gating_effective=soft_gating_effectives if soft_gating_effectives else None,
        bin_counts=bin_counts,
        n_bins=n_bins,
        diagnostics_dir=diagnostics_dir,
    )

    timeF = (time.time() - timei) / 60
    log.info("Finished Recon in {} minutes".format(timeF))


def main():
    """CLI entry point for the iMoCo reconstruction pipeline."""
    parser = argparse.ArgumentParser(
        description="iMoCo: Iterative Motion Compensation for pulmonary UTE MRI"
    )
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--raw_dir", type=str, default=None, help="Override raw data directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--postfix", type=str, default=None, help="Override postfix string")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.raw_dir:
        cfg["raw_dir"] = args.raw_dir
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.postfix is not None:
        cfg["postfix"] = args.postfix

    if not cfg["raw_dir"] or not cfg["out_dir"]:
        parser.error("raw_dir and out_dir must be specified (in YAML or via CLI)")

    Path(os.path.join(cfg["out_dir"], f"diagnostics{cfg['postfix']}")).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(cfg["out_dir"], f"diagnostics{cfg['postfix']}", "recon_log.txt"), mode="a"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Configuration: {cfg}")
    run(cfg)


if __name__ == "__main__":
    main()
