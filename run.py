import os
import nibabel as nib
from convertUTE import convertUTE
from autofov import autofov
from estimate_resp import estimate_resp
from estimate_respSavitzkyGolay import estimate_respSavitzkyGolay
from bin_motion_states import bin_motion_states
from xdgrasp import xdgrasp
from imoco import imoco
from moco import moco
from gatedRecon import gatedRecon
import logging
from pathlib import Path
import sigpy as sp
import numpy as np
import time
import matplotlib.pyplot as plt
import subprocess
from read_pcvipr_header import read_pcvipr_header

logging.basicConfig(level=logging.INFO)

rawDir = "/home/ltorres/data/rawdata/nicu/P24/"
outDir = "/home/ltorres/data/recon/nicu/P24/"

# rawDir = "/home/ltorres/data/forNara/Patient153/"
# outDir = "/home/ltorres/data/forNara/Patient153/reconOut"

# rawDir = "/home/ltorres/data/forNara/Patient95/"
# outDir = "/home/ltorres/data/forNara/Patient95/reconOut"

# rawDir = "/home/ltorres/data/forNara/Patient137/"
# outDir = "/home/ltorres/data/forNara/Patient137/reconOut"

# set device
device = 0
nBins = 8
nCoils = 1
postfix = ""
register = 1
dc_signal = 1
nRef = -1  # Reference Frame (last ie expiratory)
xp = sp.Device(device).xp
spokesDSF = 1.0
fovthresh = 0.1
fovNReadout = 70
# 1.5)
tr = 0.00502  # nicu
imoco_lambdas = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
xdgrasp_lambda = 0.01
lowRes_xdgrasp_lambda = xdgrasp_lambda * 0.75
logging.info("Low Res XDGRASP Lambda: {}".format(lowRes_xdgrasp_lambda))
tv_device = 0

overwrite_raw = False
overWriteNoGating = True
overWriteHardGating = True
overWriteSoftGating = True
overWriteLowRes = True
overWriteiMoCoExp = True
overWriteiMoCoInsp = False
overWriteHighRes = True
overWriteMoCoExp = True
overWriteMoCoInsp = False

try:
    timei = time.time()
    # print(visit)
    motionResolvedDir = os.path.join(outDir, "MotionResolved" + postfix)
    iterativeMocoDir = os.path.join(outDir, "IterativeMoCo" + postfix)
    iterativeMocoInspDir = os.path.join(outDir, "IterativeMoCoInsp" + postfix)
    mocoDir = os.path.join(outDir, "MoCo" + postfix)
    mocoInspDir = os.path.join(outDir, "MoCoInsp" + postfix)
    noGateDir = os.path.join(outDir, "NoGate" + postfix)
    hardGateDir = os.path.join(outDir, "HardGate" + postfix)
    softGateDir = os.path.join(outDir, "SoftGate" + postfix)
    Path(rawDir).mkdir(parents=True, exist_ok=True)
    # print(rawDir)

    # Copy Raw Data
    fileList = os.listdir(rawDir)
    if "MRI_Raw.h5" in fileList:
        print("File Exists, Not Copying!")
    else:
        currDir = os.getcwd()
        print("need to fix this.")
        # os.chdir(rawDir)
        # subprocess.call(["pcvipr_recon_binary", "-f ", rawDir+"P\*", "-export_kdata"])
        # os.chdir(currDir)

    fileList = os.listdir(rawDir)
    if "MRI_Raw.h5" in fileList:
        print("File Exists, Begin!")
    else:
        pass
    # Set up data paths
    h5Path = os.path.join(rawDir, "MRI_Raw.h5")
    diagnostics_dir = os.path.join(outDir, "diagnostics/")
    mrimgPath = os.path.join(motionResolvedDir, "MotionResolved.nii.gz")
    mrimgLPath = os.path.join(motionResolvedDir, "MotionResolvedLowRes.nii.gz")
    imgMocoPath = os.path.join(mocoDir, "MoCo.nii.gz")
    imgPath = os.path.join(iterativeMocoDir, "iMoCo.nii.gz")
    respPath = os.path.join(diagnostics_dir, "resp.npy")
    imgInspPath = os.path.join(iterativeMocoInspDir, "iMoCo.nii.gz")
    imgMocoInspPath = os.path.join(mocoInspDir, "MoCo.nii.gz")
    imgNoGatePath = os.path.join(noGateDir, "noGate.nii.gz")
    imgHardGatePath = os.path.join(hardGateDir, "hardGate.nii.gz")
    imgSoftGatePath = os.path.join(softGateDir, "softGate.nii.gz")
    ksp_file = os.path.join(rawDir, "ksp.npy")
    coord_file = os.path.join(rawDir, "coord.npy")
    dcf_file = os.path.join(rawDir, "dcf.npy")
    resp_file = os.path.join(rawDir, "resp.npy")

    # Create outpath if doesn't exist.
    Path(motionResolvedDir).mkdir(parents=True, exist_ok=True)
    Path(iterativeMocoDir).mkdir(parents=True, exist_ok=True)
    Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)
    Path(mocoDir).mkdir(parents=True, exist_ok=True)
    # Path(iterativeMocoInspDir).mkdir(parents=True, exist_ok=True)
    # Path(mocoInspDir).mkdir(parents=True, exist_ok=True)
    Path(noGateDir).mkdir(parents=True, exist_ok=True)
    Path(hardGateDir).mkdir(parents=True, exist_ok=True)
    Path(softGateDir).mkdir(parents=True, exist_ok=True)

    # 1) Convert MRI_Raw.h5 to cfl and read resp waveform.
    if os.path.isfile(ksp_file) is False or overwrite_raw is True:
        logging.info("Loading and Saving.....")
        logging.info("Running File Conversion...")
        ksp, coord, dcf, resp = convertUTE(h5Path, nCoils, dsfSpokes=spokesDSF)
        try:
            os.remove(ksp_file)
            os.remove(coord_file)
            os.remove(dcf_file)
            os.remove(resp_file)
        except OSError:
            pass
        np.save(ksp_file, ksp)
        np.save(coord_file, coord)
        np.save(dcf_file, dcf)
        np.save(resp_file, resp)
    else:
        ksp = np.load(ksp_file)
        coord = np.load(coord_file)
        dcf = np.load(dcf_file)
        resp = np.load(resp_file)

    # Scale DCF for improved convergence
    dcf **= 0.5

    # Read Affine Transformation
    # header = read_pcvipr_header(rawDir)
    # affine_t = np.array(
    #     [
    #         [header["ix"], header["iy"], header["iz"], 0],
    #         [header["jx"], header["jy"], header["jz"], 0],
    #         [header["kx"], header["ky"], header["kz"], 0],
    #         [header["sx"], header["sy"], header["sz"], 1],
    #     ]
    # )
    affine_t = np.eye(4)
    # print(affine_t.shape)
    logging.info("Kspace Shape: {}...".format(ksp.shape))
    logging.info("trajectory Shape: {}...".format(coord.shape))
    logging.info("DCF Shape: {}....".format(dcf.shape))

    if dc_signal == 1:
        logging.info("Estimating Resp Waveform from DC signal...")
        logging.info("Using TR: {} seconds".format(tr))
        # resp = estimate_resp(ksp[:, :, 0], tr * 2, fl=0.25, fh=1.2, fw=0.01, usePhase=False)
        resp = estimate_respSavitzkyGolay(
            ksp[:, :, 0],
            tr,
            window=0.8,
            order=2,
            detrend_window=10.0,
            usePhase=False,
            useDetrend=True,
        )

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
        dcf ** 2,
        diagnostics_dir,
        num_ro=fovNReadout,
        thresh=fovthresh,
        device=device,
        radial=False,
    )

    # 3) noGating Recon
    if os.path.exists(imgNoGatePath) is False or overWriteNoGating is True:
        imgNoGate = gatedRecon(ksp, coord, dcf, resp, gating_type="none", device=device, flip=True)
        imgNoGate = sp.resize(np.abs(imgNoGate), (256, 256, 256))
        imgNoGate = nib.Nifti1Image(imgNoGate, affine_t)
        nib.save(imgNoGate, imgNoGatePath)
        del imgNoGate

    # 4) hardGating Recon
    if os.path.exists(imgHardGatePath) is False or overWriteHardGating is True:
        imgHardGate = gatedRecon(ksp, coord, dcf, resp, gating_type="hard", gating_thresh=50, device=device, flip=True)
        imgHardGate = sp.resize(np.abs(imgHardGate), (256, 256, 256))
        imgHardGate = nib.Nifti1Image(imgHardGate, affine_t)
        nib.save(imgHardGate, imgHardGatePath)
        del imgHardGate

    # 5) softGating Recon
    if os.path.exists(imgSoftGatePath) is False or overWriteSoftGating is True:
        imgSoftGate = gatedRecon(
            ksp,
            coord,
            dcf,
            resp,
            gating_type="soft",
            gating_thresh=25,
            gating_weight=1,
            device=-device,
            flip=True,
        )
        imgSoftGate = sp.resize(np.abs(imgSoftGate), (256, 256, 256))
        imgSoftGate = nib.Nifti1Image(imgSoftGate, affine_t)
        nib.save(imgSoftGate, imgSoftGatePath)
        del imgSoftGate

    # 6) Bin Motion States
    logging.info("Running bin_motion_states...")
    ksp, coord, dcf = bin_motion_states(
        ksp, coord, dcf, resp, nBins, diagnostics_dir, filter_bulk=True, filter_extremes=True
    )

    # 7) Low Res xdgrasp recon
    if os.path.exists(mrimgLPath) is False or overWriteLowRes is True:
        logging.info("Running Low Res XDGrasp Reconstruction...")
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
        mrimgL = sp.resize(mrimg, (nBins, 192, 192, 192))
        mrimgL = np.moveaxis(np.abs(mrimgL), 0, -1)
        mrimgL = np.transpose(mrimgL, (2, 1, 0, 3))
        mrimgL = np.flip(mrimgL, (0, 1, 2))
        mrimgL = nib.Nifti1Image(mrimgL, affine_t)
        nib.save(mrimgL, mrimgLPath)
        del mrimgL

    # 8) iMoCo recon expir
    if os.path.exists(imgPath) is False or overWriteiMoCoExp is True:
        logging.info("Running iMoCo Reconstruction...")
        counter = 0
        for imoco_lambda in imoco_lambdas[::-1]:
            if len(imoco_lambdas) != 1:
                imgPath = os.path.join(iterativeMocoDir, "iMoCo{}.nii.gz".format(imoco_lambda))
            if counter != 0:
                register_imoco = 0
            else:
                register_imoco = register
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
                nRef=nRef,
                reg_flag=register_imoco,
            )
            img = sp.resize(np.abs(img), (256, 256, 256))
            img = nib.Nifti1Image(img, np.eye(4))
            nib.save(img, imgPath)
            counter += 1
            del img

    # # 9) iMoCo recon Insp
    # if os.path.exists(imgInspPath) is False or overWriteiMoCoInsp is True:
    #     logging.info("Running iMoCo Inspiratory Reconstruction...")
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
    if os.path.exists(mrimgPath) is False or overWriteHighRes is True:
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
        mrimg = np.moveaxis(np.abs(mrimg), 0, -1)
        mrimg = np.transpose(mrimg, (2, 1, 0, 3))
        mrimg = np.flip(mrimg, (0, 1, 2))
        mrimg = nib.Nifti1Image(mrimg, affine_t)
        nib.save(mrimg, mrimgPath)
        del ksp, coord, dcf, mrimg

    # 11) Full Res MoCo Expiratory
    if os.path.exists(imgMocoPath) is False or overWriteMoCoExp is True:
        imgMoco = moco(mrimgPath, diagnostics_dir, nRef=nRef)
        imgMoco = nib.Nifti1Image(imgMoco, affine_t)
        nib.save(imgMoco, imgMocoPath)
        del imgMoco

    # # 12) Full Res MoCo Inspiratory
    # if os.path.exists(imgMocoInspPath) is False or overWriteMoCoInsp is True:
    #     imgMocoInsp = moco(mrimgPath, diagnostics_dir, nRef=0)
    #     imgMocoInsp = nib.Nifti1Image(imgMocoInsp, np.eye(4))
    #     nib.save(imgMocoInsp, imgMocoInspPath)
    #     del imgMocoInsp

    timeF = (time.time() - timei) / 60
    logging.info("Finshed Recon in {} minutes".format(timeF))
    # destroy data every loop
except KeyboardInterrupt:
    print("interrupted!")
