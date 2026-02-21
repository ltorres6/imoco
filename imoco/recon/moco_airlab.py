"""MoCo reconstruction using AirLab registration (experimental).

This module is experimental and was not described in the published paper.
It uses AirLab instead of ANTs for the registration step within the
post-hoc MoCo framework. Requires ``airlab`` and ``torch``.
"""

import logging
import os
import time

import nibabel as nib
import numpy as np
import sigpy as sp
import torch as th
from scipy.ndimage import median_filter
from tqdm import trange

from imoco.registration.airlab import regAirlab
from imoco.registration.interpolation import interp_op
from imoco.utils.normalize import normalize

try:
    import airlab as al
except ImportError:
    al = None


def moco(mrimgPath, mf_dir, nRef=-1, reg_flag=1, res_scale=1.0):
    """Post-hoc MoCo using AirLab Demons registration (experimental).

    Args:
        mrimgPath (str): Path to motion-resolved NIfTI image.
        mf_dir (str): Directory for motion field diagnostics.
        nRef (int): Reference frame index.
        reg_flag (int): 1 to compute, 0 to load.
        res_scale (float): Resolution scale.

    Returns:
        ndarray: Motion-compensated 3D image.
    """
    if al is None:
        raise ImportError("airlab is required for this module")

    timeStart = time.time()
    mrimg = nib.load(mrimgPath).get_fdata()
    mrimg = np.moveaxis(np.abs(mrimg), -1, 0)
    nPhases = mrimg.shape[0]
    tshape = mrimg.shape[1:]
    vox_res = [n * res_scale for n in [1, 1, 1]]
    logging.info("Registration...")
    M_fields = []

    if reg_flag == 1:
        timei = time.time()
        pbar = trange(nPhases, leave=True, ncols=80)
        for ii in pbar:
            pbar.set_description(f"Registering Frame: {ii}...")
            M_field, _ = regAirlab(
                median_filter(np.abs(mrimg[nRef]), [3, 3, 3]),
                median_filter(np.abs(mrimg[ii]), [3, 3, 3]),
                vox_res=vox_res,
            )
            M_fields.append(M_field)
        del M_field

        logging.info("Motion Field scaling...")
        M_fields = [
            np.flip(
                np.squeeze(al.transformation.utils.upsample_displacement(M, tshape, interpolation="linear").cpu().numpy()),
                -1,
            )
            for M in M_fields
        ]
        th.cuda.empty_cache()

        logging.info("Saving Motion Fields as nii...")
        tmp = np.asarray(M_fields)
        tmp = np.moveaxis(tmp, 0, -1)
        tmp = np.transpose(tmp, (2, 1, 0, 3, 4))
        tmp = np.flip(tmp, (0, 1, 2))
        tmp = nib.Nifti1Image(tmp, np.eye(4))
        nib.save(tmp, os.path.join(mf_dir, "M_mr.nii.gz"))

        timeF = (time.time() - timei) / 60
        logging.info("Finished Registration in {} minutes".format(timeF))
        del tmp
    else:
        logging.info("Reading Motion Fields from disk...")
        M_fields = nib.load(os.path.join(mf_dir, "M_mr.nii.gz")).get_fdata()
        M_fields = np.flip(M_fields, (0, 1, 2))
        M_fields = np.transpose(M_fields, (2, 1, 0, 3, 4))
        M_fields = np.moveaxis(M_fields, -1, 0)

    img = 0
    pbar = trange(nPhases, leave=True)
    for ii in pbar:
        pbar.set_description("Applying Warp To Frame # {}...".format(ii))
        M = interp_op(tshape, M_fields[ii])
        img += M * mrimg[ii]
    img = img / nPhases

    timeFinish = time.time()
    logging.info("Moco Registration Finished in: {} min...".format((timeFinish - timeStart) / 60))
    return img
