import logging
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import median_filter
from tqdm import trange

from imoco.registration.ants import ANTsReg
from imoco.registration.interpolation import interp_op
from imoco.utils.normalize import normalize


def moco(
    mrimgPath, mf_dir=None, nRef=-1, reg_flag=1, res_scale=1.0, resolution=[1.25, 1.25, 1.25]
):
    """Post-hoc motion compensation by registering and averaging motion states.

    Loads a motion-resolved image (e.g., from XD-GRASP), registers each
    motion state to a reference frame, and averages the warped images.

    Reference: Section II-H of the JMRI paper.

    Args:
        mrimgPath (str): Path to a motion-resolved NIfTI image (4D).
        mf_dir (str, optional): Directory for motion field diagnostics.
        nRef (int): Reference frame index (-1 for last frame = end-expiration).
        reg_flag (int): 1 to compute registration, 0 to skip.
        res_scale (float): Resolution scale factor.
        resolution (list): Voxel resolution in mm, length 3.

    Returns:
        ndarray: Motion-compensated 3D image.
    """
    timeStart = time.time()
    mrimg = nib.load(mrimgPath).get_fdata()
    mrimg = np.moveaxis(np.abs(mrimg), -1, 0)
    nPhases = mrimg.shape[0]
    tshape = mrimg.shape[1:]
    logging.info("Registration...")
    M_fields = []
    iM_fields = []
    vox_res = [r / res_scale for r in resolution]

    if reg_flag == 1:
        timei = time.time()
        pbar = trange(nPhases, leave=True)
        for ii in pbar:
            pbar.set_description("Registering Frame # {}...".format(ii))
            M_field, iM_field = ANTsReg(
                normalize(median_filter(np.abs(mrimg[nRef]), 3), 0, 1),
                normalize(median_filter(np.abs(mrimg[ii]), 3), 0, 1),
                vox_res=vox_res,
            )
            M_fields.append(M_field)
            iM_fields.append(iM_field)
        M_fields = np.asarray(M_fields)
        iM_fields = np.asarray(iM_fields)
        timeF = (time.time() - timei) / 60
        logging.info("Finished Registration in {} minutes".format(timeF))

    iM_fields = [iM_fields[i] for i in range(iM_fields.shape[0])]
    M_fields = [M_fields[i] for i in range(M_fields.shape[0])]

    img = 0
    pbar = trange(nPhases, leave=True)
    for ii in pbar:
        pbar.set_description("Applying Warp To Frame # {}...".format(ii))
        M = interp_op(tshape, M_fields[ii])
        img += M * mrimg[ii]
    img = img / nPhases

    timeFinish = time.time()
    logging.info(
        "Moco Registration Finished in: {} min...".format((timeFinish - timeStart) / 60)
    )
    return img
