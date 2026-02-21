import logging
import os

import numpy as np
import sigpy as sp
from PIL import Image
from scipy import ndimage
from skimage import measure, transform

from imoco.utils.normalize import normalize


def getLargestCC(mask):
    labels = measure.label(mask)
    assert labels.max() != 0
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def autofov(ksp, coord, dcf, diagnostics_dir, num_ro=100, device=-1, thresh=0.4, radial=False):
    """Automatic estimation of field-of-view (FOV).

    FOV is estimated by thresholding a low resolution gridded image.
    coord will be modified in-place.

    Args:
        ksp (ndarray): K-space measurements of shape (C, num_tr, num_ro).
        coord (ndarray): K-space coordinates of shape (num_tr, num_ro, D).
        dcf (ndarray): Density compensation factor of shape (num_tr, num_ro).
        diagnostics_dir (str): Directory to save diagnostic images.
        num_ro (int): Number of readout points for low-res estimation.
        device (int): Computing device (-1 for CPU, >=0 for GPU).
        thresh (float): Threshold between 0 and 1 for FOV mask.
        radial (bool): Whether data is radially sampled from center.

    Returns:
        ndarray: Modified coordinates scaled to the estimated FOV.
    """
    device = sp.Device(device)
    xp = device.xp
    with device:
        if radial is True:
            ro_center = ksp.shape[2] // 2
            ro_range = slice(ro_center - num_ro // 2, ro_center + num_ro // 2, 1)
        else:
            ro_range = slice(0, num_ro, 1)
        logging.info("AutoFov Input Shape: {}".format(sp.estimate_shape(coord)))

        kspc = ksp[:, :, ro_range]
        coordc = coord[:, ro_range, :]
        dcfc = dcf[:, ro_range]
        coordc2 = sp.to_device(coordc * 2, device)
        num_coils = len(kspc)
        imgc_shape = np.array(sp.estimate_shape(coordc))
        imgc2_shape = sp.estimate_shape(coordc2)
        imgc2_center = [i // 2 for i in imgc2_shape]
        logging.info("Adjoint Nufft 1")
        imgc2 = sp.nufft_adjoint(sp.to_device(dcfc * kspc, device), coordc2, [num_coils] + imgc2_shape)
        imgc2 = xp.sum(xp.abs(imgc2) ** 2, axis=0) ** 0.5
        imgc2 = sp.to_device(imgc2)
        imgc2 = ndimage.median_filter(imgc2, (3, 3, 3))
        imgc2 /= imgc2.max()

        imc = normalize(imgc2[:, imgc2.shape[1] // 2, :], 0, 255)
        imc = Image.fromarray(transform.resize(imc, (256, 256)))
        imc = imc.convert("L")
        imc.save(os.path.join(diagnostics_dir, "autofov_lowResCoronal.jpg"))

        ims = normalize(imgc2[:, :, imgc2.shape[2] // 2], 0, 255)
        ims = Image.fromarray(transform.resize(ims, (256, 256)))
        ims = ims.convert("L")
        ims.save(os.path.join(diagnostics_dir, "autofov_lowResSaggital.jpg"))

        ima = normalize(imgc2[imgc2.shape[0] // 2, :, :], 0, 255)
        ima = Image.fromarray(transform.resize(ima, (256, 256)))
        ima = ima.convert("L")
        ima.save(os.path.join(diagnostics_dir, "autofov_lowResAxial.jpg"))

        thresh *= imgc2.max()
        boxc = imgc2 > thresh
        boxc = getLargestCC(boxc).astype(float)
        imc = normalize(boxc[:, boxc.shape[1] // 2, :], 0, 255)
        imc = Image.fromarray(transform.resize(imc, (256, 256)))
        imc = imc.convert("1")
        imc.save(os.path.join(diagnostics_dir, "autofov_maskCoronal.jpg"))

        ims = normalize(boxc[:, :, boxc.shape[2] // 2], 0, 255)
        ims = Image.fromarray(transform.resize(ims, (256, 256)))
        ims = ims.convert("1")
        ims.save(os.path.join(diagnostics_dir, "autofov_maskSaggital.jpg"))

        ima = normalize(boxc[boxc.shape[0] // 2, :, :], 0, 255)
        ima = Image.fromarray(transform.resize(ima, (256, 256)))
        ima = ima.convert("1")
        ima.save(os.path.join(diagnostics_dir, "autofov_maskAxial.jpg"))

        boxc_idx = np.nonzero(boxc)
        boxc_shape = np.array([int(np.abs(boxc_idx[i] - imgc2_center[i]).max()) * 2 for i in range(imgc2.ndim)])
        img_scale = boxc_shape / imgc_shape
        if radial:
            img_scale *= 2
        coord *= img_scale

        coordc = coord[:, ro_range, :]
        coordc = sp.to_device(coordc, device)
        num_coils = len(kspc)
        imgc_shape = sp.estimate_shape(coordc)
        logging.info("Adjoint Nufft 2")
        imgc = sp.nufft_adjoint(sp.to_device(dcfc * kspc, device), coordc, [num_coils] + imgc_shape)
        imgc = xp.sum(xp.abs(imgc) ** 2, axis=0) ** 0.5
        imgc = sp.to_device(xp.abs(imgc))

        imc = normalize(imgc[:, imgc.shape[1] // 2, :], 0, 255)
        imc = Image.fromarray(transform.resize(imc, (256, 256)))
        imc = imc.convert("L")
        imc.save(os.path.join(diagnostics_dir, "autofov_croppedCoronal.jpg"))

        ims = normalize(imgc[:, :, imgc.shape[2] // 2], 0, 255)
        ims = Image.fromarray(transform.resize(ims, (256, 256)))
        ims = ims.convert("L")
        ims.save(os.path.join(diagnostics_dir, "autofov_croppedSaggital.jpg"))

        ima = normalize(imgc[imgc.shape[0] // 2, :, :], 0, 255)
        ima = Image.fromarray(transform.resize(ima, (256, 256)))
        ima = ima.convert("L")
        ima.save(os.path.join(diagnostics_dir, "autofov_croppedAxial.jpg"))

        logging.info("AutoFov Output Shape: {}".format(sp.estimate_shape(coord)))
        logging.info("Scaling Factors: {}".format(img_scale))
        np.savetxt(os.path.join(diagnostics_dir, "fovScaleFactors.txt"), img_scale)

        return coord
