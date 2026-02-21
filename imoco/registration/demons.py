import logging

import numpy as np
import sigpy as sp
from scipy import ndimage

from imoco.registration.interpolation import M_scale, interp


def imgrad3d(I):
    gx = I - sp.circshift(I, (-1,), axes=(0,))
    gx[-1, :, :] = 0
    gy = I - sp.circshift(I, (-1,), axes=(1,))
    gy[:, -1, :] = 0
    gz = I - sp.circshift(I, (-1,), axes=(2,))
    gz[:, :, -1] = 0
    return gx, gy, gz


def lap3d(I):
    gxx = sp.circshift(I, (-1,), axes=(0,)) + sp.circshift(I, (1,), axes=(0,)) - 2 * I
    gyy = sp.circshift(I, (-1,), axes=(1,)) + sp.circshift(I, (1,), axes=(1,)) - 2 * I
    gzz = sp.circshift(I, (-1,), axes=(2,)) + sp.circshift(I, (1,), axes=(2,)) - 2 * I
    return gxx + gyy + gzz


def pmask(I, sigma):
    I = np.abs(I)
    mask = np.abs(I) > sigma
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = ndimage.morphology.binary_opening(mask, structure=np.ones((5, 5, 5)))
    return mask


def DemonsReg4(Is, ref=0, level=3, device=-1):
    M_fields = []
    nphase = len(Is)
    logging.info("4D Demons registration:")
    for i in range(nphase):
        logging.info("Ref/Mov:{}/{}".format(i, ref))
        M_field = Demons(np.abs(Is[ref]), np.abs(Is[i]), level=level, device=device)
        M_fields.append(M_field)
    return np.asarray(M_fields)


def Demons(
    If,
    Im,
    level,
    device=-1,
    rho=0.7,
    sigmas_f=[2, 2, 2, 3],
    sigmas_e=[2, 2, 2, 2],
    sigmas_s=[0.5, 0.5, 1, 1],
    iters=[40, 40, 40, 20, 20],
):
    """Diffeomorphic Demons registration.

    Multi-scale Demons registration between a fixed and moving image.

    Args:
        If (ndarray): Fixed image, 3D.
        Im (ndarray): Moving image, 3D.
        level (int): Number of multi-resolution levels.
        device (int): Computing device (-1 for CPU).
        rho (float): Momentum parameter.
        sigmas_f (list): Fluid regularization sigmas per level.
        sigmas_e (list): Elastic regularization sigmas per level.
        sigmas_s (list): Image smoothing sigmas per level.
        iters (list): Number of iterations per level.

    Returns:
        ndarray: Displacement field of shape (*Im.shape, 3).
    """
    Im = np.abs(Im)
    m_scale = np.max(Im)
    Im = Im / m_scale
    If = np.abs(If)
    If = If / m_scale

    M = np.zeros(Im.shape + (3,))
    Mt = np.zeros(Im.shape + (3,))
    for k in range(level):
        logging.info("Demons Level:{}".format(k))
        scale = 2 ** (level - k - 1)
        sigma_f = sigmas_f[k]
        sigma_e = sigmas_e[k]
        sigma_s = sigmas_s[k]
        iter_each_level = iters[k]

        Ift = ndimage.zoom(If, zoom=1 / scale, order=2)
        Ift = ndimage.gaussian_filter(Ift, sigma=sigma_s, truncate=2.0)
        Imt = ndimage.zoom(Im, zoom=1 / scale, order=2)
        Imt = ndimage.gaussian_filter(Imt, sigma=sigma_s, truncate=2.0)
        Imask = pmask(Imt + Ift, 1e-2)

        Isizet = Ift.shape
        Mt = M_scale(Mt, Isizet)
        uo = np.zeros_like(Mt)
        for i in range(iter_each_level):
            Imm = interp(Imt, Mt, device=sp.Device(device), k_id=1)
            Ifm = interp(Ift, -Mt, device=sp.Device(device), k_id=1)
            dI = Ifm - Imm
            Is = (Ifm + Imm) / 2

            gIx, gIy, gIz = imgrad3d(Is)
            gI = np.sqrt(np.abs(gIx**2 + gIy**2 + gIz**2) + 1e-6)
            discriminator = gI**2 + np.abs(dI) ** 2
            dI = dI * 3.0
            ux = -dI * gIx / discriminator
            uy = -dI * gIy / discriminator
            uz = -dI * gIz / discriminator

            mask = (gI < 1e-4) | (~Imask)
            ux[np.isnan(ux) | mask] = 0
            uy[np.isnan(uy) | mask] = 0
            uz[np.isnan(uz) | mask] = 0

            ux = np.maximum(np.minimum(ux, 1), -1)
            uy = np.maximum(np.minimum(uy, 1), -1)
            uz = np.maximum(np.minimum(uz, 1), -1)
            ux = ndimage.gaussian_filter(ux, sigma=sigma_f)
            uy = ndimage.gaussian_filter(uy, sigma=sigma_f)
            uz = ndimage.gaussian_filter(uz, sigma=sigma_f)

            Mt[..., 0] = Mt[..., 0] + rho * ux + (1 - rho) * uo[..., 0]
            Mt[..., 1] = Mt[..., 1] + rho * uy + (1 - rho) * uo[..., 1]
            Mt[..., 2] = Mt[..., 2] + rho * uz + (1 - rho) * uo[..., 2]
            uo[..., 0] = ux
            uo[..., 1] = uy
            uo[..., 2] = uz

            Mt[..., 0] = ndimage.gaussian_filter(Mt[..., 0], sigma=sigma_e)
            Mt[..., 1] = ndimage.gaussian_filter(Mt[..., 1], sigma=sigma_e)
            Mt[..., 2] = ndimage.gaussian_filter(Mt[..., 2], sigma=sigma_e)

    M = M_scale(Mt * 2, Im.shape)
    return M
