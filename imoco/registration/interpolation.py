import numpy as np
import sigpy as sp
from scipy import ndimage
from sigpy import backend
from sigpy.linop import Linop


def M_scale(M, oshape, scale=1):
    """Scale a motion field to a target shape.

    Args:
        M (ndarray): Motion field of shape (..., ndim).
        oshape (tuple): Target spatial shape.
        scale (float): Additional scaling factor.

    Returns:
        ndarray: Scaled motion field of shape (*oshape, ndim).
    """
    Mscale = [oshape[i] / M.shape[i] for i in range(M.shape[-1])]
    Mo = np.zeros(oshape + (M.shape[-1],))
    for i in range(M.shape[-1]):
        M[..., i] = M[..., i] * (Mscale[i] * scale)
        Mo[..., i] = ndimage.zoom(M[..., i], zoom=tuple(Mscale), order=2)

    return Mo


class interp_op(Linop):
    """Interpolation linear operator using a displacement field.

    Args:
        ishape (tuple): Image shape.
        M_field (ndarray): Forward displacement field of shape (*ishape, ndim).
        iM_field (ndarray, optional): Inverse displacement field for the adjoint.
    """

    def __init__(self, ishape, M_field, iM_field=None):
        assert list(ishape) == list(M_field.shape[:-1]), "Dimension mismatch!"
        oshape = ishape
        self.M_field = M_field
        self.iM_field = iM_field
        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)
        with device:
            return interp(input, self.M_field, device)

    def _adjoint_linop(self):
        if self.iM_field is None:
            inv_field = -self.M_field
            M_field = None
        else:
            inv_field = self.iM_field
            M_field = self.M_field

        return interp_op(self.ishape, inv_field, M_field)


def interp(I, M_field, device=sp.Device(-1), k_id=1, deblur=True):
    """Interpolate an image using a displacement field.

    Args:
        I (ndarray): Input image.
        M_field (ndarray): Displacement field of shape (*I.shape, ndim).
        device (Device): Computing device.
        k_id (int): Interpolation kernel (0=cubic B-spline, 1=linear).
        deblur (bool): Apply deblurring convolution (only for cubic B-spline).

    Returns:
        ndarray: Warped image.
    """
    N = 64
    if k_id == 0:
        kernel = [
            (3 * (x / N) ** 3 - 6 * (x / N) ** 2 + 4) / 6 for x in range(0, N)
        ] + [(2 - x / N) ** 3 / 6 for x in range(N, 2 * N)]
        dkernel = np.array([-0.2, 1.4, -0.2])
        k_wid = 4
    else:
        kernel = [1 - x / (2 * N) for x in range(0, 2 * N)]
        dkernel = np.array([0, 1, 0])
        deblur = False
        k_wid = 2
    kernel = np.asarray(kernel)

    c_device = sp.get_device(I)
    ndim = M_field.shape[-1]

    if ndim == 3:
        dkernel = (
            dkernel[:, None, None] * dkernel[None, :, None] * dkernel[None, None, :]
        )
        Nx, Ny, Nz = I.shape
        my, mx, mz = np.meshgrid(np.arange(Ny), np.arange(Nx), np.arange(Nz))
        m = np.stack((mx, my, mz), axis=-1)
        M_field = M_field + m
    else:
        dkernel = dkernel[:, None] * dkernel[None, :]
        Nx, Ny = I.shape
        my, mx = np.meshgrid(np.arange(Ny), np.arange(Nx))
        m = np.stack((mx, my), axis=-1)
        M_field = M_field + m

    g_device = device
    I = sp.to_device(input=I, device=g_device)
    I = sp.interp.interpolate(I, sp.to_device(M_field.astype(np.float64), g_device))

    if deblur is True:
        sp.conv.convolve(I, sp.to_device(dkernel, g_device))
    I = sp.to_device(input=I, device=c_device)

    return I
