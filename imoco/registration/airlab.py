"""AirLab-based image registration (experimental).

This module is experimental and was not used in the published paper.
It requires the ``airlab`` and ``torch`` packages.
"""

import logging

import numpy as np
import sigpy as sp
import torch as th
from scipy.ndimage import median_filter
from sigpy import backend
from sigpy.linop import Linop
from skimage.exposure import match_histograms

from imoco.utils.normalize import normalize

try:
    import airlab as al
except ImportError:
    al = None


def regAirlab(fixed_image, moving_image, vox_res=[1, 1, 1]):
    """Register two 3D images using AirLab Demons.

    Args:
        fixed_image (ndarray): Fixed (reference) image, 3D.
        moving_image (ndarray): Moving image, 3D.
        vox_res (list): Voxel resolution in mm, length 3.

    Returns:
        tuple: (displacement, inv_displacement) as torch tensors.
    """
    if al is None:
        raise ImportError("airlab is required for regAirlab. Install with: pip install airlab")

    dtype = th.float32
    device = th.device("cuda:0")

    fixed_image = normalize(fixed_image, 0, 1)
    moving_image = normalize(moving_image, 0, 1)
    moving_image = match_histograms(moving_image, fixed_image)
    fixed_image = al.image_from_numpy(
        fixed_image, vox_res, [0, 0, 0], dtype=dtype, device=device
    )
    moving_image = al.image_from_numpy(
        moving_image, vox_res, [0, 0, 0], dtype=dtype, device=device
    )

    downsample_factor = [[6, 6, 6], [2, 2, 2], [1, 1, 1]]
    constant_flow = None
    number_of_iterations = [5000, 3000, 1000]
    sigma = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]

    for level, dsf in enumerate(downsample_factor):
        fix_im_level = al.create_downsampled_image(fixed_image, dsf)
        mov_im_level = al.create_downsampled_image(moving_image, dsf)
        registration = al.DemonsRegistraion(verbose=False)

        transformation = al.transformation.pairwise.NonParametricTransformation(
            mov_im_level.size, dtype=dtype, device=device, diffeomorphic=True
        )
        if level > 0:
            constant_flow = al.transformation.utils.upsample_displacement(
                constant_flow, mov_im_level.size, interpolation="linear"
            )
            transformation.set_constant_flow(constant_flow)

        registration.set_transformation(transformation)
        image_loss = al.loss.pairwise.MSE(fix_im_level, mov_im_level)
        registration.set_image_loss([image_loss])
        regulariser = al.regulariser.demons.GaussianRegulariser(
            mov_im_level.spacing, sigma=sigma[level], dtype=dtype, device=device
        )
        registration.set_regulariser([regulariser])
        optimizer = th.optim.Adam(transformation.parameters())
        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(number_of_iterations[level])
        registration.start()
        constant_flow = transformation.get_flow()

    displacement = transformation.get_displacement()
    displacement = al.create_displacement_image_from_image(
        transformation.get_displacement(), fixed_image
    )
    displacement = al.transformation.utils.unit_displacement_to_displacement(displacement)

    inv_displacement = transformation.get_inverse_displacement()
    inv_displacement = al.create_displacement_image_from_image(
        transformation.get_inverse_displacement(), moving_image
    )
    inv_displacement = al.transformation.utils.unit_displacement_to_displacement(inv_displacement)

    th.cuda.empty_cache()
    return displacement, inv_displacement


class interp_al_op(Linop):
    """AirLab-based interpolation operator (experimental)."""

    def __init__(self, ishape, M_field, iM_field=None, vox_res=[1, 1, 1]):
        assert list(ishape) == list(list(M_field.shape)[:-1]), "Dimension mismatch!"
        oshape = ishape
        self.M_field = M_field
        self.iM_field = iM_field
        self.vox_res = vox_res
        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)
        with device:
            dtype = th.float32
            th_device = th.device("cuda:0")
            input_d = sp.to_device(input)
            input_d = al.image_from_numpy(
                input_d, self.vox_res, [0, 0, 0], dtype=dtype, device=th_device
            )
            warped_im = al.transformation.utils.warp_image(input_d, self.M_field)
            return sp.to_device(warped_im.numpy(), device)

    def _adjoint_linop(self):
        if self.iM_field is None:
            iM_field = -self.M_field
            M_field = None
        else:
            iM_field = self.iM_field
            M_field = self.M_field
        return interp_al_op(self.ishape, iM_field, M_field)
