# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time

import airlab as al
import matplotlib.pyplot as plt
import numpy as np
import sigpy.plot as plt2
import torch as th
from scipy import ndimage
from scipy.stats.mstats import winsorize
from skimage.exposure import match_histograms
from skimage.filters import threshold_minimum, threshold_otsu
from skimage.morphology import ball
from normalize import normalize

from scipy.ndimage import median_filter, gaussian_filter


def estimate_mask(img):
    # thresh = threshold_otsu(img[img > 0].ravel())
    thresh = threshold_minimum(img.ravel())
    bg_mask = img >= thresh
    fill_mask = img == 0
    strel = ball(1, dtype=np.uint8)
    # plt2.ImagePlot(bg_mask)
    bg_mask = ndimage.binary_closing(bg_mask, structure=strel, iterations=10)
    bg_mask = ndimage.binary_fill_holes(bg_mask, structure=strel)
    bg_mask = bg_mask + fill_mask
    # plt2.ImagePlot(bg_mask)
    return bg_mask


def main():
    start = time.time()

    # set the used data type
    dtype = th.float32
    device = th.device("cuda:0")

    # create test image data
    img = np.load("/home/ltorres/data/recon/nicu/P110_Exam1/MotionResolved_run_airlab/MotionResolvedLowRes.npy")
    # img = np.load(
    #     "/home/ltorres/data/recon/ipf/103-042/mri/20190221/pre_contrast_ute/MotionResolved_run_6bin_airlab/MotionResolvedLowRes.npy"
    # )

    img = np.abs(img)
    avg_image = img[0]
    avg_image2 = img[0]
    fixed_mask = estimate_mask(gaussian_filter(img[0], [1, 1, 1]))
    fixed_mask = al.image_from_numpy(fixed_mask, [0.75, 0.75, 0.75], [0, 0, 0], dtype=dtype, device=device)

    for phase in range(img.shape[0]):
        moving_mask = estimate_mask(gaussian_filter(img[phase], [1, 1, 1]))
        fixed_image = winsorize(normalize(img[0], 0, 1), (0.0, 0.0))
        moving_image = winsorize(normalize(img[phase], 0, 1), (0.0, 0.0))
        moving_image = match_histograms(moving_image, fixed_image)
        fixed_image = al.image_from_numpy(fixed_image, [0.75, 0.75, 0.75], [0, 0, 0], dtype=dtype, device=device)
        moving_image = al.image_from_numpy(moving_image, [0.75, 0.75, 0.75], [0, 0, 0], dtype=dtype, device=device)
        moving_mask = al.image_from_numpy(moving_mask, [0.75, 0.75, 0.75], [0, 0, 0], dtype=dtype, device=device)

        # del img
        fixed_min = fixed_image.image.min()
        fixed_max = fixed_image.image.max()
        # plt2.ImagePlot(fixed_image.numpy())
        # fixed_image.write("fixed.nii")
        # moving_image.write("moving.nii")
        # create image pyramide size/4, size/2, size/1
        # fixed_image_pyramid = al.create_image_pyramid(fixed_image, [[4, 4, 4], [2, 2, 2]])
        # moving_image_pyramid = al.create_image_pyramid(moving_image, [[4, 4, 4], [2, 2, 2]])
        n_levels = 3
        stop_crit = 1e-9
        resolution = 1.25
        initial_spacing = 8  # mm
        vox_res = resolution / 0.75  # (mm/voxel)
        knot_spacing = initial_spacing // vox_res  # voxels/knot
        # print(knot_spacing)
        # pixels_per_knot = [[int(knot_spacing // (2 ** i))] * 3 for i in range(n_levels)]
        # print(pixels_per_knot)
        pixels_per_knot = [[6, 6, 6], [2, 2, 2], [2, 2, 2]]

        downsample_factor = [[2 ** i] * 3 for i in range(n_levels)][::-1]
        # smoothing_factor = [[2 ** i] * 3 for i in range(n_levels)][::-1]
        smoothing_factor = [[0] * 3 for i in range(n_levels)][::-1]
        constant_flow = None
        number_of_iterations = [2000, 2000, 2000]
        reg_sigma = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        # regularisation_weight = [1000, 1000, 1000]
        step_size = [1e-2] * n_levels
        # step_size_lbfgs = [1, 1e-1, 1e-1]
        for level, dsf in enumerate(downsample_factor):
            fix_im_level = al.create_downsampled_image(
                al.gaussian_blur(fixed_image, smoothing_factor[level], dtype=dtype, device=device), dsf
            )
            mov_im_level = al.create_downsampled_image(
                al.gaussian_blur(moving_image, smoothing_factor[level], dtype=dtype, device=device), dsf
            )
            fix_mask_level = al.create_downsampled_image(fixed_mask, dsf)
            mov_mask_level = al.create_downsampled_image(moving_mask, dsf)

            # plt2.ImagePlot(fix_im_level.numpy())
            registration = al.DemonsRegistration(verbose=True)

            # define the transformation
            transformation = al.transformation.pairwise.BsplineTransformation(
                mov_im_level.size, sigma=pixels_per_knot[level], order=3, dtype=dtype, device=device, diffeomorphic=True
            )
            # transformation = al.transformation.pairwise.NonParametricTransformation(
            #     mov_im_level.size, dtype=dtype, device=device, diffeomorphic=True
            # )
            if level > 0:
                constant_flow = al.transformation.utils.upsample_displacement(
                    constant_flow, mov_im_level.size, interpolation="linear"
                )
                transformation.set_constant_flow(constant_flow)

            registration.set_transformation(transformation)

            # choose the Mean Squared Error as image loss
            image_loss = al.loss.pairwise.MSE(
                fix_im_level, mov_im_level, fixed_mask=fix_mask_level, moving_mask=mov_mask_level, size_average=True
            )
            # image_loss = al.loss.pairwise.LCC(fix_im_level, mov_im_level, sigma=[7], kernel_type="box")
            # image_loss = al.loss.pairwise.NCC(fix_im_level, mov_im_level)
            # image_loss = al.loss.pairwise.NGF(fix_im_level, mov_im_level)
            # image_loss = al.loss.pairwise.MI(fix_im_level, mov_im_level, bins=64, spatial_samples=0.7)
            # image_loss = al.loss.pairwise.SSIM(fix_im_level, mov_im_level, dim=3)  # Doesn't work in 3d for now.

            registration.set_image_loss([image_loss])
            # choose a regulariser for the demons
            regulariser = al.regulariser.demons.GaussianRegulariser(
                mov_im_level.spacing, sigma=reg_sigma[level], dtype=dtype, device=device
            )
            registration.set_regulariser([regulariser])

            # # define the regulariser for the displacement
            # regulariser = al.regulariser.displacement.IsotropicTVRegulariser(mov_im_level.spacing)
            # regulariser.set_weight(regularisation_weight[level])
            # registration.set_regulariser_displacement([regulariser])

            # define the optimizer
            optimizer = th.optim.Adam(transformation.parameters(), lr=step_size[level])
            # optimizer = th.optim.LBFGS(transformation.parameters(), lr=step_size_lbfgs[level], history_size=10)

            registration.set_optimizer(optimizer)
            registration.set_number_of_iterations(number_of_iterations[level])

            registration.start(EarlyStopping=True, stop_crit=stop_crit)

            constant_flow = transformation.get_flow()
            # displacement = al.transformation.utils.unit_displacement_to_displacement(
            #     al.create_displacement_image_from_image(transformation.get_displacement(), moving_image)
            # )
            # plt2.ImagePlot(
            #     displacement.magnitude().numpy(), colormap="jet",
            # )

        # create final result
        displacement1 = transformation.get_displacement()
        # print(displacement1.cpu().numpy().shape)
        warped_image = al.transformation.utils.warp_image(moving_image, displacement1)
        displacement = al.transformation.utils.unit_displacement_to_displacement(
            al.create_displacement_image_from_image(displacement1, moving_image)
        )
        avg_image += warped_image.numpy()
        avg_image2 += moving_image.numpy()
        im_slice = fixed_image.numpy().shape[1] // 2
        plt.figure(1)
        plt.subplot(2, 5, phase + 1)
        plt.imshow(displacement.magnitude().numpy()[:, im_slice, :], cmap="jet", vmin=0, vmax=5)
        plt.title(f"Flow {phase} loss{registration.loss.item()}")
        plt.figure(2)
        plt.subplot(2, 5, phase + 1)
        plt.imshow(
            warped_image.numpy()[:, im_slice, :] - fixed_image.numpy()[:, im_slice, :], cmap="gray",
        )

        plt.title(f"Subtraction {phase}")
        plt.figure(3)
        plt.subplot(2, 5, phase + 1)
        plt.imshow(warped_image.numpy()[:, im_slice, :], cmap="gray", vmin=fixed_min, vmax=0.1 * fixed_max)
        plt.title(f"Registered {phase}")

    # avg_image /= img.shape[0]
    # avg_image2 /= img.shape[0]
    # plt.figure(1)
    # plt.subplot(2, 5, phase + 2)
    # plt.imshow(fixed_image.numpy()[:, im_slice, :], cmap="gray", vmin=fixed_min, vmax=0.1 * fixed_max)
    # plt.title(f"Fixed ")
    # plt.subplot(2, 5, phase + 3)
    # plt.imshow(avg_image[:, im_slice, :], cmap="gray", vmin=fixed_min, vmax=0.05 * fixed_max)
    # plt.title(f"Average ")
    # plt.subplot(2, 5, phase + 4)
    # plt.imshow(avg_image2[:, im_slice, :], cmap="gray", vmin=fixed_min, vmax=0.05 * fixed_max)
    # plt.title(f"Average  Uncorrected")
    # plt.figure(2)
    # plt.subplot(2, 5, phase + 2)
    # plt.imshow(fixed_image.numpy()[:, im_slice, :], cmap="gray", vmin=0.0, vmax=0.1 * fixed_max)
    # plt.title(f"Fixed ")
    # plt.subplot(2, 5, phase + 3)
    # plt.imshow(avg_image[:, im_slice, :], cmap="gray", vmin=fixed_min, vmax=0.05 * fixed_max)
    # plt.title(f"Average ")
    # plt.subplot(2, 5, phase + 4)
    # plt.imshow(avg_image2[:, im_slice, :], cmap="gray", vmin=fixed_min, vmax=0.05 * fixed_max)
    # plt.title(f"Average  Uncorrected")

    # plt2.ImagePlot(avg_image)
    # create inverse displacement field
    # inverse_displacement = transformation.get_inverse_displacement()
    # inverse_warped_image = al.transformation.utils.warp_image(fixed_image, inverse_displacement)
    # inverse_displacement = al.transformation.utils.unit_displacement_to_displacement(
    #     al.create_displacement_image_from_image(inverse_displacement, fixed_image)
    # )

    end = time.time()

    print("=================================================================")

    print("Registration done in: ", end - start)
    plt.show()

    # print(f"Final Loss: {registration.loss.item()}")
    # # plot the results
    # im_slice = 90
    # diff1 = moving_image.numpy()[:, im_slice, :] - fixed_image.numpy()[:, im_slice, :]
    # diff2 = warped_image.numpy()[:, im_slice, :] - fixed_image.numpy()[:, im_slice, :]
    # plt.subplot(231)
    # plt.imshow(fixed_image.numpy()[:, im_slice, :], cmap="gray")
    # plt.title("Fixed Image")

    # plt.subplot(232)
    # plt.imshow(moving_image.numpy()[:, im_slice, :], cmap="gray")
    # plt.title("Moving Image")

    # plt.subplot(233)
    # plt.imshow(warped_image.numpy()[:, im_slice, :], cmap="gray")
    # plt.title("Warped Shaded Moving Image")

    # plt.subplot(234)
    # plt.imshow(diff1, cmap="gray", vmin=diff1.min(), vmax=diff1.max())
    # plt.title("Subtraction Image")

    # plt.subplot(235)
    # plt.imshow(diff2, cmap="gray", vmin=diff1.min(), vmax=diff1.max())
    # plt.title("Subtraction Post Warp")

    # plt.subplot(236)
    # plt.imshow(displacement.magnitude().numpy()[:, im_slice, :], cmap="jet")
    # plt.title("Magnitude Displacement")

    # # plt.subplot(2, 5, 9)
    # # plt.imshow((warped_image.numpy()[:, im_slice, :] + fixed_image.numpy()[:, im_slice, :]) / 2, cmap="gray")
    # # plt.title("Average Image")

    # # print(displacement.magnitude().numpy().shape)
    # # print(M_fields.shape)
    # plt.show()
    # # plt.savefig(f"demons_{number_of_iterations}iter_{}sigma.png")
    # # displacement = al.create_displacement_image_from_image(
    # #     al.transformation.utils.upsample_displacement(displacement1, Mfield_size, interpolation="linear"), moving_image,
    # # )
    # # plt2.ImagePlot(M_fields)
    # plt2.ImagePlot(np.flip(np.squeeze(displacement.magnitude().numpy()), -1))


if __name__ == "__main__":
    main()
