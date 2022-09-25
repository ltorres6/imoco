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


def main():
    start = time.time()

    # set the used data type
    dtype = th.float32
    device = th.device("cuda:0")

    # create test image data
    img = np.load(
        "/home/ltorres/data/recon/ipf/103-002/mri/2015122/pre_contrast_ute/MotionResolved_run_6bin_periodic_bspline_faster_morebins/MotionResolvedLowRes.npy"
    )
    img = np.abs(img)
    fixed_image = normalize(img[0], 0, 1)
    moving_image = normalize(img[3], 0, 1)
    moving_image = match_histograms(moving_image, fixed_image)
    fixed_image = al.image_from_numpy(fixed_image, [0.75, 0.75, 0.75], [0, 0, 0], dtype=dtype, device=device)
    moving_image = al.image_from_numpy(moving_image, [0.75, 0.75, 0.75], [0, 0, 0], dtype=dtype, device=device)
    del img

    # create image pyramide size/4, size/2, size/1
    fixed_image_pyramid = al.create_image_pyramid(fixed_image, [[4, 4, 4], [2, 2, 2]])
    moving_image_pyramid = al.create_image_pyramid(moving_image, [[4, 4, 4], [2, 2, 2]])

    constant_flow = None
    number_of_iterations = [3000, 1000, 1000]
    sigma = [[3, 3, 3], [2, 2, 2], [1, 1, 1]]

    for level, (mov_im_level, fix_im_level) in enumerate(zip(moving_image_pyramid, fixed_image_pyramid)):
        # Registration type
        registration = al.DemonsRegistraion(verbose=True)

        # Transform
        transformation = al.transformation.pairwise.NonParametricTransformation(
            mov_im_level.size, dtype=dtype, device=device, diffeomorphic=True
        )

        if level > 0:
            constant_flow = al.transformation.utils.upsample_displacement(
                constant_flow, mov_im_level.size, interpolation="linear"
            )
            transformation.set_constant_flow(constant_flow)

        registration.set_transformation(transformation)

        # Loss
        image_loss = al.loss.pairwise.MSE(fix_im_level, mov_im_level)
        registration.set_image_loss([image_loss])

        # Edge updater
        edge_updater = al.regulariser.demons.EdgeUpdaterDisplacementIntensities(
            mov_im_level.spacing, mov_im_level.image
        )
        # Regularizer
        regulariser = al.regulariser.demons.GraphDiffusionRegulariser(
            mov_im_level.size, mov_im_level.spacing, edge_updater=edge_updater, dtype=dtype, device=device
        )
        registration.set_regulariser([regulariser])

        # Optimizer
        optimizer = th.optim.Adam(transformation.parameters(), lr=0.01)

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(number_of_iterations[level])

        registration.start()

        constant_flow = transformation.get_flow()


if __name__ == "__main__":
    main()
