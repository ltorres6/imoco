import sigpy as sp
import numpy as np
import os
import nibabel
from sigpy.linop import Linop
from sigpy import backend
import scipy.ndimage as ndimage
from scipy.io import loadmat
import sys
from skimage.exposure import match_histograms
import torch as th
from normalize import normalize
from scipy.stats.mstats import winsorize
import sigpy.plot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import airlab as al
# import ants

__all__ = ["interp_op", "interp", "ANTsReg", "regAirlab", "ANTsAff", "interp_affine_op"]


def M_scale2(M, oshape, scale=1):
    Mscale = [oshape[i] / M.shape[i + 1] for i in range(M.shape[0])]
    Mo = np.zeros((M.shape[0],) + oshape)
    for i in range(M.shape[0]):
        M[i] = M[i] * (Mscale[i] * scale)
        Mo[i] = ndimage.zoom(M[i], zoom=tuple(Mscale), order=1)

    return Mo


def M_scale(M, oshape, scale=1):
    Mscale = [oshape[i] / M.shape[i] for i in range(M.shape[-1])]
    Mo = np.zeros(oshape + (M.shape[-1],))
    for i in range(M.shape[-1]):
        M[..., i] = M[..., i] * (Mscale[i] * scale)
        Mo[..., i] = ndimage.zoom(M[..., i], zoom=tuple(Mscale), order=2)

    return Mo


def ANTsAff(If, Im, vox_res=[1, 1, 1], reg_level=[8, 4, 2], gauss_filt=[2, 2, 1]):
    # transfer to nifti
    Ifnft = nibabel.Nifti1Image(If, affine=np.diag(vox_res + [1]))
    Imnft = nibabel.Nifti1Image(Im, affine=np.diag(vox_res + [1]))

    nibabel.save(Ifnft, "./tmp_If.nii")
    nibabel.save(Imnft, "./tmp_Im.nii")

    reg_level_s = "x".join([str(t) for t in reg_level])
    gauss_filt_s = "x".join([str(t) for t in gauss_filt])

    ants_cmd = "antsRegistration -d 3 -m MI[ {}, {}, 1, 50 ] -t Rigid[0.1] \
    -c [ 100x100x40, 1e-6, 10 ] -s {}vox -f {} --winsorize-image-intensities [0.1,1]\
    -l 1 -u 1 -z 1 -o tmp_".format(
        "tmp_Im.nii", "tmp_If.nii", gauss_filt_s, reg_level_s
    )
    os.system(ants_cmd)
    x = loadmat("./tmp_0GenericAffine.mat")
    T = x["AffineTransform_double_3_3"].reshape([4, 3])

    # ANTs orientation
    M_rot = [[1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]]
    T = T * M_rot
    T[3, ...] = T[3, ...].dot(linalg.inv(T[:3]))

    return T


class interp_affine_op(Linop):
    def __init__(self, ishape, T):
        assert list(T.shape) == [4, 3], "Tmatrix Dimension mismatch!"
        oshape = ishape
        self.T = T
        super().__init__(oshape, ishape)

    def _apply(self, input):
        return interp_affine(input, self.T)

    def _adjoint_linop(self):
        T = self._aff_inversion(self.T)

        return interp_affine_op(self.ishape, T)

    def _aff_inversion(self, T):
        T_inv = np.zeros_like(T)
        T_inv[:3, :] = np.linalg.inv(T[:3, :])
        T_inv[3, :] = -T[3, :].dot(T[:3, :].transpose())
        return T_inv


def interp_affine(I, T, aff_order=1):
    # T should be [4,3], [:3,3] rotation, [3,:] shift
    shift_before_rot = T[3, :]
    shift_after_rot = shift_before_rot.dot(T[:3, :].transpose())
    shift_after_rot = -T[3, :]
    AT = lambda x: ndimage.affine_transform(x, T[:3, :], offset=-shift_after_rot, order=aff_order)
    if np.iscomplexobj(I) is True:
        I_aff = AT(np.real(I)) + 1j * AT(np.imag(I))
    else:
        I_aff = AT(I)

    return I_aff


def ANTsReg4(Is, ref=0):
    M_fields = []
    iM_fields = []
    nphase = len(Is)
    for i in range(nphase):
        M_field, iM_field = ANTsReg(np.abs(Is[2]), np.abs(Is[i]))

        M_fields.append(M_field)
        iM_fields.append(iM_field)
    # change
    np.save("./M_field.npy", np.asarray(M_fields))
    np.save("./iM_field.npy", np.asarray(iM_fields))


def ANTsReg(
    If,
    Im,
    fixed_mask,
    moving_mask,
    vox_res=[1, 1, 1],
    reg_level=[8, 4, 2, 1],
    gauss_filt=[6, 4, 2, 0],
    frame=None,
    fluid=0.0,
    diffusion=2.0,
    diagnostics_dir=None,
):
    cwd = os.getcwd()
    os.chdir(diagnostics_dir)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(24)
    # transfer to nifti
    Ifnft = nibabel.Nifti1Image(If, affine=np.diag(vox_res + [1]))
    Imnft = nibabel.Nifti1Image(Im, affine=np.diag(vox_res + [1]))
    fixed_mask = nibabel.Nifti1Image(fixed_mask.astype(np.int8), affine=np.diag(vox_res + [1]))
    moving_mask = nibabel.Nifti1Image(moving_mask.astype(np.int8), affine=np.diag(vox_res + [1]))

    nibabel.save(Ifnft, "./tmp_If.nii")
    nibabel.save(Imnft, "./tmp_Im.nii")
    nibabel.save(fixed_mask, "./tmp_If_mask.nii")
    nibabel.save(moving_mask, "./tmp_Im_mask.nii")

    reg_level_s = "x".join([str(t) for t in reg_level])
    gauss_filt_s = "x".join([str(t) for t in gauss_filt])

    # # neighborhood cross correlatio
    # ants_cmd = f"antsRegistration -d 3 -v 1 -m CC[ tmp_If.nii, tmp_Im.nii, 1, 4 ] -t BSplineSyN[ 0.15, 10, 0, 3 ] \
    # -c [ 5000x500x250x150, 1e-6, 10 ] -s {gauss_filt_s}vox -f {reg_level_s} --winsorize-image-intensities [0.05,1.0]\
    # -l 1 -u 1 -z 1 -x [tmp_If_mask.nii, tmp_Im_mask.nii] -o tmp_ --write-interval-volumes 5 "

    # neighborhood cross correlation
    # # Same as above but using Syn
    # ants_cmd = f"antsRegistration -d 3 -v 1 -m CC[ tmp_If.nii, tmp_Im.nii, 1, 4 ] -t SyN[ 0.25, {fluid}, {diffusion} ] \
    # -c [ 5000x500x250x150, 1e-6, 10 ] -s {gauss_filt_s}vox -f {reg_level_s} --winsorize-image-intensities [0.05,1.0]\
    # -l 1 -u 1 -z 1 -x [tmp_If_mask.nii, tmp_Im_mask.nii] -o [ tmp_, warped_{frame}_{fluid}fluid_{diffusion}diffusion.nii.gz ] "

    # mutual information
    # Same as above but using Syn
    # ants_cmd = f"antsRegistration -d 3 -v 1 -m MI[ tmp_If.nii, tmp_Im.nii, 1,  32 ] -t SyN[ 0.1, {fluid}, {diffusion} ] \
    # -c [ 500x500x250x150, 1e-6, 10 ] -s {gauss_filt_s}vox -f {reg_level_s} --winsorize-image-intensities [0.05,1.0]\
    # -l 1 -u 1 -z 1 -x [tmp_If_mask.nii, tmp_Im_mask.nii] -o tmp_ --write-interval-volumes 2 "

    # Demons
    # Using Syn
    ants_cmd = f"antsRegistration -d 3 -v 1 -m Demons[ tmp_If.nii, tmp_Im.nii, 1 ] -t SyN[ 0.15, {fluid}, {diffusion} ] \
    -c [ 1000x500x400x300, 1e-6, 10 ] -s {gauss_filt_s}vox -f {reg_level_s} --winsorize-image-intensities [0.05,1.0]\
    -l 1 -u 1 -z 1 -x [tmp_If_mask.nii, tmp_Im_mask.nii] -o [ tmp_, warped_{frame}_{fluid}fluid_{diffusion}diffusion.nii.gz ] "

    # Demons
    # Using bSplineSyn
    # ants_cmd = f"antsRegistration -d 3 -v 1 -m Demons[ tmp_If.nii, tmp_Im.nii, 1 ] -t BSplineSyN[ 0.2, 26, 0, 3 ] \
    # -c [ 500x500x250x150, 1e-6, 10 ] -s {gauss_filt_s}vox -f {reg_level_s} --winsorize-image-intensities [0.05,1.0]\
    # -l 1 -u 1 -z 1 -x [tmp_If_mask.nii, tmp_Im_mask.nii] -o tmp_ --write-interval-volumes 2 "

    # jac_cmd = f"CreateJacobianDeterminantImage 3 tmp_0Warp.nii.gz jacobian_{frame}.nii.gz 1 1"
    # ijac_cmd = f"CreateJacobianDeterminantImage 3 tmp_0InverseWarp.nii.gz ijacobian_{frame}.nii.gz 1 1"

    # apply_cmd = f"antsApplyTransforms -d 3 -i tmp_Im.nii -o warped_{frame}_{fluid}fluid_{diffusion}diffusion.nii.gz -r tmp_If.nii -t tmp_0Warp.nii.gz"
    os.system(ants_cmd)
    # os.system(jac_cmd)
    # os.system(ijac_cmd)
    # os.system(apply_cmd)
    M_field = nibabel.load("./tmp_0Warp.nii.gz")
    iM_field = nibabel.load("./tmp_0InverseWarp.nii.gz")
    # print(f"Motion Field Shape (from read): {M_field.shape}")

    Mt = M_field.get_data()
    iMt = iM_field.get_data()
    # X and Y need to be flipped?
    Mt[..., :2] = -Mt[..., :2]
    iMt[..., :2] = -iMt[..., :2]
    Mt = np.squeeze(Mt)
    iMt = np.squeeze(iMt)

    # scale by voxel resolution and final resolution scale
    scale = 1 / reg_level[-1]
    Mt = M_scale(Mt, If.shape, scale * (1 / vox_res[-1]))
    iMt = M_scale(iMt, If.shape, scale * (1 / vox_res[-1]))

    # # # Warp with sigpy and compare to ANTs warped output to verify same transformation is occuring.
    # warped = interp(Im, Mt)
    # warped = nibabel.Nifti1Image(warped, np.diag(vox_res + [1]))
    # nibabel.save(warped, f"warped_{frame}_sigpy.nii.gz")

    os.remove("./tmp_If.nii")
    os.remove("./tmp_Im.nii")
    os.remove("./tmp_If_mask.nii")
    os.remove("./tmp_Im_mask.nii")
    os.remove("./tmp_0Warp.nii.gz")
    os.remove("./tmp_0InverseWarp.nii.gz")
    os.chdir(cwd)
    return Mt, iMt


def regAirlab(fixed_image, moving_image, vox_res=[1, 1, 1]):
    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    # device = th.device("cpu")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    device = th.device("cuda:0")

    fixed_image = normalize(fixed_image, 0, 1)
    moving_image = normalize(moving_image, 0, 1)
    moving_image = match_histograms(moving_image, fixed_image)
    fixed_image = al.image_from_numpy(fixed_image, vox_res, [0, 0, 0], dtype=dtype, device=device)
    moving_image = al.image_from_numpy(moving_image, vox_res, [0, 0, 0], dtype=dtype, device=device)

    # create image pyramide size/4, size/2, size/1
    # ds_factors = [8, 4, 2, 1]
    # fixed_image_pyramid = al.create_image_pyramid(fixed_image, [[4, 4, 4], [2, 2, 2]])
    # moving_image_pyramid = al.create_image_pyramid(moving_image, [[4, 4, 4], [2, 2, 2]])
    # del fixed_image, moving_image
    downsample_factor = [[6, 6, 6], [2, 2, 2], [1, 1, 1]]
    constant_flow = None
    # regularisation_weight = [1, 1, 1, 1]
    # number_of_iterations = [10, 10, 10]
    number_of_iterations = [5000, 3000, 1000]
    sigma = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    # pixels_per_knot = [[2, 2, 2], [4, 4, 4], [8, 8, 8]]

    for level, dsf in enumerate(downsample_factor):
        fix_im_level = al.create_downsampled_image(fixed_image, dsf)
        mov_im_level = al.create_downsampled_image(moving_image, dsf)
        # registration = al.PairwiseRegistration(verbose=True)
        registration = al.DemonsRegistraion(verbose=False)
        # print(level)
        # define the transformation
        # transformation = al.transformation.pairwise.BsplineTransformation(
        #     mov_im_level.size, sigma=pixels_per_knot[level], order=3, dtype=dtype, device=device, diffeomorphic=True
        # )
        transformation = al.transformation.pairwise.NonParametricTransformation(
            mov_im_level.size, dtype=dtype, device=device, diffeomorphic=True
        )
        if level > 0:
            constant_flow = al.transformation.utils.upsample_displacement(
                constant_flow, mov_im_level.size, interpolation="linear"
            )
            transformation.set_constant_flow(constant_flow)

        registration.set_transformation(transformation)

        # choose the Mean Squared Error as image loss
        image_loss = al.loss.pairwise.MSE(fix_im_level, mov_im_level)

        registration.set_image_loss([image_loss])
        # choose a regulariser for the demons
        regulariser = al.regulariser.demons.GaussianRegulariser(
            mov_im_level.spacing, sigma=sigma[level], dtype=dtype, device=device
        )
        registration.set_regulariser([regulariser])

        # # define the regulariser for the displacement
        # regulariser = al.regulariser.displacement.DiffusionRegulariser(mov_im_level.spacing)
        # regulariser.set_weight(regularisation_weight[level])
        # registration.set_regulariser_displacement([regulariser])

        # define the optimizer
        optimizer = th.optim.Adam(transformation.parameters())

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(number_of_iterations[level])

        registration.start()

        constant_flow = transformation.get_flow()
        # del image_loss, regulariser
        # plt.ImagePlot(transformation.get_displacement().cpu().numpy())
        # plt.ImagePlot(transformation.get_inverse_displacement().cpu().numpy())

    # create final result
    displacement = transformation.get_displacement()
    displacement = al.create_displacement_image_from_image(transformation.get_displacement(), fixed_image)
    displacement = al.transformation.utils.unit_displacement_to_displacement(displacement)

    inv_displacement = transformation.get_inverse_displacement()
    inv_displacement = al.create_displacement_image_from_image(transformation.get_inverse_displacement(), moving_image)
    inv_displacement = al.transformation.utils.unit_displacement_to_displacement(inv_displacement)

    th.cuda.empty_cache()
    # return np.squeeze(displacement.numpy()), np.squeeze(inv_displacement.numpy())
    return displacement, inv_displacement


## Demons registration
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
    lapI = gxx + gyy + gzz
    return lapI


def pmask(I, sigma):
    # TODO: optimize
    I = np.abs(I)
    mask = np.abs(I) > sigma
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = ndimage.morphology.binary_opening(mask, structure=np.ones((5, 5, 5)))
    return mask


def DemonsReg4(Is, ref=0, level=3, device=-1):
    M_fields = []
    iM_fields = []
    nphase = len(Is)
    print("4D Demons registration:")
    for i in range(nphase):
        print("Ref/Mov:{}/{}".format(i, ref))
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
    ### normalization??
    Im = np.abs(Im)
    m_scale = np.max(Im)
    Im = Im / m_scale
    If = np.abs(If)
    If = If / m_scale

    ### registration
    M = np.zeros(Im.shape + (3,))
    Mt = np.zeros(Im.shape + (3,))
    for k in range(level):
        print("Demons Level:{}".format(k))
        ### hyperparameter assignment
        scale = 2 ** (level - k - 1)
        sigma_f = sigmas_f[k]
        sigma_e = sigmas_e[k]
        sigma_s = sigmas_s[k]
        iter_each_level = iters[k]

        ###
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
            # Is = ndimage.gaussian_filter((Ifm+Imm)/2,sigma=sigma_s,truncate=2.0)

            gIx, gIy, gIz = imgrad3d(Is)
            gI = np.sqrt(np.abs(gIx ** 2 + gIy ** 2 + gIz ** 2) + 1e-6)
            discriminator = gI ** 2 + np.abs(dI) ** 2
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

    ### TODO inverse combination (right now just double)
    M = M_scale(Mt * 2, Im.shape)
    return M


## interpolation operator
class interp_op(Linop):
    def __init__(self, ishape, M_field, inv_field=None):
        ndim = M_field.shape[-1]
        # print(list(ishape))
        # print(list(M_field.shape[:-1]))
        assert list(ishape) == list(M_field.shape[:-1]), "Dimension mismatch!"
        oshape = ishape
        self.M_field = M_field
        self.inv_field = inv_field
        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)

        with device:
            return interp(input, self.M_field, device)

    def _adjoint_linop(self):
        device = backend.get_device(input)
        if self.inv_field is None:
            inv_field = -self.M_field
            M_field = None
        else:
            inv_field = self.inv_field
            M_field = self.M_field

        return interp_op(self.ishape, inv_field, M_field)


def interp(I, M_field, device=sp.Device(-1), k_id=1, deblur=True):
    # b spline interpolation
    N = 64
    if k_id == 0:
        kernel = [(3 * (x / N) ** 3 - 6 * (x / N) ** 2 + 4) / 6 for x in range(0, N)] + [
            (2 - x / N) ** 3 / 6 for x in range(N, 2 * N)
        ]
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

    # 2d/3d
    if ndim == 3:
        dkernel = dkernel[:, None, None] * dkernel[None, :, None] * dkernel[None, None, :]
        Nx, Ny, Nz = I.shape
        my, mx, mz = np.meshgrid(np.arange(Ny), np.arange(Nx), np.arange(Nz))
        m = np.stack((mx, my, mz), axis=-1)
        M_field = M_field + m
    else:
        dkernel = dkernel[:, None] * dkernel[None, :]
        Nx, Ny = I.shape
        my, mx = np.meshgrid(np.arange(Ny), np.arange(Nx))
        m = np.stack((mx, my, mz), axis=-1)
        M_field = M_field + m
    # TODO remove out of range values

    # image warp

    g_device = device
    I = sp.to_device(input=I, device=g_device)
    # I = sp.interp.interpolate(I, k_wid, kernel, M_field.astype(np.float64))
    I = sp.interp.interpolate(I, sp.to_device(M_field.astype(np.float64), g_device))

    # deconv
    if deblur is True:
        sp.conv.convolve(I, sp.to_device(dkernel, g_device))
    I = sp.to_device(input=I, device=c_device)

    return I


class interp_al_op(Linop):
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
            input_d = al.image_from_numpy(input_d, self.vox_res, [0, 0, 0], dtype=dtype, device=th_device)
            warped_im = al.transformation.utils.warp_image(input_d, self.M_field)
            return sp.to_device(warped_im.numpy(), device)

    def _adjoint_linop(self):
        if self.iM_field is None:
            iM_field = -self.M_field
            M_field = None
        else:
            # swap fields
            iM_field = self.iM_field
            M_field = self.M_field

        return interp_al_op(self.ishape, iM_field, M_field)
