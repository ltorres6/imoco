import logging
import os
import tempfile

import nibabel
import numpy as np

from imoco.registration.interpolation import M_scale


def ANTsReg(
    If,
    Im,
    fixed_mask=None,
    moving_mask=None,
    vox_res=[1, 1, 1],
    reg_level=[8, 4, 2, 1],
    gauss_filt=[6, 4, 2, 0],
    frame=None,
    fluid=0.0,
    diffusion=2.0,
    diagnostics_dir=None,
):
    """Register two 3D images using ANTs SyN with Demons metric.

    Uses the ANTs command-line tool ``antsRegistration`` to compute a
    diffeomorphic displacement field aligning the moving image to the fixed
    image.  Temporary NIfTI files are written to a temporary directory
    (via ``tempfile``) to avoid polluting the working directory.

    Args:
        If (ndarray): Fixed (reference) image, 3D.
        Im (ndarray): Moving image, 3D.
        fixed_mask (ndarray, optional): Binary mask for the fixed image.
        moving_mask (ndarray, optional): Binary mask for the moving image.
        vox_res (list): Voxel resolution in mm, length 3.
        reg_level (list): Multi-resolution schedule (shrink factors).
        gauss_filt (list): Smoothing sigmas per level in voxels.
        frame (int, optional): Frame index, used for naming diagnostic outputs.
        fluid (float): Fluid regularization weight for SyN.
        diffusion (float): Diffusion regularization weight for SyN.
        diagnostics_dir (str, optional): Directory for diagnostic outputs.

    Returns:
        tuple: (M_field, iM_field) forward and inverse displacement fields,
            each of shape (*If.shape, 3), scaled to image coordinates.
    """
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(32)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write inputs as NIfTI
        affine = np.diag(vox_res + [1])
        nibabel.save(nibabel.Nifti1Image(If, affine=affine), os.path.join(tmpdir, "tmp_If.nii"))
        nibabel.save(nibabel.Nifti1Image(Im, affine=affine), os.path.join(tmpdir, "tmp_Im.nii"))

        mask_args = ""
        if fixed_mask is not None and moving_mask is not None:
            nibabel.save(
                nibabel.Nifti1Image(fixed_mask.astype(np.int8), affine=affine),
                os.path.join(tmpdir, "tmp_If_mask.nii"),
            )
            nibabel.save(
                nibabel.Nifti1Image(moving_mask.astype(np.int8), affine=affine),
                os.path.join(tmpdir, "tmp_Im_mask.nii"),
            )
            mask_args = f"-x [ {os.path.join(tmpdir, 'tmp_If_mask.nii')}, {os.path.join(tmpdir, 'tmp_Im_mask.nii')} ]"

        reg_level_s = "x".join([str(t) for t in reg_level])
        gauss_filt_s = "x".join([str(t) for t in gauss_filt])

        output_prefix = os.path.join(tmpdir, "tmp_")
        warped_name = ""
        if frame is not None and diagnostics_dir is not None:
            warped_name = os.path.join(diagnostics_dir, f"warped_{frame}_{fluid}fluid_{diffusion}diffusion.nii.gz")
            output_arg = f"[ {output_prefix}, {warped_name} ]"
        else:
            output_arg = output_prefix

        ants_cmd = (
            f"antsRegistration -d 3 -v 1"
            f" -m Demons[ {os.path.join(tmpdir, 'tmp_If.nii')}, {os.path.join(tmpdir, 'tmp_Im.nii')}, 1 ]"
            f" -t SyN[ 0.15, {fluid}, {diffusion} ]"
            f" -c [ 1000x500x400x300, 1e-6, 10 ]"
            f" -s {gauss_filt_s}vox -f {reg_level_s}"
            f" -w [ 0.05, 1.0 ]"
            f" -u 1 -z 1"
            f" {mask_args}"
            f" -o {output_arg}"
        )

        os.system(ants_cmd)

        M_field = nibabel.load(os.path.join(tmpdir, "tmp_0Warp.nii.gz"))
        iM_field = nibabel.load(os.path.join(tmpdir, "tmp_0InverseWarp.nii.gz"))

        Mt = M_field.get_fdata()
        iMt = iM_field.get_fdata()

        # ANTs convention: flip X and Y components
        Mt[..., :2] = -Mt[..., :2]
        iMt[..., :2] = -iMt[..., :2]
        Mt = np.squeeze(Mt)
        iMt = np.squeeze(iMt)

        # Scale by voxel resolution
        scale = 1 / reg_level[-1]
        Mt = M_scale(Mt, If.shape, scale * (1 / vox_res[-1]))
        iMt = M_scale(iMt, If.shape, scale * (1 / vox_res[-1]))

    return Mt, iMt
