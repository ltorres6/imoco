import argparse
import numpy as np
from imoco_e import cfl, reg
from tqdm import trange
import logging
import time
import nibabel as nib
from scipy.ndimage import median_filter
from normalize import normalize
import sigpy as sp
import torch as th
import airlab as al


def moco(mrimgPath, mf_dir, nRef=-1, reg_flag=1, res_scale=1.0):
    timeStart = time.time()
    #  Load mrimg
    mrimg = nib.load(mrimgPath).get_fdata()
    mrimg = np.moveaxis(np.abs(mrimg), -1, 0)
    nPhases = mrimg.shape[0]
    tshape = mrimg.shape[1:]
    vox_res = [n * res_scale for n in [1, 1, 1]]
    logging.info("Registration...")
    vox_res = [n * res_scale for n in [1, 1, 1]]
    M_fields = []

    if reg_flag == 1:
        timei = time.time()
        pbar = trange(nPhases, leave=True, ncols=80)
        for ii in pbar:
            pbar.set_description(f"Registering Frame: {ii}...")
            # try:
            M_field, _ = reg.regAirlab(
                median_filter(np.abs(mrimg[nRef]), [3, 3, 3]), median_filter(np.abs(mrimg[ii]), [3, 3, 3]), vox_res=vox_res,
            )

            M_fields.append(M_field)
        del M_field

        # Scale Motion field (multply values by scale and expand by scale)
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
        # print(tmp.shape)
        tmp = np.moveaxis(tmp, 0, -1)  # Move n_phases to last dim (so now should be [nx,ny,nz, ndim, nphases])
        tmp = np.transpose(tmp, (2, 1, 0, 3, 4))
        tmp = np.flip(tmp, (0, 1, 2))
        tmp = nib.Nifti1Image(tmp, np.eye(4))
        nib.save(tmp, mf_dir + "/M_mr.nii.gz")

        timeF = (time.time() - timei) / 60
        logging.info("Finshed Registration in {} minutes".format(timeF))
        del tmp
    else:
        logging.info("Reading Motion Fields from disk...")
        M_fields = nib.load(mf_dir + "/M_mr.nii.gz").get_fdata()
        M_fields = np.flip(M_fields, (0, 1, 2))
        M_fields = np.transpose(M_fields, (2, 1, 0, 3, 4))
        M_fields = np.moveaxis(M_fields, -1, 0)

    img = 0
    pbar = trange(nPhases, leave=True)
    for ii in pbar:
        pbar.set_description("Applying Warp To Frame # {}...".format(ii))
        M = reg.interp_op(tshape, M_fields[ii])
        img += M * mrimg[ii]
    img = img / nPhases

    timeFinish = time.time()
    logging.info("Moco Registration Finished in: {} min...".format((timeFinish - timeStart) / 60))
    return img


if __name__ == "__main__":
    # IO parameters
    parser = argparse.ArgumentParser(description="imoco recon.")
    parser.add_argument("ksp_file", type=str, help="k-space file.")
    parser.add_argument("coord_file", type=str, help="coordectory file.")
    parser.add_argument("dcf_file", type=str, help="dcf file.")
    parser.add_argument("img_file", type=str, help="img out file.")
    parser.add_argument("--res_scale", type=float, default=1.0, help="scale of resolution 0-1")
    parser.add_argument("--lambda_tv", type=float, default=2e-2, help="TV regularization, 0.05")
    parser.add_argument("--inner_iter", type=int, default=10, help="Num of inner Iterations.")
    parser.add_argument("--outer_iter", type=int, default=20, help="Num of outer Iterations.")
    parser.add_argument("--device", type=int, default=0, help="Computing device.")
    args = parser.parse_args()

    # Read in ksp
    ksp = np.load(args.ksp_file)
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)

    img = moco(ksp, coord, dcf, args.res_scale, args.lambda_tv, args.inner_iter, args.outer_iter, args.device,)
    print("writing ksp...")
    # plt.ImagePlot(img)
    cfl.write_cfl(args.img_file, img)
