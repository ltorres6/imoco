import argparse
import sys
import numpy as np
from imoco_e import cfl
from tqdm import trange
import logging
import time
import nibabel as nib
sys.path.append('../')
import demonsRegistration
# from demonsRegistration.demons import Demons
dir(demonsRegistration)

def mocoDemons(
    mrimgPath,
    fname,
    nRef=-1,
    reg_flag=1,
):
    timeStart = time.time()
    #  Load mrimg
    mrimg = nib.load(mrimgPath).get_fdata()
    mrimg = np.moveaxis(np.abs(mrimg), -1, 0)
    nPhases = mrimg.shape[0]
    logging.info("Registration...")
    M_fields = []
    img = 0
    if reg_flag is 1:
        pbar = trange(nPhases, leave=True)
        for ii in pbar:
            pbar.set_description("Registering Frame # {}...".format(ii))
            I_temp, M_field = Demons.run(np.abs(mrimg[nRef]), np.abs(mrimg[ii]), nLevels=3, diffusionSigmas=1.0, fluidSigmas=0.0, max_iter=300, alpha=1.0, cThresh=1e-6, variant='passive', diffeomorphic=False, compositionType="A", device=0,).run()
            M_fields.append(M_field)
            img += I_temp
    else:
        pass

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

    img = mocoDemons(
        ksp,
        coord,
        dcf,
        args.res_scale,
        args.lambda_tv,
        args.inner_iter,
        args.outer_iter,
        args.device,
    )
    print("writing ksp...")
    # plt.ImagePlot(img)
    cfl.write_cfl(args.img_file, img)
