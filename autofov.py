import argparse
import numpy as np
import sigpy as sp
import logging
from normalize import normalize
# from scipy.misc import imsave
from PIL import Image

# import sigpy.plot as plt
# import imageio


def autofov(ksp, coord, dcf, diagPath, num_ro, device, thresh, radial):
    """Automatic estimation of FOV.

    FOV is estimated by thresholding a low resolution gridded image.
    coord will be modified in-place.

    Args:
        ksp (array): k-space measurements of shape (C, num_tr, num_ro, D).
            where C is the number of channels,
            num_tr is the number of TRs, num_ro is the readout points,
            and D is the number of spatial dimensions.
        coord (array): k-space coordinates of shape (num_tr, num_ro, D).
        dcf (array): density compensation factor of shape (num_tr, num_ro).
        num_ro (int): number of read-out points.
        device (Device): computing device.
        thresh (float): threshold between 0 and 1.

    """
    device = sp.Device(device)
    xp = device.xp
    with device:
        if radial:
            ro_center = ksp.shape[2] // 2
            ro_range = slice(ro_center - num_ro // 2, ro_center + num_ro // 2, 1)
        else:
            ro_range = slice(0, num_ro, 1)

        kspc = ksp[:, :, ro_range]
        coordc = coord[:, ro_range, :]
        dcfc = dcf[:, ro_range]
        # Multiply by two so we can encompass entire FOV.
        coordc2 = sp.to_device(coordc * 2, device)
        num_coils = len(kspc)
        imgc_shape = np.array(sp.estimate_shape(coordc))
        imgc2_shape = sp.estimate_shape(coordc2)
        imgc2_center = [i // 2 for i in imgc2_shape]
        imgc2 = sp.nufft_adjoint(
            sp.to_device(dcfc * kspc, device), coordc2, [num_coils] + imgc2_shape
        )
        imgc2 = xp.sum(xp.abs(imgc2) ** 2, axis=0) ** 0.5
        filt = sp.to_device(sp.hanning((16, 16, 16)), device)
        filt = sp.resize(filt, imgc2.shape)
        # imgc2 = sp.convolve(imgc2, filt)
        imgc2 = sp.ifft(sp.fft(sp.to_device(imgc2, device), norm=None) * filt, norm=None)
        imgc2 /= imgc2.max()
        # plt.ImagePlot(imgc2)
        im = 
        im = Image.fromarray(sp.to_device(xp.abs(imgc2[:, imgc2.shape[1] // 2, :])))
        im = im.convert("L")
        im.save(diagPath + "/diag_lowResRecon.jpg")

        if imgc2.ndim == 3:
            imgc2_cor = imgc2[:, imgc2.shape[1] // 2, :]
            thresh *= imgc2_cor.max()
        else:
            thresh *= imgc2.max()
        boxc = imgc2 > thresh
        boxc = sp.to_device(boxc)
        im = Image.fromarray(sp.to_device(boxc[:, boxc.shape[1] // 2, :]))
        im = im.convert("L")
        im.save(diagPath + "/diag_fovMask.jpg")
        boxc_idx = np.nonzero(boxc)
        boxc_shape = np.array(
            [int(np.abs(boxc_idx[i] - imgc2_center[i]).max()) * 2 for i in range(imgc2.ndim)]
        )
        img_scale = boxc_shape / imgc_shape
        print(img_scale)
        print(imgc2_shape)
        if radial:
            img_scale *= 2
        coord *= img_scale
        # --------------------
        coordc = coord[:, ro_range, :]
        coordc = sp.to_device(coordc, device)
        num_coils = len(kspc)
        imgc_shape = sp.estimate_shape(coordc)
        imgc = sp.nufft_adjoint(sp.to_device(dcfc * kspc, device), coordc, [num_coils] + imgc_shape)
        imgc = xp.sum(xp.abs(imgc) ** 2, axis=0) ** 0.5
        # plt.ImagePlot(imgc)
        # imageio.imwrite(
        #     diagPath + "/diag_effectiveFOVImg.jpg", sp.to_device(imgc[:, imgc.shape[1] // 2, :])
        # )
        im = Image.fromarray(sp.to_device(xp.abs(imgc[:, imgc.shape[1] // 2, :])))
        im = im.convert("L")
        im.save(diagPath + "/diag_effectiveFOVImg.jpg")

        # --------------------


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ro", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--thresh", type=float, default=0.1)

    parser.add_argument("ksp_file", type=str)
    parser.add_argument("coord_file", type=str)
    parser.add_argument("dcf_file", type=str)
    parser.add_argument("diagnosticsDir", type=str)

    parser.add_argument("--radial", action="store_true")

    args = parser.parse_args()

    ksp = np.load(args.ksp_file)
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)
    print("Kspace Shape Original: {}".format(ksp.shape))
    print("Input Image Shape: {}".format(sp.estimate_shape(coord)))

    autofov(
        ksp,
        coord,
        dcf,
        diagPath=args.diagnosticsDir,
        num_ro=args.num_ro,
        device=args.device,
        thresh=args.thresh,
        radial=args.radial,
    )

    logging.info("Output Image shape: {}".format(sp.estimate_shape(coord)))

    logging.info("Saving data.")
    np.save(args.coord_file, coord)
