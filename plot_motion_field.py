import os

import nibabel as nib
import numpy as np
import sigpy as sp
from matplotlib import animation
from matplotlib import pyplot as plt


def animate(num, scale, sl, qr, mot_field, dim1, dim2, im, img):
    im.set_data(img[:, sl, :, num])
    # im2.set_data(img[:, sl, :, num])
    qr.set_UVC(mot_field[::scale, sl, ::scale, dim1, num], mot_field[::scale, sl, ::scale, dim2, num])
    return qr, im


sub_dir = "/home/ltorres/data/rawdata/ipf/"
ignored = []
subjects = [x for x in os.listdir(sub_dir) if x not in ignored]
subjects.sort()

try:
    for subject in subjects:
        # subject = "P179_Exam1"
        print(subject)
        file_dir = f"/home/ltorres/data/recon/ipf/{subject}/IterativeMoCo_run0"
        mr_dir = f"/home/ltorres/data/recon/ipf/{subject}/MotionResolved_run0"
        diagnostics_dir = f"/home/ltorres/data/recon/nicu/{subject}/diagnostics_run0"
        img = np.flipud(np.swapaxes(nib.load(os.path.join(mr_dir, "MotionResolved0.01.nii.gz")).get_fdata(), 0, 2))

        mot_field = nib.load(os.path.join(file_dir, f"diagnostics/iM_mr.nii.gz")).get_fdata()
        mot_field = np.flipud(
            np.swapaxes(sp.resize(mot_field, (256, 256, 256, mot_field.shape[3], mot_field.shape[4])), 0, 2)
        )
        # mot_field = np.flip(mot_field, 1)
        # print(img.shape)
        # print(mot_field.shape)
        sl = 115
        scale = 2
        dim1 = 0
        dim2 = 2
        X, Y = np.meshgrid(np.arange(0, 256, scale), np.arange(0, 256, scale))

        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(img[:, sl, :, 0], cmap="gray", vmin=0, vmax=150)
        # im2 = ax[1].imshow(img[:, sl, :, 0], cmap="gray", vmin=0, vmax=150)
        # ax.colorbar()
        qr = ax.quiver(
            X,
            Y,
            mot_field[::scale, sl, ::scale, dim1, 0],
            mot_field[::scale, sl, ::scale, dim2, 0],
            np.sqrt(mot_field[::scale, sl, ::scale, dim1, 0] ** 2 + mot_field[::scale, sl, ::scale, dim2, 0] ** 2),
            scale_units="xy",
            scale=1,
            alpha=0.8,
        )

        anim = animation.FuncAnimation(
            fig, animate, frames=mot_field.shape[4], fargs=(scale, sl, qr, mot_field, dim1, dim2, im, img),
        )
        # plt.colorbar()
        anim.save(os.path.join(diagnostics_dir, "motion_fields.gif"), writer="imagemagick", fps=6)
except:
    pass
