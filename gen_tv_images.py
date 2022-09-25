import os
from os.path import join

import nibabel as nib
import numpy as np

from sharpness_metrics import rwc
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import sigpy.plot as plt2
import sigpy as sp


def make_image(data, outputname, size=(3, 3), dpi=300, vmin=0, vmax=255, xlim=[0, 256], ylim=[0, 256]):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap("gray")
    ax.imshow(data, aspect="equal", vmin=vmin, vmax=vmax)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[1], ylim[0])
    plt.savefig(outputname, dpi=dpi)
    plt.close()


# roi labels: 1: Airway, 2: lung, 3: liver, 4:muscle, 5:aorta, 6: background
sub_dir = "/home/ltorres/data/recon/ipf/"
ignored = ["103-009"]
subjects = [x for x in os.listdir(sub_dir) if x not in ignored]
subjects.sort()
run = "final"
contrast_type = "pre_contrast"
focus_measures = []
pbar = tqdm(subjects)
filenames = [
    f"MotionResolved_{run}/MotionResolved0.050.nii.gz",
]

final_list = []
for subject in pbar:
    pbar.set_description(f"Processing: {subject}")
    # print(f"Working on subject: {subject}")
    visits = os.listdir(os.path.join(sub_dir, subject + "/mri/"))
    visits.sort()
    visits = visits[0]
    visits = [visits] if isinstance(visits, str) else visits
    for visit in visits:
        mypath = f"/home/ltorres/data/recon/ipf/{subject}/mri/{visit}/{contrast_type}"
        for filename in filenames:
            img_path = join(mypath, filename)
            img_type = filename.split("_")[0]
            img = nib.load(img_path).get_fdata()
            print(img.shape)
            TVop = sp.linop.FiniteDifference(img.shape, axes=(4,))
            img_tv = np.squeeze(TVop * img)
            print(img_tv.shape)
            plt2.ImagePlot(img_tv)
            tv_slice = np.rot90(img_tv[:, 128, :])
            plt2.ImagePlot(img_tv)

            # plt.hist(img.ravel(), 100)
            # plt2.ImagePlot(mip_min)
            # plt2.ImagePlot(img_slice)
            save_path = join(mypath, f"diagnostics_{run}/{img_type}_tv_slice.png")
            make_image(tv_slice, save_path, vmin=0, vmax=0.1 * img.max())

        # plt.imshow(mip_min, cmap="gray", vmin=0, vmax=0.35 * img.max())
        # plt.savefig(join(mypath, f"diagnostics_{run}/{img_type}_mip_min.png"))
        # plt.close()

        # plt.imshow(img_slice, cmap="gray", vmin=0, vmax=0.35 * img.max())
        # # plt.show()
        # plt.savefig(join(mypath, f"diagnostics_{run}/{img_type}_img_slice.png"))
        # plt.close()
