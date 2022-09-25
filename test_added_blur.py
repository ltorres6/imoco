import os
import re
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import skimage.filters as filters

from sharpness_metrics import rwc
from tqdm import tqdm

sub_dir = "/home/ltorres/data/recon/nicu/"
ignored = []
subjects = [x for x in os.listdir(sub_dir) if x not in ignored]
subjects.sort()
# subjects = ["P099_Exam1", "P122_Exam1"]
run = "run_external"
sigmas = np.linspace(0, 2, num=10)
pbar = tqdm(subjects)
for subject in pbar:
    pbar.set_description(f"Processing: {subject}")
    # print(f"Working on subject: {subject}")
    mypath = f"/home/ltorres/data/recon/nicu/{subject}/NoGate_{run}/"
    img_path = join(mypath, "noGate.nii.gz")
    diagnostics_path = "/home/ltorres/data/recon/nicu/{}/diagnostics_{run}/".format(subject)
    img = nib.load(img_path).get_fdata()
    focus_measures = []
    noises = []
    coeff_threshs = []
    idx = img > 0
    for blur in sigmas:
        img_blurred = filters.gaussian(img, sigma=blur)
        img_blurred[~idx] = 0
        # plt.imshow(img_blurred[..., 128])
        # plt.show()
        fm, noise, coeff_thresh = rwc(img_blurred, apply_thresh=True, donoho_noise=False)
        # print("Focus Measure Value:{} ".format(fm))
        # print("Estimated Image Noise Std Dev: {}".format(noise))
        # print("Estimated High Frequency Threshold: {}".format(coeff_thresh))
        focus_measures.append(fm)
        noises.append(noise)
        coeff_threshs.append(coeff_thresh)

    # plot relative change
    focus_measures = [x / focus_measures[0] for x in focus_measures]
    plt.figure(1)
    plt.rcParams["figure.figsize"] = (7.5, 4)
    plt.plot(sigmas, focus_measures, label=subject)
    plt.title("Sharpness vs Blur")
    plt.xlabel("Gaussian Blur Sigma")
    plt.ylabel("Sharpness")
    plt.legend()

    # plt.savefig(diagnostics_path + "sharpness_vs_blurring.png")
    # plt.show()
    # plt.close()

    # noises = [x / noises[0] for x in noises]
    plt.figure(2)
    plt.plot(sigmas, noises, label=subject)
    plt.title("Noise vs Blur")
    plt.xlabel("Gaussian Blur Sigma")
    plt.ylabel("Noise StdDev")
    plt.legend()

    # plt.savefig(diagnostics_path + "noise_vs_blur.png")
    # plt.show()
    # plt.close()

    # coeff_threshs = [x / coeff_threshs[0] for x in coeff_threshs]
    plt.figure(3)
    plt.plot(sigmas, coeff_threshs, label=subject)
    plt.title("Noise Threshold vs Blur")
    plt.xlabel("Gaussian Blur Sigma")
    plt.ylabel("Noise Exclusion Threshold")
    plt.legend()
    # plt.savefig(diagnostics_path + "noise_threshold_vs_blur.png")
    # plt.show()
    # plt.close()

    plt.figure(4)
    plt.plot(coeff_threshs, focus_measures, label=subject)
    plt.title("Sharpness vs Noise Threshold")
    plt.xlabel("Noise Exclusion Threshold")
    plt.ylabel("Sharpness")
    plt.legend()
    # plt.show()
plt.show()
