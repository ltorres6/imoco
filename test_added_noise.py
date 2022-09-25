import os
import re
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import skimage.filters as filters
import skimage.util as util
from tqdm import tqdm

from sharpness_metrics import rwc

sub_dir = "/home/ltorres/data/recon/nicu/"
ignored = []
subjects = [x for x in os.listdir(sub_dir) if x not in ignored]
subjects.sort()
# subjects = ["P099_Exam1", "P122_Exam1"]
sigmas = np.linspace(0, 10, num=10)
np.random.seed(0)
run = "run_external"
pbar = tqdm(subjects)
for subject in pbar:
    pbar.set_description(f"Processing: {subject}")
    # print(f"Working on subject: {subject}")
    mypath = f"/home/ltorres/data/recon/nicu/{subject}/NoGate_{run}/"
    img_path = join(mypath, "noGate.nii.gz")
    diagnostics_path = f"/home/ltorres/data/recon/nicu/{subject}/diagnostics_{run}/"
    img = nib.load(img_path).get_fdata()
    focus_measures = []
    measured_noises = []
    coeff_threshs = []
    idx = img > 0
    for noise in sigmas:
        img_noisy = img + np.random.normal(0, noise, img.shape)
        img_noisy[~idx] = 0
        # plt.imshow(img[..., 128])
        # plt.imshow(img_noisy[..., 128])
        # plt.show()
        fm, measured_noise, coeff_thresh = rwc(img_noisy, apply_thresh=True, donoho_noise=False)
        # print("Focus Measure Value:{} ".format(fm))
        # print("Estimated Image Noise Std Dev: {}".format(noise))
        # print("Estimated High Frequency Threshold: {}".format(coeff_thresh))
        focus_measures.append(fm / measured_noise)
        measured_noises.append(measured_noise)
        coeff_threshs.append(coeff_thresh)

    # plot relative change
    focus_measures = [x / focus_measures[0] for x in focus_measures]
    plt.figure(1)
    plt.rcParams["figure.figsize"] = (7.5, 4)
    plt.plot(sigmas, focus_measures, label=subject)
    plt.title("Sharpness vs Added Noise")
    plt.xlabel("Added Gaussian Noise Variance")
    plt.ylabel("Sharpness")
    plt.legend()

    # plt.savefig(diagnostics_path + "sharpness_vs_blurring.png")
    # plt.show()
    # plt.close()

    # measured_noises = [(x / measured_noises[0]) ** 2 for x in measured_noises]
    plt.figure(2)
    plt.plot(sigmas, measured_noises, label=subject)
    plt.title("Measured Noise vs Added Noise")
    plt.xlabel("Added Gaussian Noise Variance")
    plt.ylabel("Measured Noise Variance")
    plt.legend()

    # plt.savefig(diagnostics_path + "noise_vs_blur.png")
    # plt.show()
    # plt.close()

    # coeff_threshs = [x / coeff_threshs[0] for x in coeff_threshs]
    plt.figure(3)
    plt.plot(sigmas, coeff_threshs, label=subject)
    plt.title("Noise Threshold vs Added Noise Var")
    plt.xlabel("Added Gaussian Noise Variance")
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
plt.show()
