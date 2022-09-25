from sharpness_metrics import rwc
import nibabel as nib
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import re

sub_dir = "/home/ltorres/data/recon/nicu/"
subjects = ["P099_Exam1", "P122_Exam1", "P173_Exam3", "P179_Exam1"]
for subject in subjects:
    mypath = "/home/ltorres/data/recon/nicu/{}/SoftGate_run0/".format(subject)
    diagnostics_path = "/home/ltorres/data/recon/nicu/{}/diagnostics_run0/".format(subject)
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files = sorted(files)
    print(files)
    focus_measures = []
    noises = []
    decay_const = []
    for filename in files:
        print("Filename: {} ------------".format(filename))
        img = nib.load(mypath + "{}".format(filename)).get_fdata()
        fm, noise, coeff_thresh = rwc(img, apply_thresh=True, donoho_noise=True)
        print("Focus Measure Value:{} ".format(fm))
        print("Estimated Image Noise Std Dev: {}".format(noise))
        print("Estimated High Frequency Threshold: {}".format(coeff_thresh))
        focus_measures.append(fm)
        noises.append(noise)
        # print(re.findall('\d*\.?\d+',filename))
        decay_const.append(float(re.findall("\d*\.?\d+", filename)[0]))
    # print(decay_const[:])
    plt.plot(decay_const, focus_measures)
    plt.title(f"{subject} Sharpness vs Decay Constant")
    plt.xlabel("Decay Constant")
    plt.ylabel("Sharpness")
    plt.savefig(diagnostics_path + "sharpness_vs_decay_constant_softgating.png")
    # plt.show()
    plt.close()

    plt.plot(decay_const, noises)
    plt.title("Noise vs Decay Constant")
    plt.xlabel("Decay Constant")
    plt.ylabel("Noise StdDev")
    plt.savefig(diagnostics_path + "image_noise_vs_decay_constant_softgating.png")
    # plt.show()
    plt.close()

    plt.plot(decay_const[1:], np.diff(np.array(focus_measures)) / np.diff(np.array(noises)))
    plt.title(r"$\frac{\delta Sharpness}{\delta Noise} vs  Decay Constant$")
    plt.xlabel("Decay Constant")
    plt.ylabel("Sharpness/Noise")
    plt.savefig(diagnostics_path + "normalized_sharpness_vs_decay_constant_softgating.png")
    # plt.show()
    plt.close()

    plt.plot(np.array(noises), np.array(focus_measures))
    plt.title("Sharpness vs Noise")
    plt.xlabel("Noise Std Dev")
    plt.ylabel("Sharpness")
    plt.savefig(diagnostics_path + "sharpness_vs_image_noise_softgating.png")
    # plt.show()
    plt.close()
