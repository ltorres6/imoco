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
    mypath = "/home/ltorres/data/recon/nicu/{}/HardGate_run0/".format(subject)
    diagnostics_path = "/home/ltorres/data/recon/nicu/{}/diagnostics_run0/".format(subject)
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files.sort(key=lambda f: int(re.sub("\D", "", f)))
    print(files)
    focus_measures = []
    noises = []
    hardgate_thresh = []
    for filename in files:
        print("Filename: {} ------------".format(filename))
        img = nib.load(mypath + "{}".format(filename)).get_fdata()
        if np.isnan(np.sum(img)):
            focus_measures.append(np.nan)
            noises.append(np.nan)
        else:
            fm, noise, coeff_thresh = rwc(img, apply_thresh=True, donoho_noise=True)
            print("Focus Measure Value:{} ".format(fm))
            print("Estimated Image Noise Std Dev: {}".format(noise))
            print("Estimated High Frequency Threshold: {}".format(coeff_thresh))
            focus_measures.append(fm)
            noises.append(noise)
        # print(re.findall('\d*\.?\d+',filename))
        # hardgate_thresh.append(float(re.findall("\d*\.?\d+", filename)[0]))
        hardgate_thresh.append(float(re.findall(r"\d+", filename)[0]))
    # print(hardgate_thresh[:])
    plt.rcParams["figure.figsize"] = (7.5, 4)
    plt.plot([100 - x for x in hardgate_thresh], focus_measures)
    plt.title(f"{subject} Sharpness vs Gating Threshold")
    plt.xlabel("%% data retained (gating threshold)")
    plt.ylabel("Sharpness")
    plt.xticks(ticks=[100 - x for x in hardgate_thresh], labels=[int(x) for x in hardgate_thresh])
    plt.savefig(diagnostics_path + "sharpness_vs_hardgate_threshold.png")
    # plt.show()
    plt.close()

    plt.plot([100 - x for x in hardgate_thresh], noises)
    plt.title("Noise vs Gating Threshold")
    plt.xlabel("%% data retained (gating threshold)")
    plt.ylabel("Noise StdDev")
    plt.xticks(ticks=[100 - x for x in hardgate_thresh], labels=[int(x) for x in hardgate_thresh])
    plt.savefig(diagnostics_path + "noise_vs_hardgate_threshhold.png")
    # plt.show()
    plt.close()

    plt.plot([100 - x for x in hardgate_thresh[1:]], np.diff(np.array(focus_measures)) / np.diff(np.array(noises)))
    plt.title(r"$\frac{\delta Sharpness}{\delta Noise} vs  Gating Threshold$")
    plt.xlabel("%% data retained (gating threshold)")
    plt.xticks(ticks=[100 - x for x in hardgate_thresh], labels=[int(x) for x in hardgate_thresh])
    plt.ylabel("Sharpness/Noise")
    plt.savefig(diagnostics_path + "normalized_sharpness_vs_hardgate_threshhold.png")
    # plt.show()
    plt.close()

    plt.plot(np.array(noises), np.array(focus_measures))
    plt.title("Sharpness vs Noise")
    plt.xlabel("Noise Std Dev")
    plt.ylabel("Sharpness")
    plt.savefig(diagnostics_path + "Sharpness_vs_image_noise_hardgated.png")
    # plt.show()
    plt.close()
