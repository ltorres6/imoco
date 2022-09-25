from sharpness_metrics import rwc
import nibabel as nib
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import re

sub_dir = "/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf/"
subjects = ["103-005", "103-010", "103-038", "103-039", "103-042"]
for subject in subjects:
    # Search for visits here
    visits = os.listdir(os.path.join(sub_dir, subject + "/mri/"))
    visits.sort()
    # print(visits)
    visits = visits[0]
    visits = [visits] if isinstance(visits, str) else visits
    for visit in visits:
        mypath = "/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf/{}/mri/{}/PreContrastSoftGate_run2/".format(subject, visit)
        diagnostics_path = "/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf/{}/mri/{}/diagnostics_run2/".format(subject, visit)
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
        decay_const.append(float(re.findall('\d*\.?\d+',filename)[0]))
    # print(decay_const[:])
    plt.plot(decay_const, focus_measures)
    plt.title(f"{subject} focus measures vs Decay Constant")
    plt.xlabel("decay_constant Parameter")
    plt.ylabel("Sharpness")
    plt.savefig(diagnostics_path + "focus_measures_vs_decay_constant.png")
    # plt.show()
    plt.close()

    plt.plot(decay_const, noises)
    plt.title("Noise vs Decay Constant")
    plt.xlabel("decay_constant Parameter")
    plt.ylabel("Noise StdDev")
    plt.savefig(diagnostics_path + "image_noise_vs_decay_constant.png")
    # plt.show()
    plt.close()

    plt.plot(decay_const[1:], np.diff(np.array(focus_measures)) / np.diff(np.array(noises)))
    plt.title("d_focus measure / d_Noise vs Decay Constant")
    plt.xlabel("decay_constant Parameter")
    plt.ylabel("Sharpness/Noise")
    plt.savefig(diagnostics_path + "normalized_focus_measures_vs_decay_constant.png")
    # plt.show()
    plt.close()

    plt.plot(np.array(noises), np.array(focus_measures))
    plt.title("Noise vs fm")
    plt.xlabel("Noise Std Dev")
    plt.ylabel("Sharpness")
    plt.savefig(diagnostics_path + "focus_measures_vs_image_noise.png")
    # plt.show()
    plt.close()
