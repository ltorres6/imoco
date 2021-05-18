from sharpness_metrics import rwc
import nibabel as nib
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import re

# img = nib.load("/home/ltorres/data/recon/nicu/P24/IterativeMoCo/iMoCo.nii.gz").get_fdata()
# img = nib.load("/home/ltorres/data/recon/nicu/P24/HardGate/hardGate.nii.gz").get_fdata()
# img = nib.load("/home/ltorres/data/recon/nicu/P24/NoGate/noGate.nii.gz").get_fdata()
subjects = ["P24", "P89_Exam1", "P118_Exam2", "P153"]
for subject in subjects:
    mypath = f"/home/ltorres/data/recon/nicu/{subject}/IterativeMoCo_cchmc/"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files = sorted(files)
    focus_measures = []
    noises = []
    reg_param = []
    for filename in files:
        print("Filename: {} ------------".format(filename))
        img = nib.load(mypath + "{}".format(filename)).get_fdata()
        fm, noise, coeff_thresh = rwc(img, apply_thresh=True, donoho_noise=False)
        print("Focus Measure Value:{} ".format(fm))
        print("Estimated Image Noise Std Dev: {}".format(noise))
        print("Estimated High Frequency Threshold: {}".format(coeff_thresh))
        focus_measures.append(fm)
        noises.append(noise)
        reg_param.append(float("0." + re.split("[_.]", filename)[1]))
    # print(reg_param[:])
    plt.plot(reg_param, focus_measures)
    plt.title("focus measures vs lambda")
    plt.xlabel("Regularization Parameter")
    plt.ylabel("Sharpness")
    plt.savefig(mypath + "diagnostics/focus_measures_vs_regularization.png")
    # plt.show()
    plt.close()

    plt.plot(reg_param, noises)
    plt.title("Noise vs lambda")
    plt.xlabel("Regularization Parameter")
    plt.ylabel("Noise StdDev")
    plt.savefig(mypath + "diagnostics/image_noise_vs_regularization.png")
    # plt.show()
    plt.close()

    plt.plot(reg_param[1:], np.diff(np.array(focus_measures)) / np.diff(np.array(noises)))
    plt.title("d_focus measure / d_Noise vs lambda")
    plt.xlabel("Regularization Parameter")
    plt.ylabel("Sharpness/Noise")
    plt.savefig(mypath + "diagnostics/normalized_focus_measures_vs_regularization.png")
    # plt.show()
    plt.close()

    plt.plot(np.array(noises), np.array(focus_measures))
    plt.title("Noise vs fm")
    plt.xlabel("Noise Std Dev")
    plt.ylabel("Sharpness")
    plt.savefig(mypath + "diagnostics/focus_measures_vs_image_noise.png")
    # plt.show()
    plt.close()
