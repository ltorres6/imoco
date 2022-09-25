import os
from os.path import join

import nibabel as nib
import numpy as np
from skimage import measure

from sharpness_metrics import rwc
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import sigpy.plot as plt2
from skimage.morphology import rectangle, binary_dilation

# from skimage.filters import sobel
from scipy import signal, ndimage

plt.rcParams["figure.figsize"] = (9, 4)

# roi labels: 1: Airway, 2: lung, 3: liver, 4:muscle, 5:aorta, 6: background
sub_dir = "/home/ltorres/data/recon/nicu/"
ignored = []
subjects = [x for x in os.listdir(sub_dir) if x not in ignored]
subjects.sort()
# subjects = ["P006_Exam1", "P099_Exam1"]
run = "run_cc"
snr = []
pbar = tqdm(subjects)
filenames = [
    f"NoGate_{run}/noGate.nii.gz",
    f"HardGate_{run}/hardGate50.nii.gz",
    f"SoftGate_{run}/softGate0.8.nii.gz",
    f"MotionResolved_{run}/MotionResolved_exp0.050.nii.gz",
    # f"MoCo_{run}/MoCo0.050_frame0.nii.gz",
    f"IterativeMoCo_{run}/iMoCo0.05_frame0.nii.gz",
]
filenames_sobel = [
    f"NoGate_{run}/noGate_sobel.nii.gz",
    f"HardGate_{run}/hardGate50_sobel.nii.gz",
    f"SoftGate_{run}/softGate0.8_sobel.nii.gz",
    f"MotionResolved_{run}/MotionResolved_exp0.050_sobel.nii.gz",
    # f"MoCo_{run}/MoCo0.050_frame0.nii.gz",
    f"IterativeMoCo_{run}/iMoCo0.05_frame0_sobel.nii.gz",
]
df = []
for subject in pbar:
    pbar.set_description(f"Processing: {subject}")
    # print(f"Working on subject: {subject}")
    mypath = f"/home/ltorres/data/recon/nicu/{subject}/"
    roi_path = join(mypath, "roi.nii.gz")
    rois = nib.load(roi_path).get_fdata()
    profile_path = join(mypath, "profile.nii.gz")
    profile = nib.load(profile_path).get_fdata()
    # plt2.ImagePlot(np.max(profile, axis=1))
    # profile = binary_dilation(profile, rectangle(15, 1))
    # plt2.ImagePlot(np.max(profile2, axis=1))
    fig_path = join(mypath, f"diagnostics_{run}/")
    fig1, ax1 = plt.subplots()
    for filename, filename_sobel in zip(filenames, filenames_sobel):
        img_type = filename.split("_")[0]
        img_path = join(mypath, filename)
        img = nib.load(img_path).get_fdata()
        liver_mean = np.mean(img[rois == 3])
        img_path = join(mypath, filename_sobel)
        img = nib.load(img_path).get_fdata() / liver_mean
        img = np.abs(img) * profile
        selection = np.argwhere(profile == 1)[0]
        # print(selection[1])
        # plt.imshow(img[:, selection[1], :], cmap="gray", interpolation="none", vmin=0, vmax=0.5*np.max(img_p))
        # plt.imshow(profile[:, selection[1], :], cmap="jet", alpha=0.5, interpolation="none")
        # plt.show()
        # plt.close()
        line_profile = np.mean(img, axis=(0, 1))
        line_profile = line_profile[line_profile != 0]
        if subject != "P006_Exam1":
            line_profile = np.flip(line_profile)
        # print(sum_x.shape)
        df.append([subject, img_type, line_profile.max()])
        ax1.plot(line_profile, label=img_type)
        ax1.set(xlabel="profile", ylabel="abs(gradient)", title=f"Gradient Profile - {subject}")
        # plt.show()
    plt.legend()
    # plt.show()
    fig1.savefig(fig_path + "line_profiles.png", dpi=300)

# plt.show()
filename = "max_gradient.csv"
# print(len(df))
# writing to csv file in long format
fields = ["subject", "image_type", "max_gradient"]
with open(filename, "w") as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(df)
# print(final_list)
