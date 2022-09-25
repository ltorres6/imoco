import os
from os.path import join, exists

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

# roi labels: 1: Airway, 2: lung, 3: liver, 4:muscle, 5:aorta, 6: background
sub_dir = "/home/ltorres/data/recon/ipf/"
ignored = ["103-002", "103-009"]
subjects = [x for x in os.listdir(sub_dir) if x not in ignored]
subjects.sort()
# subjects = ["P006_Exam1", "P099_Exam1"]
run = "final"
contrast_type = "pre_contrast"
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
    visits = os.listdir(os.path.join(sub_dir, subject + "/mri/"))
    visits.sort()
    visits = visits[0]
    visits = [visits] if isinstance(visits, str) else visits
    for visit in visits:
        mypath = f"/home/ltorres/data/recon/ipf/{subject}/mri/{visit}/{contrast_type}"
        roi_path = join(mypath, "roi.nii.gz")
        rois = nib.load(roi_path).get_fdata()
        # plt2.ImagePlot(np.max(profile, axis=1))
        # profile2 = binary_dilation(profile, rectangle(15, 1))
        # plt2.ImagePlot(np.max(profile2, axis=1))

        for filename, filename_sobel in zip(filenames, filenames_sobel):
            img_type = filename.split("_")[0]
            img_path = join(mypath, filename)
            if not exists(join(mypath, filename_sobel)):
                img = nib.load(img_path).get_fdata()
                muscle_mean = np.mean(img[rois == 4])
                img /= muscle_mean
                img = ndimage.sobel(img)
                img_sob = nib.Nifti1Image(img, np.eye(4))
                nib.save(img_sob, join(mypath, filename_sobel))

