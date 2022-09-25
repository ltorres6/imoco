import os
from os.path import join

import nibabel as nib
import numpy as np

from sharpness_metrics import rwc
from tqdm import tqdm
import csv
from normalize import normalize

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

df = []
for subject in pbar:
    pbar.set_description(f"Processing: {subject}")
    # print(f"Working on subject: {subject}")
    mypath = f"/home/ltorres/data/recon/nicu/{subject}/"
    fields = ["subject", "image_type", "region", "snr", "cnr"]
    regions = {"Airway": 1, "Lung": 2, "Liver": 3, "Muscle": 4, "Aorta": 5}
    roi_path = join(mypath, "roi.nii.gz")
    rois = nib.load(roi_path).get_fdata()
    for filename in filenames:
        img_type = filename.split("_")[0]
        img_path = join(mypath, filename)
        img = nib.load(img_path).get_fdata()
        img = normalize(img, 0, 255)
        _, noise, _ = rwc(img, apply_thresh=True, donoho_noise=True)
        # noise = np.std(img[rois == 6])
        airway_snr = np.median(img[rois == 1]) / noise

        for key, value in regions.items():
            # Means
            region_mean = np.mean(img[rois == value])

            # SNR
            region_snr = region_mean / noise

            # CNR
            if key in "Airway":
                region_cnr = float("nan")
            else:
                region_cnr = region_snr - airway_snr

            df.append([subject, img_type, key, region_snr, region_cnr])
# print(final_list)
# name of csv file
filename = "snr_cnr_measures.csv"
# print(len(df))
# writing to csv file in long format
with open(filename, "w") as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(df)
# print(final_list)
