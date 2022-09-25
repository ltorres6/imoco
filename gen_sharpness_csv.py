import os
from os.path import join

import nibabel as nib
import numpy as np

from sharpness_metrics import rwc
from tqdm import tqdm
import csv
from normalize import normalize

# roi labels: 1: Airway, 2: lung, 3: liver, 4:muscle, 5:aorta, 6: background
sub_dir = "/home/ltorres/data/recon/ipf/"
# ignored = ['P179_Exam1']
ignored = ["103-002", "103-009"]
subjects = [x for x in os.listdir(sub_dir) if x not in ignored]
subjects.sort()
# subjects = ["P006_Exam1"]
run = "final"
contrast_type = "pre_contrast"
focus_measures = []
pbar = tqdm(subjects)
filenames = [
    f"NoGate_{run}/noGate.nii.gz",
    f"HardGate_{run}/hardGate50.nii.gz",
    f"SoftGate_{run}/softGate0.8.nii.gz",
    f"MotionResolved_{run}/MotionResolved_exp0.050.nii.gz",
    # f"MoCo_{run}/MoCo0.050_frame0.nii.gz",
    f"IterativeMoCo_{run}/iMoCo0.05_frame0.nii.gz",
]

final_list = []
for subject in pbar:
    pbar.set_description(f"Processing: {subject}")
    visits = os.listdir(os.path.join(sub_dir, subject + "/mri/"))
    visits.sort()
    visits = visits[0]
    visits = [visits] if isinstance(visits, str) else visits
    for visit in visits:
        mypath = f"/home/ltorres/data/recon/ipf/{subject}/mri/{visit}/{contrast_type}"
        fields = ["subject", "image_type", "fm"]
        focus_measures = []
        roi_path = join(mypath, "roi.nii.gz")
        rois = nib.load(roi_path).get_fdata()
        for filename in filenames:
            img_type = filename.split("_")[0]
            # print(img_type)
            img_path = join(mypath, filename)
            # diagnostics_path = "/home/ltorres/data/recon/nicu/{}/diagnostics_run0/".format(subject)
            img = nib.load(img_path).get_fdata()
            img = normalize(img, 0, 255)
            muscle_mean = np.mean(img[rois == 4])
            # noise = np.std(img[rois == 6])
            # img /= muscle_mean

            # print(liver_mean)
            # fm, noise, coeff_thresh = rwc(img, apply_thresh=False, donoho_noise=True, decomp_level=2)
            fm, _, coeff_thresh = rwc(img, apply_thresh=False, donoho_noise=True, decomp_level=2)
            final_list.append([subject, img_type, fm])
            # subject_list.append(subject)
            # print(focus_measures)
        # final_list.append([subject_list, focus_measures])
# print(final_list)
# name of csv file
filename = "sharpness_measures_no_threshold.csv"
# writing to csv file
with open(filename, "w") as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(final_list)
print(final_list)
