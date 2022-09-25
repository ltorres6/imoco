import nibabel as nib
from write_dicoms_4d import write_dicoms
import time

img_path = "/home/ltorres/Desktop/test_4d/MotionResolved0.050.nii.gz"
dicom_dir = "/home/ltorres/Desktop/test_4d/dicoms"

series_nums = [1]
# Same UID base for each visit for dicoms
modification_time = time.strftime("%H%M%S")
modification_date = time.strftime("%Y%m%d")
UID_base = "1.2.840.0.1.3680043.2.1125." + modification_date + ".1" + modification_time

write_dicoms(img_path, dicom_dir, UID_base=UID_base, subject_id=0, series_num=series_nums[0])
