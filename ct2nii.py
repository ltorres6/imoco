#!/usr/bin/env python


import SimpleITK as sitk
import sys
import os
from os import listdir
from os.path import isfile, join

if len(sys.argv) < 3:
    print("Usage: DicomSeriesReader <input_directory> <output_file>")
    sys.exit(1)

print("Reading Dicom directory:", sys.argv[1])
reader = sitk.ImageSeriesReader()

# dicom_names = reader.GetGDCMSeriesFileNames(sys.argv[1])
dicom_names = [join(sys.argv[1], f) for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]
dicom_names.sort(reverse=True)
reader.SetFileNames(dicom_names)

image = reader.Execute()

size = image.GetSize()
print("Image size:", size[0], size[1], size[2])

print("Writing image:", sys.argv[2])

sitk.WriteImage(image, sys.argv[2])

# if "SITK_NOSHOW" not in os.environ:
#     sitk.Show(image, "Dicom Series")
