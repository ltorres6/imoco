import os
import time

import nibabel as nib
import numpy as np
import SimpleITK as sitk


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def writeSlices(series_tag_values, new_img, path, UID_base, i):
    image_slice = new_img[:, :, i]

    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
    image_slice.SetMetaData("0008|0060", "MR")
    image_slice.SetMetaData(
        "0020|0032", "\\".join(map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i))))
    )
    image_slice.SetMetaData("0020|0013", str(i))
    image_slice.SetMetaData("0008|0018", "1.2.3" + str(i))
    image_slice.SetMetaData("0002|0003", "1.2.3" + str(i))

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(os.path.join(path, str(i) + ".dcm"))
    writer.Execute(image_slice)


def writeDicoms(imgPath, dicomDir, UID_base, subject_id, series_num):
    """Write a NIfTI image as a series of DICOM slices.

    Args:
        imgPath (str): Path to the input NIfTI file.
        dicomDir (str): Output directory for DICOM slices.
        UID_base (str): Base UID for DICOM series/study identifiers.
        subject_id (str): Subject identifier for DICOM patient tags.
        series_num (int): DICOM series number.
    """
    if not os.path.isdir(dicomDir):
        os.makedirs(dicomDir)

    img = nib.load(imgPath).get_fdata()
    img = convert(img, 0, 65535, "uint16")
    img = np.flip(np.flip(np.transpose(img, [2, 1, 0]), axis=1), axis=2)

    ishape = img.shape
    img = sitk.GetImageFromArray(img)
    origin = [-1 * (i // 2) for i in ishape]
    img.SetSpacing([1.25, 1.25, 1.25])
    img.SetOrigin(origin)

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    direction = img.GetDirection()

    series_tag_values = [
        ("0002|0002", "1.2.840.10008.5.1.4.1.1.4"),
        ("0008|0016", "1.2.840.10008.5.1.4.1.1.4"),
        ("0008|0031", modification_time),
        ("0008|0021", modification_date),
        ("0008|0008", "ORIGINAL\\SECONDARY"),
        ("0020|000e", UID_base + f".{series_num}"),
        ("0020|000d", UID_base),
        (
            "0020|0037",
            "\\".join(
                map(
                    str,
                    (
                        direction[0],
                        direction[3],
                        direction[6],
                        direction[1],
                        -direction[4],
                        direction[7],
                    ),
                )
            ),
        ),
        ("0008|1030", "IPF"),
        ("0008|103e", "3D UltraShort Echo Time"),
        ("0025|1007", str(ishape[-1])),
        ("0010|0010", f"subject_{subject_id}"),
        ("0010|0020", f"subject_{subject_id}"),
        ("0028|0030", "1.25\\1.25"),
        ("0018|0088", "1.25"),
        ("0018,0050", "1.25"),
        ("0020|0011", f"{series_num}"),
        ("0020|0012", "1"),
        ("0018|0020", "GR"),
        ("0018|0021", "SP"),
        ("0028|1050", "6000"),
        ("0028|1051", "12000"),
    ]

    list(map(lambda i: writeSlices(series_tag_values, img, dicomDir, UID_base, i), range(img.GetDepth())))
