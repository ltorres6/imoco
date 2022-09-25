import time
import os
import nibabel as nib
import SimpleITK as sitk
import numpy as np


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def write_slices(series_tag_values, new_img, path, UID_base, slice_num, frame_num):
    image_slice = new_img[:, :, slice_num, frame_num]

    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time

    image_slice.SetMetaData("0008|0060", "MR")

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    image_slice.SetMetaData(
        "0020|0032", "\\".join(map(str, new_img.TransformIndexToPhysicalPoint((0, 0, slice_num, frame_num))))
    )  # Image Position (Patient)
    # instance_num = slice_num * (frame_num + 1) + slice_num
    # image_slice.SetMetaData("0020|1041", str(slice_num * 1.25))  # Slice Location
    image_slice.SetMetaData("0020|0013", str(slice_num))  # Instance Number
    # image_slice.SetMetaData("0020|0012", str(frame_num))  # Acquisition Number

    image_slice.SetMetaData("0008|0018", "1.2.3" + str(slice_num))  # Media Storage SOP Instance UID
    image_slice.SetMetaData("0002|0003", "1.2.3" + str(slice_num))  # Media Storage SOP Instance UID
    # image_slice.SetMetaData("0020|0100", str(frame_num))  # Temporal Position Identifier
    # image_slice.SetMetaData("0020,9128", str(frame_num+1))  # Temporal Position Index
    # image_slice.SetMetaData("0020,9128", str(frame_num))  # Temporal Position Index
    # image_slice.SetMetaData("0020,9157", f"{1}/{slice_num+1}/{frame_num+1}")  # Dimension Index Values Attribute

    # image_slice.SetMetaData("0008|1155", UID_base)  # SOPInstanceUID
    # image_slice.SetMetaData("0020|9157", "0\\" + str(slice_num) + "\\0")  # Dimension Index Values

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(os.path.join(path, f"{frame_num}_{slice_num}.dcm"))
    writer.Execute(image_slice)


def write_dicoms(imgPath, dicomDir, UID_base, subject_id, series_num):
    pathExist = os.path.isdir(dicomDir)
    if pathExist is False:
        os.makedirs(dicomDir)

    img = nib.load(imgPath).get_fdata()
    # Convert to uint16
    img = convert(img, 0, 65535, "uint16")

    # Orient Properly
    img = np.flip(np.flip(np.transpose(img, [2, 1, 0, 3]), axis=1), axis=2)
    num_frames = img.shape[-1]
    ishape = img.shape
    # img = sitk.GetImageFromArray(img, isVector=True)
    # print(img.GetSize())

    images = []
    for frame in range(num_frames):
        images.append(sitk.GetImageFromArray(img[..., frame]))
    img = sitk.JoinSeries(images)
    # print(img.GetSize())

    origin = [-1 * (slice_num // 2) for slice_num in ishape]
    img.SetSpacing([0.7, 0.7, 0.7, 1])
    img.SetOrigin(origin)

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    direction = img.GetDirection()
    # UID_base = "1.2.840.0.1.3680043.2.1125." + modification_date + ".1" + modification_time
    # Study -> Series -> Imag
    series_tag_values = [
        ("0002|0002", "1.2.840.10008.5.1.4.1.1.4"),  # Media Storage SOP Class UID
        ("0008|0016", "1.2.840.10008.5.1.4.1.1.4"),  # Media Storage SOP Class UID
        ("0008|0031", modification_time),  # Series Time
        ("0008|0021", modification_date),  # Series Date
        ("0008|0008", "ORIGINAL\\SECONDARY"),  # Image Type
        ("0020|000e", UID_base + f".{series_num}"),  # Series Instance UID
        ("0020|000d", UID_base),  # Study Instance UID
        (
            "0020|0037",
            "\\".join(
                map(
                    str,
                    (
                        direction[0],
                        direction[3],
                        direction[6],  # Image Orientation (Patient)
                        -direction[1],
                        direction[4],
                        direction[7],
                    ),
                )
            ),
        ),
        ("0008|1030", "BPD"),  # Study Description
        ("0008|103e", "3D UltraShort Echo Time"),  # Series Description
        ("0025|1007", str(ishape[-2])),  # Images in Series
        ("0010|0010", f"subject_{subject_id}"),  # Patient Name
        ("0010|0020", f"subject_{subject_id}"),  # Patient ID
        ("0028|0030", "0.70\\0.70"),  # Pixel Spacing
        ("0018|0088", "0.70"),  # Slice Spacing
        ("0018,0050", "0.70"),  # Slice Thickness
        ("0020|0011", f"{series_num}"),  # Series Number
        ("0018|0020", "GR"),  # Scan Sequence Type (Gradient Recalled)
        ("0018|0021", "SP"),  # Scan Sequence Variant (Spoiled)
        # ("0018|0087", "3T"),  # Field Strength
        # ("0020|0052", UID_base + ".5",),  # Frame Of Reference for Series
        # ("0008|0070", "GE"),  # Manufacturer
        # ("0008|1090", "Unknown"),  # Manufacturer Model Name
        # ("0018|1000", "Unknown"),  # Device Serial Number
        # ("0018|1020", "Unknown"),  # Manufacturer Software Versions
        # ("0008|0023", modification_date),  # Content Date
        # ("0008|0033", modification_time),  # Content Time
        # ("0008|9205", "MONOCHROME"),  # Pixel Representation
        # ("0028|0002", "1"),  # Samples Per Pixel
        # ("0008|9206", "VOLUME"),  # Volumetric Properties
        # ("0008,9207", "NONE"),  # Volumetric Properties
        # ("0020,9221", "xx.yy.zz.1"),  # Volumetric Properties
        # ("0028|0008", f"{num_frames}"),  # Number of Frames
        # ("0028|0009", "[0020, 0100]"),  # Frame Pointer
        # ("0020|0105", f"{num_frames}"),  # Number of Temporal Positions
        ("0028|1050", "16000"),  # Window Center
        ("0028|1051", "32000"),  # Window Width
    ]

    # Write slices to output directory
    for frame_num in range(num_frames):
        for slice_num in range(img.GetDepth()):
            write_slices(series_tag_values, img, dicomDir, UID_base, slice_num, frame_num)

    # list(map(lambda slice_num: write_slices(series_tag_values, img, dicomDir, UID_base, slice_num), range(img.GetDepth()),))
