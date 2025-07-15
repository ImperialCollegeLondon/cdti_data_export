# This file will batch process multiple folders.
# The folders should be in a rigid structure:
# .
# └── group_name
#     └── dicom
#         ├── SE_full_fov
#         │   ├── subject_01
#         │   │   ├── scan_01
#         │   │   │   └── 1.dcm...
#         │   │   └── scan_02
#         │   │       └── 1.dcm...
#         │   ├── subject_02
#         │   │   ├── scan_01
#         │   │   │   └── 1.dcm...
#         │   │   └── scan_02
#         │   │       └── 1.dcm...
#         │   └── subject_03
#         │       ├── scan_01
#         │       │   └── 1.dcm...
#         │       └── scan_02
#         │           └── 1.dcm...
#         ├── SE_full_fov_slice_tracking
#         │   ├── subject_01
#         │   │   ├── scan_01
#         │   │   │   └── 1.dcm...
#         │   │   └── scan_02
#         │   │       └── 1.dcm...
#         │   ├── subject_02
#         │   │   ├── scan_01
#         │   │   │   └── 1.dcm...
#         │   │   └── scan_02
#         │   │       └── 1.dcm...
#         │   └── subject_03
#         │       ├── scan_01
#         │       │   └── 1.dcm...
#         │       └── scan_02
#         │           └── 1.dcm...
#         ├── SE_reduced_fov
#         │   ├── subject_01
#         │   │   ├── scan_01
#         │   │   │   └── 1.dcm...
#         │   │   └── scan_02
#         │   │       └── 1.dcm...
#         │   ├── subject_02
#         │   │   ├── scan_01
#         │   │   │   └── 1.dcm...
#         │   │   └── scan_02
#         │   │       └── 1.dcm...
#         │   └── subject_03
#         │       ├── scan_01
#         │       │   └── 1.dcm...
#         │       └── scan_02
#         │           └── 1.dcm...
#         └── STEAM
#             ├── subject_01
#             │   ├── scan_01
#             │   │   └── 1.dcm...
#             │   └── scan_02
#             │       └── 1.dcm...
#             ├── subject_02
#             │   ├── scan_01
#             │   │   └── 1.dcm...
#             │   └── scan_02
#             │       └── 1.dcm...
#             └── subject_03
#                 ├── scan_01
#                 │   └── 1.dcm...
#                 └── scan_02
#                     └── 1.dcm...

# command to get this: tree -P 1.dcm .

import sys
import os

from cdti_data_export import get_data_from_dicoms_and_export


# check if the number of arguments is correct
assert len(sys.argv) == 4, "Incorrect number of arguments!"

# path to group name
group_name_path = sys.argv[1]

# anonymise flag
anonymise_option = sys.argv[2]
if anonymise_option not in ["yes", "no"]:
    raise ValueError("Anonymise option must be 'yes' or 'no'!")

# overwrite flag
overwrite_option = sys.argv[3]
if overwrite_option == "yes":
    overwrite_option = True
elif overwrite_option == "no":
    overwrite_option = False
else:
    raise ValueError("Overwrite option must be 'yes' or 'no'!")

# dicom path
root_folder_path = os.path.join(group_name_path, "dicom")

# check if the path exists
assert os.path.exists(root_folder_path), f"Path {root_folder_path} does not exist!"

# list all subfolders in the group_name_path
sequence_folders = [
    f
    for f in os.listdir(root_folder_path)
    if os.path.isdir(os.path.join(root_folder_path, f))
]
sequence_folders.sort()

# loop over each sequence folder
for sequence_folder in sequence_folders:
    sequence_folder_path = os.path.join(root_folder_path, sequence_folder)

    # check if the path exists
    assert os.path.exists(
        sequence_folder_path
    ), f"Path {sequence_folder_path} does not exist!"

    # list all subject folders in the sequence folder
    subject_folders = [
        f
        for f in os.listdir(sequence_folder_path)
        if os.path.isdir(os.path.join(sequence_folder_path, f))
    ]
    subject_folders.sort()

    # loop over each subject folder
    for subject_folder in subject_folders:
        subject_folder_path = os.path.join(sequence_folder_path, subject_folder)

        # check if the path exists
        assert os.path.exists(
            subject_folder_path
        ), f"Path {subject_folder_path} does not exist!"

        # list all scan folders in the subject folder
        scan_folders = [
            f
            for f in os.listdir(subject_folder_path)
            if os.path.isdir(os.path.join(subject_folder_path, f))
        ]
        scan_folders.sort()
        # loop over each scan folder
        for scan_folder in scan_folders:
            scan_folder_path = os.path.join(subject_folder_path, scan_folder)

            # check if the path exists
            assert os.path.exists(
                scan_folder_path
            ), f"Path {scan_folder_path} does not exist!"

            # check if the scan folder contains files (hopefully DICOM files)
            assert (
                len(os.listdir(scan_folder_path)) > 0
            ), f"{scan_folder_path} is empty!"

            # get sequence substring
            sequence_substring = ""
            if "SE" in sequence_folder or "se" in sequence_folder:
                sequence_substring = "se"
            elif "STEAM" in sequence_folder or "steam" in sequence_folder:
                sequence_substring = "steam"
            else:
                raise ValueError(
                    f"Unknown sequence from folder name: {sequence_folder}. "
                    "Expected 'SE' or 'STEAM' in the folder name."
                )

            print("==================================================================")
            print("==================================================================")
            print(
                f"Processing: \n{scan_folder_path} \nwith sequence {sequence_substring} "
                f"and anonymise option {anonymise_option} and overwrite option {overwrite_option}"
            )
            print("==================================================================")

            # check if the nifti equivalent folder already contains files
            nii_folder_path = os.path.join(
                group_name_path, "nifti", sequence_folder, subject_folder, scan_folder
            )
            if os.path.exists(nii_folder_path) and len(os.listdir(nii_folder_path)) > 0:
                if overwrite_option:
                    print(f"Overwriting existing files in \n{nii_folder_path}")
                else:
                    print(
                        f"Skipping \n{nii_folder_path} \nas it already contains files. "
                        "Use 'yes' for the overwrite option to overwrite existing files."
                    )
                    continue

            # run python script to convert DICOMs to NIfTI
            # ==========================================================
            # Manual configuration of some parameters
            # if everything else fails, we can use these values
            manual_config = {"assumed_rr_interval": 1000.0, "calculated_real_b0": 30}

            get_data_from_dicoms_and_export(
                dicom_path=scan_folder_path,
                output_path=nii_folder_path,
                sequence_option=sequence_substring,
                anonymise_option=anonymise_option,
                manual_config=manual_config,
            )
