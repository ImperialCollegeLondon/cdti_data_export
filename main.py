"""
Script to acquire nominal interval data from STEAM cDTI DICOMs.
"""

import sys
import glob
import os
import pydicom
import pandas as pd


def get_nominal_interval(
    c_dicom_header: dict, dicom_type: int, frame_idx: int
) -> float:
    """
    Get the nominal interval from the DICOM header

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    Nominal interval

    """
    if dicom_type == 2:
        val = float(
            c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx][
                "CardiacSynchronizationSequence"
            ][0]["RRIntervalTimeNominal"]
        )
        return val

    elif dicom_type == 1:
        val = float(c_dicom_header["NominalInterval"])
        return val


def get_acquisition_time(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> str:
    """
    Get acquisition time string

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    Acquisition time

    """
    if dicom_type == 2:
        return c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx][
            "FrameContentSequence"
        ][0]["FrameAcquisitionDateTime"][8:]

    elif dicom_type == 1:
        return c_dicom_header["AcquisitionTime"]


def get_acquisition_date(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> str:
    """
    Get acquisition date string.

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    Acquisition date

    """
    if dicom_type == 2:
        return c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx][
            "FrameContentSequence"
        ][0]["FrameAcquisitionDateTime"][:8]

    elif dicom_type == 1:
        return c_dicom_header["AcquisitionDate"]


def dictify(ds: pydicom.dataset.Dataset) -> dict:
    """
    Turn a pydicom Dataset into a dict with keys derived from the Element tags.
    Private info is not collected, because we cannot access it with the keyword.
    So we need to manually fish the diffusion information in the old DICOMs.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The Dataset to dictify

    Returns
    -------
    DICOM header as a dict
    """

    output = dict()
    # iterate over all non private fields
    for elem in ds:
        if elem.VR != "SQ":
            output[elem.keyword] = elem.value
        else:
            output[elem.keyword] = [dictify(item) for item in elem]

    # add manually private diffusion fields if they exist
    if [0x0019, 0x100C] in ds:
        output["DiffusionBValue"] = ds[0x0019, 0x100C].value
    if [0x0019, 0x100E] in ds:
        output["DiffusionGradientDirection"] = ds[0x0019, 0x100E].value
    return output


def get_data_from_dicoms_and_export(dicom_path: str, output_path: str):
    dicom_files = glob.glob(os.path.join(dicom_path, "*.dcm"))
    dicom_files.sort()

    # collect some header info from the first DICOM
    ds = pydicom.dcmread(open(dicom_files[0], "rb"))

    # check DICOM header 1:legacy_header, 2:modern-header,
    dicom_type = 0
    if "PerFrameFunctionalGroupsSequence" in ds:
        dicom_type = 2
        # How many images in one DICOM file?
        n_images_per_file = len(ds.PerFrameFunctionalGroupsSequence)
    else:
        dicom_type = 1
        n_images_per_file = 1

    # create a dataframe with all DICOM values
    df = []
    for idx, file_name in enumerate(dicom_files):
        # read current DICOM
        ds = pydicom.dcmread(open(file_name, "rb"))
        # loop over the dictionary of header fields and collect them for this DICOM file
        c_dicom_header = dictify(ds)

        # loop over each frame within each file
        for frame_idx in range(n_images_per_file):
            # append values (will be a row in the dataframe)
            df.append(
                (
                    # file name
                    os.path.basename(file_name),
                    # nominal interval
                    get_nominal_interval(c_dicom_header, dicom_type, frame_idx),
                    # acquisition time
                    get_acquisition_time(c_dicom_header, dicom_type, frame_idx),
                    # acquisition date
                    get_acquisition_date(c_dicom_header, dicom_type, frame_idx),
                )
            )

    df = pd.DataFrame(
        df,
        columns=[
            "file_name",
            "nominal_interval",
            "acquisition_time",
            "acquisition_date",
        ],
    )

    df.to_csv(
        os.path.join(output_path, "timings.csv"),
        columns=[
            "file_name",
            "nominal_interval",
            "acquisition_time",
            "acquisition_date",
        ],
        index=False,
    )


if __name__ == "__main__":
    output_path = sys.argv[1]
    dicom_path = sys.argv[2]
    get_data_from_dicoms_and_export(dicom_path, output_path)
