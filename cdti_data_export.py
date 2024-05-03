"""
Script to convert cardiac DTI DICOMs to NIfTI plus extra side files
Including csv file with the nominal intervals and acquisition times for each image.
"""

import sys
import glob
import os
import pydicom
import pandas as pd
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
import re


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
        if "NominalInterval" in c_dicom_header:
            val = float(c_dicom_header["NominalInterval"])
        else:
            val = "None"
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


def get_series_number(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> str:
    """
    Get series number

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    Suffix string

    """
    if dicom_type == 2:
        return c_dicom_header["SeriesNumber"]

    elif dicom_type == 1:
        return c_dicom_header["SeriesNumber"]


def get_nii_file_suffix(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> str:
    """
    Build the suffix nii file name corresponding to the current DICOM image

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    Suffix string

    """
    if dicom_type == 2:
        suffix = (
            c_dicom_header["SeriesDescription"]
            + "_"
            + c_dicom_header["SeriesDate"]
            + str(round((float(c_dicom_header["StudyTime"]))))
            + "_"
            + str(c_dicom_header["SeriesNumber"])
        )
        suffix = suffix.replace(" ", "_")
        return suffix

    elif dicom_type == 1:
        suffix = (
            c_dicom_header["SeriesDescription"]
            + "_"
            + c_dicom_header["SeriesDate"]
            + str(round((float(c_dicom_header["StudyTime"]))))
            + "_"
            + str(c_dicom_header["SeriesNumber"])
        )
        suffix = suffix.replace(" ", "_")
        return suffix


def get_b_value(
    c_dicom_header: dict, dicom_type: str, dicom_manufacturer: str, frame_idx: int
) -> float:
    """
    Get b-value from a dict with the DICOM header.
    If no b-value fond, then return 0.0

    Parameters
    ----------
    c_dicom_header
    dicom_type
    dicom_manufacturer
    frame_idx

    Returns
    -------
    b_value

    """
    if dicom_type == 2:
        if (
            "DiffusionBValue"
            in c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx][
                "MRDiffusionSequence"
            ][0].keys()
        ):
            return c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx][
                "MRDiffusionSequence"
            ][0]["DiffusionBValue"]
        else:
            return 0.0

    elif dicom_type == 1:
        if dicom_manufacturer == "siemens":
            if "DiffusionBValue" in c_dicom_header.keys():
                return c_dicom_header["DiffusionBValue"]
            else:
                return 0.0
        elif dicom_manufacturer == "philips":
            return c_dicom_header["DiffusionBValue"]


def get_image_position(c_dicom_header: dict, dicom_type: str, frame_idx: int) -> Tuple:
    """
    Get the image position patient info from the DICOM header

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    image position patient

    """
    if dicom_type == 2:
        val = tuple(
            [
                float(i)
                for i in c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx][
                    "PlanePositionSequence"
                ][0]["ImagePositionPatient"]
            ]
        )

        return val

    elif dicom_type == 1:
        val = tuple([float(i) for i in c_dicom_header["ImagePositionPatient"]])

        return val


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


def add_slice_and_frame_index(df, n_images_per_file, manual_config):

    df["frame_dim_idx"] = 0
    df["slice_dim_idx"] = 0

    # for now to simplify, we will assume data has only one slice per series

    series_in_table = df["series_number"].unique()

    # loop over each series in the df
    for series in series_in_table:
        c_table = df[df["series_number"] == series]
        c_table.loc[:, "frame_dim_idx"] = np.divmod(
            np.arange(len(c_table)), n_images_per_file
        )[0]

        # slice_values = np.arange(n_images_per_file)
        # if manual_config["slice_order"] == "reverse":
        #     slice_values = slice_values[::-1]
        # c_table.loc[:, "slice_dim_idx"] = np.tile(
        #     slice_values, len(c_table) // n_images_per_file
        # )

        # df[df["series_number"] == series] = c_table
        df.loc[df["series_number"] == series, :] = c_table[:]

    return df


def sort_by_date_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the dataframe by acquisition date and time

    Parameters
    ----------
    df: dataframe with diffusion database

    Returns
    -------
    dataframe with sorted values
    """
    # create a new column with date and time, drop the previous two columns
    df["acquisition_date_time"] = df["acquisition_date"] + " " + df["acquisition_time"]

    # check if acquisition date and time information exist
    if not (df["acquisition_date"] == "None").all():
        df["acquisition_date_time"] = pd.to_datetime(
            df["acquisition_date_time"], format="%Y%m%d %H%M%S.%f"
        )
        # sort by date and time
        df = df.sort_values(["acquisition_date_time"], ascending=True)
    else:
        df["acquisition_date_time"] = "None"
        # if we don't have the acquisition time and date, then sort by series number? Not for now.
        # df = df.sort_values(["series_number"], ascending=True)

    # drop these two columns as we now have a single column with time and date
    df = df.drop(columns=["acquisition_date", "acquisition_time"])
    df = df.reset_index(drop=True)

    return df


def estimate_rr_interval(data: pd.DataFrame) -> [pd.DataFrame]:
    """
    This function will estimate the RR interval from the DICOM header
    and add it to the dataframe

    # if no nominal interval values are in the headers, then we will adjust
    # the b-values according to the RR interval by getting the time delta between images
    # convert time strings to microseconds

    Parameters
    ----------
    data: dataframe with diffusion database
    settings: dict

    Returns:
    dataframe with added estimated RR interval column
    estimated_rr_intervals_original (before adjustment, only for debug)
    """

    # check if we have acquisition date and time values
    # if so then estimate RR interval
    # if not then just copy the assumed values
    if not (data["acquisition_date_time"] == "None").all():
        # convert time to miliseconds
        time_stamps = data["acquisition_date_time"].astype(np.int64) / int(1e6)

        time_delta = np.diff(time_stamps) * 0.5

        # prepend nan to the time delta
        time_delta = np.insert(time_delta, 0, np.nan)
        # get median time delta, and replace values above 4x the median with nan
        median_time = np.nanmedian(time_delta)
        time_delta[time_delta > 4 * median_time] = np.nan
        # add time delta to the dataframe
        data["estimated_rr_interval"] = time_delta
        # replace nans with the next non-nan value
        data["estimated_rr_interval"] = data["estimated_rr_interval"].bfill()

    else:
        data["estimated_rr_interval"] = "None"
        data["nominal_interval"] = "None"

    return data


def adjust_b_val_and_dir(
    data: pd.DataFrame,
    manual_config: dict,
    info: dict,
) -> pd.DataFrame:
    """
    This function will adjust:
    . b-values according to the recorded RR interval
    . diffusion-directions rotate header directions to the image plane

    data: dataframe with diffusion database
    settings: dict
    info: dict
    data: dataframe with diffusion database
    logger: logger for console and file
    data_type: str with the type of data (dicom or nii)

    Returns
    -------
    dataframe with adjusted b-values and diffusion directions
    """

    n_entries, _ = data.shape

    # copy the b-values to another column to save the original prescribed
    # b-value
    data["b_value_original"] = data["b_value"]

    data = estimate_rr_interval(data)

    # read dicom comments to get assumed RR interval and b0 value
    if info["ImageComments"]:
        print("Dicom header comment found: " + info["ImageComments"])
        # get all numbers from comment field
        m = re.findall(r"[-+]?(?:\d*\.*\d+)", info["ImageComments"])
        m = [float(m) for m in m]
        if len(m) > 2:
            print("Header comment field is corrupted!")
            assumed_rr_int = manual_config["assumed_rr_interval"]
            calculated_real_b0 = manual_config["calculated_real_b0"]
        if len(m) == 2:
            print("Both b0 and RR interval found in header.")
            calculated_real_b0 = m[0]
            assumed_rr_int = m[1]
        elif len(m) == 1:
            print("Only b0 found in header.")
            calculated_real_b0 = m[0]
            assumed_rr_int = manual_config["assumed_rr_interval"]
        else:
            # incomplete info --> hard code numbers with the most
            # likely values
            print("No dicom header comment found!")
            assumed_rr_int = manual_config["assumed_rr_interval"]
            calculated_real_b0 = manual_config["calculated_real_b0"]
    else:
        print("No dicom header comment found!")
        # no info --> hard code numbers with the most
        # likely values
        assumed_rr_int = manual_config["assumed_rr_interval"]
        calculated_real_b0 = manual_config["calculated_real_b0"]

    print("calculated_real_b0: " + str(calculated_real_b0))
    print("assumed_rr_interval: " + str(assumed_rr_int))

    # loop through the entries and adjust b-values and directions
    print("Adjusting b-values")
    for idx in range(n_entries):
        c_b_value = data.loc[idx, "b_value"]

        # replace b0 value
        if c_b_value == 0:
            c_b_value = calculated_real_b0

        # correct b_value relative to the assumed RR interval with the nominal interval if not 0.0.
        # Otherwise use the estimated RR interval.
        c_nominal_interval = data.loc[idx, "nominal_interval"]
        c_estimated_rr_interval = data.loc[idx, "estimated_rr_interval"]
        if c_nominal_interval != 0.0:
            c_b_value = (
                c_b_value * (c_nominal_interval * 1e-3) / (assumed_rr_int * 1e-3)
            )
        else:
            c_b_value = (
                c_b_value * (c_estimated_rr_interval * 1e-3) / (assumed_rr_int * 1e-3)
            )

        # add the adjusted b-value to the database
        data.at[idx, "b_value"] = c_b_value

    return data


def get_data_from_dicoms_and_export(
    dicom_path: str, output_path: str, manual_config: dict
):

    # create output folder if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # # run the dcm2niix command
    # run_command = "dcm2niix -o " + output_path + " " + dicom_path
    # os.system(run_command)
    # print("=============================================")
    # print("dcm2niix command executed successfully!")
    # print("=============================================")

    # list all the DICOM files
    dicom_files = glob.glob(os.path.join(dicom_path, "*.dcm"))
    dicom_files.sort()

    # collect some header info from the first DICOM
    header_info = pydicom.dcmread(open(dicom_files[0], "rb"))

    # check version of dicom
    dicom_type = 0
    if "PerFrameFunctionalGroupsSequence" in header_info:
        dicom_type = 2
        print("DICOM type: Modern")
        # How many images in one file?
        n_images_per_file = len(header_info.PerFrameFunctionalGroupsSequence)
        print("Number of images per DICOM: " + str(n_images_per_file))
    else:
        dicom_type = 1
        print("DICOM type: Legacy")
        n_images_per_file = 1

    # check manufacturer
    if "Manufacturer" in header_info:
        if (
            header_info.Manufacturer == "Siemens Healthineers"
            or header_info.Manufacturer == "Siemens"
        ):
            print("Manufacturer: SIEMENS")
            dicom_manufacturer = "siemens"
        elif header_info.Manufacturer == "Philips Medical Systems":
            print("Manufacturer: Philips")
            dicom_manufacturer = "philips"
        elif header_info.Manufacturer == "GE MEDICAL SYSTEMS":
            print("Manufacturer: GE")
            sys.exit("GE DICOMs not supported yet.")
        else:
            print("Manufacturer: " + header_info.Manufacturer)
            sys.exit("Manufacturer not supported.")
    else:
        print("Manufacturer: None")
        sys.exit("Manufacturer not supported.")

    # dictify dicom header
    header_info = dictify(header_info)

    # create a table with some DICOM header fields
    df = []

    # loop over each DICOM file
    for idx, file_name in enumerate(dicom_files):

        # read current DICOM
        ds = pydicom.dcmread(open(file_name, "rb"))

        # convert header into a dict
        c_dicom_header = dictify(ds)

        # loop over each image in the current DICOM file
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
                    # series number
                    get_series_number(c_dicom_header, dicom_type, frame_idx),
                    # nii file name suffix
                    get_nii_file_suffix(c_dicom_header, dicom_type, frame_idx),
                    # b-value or zero if not a field
                    get_b_value(
                        c_dicom_header, dicom_type, dicom_manufacturer, frame_idx
                    ),
                    # image position
                    get_image_position(c_dicom_header, dicom_type, frame_idx),
                )
            )

    # column labels for the dataframe and for the csv file
    column_labels = [
        "file_name",
        "nominal_interval",
        "acquisition_time",
        "acquisition_date",
        "series_number",
        "nii_file_suffix",
        "b_value",
        "image_position",
    ]

    # create a dataframe from the list
    df = pd.DataFrame(
        df,
        columns=column_labels,
    )

    # sort the dataframe by date and time, this is needed in case we need to adjust
    # the b-values by the DICOM timings
    df = sort_by_date_time(df)

    # add slice and frame index to each row to match the nii arrays
    df = add_slice_and_frame_index(df, n_images_per_file, manual_config)

    # =========================================================
    # adjust b-values
    # =========================================================
    data = adjust_b_val_and_dir(df, manual_config, header_info)

    # create a csv file with the adjusted b-values, the frame index and slice index
    column_labels = ["b_value", "frame_dim_idx", "slice_dim_idx"]
    series_in_table = df["series_number"].unique()
    for series in series_in_table:
        c_table = df[df["series_number"] == series]
        c_nii_file_suffix = c_table["nii_file_suffix"].unique()[0]
        c_file = glob.glob(os.path.join(output_path, "**" + c_nii_file_suffix + ".nii"))
        assert len(c_file) == 1
        c_file = c_file[0]
        c_file = c_file.replace(".nii", ".csv")
        c_table.to_csv(
            c_file,
            columns=column_labels,
            index=False,
        )

    print("=============================================")
    print("csv file(s) exported successfully!")
    print("=============================================")


if __name__ == "__main__":
    # arguments from command line
    # path to where to store nii and other files
    output_path = sys.argv[1]
    # path to the DICOMs folder
    dicom_path = sys.argv[2]

    # ==========================================================
    # Manual configuration of some parameters
    manual_config = {}
    manual_config["assumed_rr_interval"] = 1000.0
    manual_config["calculated_real_b0"] = 30

    # run main function
    get_data_from_dicoms_and_export(dicom_path, output_path, manual_config)
