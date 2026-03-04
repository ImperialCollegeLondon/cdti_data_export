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

# from numpy.typing import NDArray
import re
import json
import nibabel as nib
from datetime import datetime
import yaml


def is_unique(s):
    # check if dataframe column is unique
    a = s.to_numpy()
    return (a[0] == a).all()


def detect_outliers_iqr(data, threshold=1.5):
    """
    Detect outliers using the IQR method.

    Parameters:
    - data: 1D array of numbers
    - threshold: multiplier for IQR (1.5 is standard, 3.0 is more conservative)

    Returns:
    - mask: boolean array where True indicates an outlier
    - outliers: array of outlier values
    """
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outlier_mask = (data < lower_bound) | (data > upper_bound)

    return outlier_mask, data[outlier_mask]


def detect_outliers_mad(data, threshold=3.5):
    """
    Detect outliers using Modified Z-Score (MAD method).
    More robust than standard Z-score.

    Parameters:
    - data: 1D array of numbers
    - threshold: modified z-score threshold (3.5 is recommended)

    Returns:
    - mask: boolean array where True indicates an outlier
    - outliers: array of outlier values
    """
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))

    # Modified z-score
    modified_z_scores = (
        0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)
    )

    outlier_mask = np.abs(modified_z_scores) > threshold

    return outlier_mask, data[outlier_mask]


def detect_outliers_percentage(data, threshold_percent=50):
    """
    Detect outliers based on percentage difference from median.
    Works when most data is identical.

    Parameters:
    - data: 1D array of numbers
    - threshold_percent: percentage difference threshold (e.g., 50 = 50%)

    Returns:
    - mask: boolean array where True indicates an outlier
    - outliers: array of outlier values
    """
    median = np.nanmedian(data)

    if median == 0:
        # Avoid division by zero - use absolute difference instead
        outlier_mask = np.abs(data - median) > threshold_percent
    else:
        # Calculate percentage difference
        percent_diff = np.abs((data - median) / median) * 100
        outlier_mask = percent_diff > threshold_percent

    return outlier_mask, data[outlier_mask]


def check_mostly_identical_mad(data):
    """
    Check if MAD (Median Absolute Deviation) is zero.

    Returns:
    - is_mostly_identical: boolean
    - mad: median absolute deviation
    - median: the median value
    """
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))

    is_mostly_identical = mad == 0

    return is_mostly_identical, mad, median


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
        val = c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx][
            "CardiacSynchronizationSequence"
        ][0]["RRIntervalTimeNominal"]

        if val == None:
            val = 0.0
        else:
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


def get_series_time(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> str:
    """
    Get series time string

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    Series time

    """
    if dicom_type == 1:
        return c_dicom_header["SeriesTime"]
    else:
        return c_dicom_header["SeriesTime"]


def get_series_date(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> str:
    """
    Get series date string.

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    Series date

    """
    if dicom_type == 1:
        return c_dicom_header["SeriesDate"]
    else:
        return c_dicom_header["SeriesDate"]


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


def get_series_description(
    c_dicom_header: dict, dicom_type: int, frame_idx: int
) -> str:
    """
    Get series description

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    Series description string

    """
    if "SeriesDescription" in c_dicom_header:
        return c_dicom_header["SeriesDescription"]
    else:
        return "None"


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
            "signet"
            + "_"
            # + c_dicom_header["SeriesDate"]
            # + str(round((float(c_dicom_header["StudyTime"]))))
            # + "_"
            + str(c_dicom_header["SeriesNumber"])
        )
        suffix = suffix.replace(" ", "_")
        return suffix

    elif dicom_type == 1:
        suffix = (
            "signet"
            + "_"
            # + c_dicom_header["SeriesDate"]
            # + str(round((float(c_dicom_header["StudyTime"]))))
            # + "_"
            + str(c_dicom_header["SeriesNumber"])
        )
        suffix = suffix.replace(" ", "_")
        return suffix


def get_b_value(
    c_dicom_header: dict, dicom_type: int, dicom_manufacturer: int, frame_idx: int
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
            if "b_value" in c_dicom_header.keys():
                return c_dicom_header["b_value"]
            else:
                return 0.0
        elif dicom_manufacturer == "philips":
            if "b_value" in c_dicom_header.keys():
                return c_dicom_header["b_value"]
            else:
                return 0.0


def get_diff_dir(
    c_dicom_header: dict, dicom_type: int, dicom_manufacturer: int, frame_idx: int
) -> float:
    """
    Get diffusion direction from a dict with the DICOM header.

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
            "DiffusionGradientDirectionSequence"
            in c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx][
                "MRDiffusionSequence"
            ][0].keys()
        ):
            if (
                "DiffusionGradientOrientation"
                in c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx][
                    "MRDiffusionSequence"
                ][0]["DiffusionGradientDirectionSequence"][0].keys()
            ):
                val = c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx][
                    "MRDiffusionSequence"
                ][0]["DiffusionGradientDirectionSequence"][0][
                    "DiffusionGradientOrientation"
                ]
                return tuple([float(i) for i in val])
            else:
                return (0.0, 0.0, 0.0)
        else:
            return (0.0, 0.0, 0.0)

    elif dicom_type == 1:
        if dicom_manufacturer == "siemens":
            if "diffusion_direction" in c_dicom_header.keys():
                return tuple([float(i) for i in c_dicom_header["diffusion_direction"]])
            else:
                return (0.0, 0.0, 0.0)
        elif dicom_manufacturer == "philips":
            if "diffusion_direction" in c_dicom_header.keys():
                return tuple([float(i) for i in c_dicom_header["diffusion_direction"]])
            else:
                return (0.0, 0.0, 0.0)


def get_diff_dir_philips_log(
    diff_dir: Tuple, rotation_matrix: np.ndarray, manufacturer: str
) -> Tuple:
    """
    Get the diffusion direction rotated by the rotation matrix

    Parameters
    ----------
    diff_dir: tuple with the diffusion direction
    rotation_matrix: 3x3 rotation matrix

    Returns
    -------
    Rotated diffusion direction

    """
    if manufacturer != "philips":
        None
    else:
        if diff_dir == (0.0, 0.0, 0.0):
            return diff_dir
        else:
            rotated_diff_dir = np.matmul(np.array(diff_dir), rotation_matrix)
            # round values to 3 decimal places
            rotated_diff_dir = np.round(rotated_diff_dir, 5)
            rotated_diff_dir = [
                rotated_diff_dir[1],
                -rotated_diff_dir[0],
                rotated_diff_dir[2],
            ]
            return tuple(rotated_diff_dir)


def get_image_position(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> Tuple:
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
        if "ImagePositionPatient" in c_dicom_header:
            val = tuple([float(i) for i in c_dicom_header["ImagePositionPatient"]])
        else:
            val = (0.0, 0.0, 0.0)

        return val


def get_image_type(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> Tuple:
    """
    Get image type from the DICOM header

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
        val = c_dicom_header["ImageType"][0]

    elif dicom_type == 1:
        val = c_dicom_header["ImageType"][0]

    return val


# get DICOM header fields
def dictify(ds: pydicom.dataset.Dataset, manufacturer: str, dicom_type: str) -> dict:
    """Turn a pydicom Dataset into a dict with keys derived from the Element tags.
    Private info is not collected, because we cannot access it with the keyword.
    So we need to manually fish the diffusion information in the old DICOMs.

    Args:
        ds: The Dataset to dictify
        manufacturer: Manufacturer of the DICOM files (siemens, philips, ge, uih)
        dicom_type: DICOM type (legacy or enhanced)

    Returns:
        output: A dictionary with the DICOM header information

    """

    output = dict()
    # iterate over all non private fields
    for elem in ds:
        if elem.VR != "SQ":
            output[elem.keyword] = elem.value
        else:
            output[elem.keyword] = [
                dictify(item, manufacturer, dicom_type) for item in elem
            ]

    # add manually private diffusion fields if they exist for legacy DICOMs
    if dicom_type == 1:
        if manufacturer == "siemens":
            if [0x0019, 0x100C] in ds:
                output["b_value"] = ds[0x0019, 0x100C].value
            if [0x0019, 0x100E] in ds:
                output["diffusion_direction"] = ds[0x0019, 0x100E].value

        if manufacturer == "philips":
            if [0x0018, 0x9087] in ds:
                output["b_value"] = ds[0x0018, 0x9087].value
            if [0x0018, 0x9089] in ds:
                output["diffusion_direction"] = ds[0x0018, 0x9089].value

        if manufacturer == "ge":
            if [0x0018, 0x9087] in ds:
                output["b_value"] = ds[0x0018, 0x9087].value
            if (
                [0x0019, 0x10BB] in ds
                and [0x0019, 0x10BC] in ds
                and [0x0019, 0x10BD] in ds
            ):
                output["diffusion_direction"] = [
                    ds[0x0019, 0x10BB].value,
                    ds[0x0019, 0x10BC].value,
                    ds[0x0019, 0x10BD].value,
                ]
                # convert list of strings to list of floats
                output["diffusion_direction"] = [
                    float(i) for i in output["diffusion_direction"]
                ]

        if manufacturer == "uih":
            # I was told by UIH team that the real DiffusionBValue is in the following tag [0x0065, 0x1009].
            # There is also the tag DiffusionBValue [0x0018, 0x9087], but this one seems to have approximate
            # b-values. So I am using the first one:
            if [0x0065, 0x1009] in ds:
                output["b_value"] = ds[0x0065, 0x1009].value
            if [0x0018, 0x9089] in ds:
                output["diffusion_direction"] = ds[0x0018, 0x9089].value

            # I was also told by UIH team that the DiffusionGradientDirection is in the following
            # tag [0x0065, 0x1037] and the directions are in the image coordinate system.
            # But the header already contains another field called DiffusionGradientOrientation,
            # so I am using that one instead, which seems to be in the magnetic coordinate system.
            # if [0x0065, 0x1037] in ds:
            #     output["DiffusionGradientDirection"] = ds[0x0065, 0x1037].value

    return output


def add_slice_and_frame_index(
    df: pd.DataFrame, n_images_per_file: int, manual_config: dict
) -> pd.DataFrame:
    """
    Add slice and frame index to the dataframe.
    # For now, this is going to be very simple.
        # We are considering only one slice
        # Frame idx increases for each row and resets for a new series

    Parameters
    ----------
    df
    n_images_per_file
    manual_config

    Returns
    -------
    table with added slice and frame index

    """

    df["frame_dim_idx"] = 0
    df["slice_dim_idx"] = 0

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

        # remove outliers, first detect if this is in-vivo or phantom
        is_mostly_identical, mad, median = check_mostly_identical_mad(time_delta)

        if is_mostly_identical:
            mask, outliers = detect_outliers_percentage(time_delta)
        else:
            mask, outliers = detect_outliers_mad(time_delta)

        # change to nan outliers in the time delta
        time_delta[mask] = np.nan

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
    This is for SIEMENS STEAM data.
    This function will adjust:
    - first it will change the b0 images from 0 to a >0 number
    - all b-values will also be adjusted from variations on the RR-interval

    Values are fished from the ImageComments DICOM field first, if not present
    then the manual_config values are used.

    data: dataframe with diffusion database
    manual_config: dict
    info: dict

    Returns
    -------
    dataframe with adjusted b-values
    """

    n_entries, _ = data.shape

    # copy the b-values to another column to save the original prescribed
    # b-value
    data["b_value_original"] = data["b_value"]

    data = estimate_rr_interval(data)

    # read dicom comments to get assumed RR interval and b0 value
    if "ImageComments" in info:
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
        # otherwise use the estimated RR interval.
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


def check_dicom_version_and_manufacturer(
    header_info: pydicom.dataset.Dataset,
) -> [int, int, str]:
    """
    Check the DICOM version, number of images per DICOM and manufacturer

    Parameters
    ----------
    header_info

    Returns
    -------
    dicom_type
    n_images_per_file
    dicom_manufacturer

    """

    dicom_type = None
    if "PerFrameFunctionalGroupsSequence" in header_info:
        dicom_type = 2
        print("DICOM type: Enhanced")
        # How many images in one file?
        n_images_per_file = len(header_info.PerFrameFunctionalGroupsSequence)
        print("Number of images per DICOM: " + str(n_images_per_file))
    else:
        dicom_type = 1
        print("DICOM type: Legacy")
        n_images_per_file = 1
        print("Number of images per DICOM: " + str(n_images_per_file))

    # check manufacturer
    if "Manufacturer" in header_info:
        val = header_info["Manufacturer"].value
        if val == "Siemens Healthineers" or val == "Siemens" or val == "SIEMENS":
            manufacturer = "siemens"
            print("Manufacturer: Siemens")
        elif val == "Philips Medical Systems" or val == "Philips":
            manufacturer = "philips"
            print("Manufacturer: Philips")
        elif val == "GE MEDICAL SYSTEMS" or val == "GE":
            manufacturer = "ge"
            print("Manufacturer: GE")
        elif val == "UIH" or val == "United Imaging Healthcare":
            manufacturer = "uih"
            print("Manufacturer: United Imaging Healthcare")
        else:
            raise ValueError("Manufacturer not supported.")
    else:
        raise ValueError("Manufacturer field not found in header.")

    return dicom_type, n_images_per_file, manufacturer


def export_csv_files(df: pd.DataFrame, output_path: str):
    """
    Export the dataframe to CSV files.

    Parameters
    ----------
    df
    output_path

    Returns
    -------

    """

    column_labels = [
        "b_value",
        "frame_dim_idx",
        "slice_dim_idx",
        "nominal_interval",
        # "acquisition_date_time",
    ]

    series_in_table = df["series_number"].unique()

    for series in series_in_table:
        c_table = df[df["series_number"] == series]
        c_nii_file_suffix = c_table["nii_file_suffix"].unique()[0]
        c_file = glob.glob(os.path.join(output_path, "**" + c_nii_file_suffix + ".nii"))
        assert len(c_file) == 1, "More than one file found for this series!"
        c_file = c_file[0]
        c_file = c_file.replace(".nii", ".csv")
        c_table.to_csv(
            c_file,
            columns=column_labels,
            index=False,
        )


def export_csv_files_philips(df: pd.DataFrame, df_log: pd.DataFrame, output_path: str):
    """
    Export the dataframe to CSV files.

    Parameters
    ----------
    df
    df_log
    output_path

    Returns
    -------

    """

    column_labels = [
        "b_value",
        "frame_dim_idx",
        "slice_dim_idx",
        "nominal_interval",
    ]

    # get series numbers from the log table
    series_numbers = df_log["average_number"].unique().tolist()

    # create a new column in the df with the values from the series_description column
    df["series_number_matching_log"] = df["series_description"]
    # in this new column remove all non-numeric characters and convert to int
    df["series_number_matching_log"] = (
        df["series_number_matching_log"].str.replace("\\D", "", regex=True).astype(int)
    )
    # drop the number by one, in order to match the average number in the log table
    df["series_number_matching_log"] = df["series_number_matching_log"] - 1

    # loop over the series numbers and export the corresponding csv files
    for series in series_numbers:
        c_table = df[df["series_number_matching_log"] == series]
        c_table_log = df_log[df_log["average_number"] == series]
        # reset index of c_table and c_table_log
        c_table = c_table.reset_index(drop=True)
        c_table_log = c_table_log.reset_index(drop=True)
        # # separate in c_table the column "diffusion_direction_rotated" in three columns "diff_vector_x", "diff_vector_y", "diff_vector_z"
        # c_table[["diff_vector_x", "diff_vector_y", "diff_vector_z"]] = pd.DataFrame(
        #     c_table["diffusion_direction_rotated"].tolist(), index=c_table.index
        # )

        # transform in c_table the column "diffusion_direction_rotated" from a tuple of numpy arrays, into a numpy array
        c_table["diffusion_direction_rotated"] = c_table[
            "diffusion_direction_rotated"
        ].apply(
            lambda x: np.array(x) if isinstance(x, tuple) else np.array([0.0, 0.0, 0.0])
        )

        # in c_table_log also create a tuple column with the diffusion directions
        c_table_log["diffusion_direction"] = [
            x
            for x in c_table_log[
                ["diff_vector_x", "diff_vector_y", "diff_vector_z"]
            ].to_numpy()
        ]

        # check the column "nii_file_suffix" in c_table is always the same
        assert (
            len(c_table["nii_file_suffix"].unique()) == 1
        ), "More than one nii file suffix found for series " + str(series)

        c_csv_table = []
        frame_idx = 0
        slice_idx = 0
        # loop over each row in c_table
        for idx in range(len(c_table)):
            c_diff_vector = c_table.loc[idx, "diffusion_direction_rotated"]

            # look for the row in c_table_log with the closest diffusion direction
            c_table_log["diff_vector_distance"] = c_table_log[
                "diffusion_direction"
            ].apply(lambda row: np.linalg.norm(c_diff_vector - np.array(row)))
            c_closest_row = c_table_log.loc[
                c_table_log["diff_vector_distance"].idxmin()
            ]

            # assert b_value is the same
            assert (
                c_table.loc[idx, "b_value"] == c_closest_row["b_value"]
            ), f"b_value mismatch for series {series}, row {idx}"

            # get adjusted b_value from c_closest_row
            adjusted_b_value = (
                c_closest_row["b_value"]
                * c_closest_row["mixing_time"]
                / c_closest_row["assumed_RR"]
            )

            # add frame_idx, slice_idx and nominal_interval. frame_idx is a
            # counter of the image, slice_idx will be 0 for all assuming we are only acquiring one slice.
            nominal_interval = c_table.loc[idx, "nominal_interval"]
            c_csv_table.append(
                [
                    float(adjusted_b_value),
                    frame_idx,
                    slice_idx,
                    nominal_interval,
                ]
            )

            frame_idx += 1

        c_nii_file_suffix = c_table["nii_file_suffix"].unique()[0]
        c_file = glob.glob(os.path.join(output_path, "**" + c_nii_file_suffix + ".nii"))
        assert len(c_file) == 1, "More than one file found for this series!"
        c_file = c_file[0]
        c_file = c_file.replace(".nii", ".csv")
        c_csv_table = pd.DataFrame(
            c_csv_table,
            columns=column_labels,
        )
        c_csv_table.to_csv(c_file, index=False)


def adjust_philips_b_values(b0_val, df_dicom, df_mt):
    # get new column with average number (DICOM files)
    df_dicom["average"] = (
        df_dicom["nii_file_suffix"]
        .str.split("AVG")
        .str[-1]
        .str.split("_")
        .str[0]
        .astype(int)
        - 1
    )
    # get mixing time values and assumed RR from df_mt table to df. I have to match average and direction idx
    df_dicom["nominal_interval"] = 0
    df_dicom["assumed_RR"] = 0
    df_dicom["b_value_original"] = df_dicom["b_value"]
    df_dicom["b_value"] = 0
    df_dicom["b_value"] = df_dicom["b_value"].astype(float)
    # loop over the rows in the df
    for idx in range(len(df_dicom)):
        c_average = df_dicom.loc[idx, "average"]
        c_direction = df_dicom.loc[idx, "frame_dim_idx"]
        c_mixing_time = df_mt.loc[
            (df_mt["average"] == c_average) & (df_mt["direction"] == c_direction),
            "mixing_time",
        ].values[0]
        c_assumed_RR = df_mt.loc[
            (df_mt["average"] == c_average) & (df_mt["direction"] == c_direction),
            "ref RR",
        ].values[0]
        df_dicom.at[idx, "nominal_interval"] = c_mixing_time
        df_dicom.at[idx, "assumed_RR"] = c_assumed_RR
        c_bval = df_dicom.loc[idx, "b_value_original"]
        if c_bval == 0:
            c_bval = b0_val
        c_bval = c_bval * (c_mixing_time / c_assumed_RR)
        df_dicom.at[idx, "b_value"] = c_bval.astype(float)

    return df_dicom


def separate_philips_log_table(df_dicom, df_log):
    # separate table into mixing time, b0 values, and scan names tables
    # mixing time
    df_mt = df_log[df_log[9].str.contains("CS-SCMR-SIG mixing time:") == True]
    df_mt = df_mt[9].str.split(" ", expand=True)
    df_mt[4] = df_mt[4].str.replace(";", "", regex=True).astype(int)
    df_mt[7] = df_mt[7].str.replace(";", "", regex=True).astype(int)
    df_mt[9] = df_mt[9].str.replace(";", "", regex=True).astype(float)
    df_mt[12] = df_mt[12].str.replace(";", "", regex=True).astype(float)
    df_mt = df_mt.rename(
        columns={4: "average", 7: "direction", 9: "mixing_time", 12: "ref RR"}
    )
    # keep only the n_dicom rows of the b-values table
    n_dicoms = len(df_dicom)
    df_mt = df_mt.head(n_dicoms)
    # b0 value
    df_b0 = df_log[df_log[9].str.contains("CS-STE-DIFF: value b0:") == True]
    df_b0 = df_b0[9].str.split(" ", expand=True)
    df_b0 = df_b0.rename(columns={3: "b0 value"})
    assert is_unique(df_b0["b0 value"]), "b0 values are not unique in table!"
    b0_val = float(df_b0["b0 value"].values[0])
    # scan name
    df_scan_name = df_log[df_log[9].str.contains("scan_name") == True]

    return b0_val, df_mt, df_scan_name


def philips_export_csv_tables(
    df_dicom: pd.DataFrame,
    header_info: dict,
    dicom_path: str,
    n_images_per_file: int,
    output_path: str,
):
    """
    Export the dataframe to CSV files.
    This is for PHILIPS data

    Parameters
    ----------
    df_dicom
    header_info
    dicom_path
    n_images_per_file

    """

    # read the log file
    df_log = read_philips_steam_log(df_dicom, dicom_path)

    # create csv tables with the adjusted b-values, the frame index and slice index for each nii file
    export_csv_files_philips(df_dicom, df_log, output_path)

    # b0_val, df_mt, df_scan_names = separate_philips_log_table(df_dicom, df_log)

    # # add slice and frame index to each row to match the nii arrays
    # df_dicom = add_slice_and_frame_index(df_dicom, n_images_per_file, manual_config)

    # # add adjusted b-values to df_dicom
    # df_dicom = adjust_philips_b_values(b0_val, df_dicom, df_mt)

    # # export df_dicom to csv tables
    # export_csv_files(df_dicom, output_path)


def dicom_info_table(
    dicom_files: list,
    dicom_manufacturer: str,
    dicom_type: int,
    n_images_per_file: int,
    rotation_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Create a table with DICOM information

    Parameters
    ----------
    dicom_files
    dicom_manufacturer
    dicom_type
    n_images_per_file
    rotation_matrix

    Returns
    df_dicom
    -------

    """
    # instantiate a table to store DICOM info
    df_dicom = []

    # loop over each DICOM file
    for idx, file_name in enumerate(dicom_files):

        # read current DICOM
        ds = pydicom.dcmread(open(file_name, "rb"))
        # convert header into a dict
        c_dicom_header = dictify(ds, dicom_manufacturer, dicom_type)

        # loop over each image in the current DICOM file
        for frame_idx in range(n_images_per_file):

            # append values (will be a row in the dataframe)
            df_dicom.append(
                (
                    # file name
                    os.path.basename(file_name),
                    # nominal interval
                    get_nominal_interval(c_dicom_header, dicom_type, frame_idx),
                    # acquisition time
                    get_acquisition_time(c_dicom_header, dicom_type, frame_idx),
                    # acquisition date
                    get_acquisition_date(c_dicom_header, dicom_type, frame_idx),
                    # series time
                    get_series_time(c_dicom_header, dicom_type, frame_idx),
                    # series date
                    get_series_date(c_dicom_header, dicom_type, frame_idx),
                    # series number
                    get_series_number(c_dicom_header, dicom_type, frame_idx),
                    # nii file name suffix
                    get_nii_file_suffix(c_dicom_header, dicom_type, frame_idx),
                    # b-value or zero if not a field
                    get_b_value(
                        c_dicom_header, dicom_type, dicom_manufacturer, frame_idx
                    ),
                    # diffusion direction
                    get_diff_dir(
                        c_dicom_header, dicom_type, dicom_manufacturer, frame_idx
                    ),
                    # get diffusion_direction rotated
                    get_diff_dir_philips_log(
                        get_diff_dir(
                            c_dicom_header, dicom_type, dicom_manufacturer, frame_idx
                        ),
                        rotation_matrix,
                        dicom_manufacturer,
                    ),
                    # image position
                    get_image_position(c_dicom_header, dicom_type, frame_idx),
                    # get image type
                    get_image_type(c_dicom_header, dicom_type, frame_idx),
                    # get series description
                    get_series_description(c_dicom_header, dicom_type, frame_idx),
                )
            )
    # column labels for the dataframe and for the csv file
    column_labels = [
        "file_name",
        "nominal_interval",
        "acquisition_time",
        "acquisition_date",
        "series_time",
        "series_date",
        "series_number",
        "nii_file_suffix",
        "b_value",
        "diffusion_direction",
        "diffusion_direction_rotated",
        "image_position",
        "image_type",
        "series_description",
    ]
    # create a dataframe from the list
    df_dicom = pd.DataFrame(
        df_dicom,
        columns=column_labels,
    )
    return df_dicom


def siemens_export_csv_tables(
    df_dicom: pd.DataFrame,
    header_info: dict,
    manual_config: dict,
    n_images_per_file: int,
    output_path: str,
):
    """
    Export csv tables with the adjusted b-values
    This is for SIEMENS data

    Parameters
    ----------
    df_dicom
    header_info
    manual_config
    n_images_per_file
    output_path

    Returns
    -------

    """
    # sort the dataframe by date and time, this is needed in case we need to adjust
    # the b-values by the DICOM timings
    df_dicom = sort_by_date_time(df_dicom)
    # add slice and frame index to each row to match the nii arrays
    df_dicom = add_slice_and_frame_index(df_dicom, n_images_per_file, manual_config)
    # adjust b-values
    df_dicom = adjust_b_val_and_dir(df_dicom, manual_config, header_info)
    # create a csv file with the adjusted b-values,
    # the frame index and slice index for each nii file
    export_csv_files(df_dicom, output_path)


def read_philips_steam_log(df_dicom: pd.DataFrame, dicom_path: str) -> pd.DataFrame:
    """
    Read Philips STEAM scan log file. Collect important diffusion information including:
    - scan order
    - mixing time and assumed RR intervals
    - b0 value

    Parameters
    ----------
    df_dicom
    dicom_path

    Returns
    -------
    df_log

    """

    # look for a file in the dicom_path that starts with devlogcurrent and ends with .csv
    log_files = glob.glob(os.path.join(dicom_path, "devlogcurrent*.csv"))
    assert len(log_files) == 1, "More than one log file found in the dicom folder!"
    log_file = log_files[0]
    df_log = pd.read_csv(log_file, encoding="ISO-8859-1", header=None)

    # get the number of columns in the log file
    n_cols = df_log.shape[1]

    # assert the number is always 20, otherwise we need to adjust the code below
    assert (
        n_cols == 20
    ), "Number of columns in the log file is not 20! I am counting it is always 20, otherwise we need column names!"

    # give names to all 20 columns
    column_names = df_log.columns.tolist()
    column_names[5] = "average_number"
    column_names[8] = "diff_vector"
    column_names[10] = "mixing_time"
    column_names[13] = "assumed_RR"
    column_names[15] = "b_value"
    column_names[17] = "diff_vector_x"
    column_names[18] = "diff_vector_y"
    column_names[19] = "diff_vector_z"

    df_log.columns = column_names

    # # open log to text list
    # # this file is assumed to be in the same folder as the input DICOMs
    # with open(
    #     os.path.join(dicom_path, "devlogcurrent.log"), encoding="ISO-8859-1"
    # ) as f:
    #     txt = f.readlines()

    # # remove new lines and separate text by tab
    # txt = [t.strip() for t in txt]
    # txt = [t.split("\t") for t in txt]

    # # convert list to pandas dataframe
    # df_log = pd.DataFrame(txt)

    # # remove some irrelevant columns
    # df_log = df_log.drop(columns=[0, 3, 4, 6, 7, 8, 10, 11, 12, 13])

    # # get table with only relevant diffusion information
    # searchfor = [
    #     "CS-SCMR-SIG mixing time:",
    #     "CS-STE-DIFF: value b0:",
    #     "00 00 00: scan_name",
    #     "00 00 00: start_scan_date_time",
    # ]
    # df_log = df_log[df_log[9].str.contains("|".join(searchfor)) == True]

    # # # ===========================================================
    # # # TODO this needs to be removed once fixed by C Stoeck
    # # print("===========================================================")
    # # print("Hard coded values for the log file!")
    # # print("Remove once fixed!")
    # # print("===========================================================")
    # # df_dicom["series_date"] = "20240606"
    # # # ===========================================================

    # # the DICOMS should have a fixed date and time, so we can filter the log file
    # assert is_unique(
    #     df_dicom["series_date"]
    # ), "series_date values are not unique in table!"
    # SeriesDate = df_dicom["series_date"].values[0]
    # assert is_unique(
    #     df_dicom["series_time"]
    # ), "series_time values are not unique in table!"
    # SeriesTime = df_dicom["series_time"].values[0]

    # # combine time and date
    # datetime_str = SeriesDate + " " + SeriesTime
    # datetime_object = datetime.strptime(datetime_str, "%Y%m%d %H%M%S.%f")
    # # remove anything before the series date and time
    # df_log["acquisition_date_time"] = df_log[1] + " " + df_log[2]
    # df_log["acquisition_date_time"] = pd.to_datetime(
    #     df_log["acquisition_date_time"], format="%Y-%m-%d %H:%M:%S.%f"
    # )
    # df_log = df_log.drop(columns=[1, 2])
    # df_log = df_log[df_log["acquisition_date_time"] > datetime_object]

    return df_log


def get_data_from_dicoms_and_export(
    dicom_path: str,
    output_path: str,
    sequence_option: str,
    anonymise_option: str,
    manual_config: dict,
):
    """
    # main function. This function will convert DICOMs to NIfTIs and
    # if steam data also create csv tables with the adjusted b-values

    Parameters
    ----------
    dicom_path: str path to dicom files
    output_path: str path to output folder for nifti files
    sequence_option: "se" or "steam"
    anonymise_option: "yes" or "no"
    manual_config: dict with manual configuration of some parameters

    """

    # create output folder if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # # if it exists, clear previous files
    # else:
    #     files = glob.glob(os.path.join(output_path, "*"))
    #     for f in files:
    #         os.remove(f)

    # create a dictionary with a checklist of the anonymisation pipeline
    anon_pipeline = {
        "running_dcm2niix_command": "no",
        "anonymising_json_files": "no",
        "anonymising_nii_files": "no",
        "csv_files_created": "no",
        "renaming_nii_json_bval_bvec_files": "no",
        "renaming_csv_files": "no",
    }

    # before running dcm2niix, build dicom database and remove any dicoms that are derived maps
    header_info, n_images_per_file, dicom_manufacturer, df_dicom = dicom_database(
        dicom_path
    )

    # create folder for derived maps dicoms
    folder_one_down = os.path.dirname(dicom_path)
    derived_maps_folder = os.path.join(folder_one_down, "derived_maps_dicoms")
    if not os.path.exists(derived_maps_folder):
        os.makedirs(derived_maps_folder)
    # iterate over the dicom database and move any derived maps dicoms to the new folder
    for idx in range(len(df_dicom)):
        c_image_type = df_dicom.loc[idx, "image_type"]
        c_file_name = df_dicom.loc[idx, "file_name"]
        if "DERIVED" in c_image_type:
            # check file hasn't already been moved
            if os.path.exists(os.path.join(dicom_path, c_file_name)):
                # move the dicom to the derived maps folder
                os.rename(
                    os.path.join(dicom_path, c_file_name),
                    os.path.join(derived_maps_folder, c_file_name),
                )

    # we also need to remove entries from the df_dicom that are b0 images but the b-value
    # is not 0.0 (some manufacturers store b0 images with a small b-value >0)
    if dicom_manufacturer == "siemens" and sequence_option == "steam":
        # create folder for derived maps dicoms
        b0s_data_folder = os.path.join(folder_one_down, "b0s_dicoms")
        if not os.path.exists(b0s_data_folder):
            os.makedirs(b0s_data_folder)
        # loop through the entries and remove any b0 images with b-value = 50
        # this is given by the direction [0.58, 0.58, 0.58] in Siemens STEAM data
        # this only happens for the STEAM phantom protocol
        for idx in range(len(df_dicom)):
            c_file_name = df_dicom.loc[idx, "file_name"]
            c_b_value = df_dicom.loc[idx, "b_value"]
            c_diff_dir = df_dicom.loc[idx, "diffusion_direction"]
            # round up direction to the second decimal
            c_diff_dir = list(c_diff_dir)
            c_diff_dir = [np.abs(np.round(i, 2)) for i in c_diff_dir]
            if c_diff_dir == [0.58, 0.58, 0.58] and c_b_value == 50:
                # check file hasn't already been moved
                if os.path.exists(os.path.join(dicom_path, c_file_name)):
                    # move the dicom to the b0s_data_folder
                    os.rename(
                        os.path.join(dicom_path, c_file_name),
                        os.path.join(b0s_data_folder, c_file_name),
                    )

    # run the dcm2niix command
    run_command = "dcm2niix -ba y -f signet_%s -o " + output_path + " " + dicom_path
    os.system(run_command)
    print("=============================================")
    print("dcm2niix command done.")
    print("=============================================")
    anon_pipeline["running_dcm2niix_command"] = "yes"

    # check at least one nifti file was created
    nii_files = glob.glob(os.path.join(output_path, "*.nii"))
    assert len(nii_files) > 0, "No NIfTI files found in the folder!"

    # create bool option for anonymisation
    if anonymise_option == "yes":
        anonymise_option = True
    else:
        anonymise_option = False

    if anonymise_option:
        # remove the acquisition date and time info from the json files
        json_files = glob.glob(os.path.join(output_path, "*.json"))
        for json_file in json_files:
            # json file to dict
            json_data = open(json_file)
            json_data = json_data.read()
            json_data = json.loads(json_data)

            # list of keys not to be removed
            keys_to_keep = [
                "ImageComments",
                "ImageOrientationPatientDICOM",
                "SliceThickness",
                "SeriesNumber",
            ]
            # remove all keys not in the list
            for key in list(json_data.keys()):
                if key not in keys_to_keep:
                    json_data.pop(key)

            # dict to json file
            json_string = json.dumps(json_data, indent=4)
            with open(json_file, "w") as f:
                f.write(json_string)

        print("=============================================")
        print("json files cleaned.")
        print("=============================================")
        anon_pipeline["anonymising_json_files"] = "yes"

        # Remove the "descrip" header field from the nifti files
        # it may contain acquisition time.
        nii_files = glob.glob(os.path.join(output_path, "*.nii"))
        for nii_file in nii_files:
            img = nib.load(nii_file)
            nii_hdr = img.header.copy()
            if "descrip" in nii_hdr:
                nii_hdr["descrip"] = "removed"
            new_img = nib.Nifti1Image(img.get_fdata(), img.affine, nii_hdr)
            nib.save(new_img, nii_file)

        print("=============================================")
        print("nii files cleaned.")
        print("=============================================")
        anon_pipeline["anonymising_nii_files"] = "yes"

    # if STEAM sequence, create csv tables with the adjusted b-values
    if sequence_option == "steam":

        # list all the DICOM files with extensions .dcm, .DCM,, .ima, or .IMA
        header_info, n_images_per_file, dicom_manufacturer, df_dicom = dicom_database(
            dicom_path
        )

        if dicom_manufacturer == "siemens":
            siemens_export_csv_tables(
                df_dicom, header_info, manual_config, n_images_per_file, output_path
            )
        elif dicom_manufacturer == "philips":

            philips_export_csv_tables(
                df_dicom, header_info, dicom_path, n_images_per_file, output_path
            )

        print("=============================================")
        print("csv file(s) exported successfully!")
        print("=============================================")
        anon_pipeline["csv_files_created"] = "yes"

    if anonymise_option:
        # rename all files to avoid potential identifiers
        def custom_file_rename(list_of_files, output_path):
            list_of_files.sort()
            for idx, file in enumerate(list_of_files):
                c_file = os.path.basename(file)

                filename, file_extension = os.path.splitext(c_file)
                new_name = "cdti_sig_data_" + str(idx + 1).zfill(3) + file_extension
                os.rename(file, os.path.join(output_path, new_name))

        # nii files
        nii_files = glob.glob(os.path.join(output_path, "*.nii"))
        custom_file_rename(nii_files, output_path)
        # json files
        json_files = glob.glob(os.path.join(output_path, "*.json"))
        custom_file_rename(json_files, output_path)
        # bval and bvec files
        bval_files = glob.glob(os.path.join(output_path, "*.bval"))
        custom_file_rename(bval_files, output_path)
        bvec_files = glob.glob(os.path.join(output_path, "*.bvec"))
        custom_file_rename(bvec_files, output_path)

        print("=============================================")
        print("nii, json, bval, bvec file(s) renamed successfully!")
        print("=============================================")
        anon_pipeline["renaming_nii_json_bval_bvec_files"] = "yes"

        if sequence_option == "steam":

            # csv files
            csv_files = glob.glob(os.path.join(output_path, "*.csv"))
            custom_file_rename(csv_files, output_path)

            print("=============================================")
            print("csv file(s) renamed successfully!")
            print("=============================================")
            anon_pipeline["renaming_csv_files"] = "yes"

    # save the anonymisation pipeline as a yaml file
    anon_pipeline_file = os.path.join(output_path, "anon_pipeline.yaml")
    with open(anon_pipeline_file, "w") as f:
        yaml.dump(anon_pipeline, f, default_flow_style=False)
    print("=============================================")
    print("Anonymisation pipeline saved and end of anonymisation pipeline.")
    print("=============================================")
    print("=============================================")
    print("=============================================")


def dicom_database(dicom_path):
    included_extensions = ["dcm", "DCM", "IMA", "ima"]
    dicom_files = [
        os.path.join(dicom_path, fn)
        for fn in os.listdir(dicom_path)
        if any(fn.endswith(ext) for ext in included_extensions)
    ]
    dicom_files.sort()

    assert len(dicom_files) > 0, "No DICOM files found in the folder!"

    # collect header info from the first DICOM
    header_info = pydicom.dcmread(open(dicom_files[0], "rb"))

    # check version, number of images per DICOM and manufacturer
    dicom_type, n_images_per_file, dicom_manufacturer = (
        check_dicom_version_and_manufacturer(header_info)
    )

    # get image orientation and rotation matrix
    if dicom_type == 2:
        iop = header_info["PerFrameFunctionalGroupsSequence"][0][
            "PlaneOrientationSequence"
        ][0]["ImageOrientationPatient"].value

    elif dicom_type == 1:
        iop = header_info["ImageOrientationPatient"].value

    first_column = np.array(iop[0:3])
    second_column = np.array(iop[3:6])
    third_column = np.cross(first_column, second_column)
    rotation_matrix = np.stack((first_column, second_column, third_column), axis=-1)

    # dictify dicom header
    header_info = dictify(header_info, dicom_manufacturer, dicom_type)

    # dataframe with DICOM info
    df_dicom = dicom_info_table(
        dicom_files, dicom_manufacturer, dicom_type, n_images_per_file, rotation_matrix
    )

    return header_info, n_images_per_file, dicom_manufacturer, df_dicom


if __name__ == "__main__":
    # arguments from command line

    # check if the number of arguments is correct
    assert len(sys.argv) == 5, "Incorrect number of arguments!"

    # path to where to store nii and other files
    output_path = sys.argv[2]
    # path to the DICOMs folder
    dicom_path = sys.argv[1]
    # sequence option
    sequence_option = sys.argv[3]
    # anonymise option
    anonymise_option = sys.argv[4]

    # checks of the input arguments
    assert os.path.exists(dicom_path), "DICOM path does not exist!"
    # assert not os.path.exists(output_path), "Output path already exists!"
    assert sequence_option in ["se", "steam"], "Sequence option not recognised!"
    assert anonymise_option in ["yes", "no"], "Anonymise option not recognised!"
    # check output argument folder doesn't exist or is empty
    if os.path.exists(output_path):
        assert len(os.listdir(output_path)) == 0, "Output folder is not empty!"

    # print the input arguments
    print("=============================================")
    print("DICOM path: " + dicom_path)
    print("Output path: " + output_path)
    print("Sequence option: " + sequence_option)
    print("Anonymise option: " + anonymise_option)
    print("=============================================")

    # ==========================================================
    # Manual configuration of some parameters
    # if everything else fails, we can use these values
    manual_config = {"assumed_rr_interval": 1000.0, "calculated_real_b0": 35.19}

    # run main function
    get_data_from_dicoms_and_export(
        dicom_path,
        output_path,
        sequence_option,
        anonymise_option,
        manual_config,
    )
