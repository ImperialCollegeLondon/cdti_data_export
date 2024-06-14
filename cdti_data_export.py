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
import json
import nibabel as nib
from datetime import datetime


def is_unique(s):
    # check if dataframe column is unique
    a = s.to_numpy()
    return (a[0] == a).all()


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
            # + c_dicom_header["SeriesDate"]
            # + str(round((float(c_dicom_header["StudyTime"]))))
            # + "_"
            + str(c_dicom_header["SeriesNumber"])
        )
        suffix = suffix.replace(" ", "_")
        return suffix

    elif dicom_type == 1:
        suffix = (
            c_dicom_header["ProtocolName"]
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
            if "DiffusionBValue" in c_dicom_header.keys():
                return c_dicom_header["DiffusionBValue"]
            else:
                return 0.0
        elif dicom_manufacturer == "philips":
            return c_dicom_header["DiffusionBValue"]


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


def check_dicom_version(header_info: pydicom.dataset.Dataset) -> [int, int, str]:
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
            or header_info.Manufacturer == "SIEMENS"
        ):
            print("Manufacturer: SIEMENS")
            dicom_manufacturer = "siemens"
        elif (
            header_info.Manufacturer == "Philips Medical Systems"
            or header_info.Manufacturer == "Philips"
        ):
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

    return dicom_type, n_images_per_file, dicom_manufacturer


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
    df_dicom: pd.DataFrame, dicom_path: str, n_images_per_file: int, output_path: str
):
    """
    Export the dataframe to CSV files.
    This is for PHILIPS data

    Parameters
    ----------
    df_dicom
    dicom_path
    n_images_per_file

    """

    # read the log file
    df_log = read_philips_steam_log(df_dicom, dicom_path)

    b0_val, df_mt, df_scan_names = separate_philips_log_table(df_dicom, df_log)

    # add slice and frame index to each row to match the nii arrays
    df_dicom = add_slice_and_frame_index(df_dicom, n_images_per_file, manual_config)

    # add adjusted b-values to df_dicom
    df_dicom = adjust_philips_b_values(b0_val, df_dicom, df_mt)

    # export df_dicom to csv tables
    export_csv_files(df_dicom, output_path)


def dicom_info_table(
    dicom_files: list, dicom_manufacturer: str, dicom_type: int, n_images_per_file: int
) -> pd.DataFrame:
    """
    Create a table with DICOM information

    Parameters
    ----------
    dicom_files
    dicom_manufacturer
    dicom_type
    n_images_per_file

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
        c_dicom_header = dictify(ds)

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
        "series_time",
        "series_date",
        "series_number",
        "nii_file_suffix",
        "b_value",
        "image_position",
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
    # open log to text list
    # this file is assumed to be in the same folder as the input DICOMs
    with open(
        os.path.join(dicom_path, "devlogcurrent.log"), encoding="ISO-8859-1"
    ) as f:
        txt = f.readlines()

    # remove new lines and separate text by tab
    txt = [t.strip() for t in txt]
    txt = [t.split("\t") for t in txt]

    # convert list to pandas dataframe
    df_log = pd.DataFrame(txt)

    # remove some irrelevant columns
    df_log = df_log.drop(columns=[0, 3, 4, 6, 7, 8, 10, 11, 12, 13])

    # get table with only relevant diffusion information
    searchfor = [
        "CS-SCMR-SIG mixing time:",
        "CS-STE-DIFF: value b0:",
        "00 00 00: scan_name",
        "00 00 00: start_scan_date_time",
    ]
    df_log = df_log[df_log[9].str.contains("|".join(searchfor)) == True]

    # ===========================================================
    # TODO this needs to be removed once fixed by C Stoeck
    df_dicom["series_date"] = "20240606"
    # ===========================================================

    # the DICOMS should have a fixed date and time, so we can filter the log file
    assert is_unique(
        df_dicom["series_date"]
    ), "series_date values are not unique in table!"
    SeriesDate = df_dicom["series_date"].values[0]
    assert is_unique(
        df_dicom["series_time"]
    ), "series_time values are not unique in table!"
    SeriesTime = df_dicom["series_time"].values[0]

    # combine time and date
    datetime_str = SeriesDate + " " + SeriesTime
    datetime_object = datetime.strptime(datetime_str, "%Y%m%d %H%M%S.%f")
    # remove anything before the series date and time
    df_log["acquisition_date_time"] = df_log[1] + " " + df_log[2]
    df_log["acquisition_date_time"] = pd.to_datetime(
        df_log["acquisition_date_time"], format="%Y-%m-%d %H:%M:%S.%f"
    )
    df_log = df_log.drop(columns=[1, 2])
    df_log = df_log[df_log["acquisition_date_time"] > datetime_object]

    return df_log


def get_data_from_dicoms_and_export(
    dicom_path: str, output_path: str, steam_sequence_option: bool, manual_config: dict
):
    """
    # main function. This function will convert DICOMs to NIfTIs and
    # if steam data also create csv tables with the adjusted b-values

    Parameters
    ----------
    dicom_path: str path to dicom files
    output_path: str path to output folder for nifti files
    steam_sequence_option: bool if the sequence is STEAM
    manual_config: dict with manual configuration of some parameters

    """

    # create output folder if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # if it exists, clear previous files
    else:
        files = glob.glob(os.path.join(output_path, "*"))
        for f in files:
            os.remove(f)

    # run the dcm2niix command
    run_command = "dcm2niix -ba y -f %f_%p_%s -o " + output_path + " " + dicom_path
    os.system(run_command)
    print("=============================================")
    print("dcm2niix command done.")
    print("=============================================")

    # we need to remove the acquisition date and time info from the json files
    json_files = glob.glob(os.path.join(output_path, "*.json"))
    for json_file in json_files:
        # json file to dict
        json_data = open(json_file)
        json_data = json_data.read()
        json_data = json.loads(json_data)

        # remove acquisition date and time
        if "AcquisitionDate" in json_data:
            json_data.pop("AcquisitionDate")
        if "AcquisitionTime" in json_data:
            json_data.pop("AcquisitionTime")

        # dict to json file
        json_string = json.dumps(json_data, indent=4)
        with open(json_file, "w") as f:
            f.write(json_string)

    print("=============================================")
    print("json files cleaned.")
    print("=============================================")

    # I also need to remove the "descrip" header field from the nifti files
    # it may contain acquisition time.
    nii_files = glob.glob(os.path.join(output_path, "*.nii"))
    for nii_file in nii_files:
        img = nib.load(nii_file)
        nii_hdr = img.header
        if "descrip" in nii_hdr:
            nii_hdr["descrip"] = "removed"
        new_img = nib.Nifti1Image(img.get_fdata(), img.affine, nii_hdr)
        nib.save(new_img, nii_file)

    print("=============================================")
    print("nii files cleaned.")
    print("=============================================")

    # if STEAM sequence, create csv tables with the adjusted b-values
    if steam_sequence_option:

        # list all the DICOM files
        dicom_files = glob.glob(os.path.join(dicom_path, "*.dcm"))
        dicom_files.sort()

        assert len(dicom_files) > 0, "No DICOM files found in the folder!"

        # collect header info from the first DICOM
        header_info = pydicom.dcmread(open(dicom_files[0], "rb"))

        # check version, number of images per DICOM and manufacturer
        dicom_type, n_images_per_file, dicom_manufacturer = check_dicom_version(
            header_info
        )

        # dictify dicom header
        header_info = dictify(header_info)

        # dataframe with DICOM info
        df_dicom = dicom_info_table(
            dicom_files, dicom_manufacturer, dicom_type, n_images_per_file
        )

        if dicom_manufacturer == "siemens":
            siemens_export_csv_tables(
                df_dicom, header_info, manual_config, n_images_per_file, output_path
            )
        elif dicom_manufacturer == "philips":
            philips_export_csv_tables(
                df_dicom, dicom_path, n_images_per_file, output_path
            )

        print("=============================================")
        print("csv file(s) exported successfully!")
        print("=============================================")


if __name__ == "__main__":
    # arguments from command line
    # path to where to store nii and other files
    output_path = sys.argv[2]
    # path to the DICOMs folder
    dicom_path = sys.argv[1]
    # steam sequence option
    if len(sys.argv) > 3 and sys.argv[3] == "steam":
        steam_sequence_option = True
        print("STEAM sequence option enabled.")
    else:
        steam_sequence_option = False
        print("SE sequence.")

    # ==========================================================
    # Manual configuration of some parameters
    # if everything else fails, we can use these values
    manual_config = {"assumed_rr_interval": 1000.0, "calculated_real_b0": 30}

    # run main function
    get_data_from_dicoms_and_export(
        dicom_path,
        output_path,
        steam_sequence_option,
        manual_config,
    )
