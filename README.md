# Exporting anonymised Cardiac Diffusion Tensor data

All cardiac DTI data should be anonymised before being shared with external collaborators. 
This document describes the steps to anonymise the data and export it to a format that can be shared with collaborators.

## Introduction

Cardiac DTI DICOM data should be converted to NIfTI format without any personal information. 
[dcm2niix](https://github.com/rordenlab/dcm2niix) is going to be used for this purpose. In addition to the NIfTI files which contain 
the pixel values and a minimal header with metadata information, this tool also exports the b-values, 
diffusion directions (in the image plane) and the acquisition parameters in a JSON file.

In addition to the files mentioned above, we should also export the RR-interval 
information of the all images acquired. This is a requirement for cDTI data acquired with a STEAM sequence, and 
optional for the Spin-Echo (SE) sequence. The RR-interval information is used to correct the b-values of 
the diffusion weighted images. Even though this correction is not required for SE data, it is still useful
to have the RR-interval information for quality control purposes. So we advise to export the RR-interval information 
for all protocols.

### Option 1 (recommended for all and required for STEAM)

This repository contains a Python script that will:
- run the appropriate `dcm2niix` command to convert the DICOM files to NIfTI format.
- read the RR-interval information from the DICOM files and exports it to a CSV file.


### Option 2 (alternative for SE)

Alternatively and for SE data only, the `dcm2niix` command can be run directly to convert the 
DICOM files to NIfTI format. This option does not require any Python, but we do not export the RR-intervals.

---

## Installation

### Option 1: Run a Python script (required for STEAM, recommended for SE)

Install the `dcm2niix` tool. More information on how to install it can be 
found in [this link](https://github.com/rordenlab/dcm2niix?tab=readme-ov-file#install).

In addition, we need a recent version of Python 3 installed in the system. Check this link for 
[Python installation instructions](https://realpython.com/installing-python/).

Clone this repository, go to the project folder and create a virtual environment and install the dependencies with
the following terminal commands:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel pip-tools
pip install -r requirements.txt
```

### Option 2: Run only the `dcm2niix` command (alternative for SE): 

Install the `dcm2niix` tool. More information on how to install it can be 
found in [this link](https://github.com/rordenlab/dcm2niix?tab=readme-ov-file#install).

---

## Running

Whe need to have the full path for the input and output folders.
- `<input_folder>` is the path to the folder where the DICOM files are located.
- `<output_folder>` is the path to the folder where the nii files will be stored.


>[!WARNING]
> Please make sure `<output_folder>` exists at this stage.

### Option 1: Run a Python script (required for STEAM, recommended for SE)

>[!WARNING]
> Make sure you are using the python virtual environment created in the repository folder:
```bash
cd <project_folder>
source .venv/bin/activate
```

Then run the following command:

```bash
python cdti_data_export.py <output_folder> <input_folder>
```

### Option 2: Run only the `dcm2niix` command (only for SE): 

```bash
dcm2niix -o <output_folder> <input_folder>
```

---
### Output

Both **option 1** and **option 2** should create the `*.nii` files and associated `*.bvec, *.bval, *.json` 
files in the `<output folder>`. 
In addition, for **option 1** a `rr_timings.csv` file will also be created in the `<output folder>` 
with content similar to this:

|file_name   |nominal_interval|acquisition_time|acquisition_date|nii_file_suffix                |
|------------|----------------|----------------|----------------|-------------------------------|
|MR_00002.dcm|1000            |132309.1075     |20240223        |STEAM_standard_20240223131230_7|
|MR_00013.dcm|1002            |132311.1175     |20240223        |STEAM_standard_20240223131230_7|
|MR_00022.dcm|1000            |132313.1525     |20240223        |STEAM_standard_20240223131230_7|
|MR_00034.dcm|1007            |132315.1525     |20240223        |STEAM_standard_20240223131230_7|
|MR_00046.dcm|997             |132317.1625     |20240223        |STEAM_standard_20240223131230_7|
...
