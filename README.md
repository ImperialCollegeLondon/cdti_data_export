# Exporting Anonymised Cardiac Diffusion Tensor Data

<p align="center">
  <img src="assets/main_fig/main_fig.png">
</p>

<p align="center">
    Anonymisation of cardiac diffusion tensor imaging data.<br>
</p>

This repository provides a Python script to anonymise cardiac DTI data and convert DICOM files to NIFTI format.

This is the standard procedure for sharing cardiac DTI data in the multicentre study of the [Cardiac Diffusion Special Interest Group (SCMR)](https://scmr.site-ym.com/group/Diffusion).

Below are the steps to install and run the script.

---

## Introduction

Cardiac DTI DICOM data should be converted to NIFTI format without any personal information.  
This Python script uses the [dcm2niix](https://github.com/rordenlab/dcm2niix) tool to:

- Export NIFTI files containing pixel data and minimal metadata
- Export b-values
- Export diffusion directions
- Save extra metadata in a JSON file
- Save adjusted b-values in a CSV file (STEAM sequences only)
- Save a YAML file with details of the anonymisation steps performed

> [!NOTE]
> The diffusion direction files produced by `dcm2niix` are already rotated to the image plane.

> [!WARNING]
> Enhanced multi-image DICOMs are not currently supported.  
> Philips STEAM data is work in progress, please report any issues.

---

## Installation

### dcm2niix

Install the `dcm2niix` tool. See [installation instructions](https://github.com/rordenlab/dcm2niix?tab=readme-ov-file#install).

For macOS with Homebrew:

```bash
brew install dcm2niix
```

### Python Environment

A recent version of Python 3 is required (developed with Python 3.12 on macOS).  
See [Python installation instructions](https://realpython.com/installing-python/) if needed.

### Setup Steps

1. **Clone or download this repository.**
2. **Create a virtual environment and install dependencies.**

If you have `git`, run:

```bash
git clone https://github.com/ImperialCollegeLondon/cdti_data_export.git
cd cdti_data_export
```

Create the virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel pip-tools
pip install -r requirements.txt
```

> [!NOTE]
> You may need to use `python3` instead of `python` on some systems.

---

## Running the Scripts

There are two scripts available in this repository:

- **Single folder mode** `cdti_data_export.py`: Script for exporting DICOM data to NIFTI format in one folder.
- **Multiple folders mode** `batch_process_multiple_folders.py`: Batch processing script for multiple folders, this script calls `cdti_data_export.py` for each folder in a pre-determined folder structure.

---

### Single Folder Mode

You will need these arguments to run the script:

- `<input_folder>`: Folder containing the DICOM files (all files should be at the root, not in subfolders).
- `<output_folder>`: Folder where the NIFTI files will be created.
- `sequence`: Either `se` or `steam`, depending on the sequence.
- `anonymisation`: `yes` or `no` (generally, use `yes`).

> [!WARNING]
> Ensure you have activated the Python virtual environment in the repository folder:
>
> ```bash
> cd <repository_folder>
> source .venv/bin/activate
> ```

> [!NOTE]
> For Philips STEAM data. 🚧 WORK IN PROGRESS 🚧
>
> Currently developing a way to create the adjusted b-value tables...

Run the script (examples):

```bash
# To anonymise SE data
python cdti_data_export.py <input_folder> <output_folder> se yes

# To anonymise STEAM data
python cdti_data_export.py <input_folder> <output_folder> steam yes
```

---

### Multiple Folders Mode

This script processes multiple folders in a pre-determined structure.
The folders must be in a rigid structure as follows:

```bash
.
└── group_name
    └── dicom
        ├── SE_full_fov
        │   ├── subject_01
        │   │   ├── scan_01
        │   │   │   └── 1.dcm...
        │   │   └── scan_02
        │   │       └── 1.dcm...
        │   ├── subject_02
        │   │   ├── scan_01
        │   │   │   └── 1.dcm...
        │   │   └── scan_02
        │   │       └── 1.dcm...
        │   └── subject_03
        │       ├── scan_01
        │       │   └── 1.dcm...
        │       └── scan_02
        │           └── 1.dcm...
        ├── SE_full_fov_slice_tracking
        │   ├── subject_01
        │   │   ├── scan_01
        │   │   │   └── 1.dcm...
        │   │   └── scan_02
        │   │       └── 1.dcm...
        │   ├── subject_02
        │   │   ├── scan_01
        │   │   │   └── 1.dcm...
        │   │   └── scan_02
        │   │       └── 1.dcm...
        │   └── subject_03
        │       ├── scan_01
        │       │   └── 1.dcm...
        │       └── scan_02
        │           └── 1.dcm...
        ├── SE_reduced_fov
        │   ├── subject_01
        │   │   ├── scan_01
        │   │   │   └── 1.dcm...
        │   │   └── scan_02
        │   │       └── 1.dcm...
        │   ├── subject_02
        │   │   ├── scan_01
        │   │   │   └── 1.dcm...
        │   │   └── scan_02
        │   │       └── 1.dcm...
        │   └── subject_03
        │       ├── scan_01
        │       │   └── 1.dcm...
        │       └── scan_02
        │           └── 1.dcm...
        └── STEAM
            ├── subject_01
            │   ├── scan_01
            │   │   └── 1.dcm...
            │   └── scan_02
            │       └── 1.dcm...
            ├── subject_02
            │   ├── scan_01
            │   │   └── 1.dcm...
            │   └── scan_02
            │       └── 1.dcm...
            └── subject_03
                ├── scan_01
                │   └── 1.dcm...
                └── scan_02
                    └── 1.dcm...
```

You don't need all the sequence folders to be present, but they must contain the capitalised string `SE` or `STEAM` for identification.

You also don't need to have repeat scans per subject, but you must have the same levels of hierarchy.

The DICOM files should be inside level 5 of the hierarchy, as shown above. Not in further subfolders.

You will need the following arguments to run the script:

- `<root_folder>`: full path for the root folder that contains the `group_name` subfolder.
- `anonymisation`: `yes` or `no` (generally, use `yes`).
- `overwrite`: `yes` or `no` (if you want to overwrite potentially existing NIFTI files, use `yes`).

Run the script example:

```bash
# To anonymise multiple folders and overwrite existing files
python batch_process_multiple_folders.py <root_folder> yes yes
```

---

## Output

If the scripts runs successfully, the nifti data folder(s) should contain:

- NIFTI files: `*.nii`
- b-values: `*.bval`
- diffusion directions: `*.bvec`
- Extra metadata: `*.json`
- Adjusted b-value tables: `*.csv` (STEAM sequences only)
- YAML file with anonymisation information: `anon_pipeline.yml`

**Please double-check that no private data (including acquisition date and time) is present in the output files.**
