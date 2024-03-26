# Reading STEAM cDTI RR-interval information

Collect RR-interval and acquisition time information from STEAM cardiac diffusion scans.

The b-value set in the protocol assumes a certain RR-interval which is set at 1sec.
With the recorded RR-interval information, we can correct the b-value of each diffusion weighted image
due to RR-interval deviations from the assumed time of 1sec.

The nominal interval field in the DICOM header is used to calculate the RR-interval. 
As a fallback option, we can also use the acquisition times.

## Installation

We need python installed in the system. This script is known to work with python 3.10.

Go to the project folder and create a virtual environment and install the dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel pip-tools
pip install -r requirements.txt
```




## Running

Before running the Python script, please convert the DICOM files to NIfTI format if not yet done.
- Install [dcm2niix](https://github.com/rordenlab/dcm2niix)
- Run the code below to convert the DICOM files to NIfTI format.

```bash
dcm2niix -o <output_folder> <input_folder>
```

Where:
- `<output_folder>` is the path to the folder where the nii files will be stored
- `<input_folder>` is the path to the folder where the DICOM files are located.

>[!WARNING]
> Please make sure `<output_folder>` exists before running the `dcm2niix` command.

Once the `*.nii` files and associated `*.bvec, *.bval, *.json` files are created, we can run the python script.

>[!WARNING]
> Make sure you are using the python virtual environment created at the project folder:
```bash
cd <project_folder>
source .venv/bin/activate
```

Then run the following command:

```bash
python read_rr_intervals.py <output_folder> <input_folder>
```

Where: `<output_folder>` and `<input_folder>` are the paths used above with the `dcm2niix` command.

The script will create an `rr_intervals.csv` file in the NIfTI data folder.