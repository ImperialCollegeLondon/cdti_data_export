# STEAM cDTI

Collect important information from STEAM cardiac diffusion scans:
- RR-interval for every image
- Assumed RR-interval of the set b-values.

With this information we can correct the diffusion weighting for RR-interval variations.

### Installation in macOS (Intel and Apple silicon)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel pip-tools
pip install -r requirements.txt
```

## Running

```bash
python main_script.py output_folder input_folder
```
