# iMoCo

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

**Iterative Motion Compensation (iMoCo)** for pulmonary UTE MRI reconstruction.

This repository implements a suite of retrospective motion compensation techniques for free-breathing 3D ultra-short echo time (UTE) MRI, as described in:

> Torres LA, et al. "Comparison of Retrospective Motion Compensation Techniques for Pulmonary Dynamic UTE MRI in Idiopathic Pulmonary Fibrosis." *Journal of Magnetic Resonance Imaging (JMRI)*.

## Citation

```bibtex
@article{torres2024imoco,
  title={Comparison of Retrospective Motion Compensation Techniques for Pulmonary Dynamic UTE MRI in Idiopathic Pulmonary Fibrosis},
  author={Torres, Luis A and others},
  journal={Journal of Magnetic Resonance Imaging},
  year={2024}
}
```

## Reconstruction Methods

The following methods from the paper are implemented:

1. **No Gating** -- NUFFT adjoint using all acquired spokes
2. **Hard Gating** -- Binary selection of spokes near end-expiration
3. **Soft Gating** -- Exponential weighting favoring end-expiration
4. **XD-GRASP** -- Motion-resolved compressed sensing with temporal TV regularization
5. **iMoCo** -- Iterative motion compensation combining XD-GRASP with deformable registration

Additionally, the following experimental methods (not described in the paper) are included:

- **iMoCo (AirLab)** -- iMoCo using AirLab Demons registration instead of ANTs
- **MoCo (AirLab)** -- Post-hoc motion compensation using AirLab
- **Gridded Motion-Resolved** -- Non-iterative gridded reconstruction per motion state

## Installation

### Prerequisites

- Python 3.9--3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [ANTs](https://github.com/ANTsX/ANTs) (required for iMoCo and MoCo registration)

### Install the package

```bash
uv sync
```

### Install ANTs

The default iMoCo and MoCo pipelines call `antsRegistration` from the command line, so it must be on your `PATH`. The AirLab-based variants (`imoco_airlab`, `moco_airlab`) do not require ANTs.

**Option A -- conda-forge (easiest):**

```bash
conda install -c conda-forge ants
```

**Option B -- pre-built binaries:**

Download a release from [ANTsX/ANTs releases](https://github.com/ANTsX/ANTs/releases), extract it, and add the `bin/` directory to your `PATH`:

```bash
export ANTSPATH=/path/to/ants/bin
export PATH=$ANTSPATH:$PATH
```

**Option C -- build from source:**

Follow the [ANTs build instructions](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS).

**Verify** that ANTs is installed:

```bash
antsRegistration --version
```

## Usage

### CLI

```bash
# Run with just a data directory (uses default parameters, output goes to raw_dir/output/)
uv run imoco --raw_dir /data/subject01

# Run with a config file
uv run imoco config.yaml

# Override paths from the command line
uv run imoco config.yaml --raw_dir /data/subject01 --out_dir /output/subject01
```

See [`examples/example_config.yaml`](examples/example_config.yaml) for all available configuration options.

### Python API

```python
from imoco.pipeline import load_config, run

cfg = load_config("config.yaml")
cfg["raw_dir"] = "/data/subject01"
cfg["out_dir"] = "/output/subject01"
run(cfg)
```

## Data Format

**Input:**
- `MRI_Raw.h5` -- Raw UTE k-space data (HDF5), or pre-extracted `.npy` files (`ksp.npy`, `coord.npy`, `dcf.npy`)

**Output:**
- NIfTI images (`.nii.gz`) for each reconstruction method
- DICOM series (optional, requires `UID_base` and `subject_id` in config)
- Diagnostic plots and data usage reports in the `diagnostics/` directory

## Acknowledgments

This implementation builds on and was inspired by:

- [`mikgroup/extreme_mri`](https://github.com/mikgroup/extreme_mri) -- Respiratory signal estimation and auto-FOV
- [`PulmonaryMRI/imoco_recon`](https://github.com/PulmonaryMRI/imoco_recon) -- Original iMoCo framework
- [`sigpy`](https://github.com/mikgroup/sigpy) -- Signal processing and MRI reconstruction library (BSD 3-Clause)

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
