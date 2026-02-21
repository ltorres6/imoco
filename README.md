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

- [Conda](https://docs.conda.io/) environment manager
- [Poetry](https://python-poetry.org/) dependency manager
- [ANTs](https://github.com/ANTsX/ANTs) (for the default registration backend)

### Steps

```bash
# 1. Create and activate a conda environment
conda create -n imoco python=3.9
conda activate imoco

# 2. Install the package
poetry install

# 3. Install CuPy for GPU acceleration
conda install -c conda-forge cupy cudnn

# 4. (Optional) Install ANTs
#    Follow instructions at: https://github.com/ANTsX/ANTs
#    Ensure antsRegistration is on your PATH
```

## Usage

### YAML Configuration

The pipeline is driven by a single YAML configuration file that specifies paths, reconstruction parameters, and pipeline flags. See [`examples/example_config.yaml`](examples/example_config.yaml) for all available options.

### CLI

```bash
# Run with a config file
imoco examples/example_config.yaml

# Override paths from the command line
imoco config.yaml --raw_dir /data/subject01 --out_dir /output/subject01
```

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
