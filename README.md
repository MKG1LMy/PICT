# PICT Model

Optical-parameter estimation for layered time-domain near-infrared spectroscopy (TD-NIRS), with four models:

- **PICT Model:** measured IRF inputs, optional exact parameter priors and joint posterior samples.
- **DTOF-only:** posterior model without IRF inputs.
- **CNN / ANN:** deterministic DTOF-only baselines without parameter priors.

Inputs use **2500 bins at 4 ps/bin** and any nonempty subset of **10, 20, 30 and 40 mm SDS**. Each available DTOF channel has a paired measured IRF. Layer counts N=1, 2 and 3 are supported. Time alignment is disabled by default.

## Install

Requires Windows or Linux and Miniconda/Anaconda. Setup creates the dedicated `pict_model` environment with Python 3.12 and PyTorch 2.11.0 + CUDA 13.0. A separate CUDA Toolkit is unnecessary; driver checks produce warnings.

Retrieve the included data with Git LFS:

```text
git lfs install
git clone https://github.com/MKG1LMy/PICT.git
cd PICT
git lfs pull
```

| Platform | Install | Train |
|---|---|---|
| Windows | Double-click `setup.bat` | Double-click `train.bat` |
| Linux | `bash setup.sh` | `bash train.sh` |

Deployment logs are saved in `deployment-logs/`. For GPUs without BF16 support, set `training.amp` and `evaluation.amp` to `false` for FP32.

## Configure and train

Edit `configs/research.yaml` for shared settings and the model-specific YAML files for overrides. `configs/pipeline.json` selects which models to run.

The default queue runs **main → DTOF-only → CNN → ANN** for each training seed (**20736, 78631, 26605**). Split, validation and test seeds stay fixed at **2026**. Each run has its own `training.output_dir/seed_<seed>/` directory.

```text
train.bat --models main
train.bat --models cnn ann
train.bat --models main --run-epochs 2 --no-test
```

On Linux, substitute `bash train.sh`. Training resumes from the last complete epoch; use a new output directory when changing experiment settings. Final tests run after completion or early stopping, but not after an intermediate `--run-epochs` pause. For limited VRAM, reduce `training.micro_batch_size` and keep `effective_batch_size` an integer multiple.

## Data and results

`Data/` contains **6000 phantoms** in eight HDF5 files. The fixed train/validation/test split is **4800/600/600**, recorded in `artifacts/data/`. File checksums are listed in `DATA_SHA256.json`.

Tests produce offline HTML reports plus JSON, CSV and HDF5 records covering accuracy, prior and IRF interventions, channel subsets, calibration and candidate posterior modes. Finite out-of-domain samples are retained and flagged. Cross-model comparisons are separate commands:

```text
python -m tdnirs_posterior.cli.compare_irf_ablation --help
python -m tdnirs_posterior.cli.compare_baselines --help
```

Use matching test cases and observations for comparisons. Candidate modes require forward validation before they can be interpreted as equivalent physical solutions.

Run code checks with `python -m pytest -q`; see [Project checks](VALIDATION.md). Reduced-size configurations are available in `configs/smoke*.yaml`.
