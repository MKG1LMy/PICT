# Validation

Date: 2026-10-13

## Automated checks

- 136 tests passed; one CUDA-specific test was skipped with the CPU-only test interpreter.
- Coverage includes small training, checkpoint and report checks for all four models; all 15 channel subsets; training resume; exact-prior and IRF interventions; and native-posterior diagnostics.
- GMM checks covered single, separated double and out-of-support candidate modes, including failure and retry tracking.
- All eight HDF5 files passed SHA-256 verification. The manifest contains 6000 cases, split into 4800 training, 600 validation and 600 test cases. Hashes are recorded in `DATA_SHA256.json`.
- All 52 Python package modules parsed successfully; offline report JavaScript passed syntax checks.

The full suite ran on Windows with Python 3.11, PyTorch 2.11.0+cpu, NumPy 2.4.6, SciPy 1.17.1 and scikit-learn 1.9.1. These checks establish code behavior, not full-training convergence or scientific accuracy.

## Deployment and launchers

Installation completed in a dedicated Conda environment with Python 3.12.15 and PyTorch 2.11.0+cu130. `pip check` found no broken requirements. A small CUDA matrix operation returned finite values, and BF16 support was available.

Windows deployment and training use the environment's Python directly. Both `setup.bat` and `train.bat --help` succeeded when TEMP, TMP and TMPDIR pointed to an invalid location, exercising the project-local temporary-directory fallback. Scoped launcher and deployment regression tests: 18 passed, 3 skipped.

Driver checks recognize both `CUDA Version` and `CUDA UMD Version` in `nvidia-smi` output. Missing or insufficient driver support produces a warning; package installation failures return a nonzero exit status.

Bash syntax and argument/error forwarding were checked with Git Bash. Native Linux deployment and a full CUDA regression suite have not been verified. The checks above did not run formal model training.
