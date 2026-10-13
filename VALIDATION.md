# Project checks

Run the automated checks from the repository root in the installed environment:

```text
python -m pytest -q
```

The suite covers model inputs, channel subsets, parameter masking, observation generation, checkpoint loading and resume, evaluation metrics, prior and IRF interventions, report generation, and deployment launchers. Small training fixtures use reduced sample counts and model sizes.

Some checks require CUDA, a native operating-system launcher or an optional GMM backend. Pytest lists unavailable checks as skipped. To include the optional GMM checks, set `TDNIRS_TEST_SKLEARN=1` before running the suite.

Check installed dependencies with:

```text
python -m pip check
```

For a launcher check that does not start training, use `train.bat --help` on Windows or `bash train.sh --help` on Linux.

`DATA_SHA256.json` contains checksums for the eight HDF5 source files. The dataset manifest in `artifacts/data/` defines the fixed training, validation and test assignments.

Training and evaluation write their own logs, checkpoints and reports under the configured output directories. Numerical failures are recorded in those outputs.
