# Public release verification

Date: 2026-10-13

## Verified

- 136 tests passed; one CUDA-specific test skipped because the test interpreter had a CPU-only PyTorch build.
- The suite includes tiny real training, checkpoint and report checks for all four models; all 15 channel subsets; training resume; exact-prior and IRF interventions; native-posterior diagnostics; and the public launcher/deployment contracts.
- Native GMM tests passed for single, separated double and out-of-support candidate modes. Failure/retry tracking is also tested.
- All eight original HDF5 files match their source SHA-256 hashes. The included manifest contains 6000 cases, split 4800/600/600. See `DATA_SHA256.json`.
- Parsed all 52 Python package modules and checked the offline report JavaScript syntax. Source and configuration text contains no Chinese or machine-specific Windows paths.
- Scientific configuration values match the source project apart from the requested training seeds: 20736, 78631 and 26605. Split, validation and evaluation seeds remain 2026.
- The installer dry-run and mocked installation tests verify that missing/old drivers produce advisory messages without causing installation failure. Actual package installation failures remain failures.

## Confirmed fixes and public-copy changes

- GMM soft random-responsibility initialization missed a clearly separated synthetic double mode. Initialization now selects observed draws (`random_from_data`), retaining deterministic multiple starts and retry diagnostics without KMeans initialization. The evaluation signature records this choice, preventing mixed-protocol resume. This changes mode diagnostics, not training or posterior sampling.
- Report conclusions are data-driven; fixed claims of high N=1 accuracy or necessarily declining multilayer performance were removed.
- English offline reports retain accuracy, calibration, priors, IRF interventions, thickness, channel-subset and candidate-mode tables. Known parameters remain excluded from accuracy.
- One public training launcher checks common data/acquisition/seeds, isolates model output directories and runs models/seeds serially. Internal CLI modules remain implementation details.
- No changes were made to the source project's files or formal runs. No formal training was started for this release.

## Validation boundaries

Tests ran on Windows using Python 3.11, PyTorch 2.11.0+cpu, NumPy 2.4.6, SciPy 1.17.1 and scikit-learn 1.9.1 with isolated temporary test dependencies. The release environment targets Python 3.12/3.13; a fresh installation of that target, CUDA execution and native Linux execution were not performed here. Cross-platform deployment uses standard Python/Conda commands rather than PowerShell or batch-only logic, but this is not a claim of native Linux validation.

An initial optional GMM probe in the pre-existing local scientific environment encountered a SciPy DLL failure. Repeating with isolated NumPy/SciPy wheels removed that environment issue; no unsafe duplicate-OpenMP override was used and no existing Conda environment was modified.

## Deployment and launcher follow-up (2026-10-13)

An authorized real installation in the dedicated `pict_model` environment completed successfully with Python 3.12.15 and PyTorch 2.11.0+cu130. `pip check` reported no broken requirements. A small CUDA matrix operation returned finite values and BF16 support was available. This is an installation check, not formal model training or a full CUDA regression suite; the earlier full-suite record above remains unchanged.

The previous unlogged deployment failure could not be reproduced. One confirmed advisory-check defect was repaired: newer `nvidia-smi` output uses `CUDA UMD Version`, whereas the old parser recognized only `CUDA Version`. Both forms are now tested. Driver failures remain advisory. New deployments save console and subprocess output under `deployment-logs/`.

Windows `train.bat` and Linux `train.sh` share the internal Python controller; `setup.sh` is also provided. Windows argument forwarding, error propagation and deployment logging passed automated checks. Bash syntax and argument/error forwarding were checked using Git Bash and a fake Conda command, including a path containing spaces. Native Linux deployment remains unverified. No model training was started during this follow-up.

Passing these tests does not establish long-run numerical stability, convergence or scientific validity. Choose the software and dataset licenses before publication. No remote repository was created or pushed.
