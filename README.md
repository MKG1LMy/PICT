# PICT Model

Research code for layered time-domain near-infrared spectroscopy (TD-NIRS). The project contains an explicit-IRF conditional posterior model, a DTOF-only posterior ablation, and deterministic CNN and ANN baselines.

## Scope

Inputs use the fixed SDS order **10, 20, 30, 40 mm**, **2500 time bins**, and **4 ps/bin**. Any nonempty channel subset is supported. The full model receives a measured IRF for each available DTOF channel. DTOF-only, CNN and ANN never receive IRF tensors, but use the same simulated DTOF protocol. Algorithmic time alignment is disabled by default.

The posterior models accept the layer count (N=1, 2 or 3) and optional exact parameter priors. They return joint posterior draws, not a list of discrete solutions with known probabilities. Baselines return point estimates and do not accept parameter priors. Finite out-of-domain draws are retained and flagged rather than rejected. This is research software, not a clinical device.

## Requirements

- Windows or Linux, with Miniconda or Anaconda already installed.
- Python 3.12 or 3.13 in a dedicated environment.
- The default GPU build is PyTorch 2.11.0 with CUDA 13.0 (`cu130`). No GPU product name is required.

The CUDA runtime comes with the PyTorch wheel; a matching system CUDA Toolkit is not required. Driver compatibility is checked at the end of installation and is **advisory only**. Missing NVIDIA hardware, an unavailable driver query or an older driver does not abort installation. Driver compatibility alone does not establish that a wheel supports a GPU architecture or that enough memory is available.

The default training configuration uses BF16. On a GPU without BF16 support, set both `training.amp` and `evaluation.amp` to `false` before training to use FP32. Do not substitute FP16: this model uses a stability protocol designed around BF16/FP32. The training code can also execute on CPU, but formal training is expensive.

## Installation

On **Windows**, double-click `setup.bat`. It locates Conda/Miniconda and runs the internal installer using Conda's base Python; a separate system Python installation is not needed. The window stays open at the end so you can read warnings or failures. For a custom Conda location, launch it from a Conda Prompt.

Optional Windows command-line invocation:

```text
setup.bat --env pict_model --torch-version 2.11.0 --cuda 130
```

On **Linux**, run the internal installer from a Conda-enabled terminal:

```text
bash setup.sh
conda activate pict_model
```

To select another officially published PyTorch/CUDA wheel combination or environment name on Linux:

```text
bash setup.sh --env pict_model --torch-version 2.11.0 --cuda 130
```

`--dry-run` prints commands without creating an environment or installing packages (Windows: `setup.bat --dry-run`). For unattended Windows use, set `PICT_SETUP_NO_PAUSE=1`. The installer uses conda-forge for Python/pip and official PyTorch wheels for torch. Other Python dependencies are installed with pip. It never deletes an existing environment. Use a fresh dedicated environment to avoid OpenMP library conflicts inherited from unrelated packages. Alternative PyTorch builds are configurable but are not automatically validated research equivalents.

Hardware diagnostics are warnings. **Download errors, unavailable package versions, unresolved dependencies and other actual installation failures still produce a nonzero exit status**; they are not reported as successful deployment. Conda channel terms or organization policies must be handled by the user; this script does not accept legal terms on the user's behalf.

The installer saves console output and subprocess errors to `deployment-logs/setup-<timestamp>-<pid>.log`. If deployment fails, retain that log. The CUDA PyTorch download is large and may take several minutes without a progress bar in redirected output.

The installer invokes the target environment's Python directly for pip installation. The Windows training launcher uses the same approach, avoiding temporary `conda activate` batch scripts. These scripts check temporary storage and fall back to `.runtime/tmp` inside this project if the configured locations are unusable. TEMP/TMP changes apply only to the launched process and its children; Windows user and system settings are not changed. This bootstrap supplies the project environment's executable and DLL paths; it does not execute arbitrary third-party Conda activation hooks.

## One training entry point

On **Windows**, double-click `train.bat`, or run it from a terminal:

```text
train.bat
```

On **Linux**, run `bash train.sh`. These platform launchers use Conda's `pict_model` environment without requiring manual activation and call the same internal Python training controller. `configs/pipeline.json` selects models and configuration files. By default, it runs **main → DTOF-only → CNN → ANN for each training seed**, including the configured final tests, then moves to the next seed. Each model is a separate process. There is no concurrent multi-model training.

If deployment used a custom environment, set `PICT_ENV` to that name before launching (PowerShell: `$env:PICT_ENV="my_env"`; Linux: `PICT_ENV=my_env bash train.sh`). Set `PICT_TRAIN_NO_PAUSE=1` for unattended Windows use; otherwise the BAT pauses at exit so errors remain visible.

The three default training seeds are **20736, 78631, 26605**, matching the archived 261008 run configurations. Split, validation and evaluation seeds remain **2026**. Results are written to each model's `training.output_dir/seed_<seed>/`; different seeds and models do not overwrite one another.

Optional invocations use the same entry point:

```text
train.bat --models main
train.bat --models cnn ann
train.bat --models main --run-epochs 2 --no-test
```

The last command pauses after at most two additional epochs of each selected run; it is not an accuracy experiment. The normal launcher resumes a run from its last complete epoch and skips completed training stages. A stage failure stops the queue; other models are not silently marked successful. Keep settings fixed when resuming. Use a new output directory when changing the scientific configuration.

### Configuration

`configs/research.yaml` defines the common data, observation, training and evaluation protocol. The other three research files inherit it and override only model-specific fields. English comments explain the main parameters.

| Setting | Default / meaning |
|---|---|
| `seeds.training` | Three independent training repetitions |
| `training.epochs` | 10000 maximum epochs, subject to early stopping |
| `training.warmup_epochs` | 100 fixed warmup epochs |
| `training.micro_batch_size` | 64 cases per forward/backward pass |
| `training.effective_batch_size` | 128 cases per optimizer update |
| `training.learning_rate` | Main/DTOF-only: 2e-4; CNN: 1e-4; ANN: 3e-4 |
| `training.early_stopping.patience_epochs` | 50 EMA-monitor epochs |
| `training.posterior_validation.patience_evaluations` | 7 posterior evaluations |
| `simulation.preprocessing.time_alignment_enabled` | false |
| `evaluation.auto_run` | true after terminal training, not an intermediate pause |

For limited VRAM, lower the micro-batch size before starting a run and keep the effective batch size an integer multiple. This may affect throughput and floating-point trajectories. Choose DataLoader workers for the target CPU/storage rather than assuming more is better.

Available-channel counts 4/3/2/1 have training probabilities 40/30/15/15%. Layer counts are balanced in the objective. Within N=2, d1 <=15 mm is the core thickness range; within N=3, d1+d2 in [10,20] mm is the core range. Core/other thicknesses have 70/30 weight. Full tests still retain all thickness ranges. These are research choices, not universal physical limits.

## Included data

`Data/` contains eight original, unmodified `mixed-dataset.h5` files: **6000 phantoms**, with layer counts **2002/2000/1998**. The frozen split in `artifacts/data/manifest.jsonl` contains **4800/600/600** training/validation/test cases. Paths are relative; the project does not import code or data from another checkout.

Each HDF5 file stores TPSFs at 16 SDS values (10–40 mm at 2-mm spacing), time edges, layer thicknesses, absorption and reduced scattering coefficients. The loader selects the four SDS values above. TPSFs are convolved with noise-free true IRFs before acquisition noise is applied; separate noisy measured IRFs are model inputs. Both measurement views share phantom parameters, channel availability and exact-prior mask. A/C share one measurement and B/D another; A/B have no priors, C/D share a nonempty prior mask.

Simulation job logs and private host paths are not required to train and are omitted. Raw HDF5 content is preserved byte-for-byte. `DATA_SHA256.json` lists hashes for auditing. The working directory contains the actual data; Git tracks HDF5 with Git LFS. When cloning a published repository, install Git LFS and run `git lfs pull` to retrieve the data (subject to the host's storage quota). An archive may instead ship the actual HDF5 files without requiring Git LFS. Do not split phantoms anew merely to upload the repository.

## Evaluation and reports

Final evaluation includes layer/thickness strata, all legal exact-prior masks, named prior-value interventions, IRF interventions, channel subsets, marginal/joint calibration, native-sample diagnostics and candidate GMM modes. Reports are offline English HTML with embedded tables, accompanied by JSON, CSV and HDF5. Numerical failures remain visible. A candidate second mode is not proof of an equivalent physical solution without forward validation.

The detailed evaluation modules remain available for manual use:

```text
python -m tdnirs_posterior.cli.test --help
python -m tdnirs_posterior.cli.compare_irf_ablation --help
python -m tdnirs_posterior.cli.compare_baselines --help
```

Use strict paired comparisons: matching case IDs, observations, channel protocol and fixed evaluation seeds. The launcher executes each model's final test; cross-model comparison reports remain explicit analysis commands. Model selection uses validation only, never formal test outcomes.

## Tests and numerical stability

```text
python -m pytest -q
```

Tests include small real forward/backward/checkpoint/report checks. They do not establish full-run convergence or scientific accuracy. Some GPU and optional platform GMM tests can be skipped if their backend is unavailable. `configs/smoke*.yaml` are small code-validation configurations, not publishable experiment results.

The posterior encoders normalize waveform latents and use bounded Q/K cross-attention with explicit FP32 arithmetic in that module. Gradient norms use FP64 reductions; nonfinite or explosive gradients stop training before an optimizer update. Logs contain per-epoch numerical summaries, and failures write `training_failure.json`. Checkpoints from before this normalization protocol are not interchangeable with the current model.

## Repository boundaries

The public copy includes source, tests, research/smoke configurations, fixed data manifests, original HDF5 data and deployment support. It excludes private experiment outputs, local environments, caches, historical reports and old deployment archives. The Python import namespace remains `tdnirs_posterior` to avoid an unnecessary checkpoint/module migration; the distribution and repository name is PICT Model.

No software or dataset redistribution license is inferred by this preparation. The rights holder must add the intended license and attribution before public release.
