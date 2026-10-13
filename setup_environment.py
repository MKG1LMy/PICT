"""Install an isolated Conda environment; hardware checks are advisory only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def find_conda() -> str:
    """Use an activated Conda installation or an executable on PATH."""
    configured = os.environ.get("CONDA_EXE")
    if configured and Path(configured).is_file():
        return configured
    executable = shutil.which("conda.exe") or shutil.which("conda")
    if executable:
        return executable
    raise RuntimeError("Conda was not found. Open a Conda/Miniconda prompt or add Conda to PATH.")


def driver_warnings(cuda: str, runner=subprocess.run) -> list[str]:
    """Compare requested CUDA with the driver's advertised maximum version.

    This is advisory: it does not prove wheel/GPU architecture compatibility.
    Neither missing nvidia-smi nor an old driver prevents package installation.
    """
    requested = (int(cuda[:-1]), int(cuda[-1]))
    try:
        result = runner(["nvidia-smi"], capture_output=True, text=True, check=False, timeout=20)
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
        if result.returncode or not match:
            return ["Could not determine CUDA driver support. Check nvidia-smi before GPU training."]
        supported = tuple(map(int, match.groups()))
        if supported < requested:
            return [f"Driver advertises CUDA {supported[0]}.{supported[1]}, below requested "
                    f"{requested[0]}.{requested[1]}. Update the driver before GPU training."]
        return []
    except (OSError, subprocess.TimeoutExpired) as error:
        return [f"CUDA driver check unavailable: {error}"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="pict_model", help="Dedicated Conda environment name")
    parser.add_argument("--python-version", default="3.12", choices=["3.12", "3.13"])
    parser.add_argument("--torch-version", default="2.11.0")
    parser.add_argument("--cuda", default="130", help="Official PyTorch CUDA wheel suffix, e.g. 130")
    parser.add_argument("--dry-run", action="store_true", help="Print installation commands only")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.env) or args.env == "base":
        parser.error("Choose a dedicated environment name (not base)")
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.torch_version) or not re.fullmatch(r"\d{3}", args.cuda):
        parser.error("Use a numeric torch version (e.g. 2.11.0) and CUDA suffix (e.g. 130)")
    warnings = []
    completed = False
    try:
        conda = find_conda()
        if args.dry_run:
            exists = False
        else:
            result = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True, check=True)
            exists = any(Path(p).name == args.env for p in json.loads(result.stdout)["envs"])
        command = [conda, "install" if exists else "create", "-n", args.env, "--yes",
                   "--override-channels", "-c", "conda-forge", f"python={args.python_version}", "pip"]
        prefix = [conda, "run", "--no-capture-output", "-n", args.env, "python", "-m", "pip"]
        commands = [command,
                    [*prefix, "install", f"torch=={args.torch_version}", "--index-url",
                     f"https://download.pytorch.org/whl/cu{args.cuda}"],
                    [*prefix, "install", "--editable", f"{ROOT}[dev]"]]
        for command in commands:
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True, cwd=ROOT)
        completed = True
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Installation did not complete: {error}", file=sys.stderr)
    finally:
        if not args.dry_run:
            warnings.extend(driver_warnings(args.cuda))
        warnings.append("Driver support alone does not verify wheel/GPU architecture support, BF16 or available memory. "
                        "If BF16 is unavailable, set training.amp and evaluation.amp to false before training.")
        print("\nDeployment warnings / limitations:")
        for message in warnings:
            print(f"- {message}")
        if completed:
            print("\nDry run completed." if args.dry_run else f"Installation completed. Environment: {args.env}")
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
