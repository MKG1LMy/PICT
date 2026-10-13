"""Install an isolated Conda environment; hardware checks are advisory only."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def run_command(command):
    """Stream both subprocess outputs to the console and the deployment log."""
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", env=environment) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def logged_main(argv=None):
    directory = ROOT / "deployment-logs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"setup-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}.log"
    with path.open("w", encoding="utf-8") as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            print(f"Deployment log: {path}")
            return main(argv)


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
        match = re.search(r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)", result.stdout)
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
                run_command(command)
        completed = True
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            if error.stdout:
                print(error.stdout)
            if error.stderr:
                print(error.stderr, file=sys.stderr)
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
    raise SystemExit(logged_main())
