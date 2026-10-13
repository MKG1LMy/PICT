"""Windows Conda-environment bootstrap without temporary activation scripts."""
from pathlib import Path
import subprocess
import sys
import os

from setup_environment import find_conda, prepare_temp, environment_python, environment_variables


def main():
    try:
        prepare_temp()
        python = environment_python(find_conda(), os.environ.get("PICT_ENV", "pict_model"))
        return subprocess.call([str(python), str(Path(__file__).with_name("run_training.py")),
                                *sys.argv[1:]], env=environment_variables(python))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Could not start training: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
