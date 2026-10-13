#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_dir/deploy/find_conda.sh"
pict_find_conda
export CONDA_EXE="$PICT_CONDA"
export PYTHONIOENCODING=utf-8
exec "$PICT_CONDA" run --no-capture-output -n base python "$project_dir/deploy/setup_environment.py" "$@"
