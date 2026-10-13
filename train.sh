#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_dir/deploy/find_conda.sh"
pict_find_conda
export PYTHONIOENCODING=utf-8
exec "$PICT_CONDA" run --no-capture-output -n "${PICT_ENV:-pict_model}" python "$project_dir/deploy/run_training.py" "$@"
