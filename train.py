"""Single training launcher for Windows and Linux; run inside the Conda environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
# Allow source checkouts to run without relying on another editable installation.
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, default=ROOT / "configs/pipeline.json")
    parser.add_argument("--models", nargs="+", choices=["main", "dtof_only", "cnn", "ann"],
                        help="Override the models selected in pipeline.json")
    parser.add_argument("--run-epochs", type=int, help="Pause after N additional epochs; no automatic final test")
    parser.add_argument("--no-test", action="store_true", help="Disable automatic tests for this invocation")
    args = parser.parse_args(argv)
    # Existing tests and relative auxiliary paths assume the project root.
    pipeline = args.pipeline.resolve()
    os.chdir(ROOT)
    from tdnirs_posterior.pipeline import run_pipeline
    run_pipeline(pipeline, models=args.models, run_epochs=args.run_epochs, no_test=args.no_test)


if __name__ == "__main__":
    main()
