from __future__ import annotations

import argparse

from ..baseline_testing import run_baseline_test
from ..config import load_config


def main(expected_architecture: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Test a deterministic TD-NIRS baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--weights", choices=("raw", "ema"), default="ema")
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    architecture = str(config.get("baseline", {}).get("architecture", ""))
    if expected_architecture is not None and architecture != expected_architecture:
        parser.error(f"This entry point requires baseline.architecture={expected_architecture}, received {architecture}")
    report = run_baseline_test(config, args.checkpoint, args.output_dir, weights=args.weights, max_test_samples=args.max_test_samples)
    print(f"report={report}")
