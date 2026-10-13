from __future__ import annotations

import argparse

from ..baseline_training import train_baseline
from ..config import load_config


def main(expected_architecture: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a deterministic TD-NIRS baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--init-weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--run-epochs", type=int)
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    architecture = str(config.get("baseline", {}).get("architecture", ""))
    if expected_architecture is not None and architecture != expected_architecture:
        parser.error(f"This entry point requires baseline.architecture={expected_architecture}, received {architecture}")
    if args.output_dir is not None:
        config["training"]["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seeds"]["training"] = args.seed
    else:
        import sys
        from ..repeats import run_repetitions
        run_repetitions(config, "tdnirs_posterior.cli.train_" + architecture, sys.argv[1:])
        return
    if expected_architecture is not None and architecture != expected_architecture:
        parser.error(f"This entry point requires baseline.architecture={expected_architecture}, received {architecture}")
    checkpoint = train_baseline(
        config,
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
        init_weights=args.init_weights,
        run_epochs=args.run_epochs,
        auto_test=not args.no_test,
    )
    print(f"checkpoint={checkpoint}")
