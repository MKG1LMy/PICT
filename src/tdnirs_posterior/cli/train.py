from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config, project_path
from ..testing import run_test
from ..training import train


def main(expected_variant: str | None = "explicit_irf") -> None:
    parser = argparse.ArgumentParser(description="Train PICT Model")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument(
        "--run-epochs",
        type=int,
        help="Run at most N additional epochs without changing the scheduler horizon",
    )
    parser.add_argument(
        "--init-checkpoint",
        help=(
            "Start a new experiment from model weights in an existing checkpoint; "
            "optimizer, scheduler and early-stopping state are reset"
        ),
    )
    parser.add_argument(
        "--init-weights",
        choices=("raw", "ema"),
        default="raw",
        help="Which weights to copy when using --init-checkpoint",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override evaluation.auto_run after a terminal training run",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume complete training state; omit PATH to use output_dir/last.pt",
    )
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, help="Internal single-run training seed override")
    args = parser.parse_args()
    config = load_config(args.config)
    actual_variant = str(config.get("model", {}).get("variant", "explicit_irf"))
    if expected_variant is not None and actual_variant != expected_variant:
        parser.error(
            f"This entry point requires model.variant={expected_variant}, "
            f"but the configuration selects {actual_variant}"
        )
    overrides = {
        "run_epochs": args.run_epochs,
        "micro_batch_size": args.micro_batch_size,
        "effective_batch_size": args.effective_batch_size,
        "num_workers": args.num_workers,
        "output_dir": args.output_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            config["training"][key] = value
    if args.seed is not None:
        config["seeds"]["training"] = args.seed
    else:
        import sys
        from ..repeats import run_repetitions
        module = "tdnirs_posterior.cli.train_dtof_only" if actual_variant == "dtof_only" else "tdnirs_posterior.cli.train"
        run_repetitions(config, module, sys.argv[1:])
        return
    if args.resume is not None and args.init_checkpoint is not None:
        parser.error("--resume and --init-checkpoint cannot be used together")
    checkpoint = train(
        config,
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
        init_weights=args.init_weights,
    )
    print(f"checkpoint={checkpoint}")
    summary_path = project_path(config, config["training"]["output_dir"]) / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    terminal = summary.get("status") == "completed"
    configured_auto_test = bool(config.get("evaluation", {}).get("auto_run", False))
    should_test = configured_auto_test if args.test is None else bool(args.test)
    if terminal and should_test:
        print(
            json.dumps(
                {"event": "automatic_test_start", "checkpoint": str(checkpoint)}
            ),
            flush=True,
        )
        report = run_test(config, Path(checkpoint))
        print(f"test_report={report}")
    elif should_test and not terminal:
        print(
            json.dumps(
                {
                    "event": "automatic_test_skipped",
                    "reason": "training_paused_by_run_epochs",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
