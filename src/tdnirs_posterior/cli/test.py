from __future__ import annotations

import argparse

from ..config import load_config
from ..testing import run_test


def main(expected_variant: str | None = "explicit_irf") -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete deterministic PICT Model test-set evaluation"
    )
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output-dir",
        help="Defaults to a test directory next to the checkpoint",
    )
    parser.add_argument("--weights", choices=("raw", "ema"))
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    actual_variant = str(config.get("model", {}).get("variant", "explicit_irf"))
    if expected_variant is not None and actual_variant != expected_variant:
        parser.error(
            f"This entry point requires model.variant={expected_variant}, "
            f"but the configuration selects {actual_variant}"
        )
    report = run_test(
        config,
        args.checkpoint,
        args.output_dir,
        weights=args.weights,
        max_test_samples=args.max_test_samples,
    )
    print(f"report={report}")


if __name__ == "__main__":
    main()
