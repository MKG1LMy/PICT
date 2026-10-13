from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..evaluation import evaluate_posterior


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved posterior samples")
    parser.add_argument(
        "--input",
        required=True,
        help="NPZ with posterior_samples, truth, active_mask",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = np.load(args.input)
    report = evaluate_posterior(
        payload["posterior_samples"], payload["truth"], payload["active_mask"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
