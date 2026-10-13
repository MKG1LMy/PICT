from __future__ import annotations

import argparse

from ..baseline_comparison import compare_deterministic_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PICT Model with deterministic CNN and ANN baselines")
    parser.add_argument("--explicit-irf-test", required=True)
    parser.add_argument("--cnn-test", required=True)
    parser.add_argument("--ann-test", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = compare_deterministic_baselines(args.explicit_irf_test, args.cnn_test, args.ann_test, args.output_dir)
    print(f"report={report}")


if __name__ == "__main__":
    main()
