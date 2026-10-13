import argparse
from pathlib import Path

from ..ablation_comparison import compare_irf_ablation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired explicit-IRF and DTOF-only test runs"
    )
    parser.add_argument("--explicit-irf-test", required=True, type=Path)
    parser.add_argument("--dtof-only-test", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(
        compare_irf_ablation(
            args.explicit_irf_test, args.dtof_only_test, args.output_dir
        )
    )


if __name__ == "__main__":
    main()
