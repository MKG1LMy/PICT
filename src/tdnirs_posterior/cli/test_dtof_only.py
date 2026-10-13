from .test import main as _test_main


def main() -> None:
    _test_main(expected_variant="dtof_only")


if __name__ == "__main__":
    main()
