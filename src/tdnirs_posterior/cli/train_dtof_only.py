from .train import main as _train_main


def main() -> None:
    _train_main(expected_variant="dtof_only")


if __name__ == "__main__":
    main()
