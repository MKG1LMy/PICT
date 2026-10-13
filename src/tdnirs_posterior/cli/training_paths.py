from __future__ import annotations

import argparse
import json

from ..config import load_config, project_path


def resolve_training_paths(config_path: str) -> dict[str, str]:
    config = load_config(config_path)
    run_directory = project_path(config, config["training"]["output_dir"]).resolve()
    return {"run_directory": str(run_directory)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve the configured training output path"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    payload = resolve_training_paths(args.config)
    print(f"training_paths={json.dumps(payload, ensure_ascii=True)}")


if __name__ == "__main__":
    main()
