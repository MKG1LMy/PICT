from __future__ import annotations

import argparse
import json

from ..config import config_seed, load_config, project_path
from ..data import prepare_dataset_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and index the MCX HDF5 datasets")
    parser.add_argument("--config", default="configs/research.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = project_path(config, config["data"]["root"])
    output = project_path(config, config["data"]["manifest_dir"])
    index = prepare_dataset_index(
        data_root, config["data"], config_seed(config, "data_split")
    )
    manifest, summary = index.write(output)
    print(json.dumps({"manifest": str(manifest), "summary": str(summary)}, indent=2))


if __name__ == "__main__":
    main()
