"""Independent model stages with common data and fixed evaluation seeds."""
from __future__ import annotations

import json
from pathlib import Path

from .config import load_config
from .repeats import run_repetitions

MODULES = {"main": "train", "dtof_only": "train_dtof_only", "cnn": "train_cnn", "ann": "train_ann"}


def run_pipeline(path: Path, *, models=None, run_epochs=None, no_test=False):
    specification = json.loads(path.read_text(encoding="utf-8"))
    selected = specification["models"] if models is None else models
    if not selected or len(set(selected)) != len(selected) or any(m not in MODULES for m in selected):
        raise ValueError("Select unique models from main, dtof_only, cnn, ann")
    if run_epochs is not None and run_epochs <= 0:
        raise ValueError("run_epochs must be positive")
    paths = {name: (path.parent / specification["configs"][name]).resolve() for name in selected}
    configs = {name: load_config(p) for name, p in paths.items()}
    reference = configs[selected[0]]
    for name, config in configs.items():
        for key in ("data", "simulation", "seeds"):
            if config[key] != reference[key]:
                raise ValueError(f"Pipeline models must share {key}: {name}")
    outputs = [(Path(c["_project_root"]) / c["training"]["output_dir"]).resolve() for c in configs.values()]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Pipeline models require distinct training output directories")
    seeds = reference["seeds"]["training"]
    seeds = seeds if isinstance(seeds, list) else [seeds]
    for seed in seeds:
        for name in selected:
            config = load_config(paths[name])
            config["seeds"]["training"] = [seed]
            arguments = ["--config", str(paths[name])]
            if run_epochs is not None:
                config["training"]["run_epochs"] = run_epochs
                arguments += ["--run-epochs", str(run_epochs)]
            if no_test:
                arguments.append("--no-test")
            run_repetitions(config, "tdnirs_posterior.cli." + MODULES[name], arguments)
