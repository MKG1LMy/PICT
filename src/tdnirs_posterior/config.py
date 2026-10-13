from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


SEED_DOMAINS = ("data_split", "training", "validation", "evaluation")
FIXED_TIME_BINS = 2500
FIXED_TIME_BIN_PS = 4.0
FIXED_TIME_START_PS = 0.0


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    inherited = config.pop("extends", None)
    if inherited is not None:
        base_path = Path(str(inherited))
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        base = load_config(base_path)
        base.pop("_config_path", None)
        base.pop("_project_root", None)
        config = _deep_merge(base, config)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    seeds = config.get("seeds")
    if not isinstance(seeds, dict):
        raise ValueError("Configuration must define a seeds mapping")
    missing = [name for name in SEED_DOMAINS if name not in seeds]
    if missing:
        raise ValueError(f"Configuration is missing seed domains: {missing}")
    for name in SEED_DOMAINS:
        value = seeds[name]
        if name == "training" and isinstance(value, list):
            if not value or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 2**32 for v in value) or len(set(value)) != len(value):
                raise ValueError("seeds.training must contain unique integers in [0,2**32)")
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"seeds.{name} must be a non-negative integer")
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("Configuration must define a data mapping")
    actual_grid = (
        data.get("time_bins"),
        data.get("time_bin_ps"),
        data.get("time_start_ps"),
    )
    expected_grid = (FIXED_TIME_BINS, FIXED_TIME_BIN_PS, FIXED_TIME_START_PS)
    if actual_grid != expected_grid:
        raise ValueError(
            "This model family requires data.time_bins=2500, "
            "data.time_bin_ps=4.0, and data.time_start_ps=0.0; externally "
            "resample and crop heterogeneous data before use"
        )
    return config


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def config_seed(config: dict[str, Any], domain: str) -> int:
    """Return one explicit random domain without silently coupling experiments."""
    if domain not in SEED_DOMAINS:
        raise ValueError(f"Unknown seed domain: {domain}")
    if isinstance(config["seeds"][domain], list):
        raise ValueError("Training seed list must be expanded by the repeat runner")
    return int(config["seeds"][domain])
