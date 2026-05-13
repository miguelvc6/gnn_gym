from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]


def load_yaml(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: Config, override: Config) -> Config:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def set_dotted(config: Config, dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override must be key=value, got: {raw}")
    key, value = raw.split("=", 1)
    return key, yaml.safe_load(value)


def load_run_config(
    model: str,
    dataset: str,
    overrides: list[str] | None = None,
    config_dir: str | Path = "configs",
) -> Config:
    root = Path(config_dir)
    config = load_yaml(root / "default.yaml")
    config = deep_merge(config, load_yaml(root / "models" / f"{model}.yaml"))
    dataset_file = dataset.replace("-", "_")
    config = deep_merge(config, load_yaml(root / "datasets" / f"{dataset_file}.yaml"))
    if overrides:
        for raw in overrides:
            key, value = parse_override(raw)
            set_dotted(config, key, value)
    return config
