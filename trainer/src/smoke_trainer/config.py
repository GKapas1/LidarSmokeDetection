from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tomllib


def load_config(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        config = tomllib.load(stream)
    for section, key in (("data", "manifest"), ("data", "split"), ("output", "root")):
        value = Path(config[section][key]).expanduser()
        if not value.is_absolute():
            value = source.parent / value
        config[section][key] = str(value.resolve())
    config["config_path"] = str(source)
    return config


def public_config(config: dict) -> dict:
    result = deepcopy(config)
    return result

