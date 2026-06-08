from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    selected_path = Path(config_path)
    if not selected_path.exists():
        raise FileNotFoundError(f"Config file not found: {selected_path}")

    with selected_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must load to a dictionary: {selected_path}")

    return config
