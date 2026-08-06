from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    probabilities = config["dataset"]["phase_count_probability"]
    if len(probabilities) != len(config["phases"]):
        raise ValueError("phase_count_probability must cover 1..N phases")
    if abs(sum(probabilities) - 1.0) > 1e-9:
        raise ValueError("phase_count_probability must sum to one")
    return config

