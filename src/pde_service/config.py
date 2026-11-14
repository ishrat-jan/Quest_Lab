"""Load configs/*.yaml. Settings live in YAML; code only reads them."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | None = None) -> dict[str, Any]:
    """Config from `path`, else $PDE_SERVICE_CONFIG, else configs/default.yaml."""
    return _load(path or os.environ.get("PDE_SERVICE_CONFIG", str(DEFAULT_CONFIG)))


@lru_cache(maxsize=8)
def _load(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
