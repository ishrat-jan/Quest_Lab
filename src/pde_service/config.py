"""Load configs/*.yaml. Settings live in YAML; code only reads them."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict[str, Any]:
    path = path or os.environ.get("PDE_SERVICE_CONFIG", str(DEFAULT_CONFIG))
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
