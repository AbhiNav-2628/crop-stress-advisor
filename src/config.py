"""Loading of YAML configuration files and project paths."""

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEMO_DATA_DIR = PROJECT_ROOT / "data" / "demo"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def load_settings() -> dict:
    """Return the main settings dictionary (cached after the first call)."""
    return load_yaml(CONFIG_DIR / "settings.yaml")


def load_advisory_rules(path: Path | None = None) -> list[dict]:
    data = load_yaml(path or CONFIG_DIR / "advisory_rules.yaml")
    return data.get("rules", [])


def resolve_path(relative: str) -> Path:
    """Turn a path from settings.yaml (relative to the project root) into an absolute one."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path
