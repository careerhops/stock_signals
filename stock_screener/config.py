from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(path: str | Path = "config/settings.yaml") -> dict[str, Any]:
    load_dotenv()
    config_path = Path(path)
    if not config_path.exists():
      raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    return config


def get_data_root(config: dict[str, Any]) -> Path:
    data_cfg = config.get("data", {})
    env_name = data_cfg.get("data_root_env", "DATA_ROOT")
    root = os.getenv(env_name, "data")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

