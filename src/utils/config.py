"""Central configuration loader — single source of truth for all pipeline settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT_DIR / "configs"


def _resolve_env_vars(obj):
    """Resolve ${VAR:-default} patterns in YAML values."""
    if isinstance(obj, str) and obj.startswith("${"):
        # Extract VAR and default from ${VAR:-default}
        inner = obj[2:-1]
        if ":-" in inner:
            var, default = inner.split(":-", 1)
        else:
            var, default = inner, ""
        return os.environ.get(var, default)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def _load_yaml(filename: str) -> dict:
    path = CONFIGS_DIR / filename
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _resolve_env_vars(raw)


@lru_cache
def get_tickers_config() -> dict:
    return _load_yaml("tickers.yaml")


@lru_cache
def get_data_config() -> dict:
    return _load_yaml("data_config.yaml")


@lru_cache
def get_model_config() -> dict:
    return _load_yaml("model_config.yaml")


# Convenience accessors
def get_ticker_list() -> list[str]:
    return get_tickers_config()["tickers"]


def get_macro_indicators() -> dict[str, str]:
    return get_tickers_config()["macro_indicators"]


def get_db_url() -> str:
    cfg = get_data_config()["database"]
    return (
        f"postgresql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['name']}"
    )
