"""Read local YAML settings with environment fallback for legacy setups."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


CONFIG_PATH = Path("config.yaml")
TEST_CONFIG_PATH = Path("config.test.yaml")


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    source = Path(path)
    if not source.is_file():
        return {}
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {source}")
    if any(not isinstance(key, str) for key in data):
        raise ValueError(f"Configuration keys must be strings: {source}")
    if "CHANGE_PASSWORD" in str(data.get("mysql_url", "")):
        raise ValueError("Replace the example mysql_url password in config.yaml")
    if data.get("rag_internal_token") == "REPLACE_WITH_SHARED_TOKEN":
        raise ValueError("Replace the example rag_internal_token in config.yaml")
    return data


def setting(name: str, default: str | None = None, *, test: bool = False) -> str | None:
    """Use the lowercase YAML key, then a legacy environment fallback."""
    config = load_config(TEST_CONFIG_PATH if test else CONFIG_PATH)
    if name.lower() not in config:
        return os.getenv(name, default)
    value = config[name.lower()]
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
