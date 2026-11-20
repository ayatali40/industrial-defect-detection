"""Configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file and attach project-root metadata."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"Configuration must be a mapping, got {type(config).__name__}")
    for key in ("project", "paths", "data", "models", "thresholds", "evaluation", "inference"):
        if key not in config:
            raise ConfigError(f"Missing required configuration section: '{key}'")
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def project_root(config: dict[str, Any]) -> Path:
    return Path(config.get("_project_root", PROJECT_ROOT))


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a possibly relative path against the project root."""
    path = Path(value)
    return path if path.is_absolute() else project_root(config) / path


def configured_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path stored under ``paths.<key>``."""
    try:
        value = config["paths"][key]
    except KeyError as exc:
        raise ConfigError(f"Missing paths.{key} in configuration") from exc
    return resolve_path(config, value)


def category_dir(config: dict[str, Any], category: str | None = None) -> Path:
    """Return the folder for ``category`` (defaults to ``data.category``).

    ``data.category_roots`` may map a category to an explicit folder. Otherwise
    ``paths.data_root`` is treated as either an MVTec AD root or a single
    category folder, preserving the original single-category behaviour.
    """
    from industrial_defect.dataset import DatasetLayoutError, resolve_category_dir

    name = str(category or config["data"]["category"])
    overrides = config["data"].get("category_roots") or {}
    if name in overrides:
        path = resolve_path(config, overrides[name])
        if not (path / "train").is_dir() or not (path / "test").is_dir():
            raise DatasetLayoutError(
                f"data.category_roots['{name}'] points to {path}, which is not an MVTec "
                "category folder (train/ and test/ are required)"
            )
        return path
    root = configured_path(config, "data_root")
    return resolve_category_dir(root, name)


def get_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("project", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigError("CUDA was requested but is not available")
    if requested not in {"cpu", "cuda"}:
        raise ConfigError(f"Unsupported device: {requested}")
    return torch.device(requested)


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for key in ("models_dir", "figures_dir", "metrics_dir", "predictions_dir", "processed_dir"):
        configured_path(config, key).mkdir(parents=True, exist_ok=True)
    (configured_path(config, "processed_dir") / "splits").mkdir(parents=True, exist_ok=True)


def config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Serialisable copy of the user configuration (without private runtime keys)."""
    return {key: value for key, value in config.items() if not key.startswith("_")}
