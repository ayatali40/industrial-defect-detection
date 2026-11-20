"""Category inventory: dataset presence, trained artifacts and evaluation status."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .categories import supported_categories
from .config import category_dir, configured_path
from .dataset import DatasetLayoutError, DatasetNotFoundError
from .training import MODEL_NAMES, artifact_path, thresholds_path
from .utils import load_json


def metrics_dir(config: dict[str, Any], category: str) -> Path:
    return configured_path(config, "metrics_dir") / category


def predictions_dir(config: dict[str, Any], category: str) -> Path:
    return configured_path(config, "predictions_dir") / category


def category_metrics(
    config: dict[str, Any], category: str, model_name: str
) -> dict[str, Any] | None:
    path = metrics_dir(config, category) / f"{model_name}_metrics.json"
    return load_json(path) if path.is_file() else None


def category_status(config: dict[str, Any], category: str) -> dict[str, Any]:
    try:
        path = category_dir(config, category)
        data_available = True
        data_error = None
    except (DatasetNotFoundError, DatasetLayoutError, OSError) as exc:
        path = None
        data_available = False
        data_error = str(exc)
    models: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        artifact = artifact_path(config, category, model_name)
        thresholds = thresholds_path(config, category, model_name)
        metrics = category_metrics(config, category, model_name)
        models[model_name] = {
            "artifact": artifact.is_file(),
            "thresholds": thresholds.is_file(),
            "evaluated": metrics is not None,
            "artifact_path": artifact.as_posix(),
            "metrics_path": (
                (metrics_dir(config, category) / f"{model_name}_metrics.json").as_posix()
                if metrics is not None
                else None
            ),
        }
    trained = [
        name
        for name, item in models.items()
        if item["artifact"] and item["thresholds"]
    ]
    return {
        "category": category,
        "data_available": data_available,
        "data_path": path.as_posix() if path is not None else None,
        "data_error": data_error,
        "models": models,
        "trained_models": trained,
        "evaluated_models": [name for name, item in models.items() if item["evaluated"]],
    }


def category_inventory(
    config: dict[str, Any], categories: list[str] | tuple[str, ...] | None = None
) -> dict[str, dict[str, Any]]:
    return {
        category: category_status(config, category)
        for category in (categories or supported_categories(config))
    }


def evaluated_categories(config: dict[str, Any]) -> list[str]:
    return [
        category
        for category in supported_categories(config)
        if category_status(config, category)["evaluated_models"]
    ]


def trained_categories(config: dict[str, Any]) -> list[str]:
    return [
        category
        for category in supported_categories(config)
        if category_status(config, category)["trained_models"]
    ]
