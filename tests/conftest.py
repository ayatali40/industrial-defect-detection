"""Shared pytest fixtures."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from industrial_defect.config import category_dir, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config(PROJECT_ROOT / "configs" / "config.yaml")


@pytest.fixture(scope="session")
def real_category_dir(config: dict) -> Path:
    path = category_dir(config)
    if not path.is_dir():
        pytest.skip("MVTec dataset is not available in this checkout")
    return path


@pytest.fixture()
def tmp_config(config, tmp_path):
    """A configuration whose inputs and outputs live under ``tmp_path``."""
    cfg = deepcopy(config)
    cfg["_project_root"] = str(tmp_path)
    cfg["paths"] = {
        "data_root": "data",
        "models_dir": "models",
        "reports_dir": "reports",
        "figures_dir": "reports/figures",
        "metrics_dir": "reports/metrics",
        "predictions_dir": "reports/predictions",
        "processed_dir": "data/processed",
    }
    cfg["data"]["category"] = "transistor"
    cfg["data"]["category_roots"] = {}
    return cfg
