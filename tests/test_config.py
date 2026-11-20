"""Configuration and path resolution tests."""

from __future__ import annotations

import pytest
import torch

from industrial_defect.config import (
    ConfigError,
    configured_path,
    get_device,
    load_config,
    project_root,
    resolve_path,
)


def test_load_config_has_required_sections(config):
    for section in ("project", "paths", "data", "models", "thresholds", "evaluation", "inference"):
        assert section in config


def test_missing_config_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_incomplete_config_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("project:\n  seed: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Missing required configuration section"):
        load_config(path)


def test_non_mapping_config_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(path)


def test_relative_paths_resolve_against_project_root(config):
    resolved = configured_path(config, "metrics_dir")
    assert resolved.is_absolute()
    assert resolved == project_root(config) / config["paths"]["metrics_dir"]
    assert resolve_path(config, resolved) == resolved


def test_get_device_cpu(config):
    updated = {**config, "project": {**config["project"], "device": "cpu"}}
    assert get_device(updated) == torch.device("cpu")


def test_get_device_rejects_unknown(config):
    updated = {**config, "project": {**config["project"], "device": "tpu"}}
    with pytest.raises(ConfigError, match="Unsupported device"):
        get_device(updated)


def test_cuda_requested_when_unavailable(config):
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    updated = {**config, "project": {**config["project"], "device": "cuda"}}
    with pytest.raises(ConfigError, match="CUDA was requested"):
        get_device(updated)
