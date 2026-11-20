"""Model construction, fitting and artifact persistence."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .categories import supported_categories
from .config import config_snapshot, configured_path, project_root
from .models import FeatureExtractor, PaDiMModel, PatchCoreModel
from .preprocessing import IMAGENET_MEAN, IMAGENET_STD
from .utils import environment_info, load_json, save_json

MODEL_NAMES = ("patchcore", "padim")
ARTIFACT_FORMAT_VERSION = 1


def build_extractor(config: dict[str, Any]) -> FeatureExtractor:
    backbone = config["models"]["backbone"]
    return FeatureExtractor(
        backbone=str(backbone["name"]),
        pretrained=bool(backbone["pretrained"]),
        layers=tuple(backbone["layers"]),
    )


def build_model(
    config: dict[str, Any], model_name: str, extractor: FeatureExtractor
) -> PatchCoreModel | PaDiMModel:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'; choose from {MODEL_NAMES}")
    settings = config["models"][model_name]
    image_size = int(config["data"]["image_size"])
    seed = int(config["project"]["seed"])
    if model_name == "patchcore":
        return PatchCoreModel(
            extractor,
            image_size=image_size,
            projection_dim=int(settings["projection_dim"]),
            smooth_sigma=float(settings["smooth_sigma"]),
            seed=seed,
        )
    return PaDiMModel(
        extractor,
        image_size=image_size,
        n_features=int(settings["n_features"]),
        covariance_epsilon=float(settings["covariance_epsilon"]),
        smooth_sigma=float(settings["smooth_sigma"]),
        seed=seed,
    )


def fit_model(
    model: PatchCoreModel | PaDiMModel, loader: DataLoader, config: dict[str, Any]
) -> dict[str, Any]:
    if isinstance(model, PatchCoreModel):
        settings = config["models"]["patchcore"]
        return model.fit(
            loader,
            max_candidates=int(settings["max_candidate_patches"]),
            coreset_ratio=float(settings["coreset_ratio"]),
        )
    return model.fit(loader)


def model_dir(
    config: dict[str, Any], category: str, model_name: str, tag: str | None = None
) -> Path:
    """Category- and model-specific directory, e.g. ``models/capsule/padim``."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'; choose from {MODEL_NAMES}")
    suffix = f"-{tag}" if tag else ""
    return configured_path(config, "models_dir") / category / f"{model_name}{suffix}"


def artifact_path(
    config: dict[str, Any], category: str, model_name: str, tag: str | None = None
) -> Path:
    return model_dir(config, category, model_name, tag) / "artifact.pt"


def thresholds_path(
    config: dict[str, Any], category: str, model_name: str, tag: str | None = None
) -> Path:
    return model_dir(config, category, model_name, tag) / "thresholds.json"


def training_record_path(
    config: dict[str, Any], category: str, model_name: str, tag: str | None = None
) -> Path:
    return model_dir(config, category, model_name, tag) / "training.json"


def legacy_paths(config: dict[str, Any], category: str, model_name: str) -> dict[str, Path]:
    """Pre-expansion flat layout (``models/<category>/<model>.pt``)."""
    base = configured_path(config, "models_dir") / category
    return {
        "artifact": base / f"{model_name}.pt",
        "thresholds": base / f"{model_name}_thresholds.json",
        "training": base / f"{model_name}_training.json",
    }


def available_models(
    config: dict[str, Any], category: str, require_thresholds: bool = True
) -> list[str]:
    """Models with a loadable artifact (and calibrated thresholds by default)."""
    available = []
    for model_name in MODEL_NAMES:
        if not artifact_path(config, category, model_name).is_file():
            continue
        if require_thresholds and not thresholds_path(config, category, model_name).is_file():
            continue
        available.append(model_name)
    return available


def trained_categories(config: dict[str, Any], require_thresholds: bool = True) -> list[str]:
    return [
        category
        for category in supported_categories(config)
        if available_models(config, category, require_thresholds=require_thresholds)
    ]


def save_artifact(
    path: str | Path,
    model: PatchCoreModel | PaDiMModel,
    category: str,
    config: dict[str, Any],
    training: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model_name": model.name,
        "category": category,
        "backbone": model.extractor.metadata(),
        "model_config": model.model_config(),
        "preprocessing": {
            "image_size": int(config["data"]["image_size"]),
            "resize": "bilinear",
            "normalization": "imagenet",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "color_mode": "RGB",
        },
        "state": model.state_dict(),
        "training": training,
        "environment": environment_info(),
        "config_snapshot": config_snapshot(config),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_artifact(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[PatchCoreModel | PaDiMModel, dict[str, Any]]:
    """Load a trusted local artifact. Returns the model and metadata (without tensors)."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(f"Unsupported artifact format: {payload.get('format_version')}")
    model_name = payload.get("model_name")
    backbone = payload["backbone"]
    extractor = FeatureExtractor(
        backbone=str(backbone["name"]),
        pretrained=bool(backbone["pretrained"]),
        layers=tuple(backbone["layers"]),
    )
    model_config = dict(payload["model_config"])
    if model_name == "patchcore":
        model: PatchCoreModel | PaDiMModel = PatchCoreModel(extractor, **model_config)
    elif model_name == "padim":
        model = PaDiMModel(extractor, **model_config)
    else:
        raise ValueError(f"Unknown model in artifact: {model_name}")
    state = payload.pop("state")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    payload.pop("config_snapshot", None)
    return model, payload


def migrate_legacy_artifact(
    config: dict[str, Any], category: str, model_name: str
) -> dict[str, Any]:
    """Move a pre-expansion flat artifact into ``models/<category>/<model>/``.

    Model weights and fitted statistics are copied byte-for-byte by moving the
    files; only the recorded relative artifact paths are rewritten. No
    retraining or recalibration happens here.
    """
    legacy = legacy_paths(config, category, model_name)
    target = {
        "artifact": artifact_path(config, category, model_name),
        "thresholds": thresholds_path(config, category, model_name),
        "training": training_record_path(config, category, model_name),
    }
    if target["artifact"].is_file():
        return {"status": "already_migrated", "artifact": target["artifact"]}
    if not legacy["artifact"].is_file():
        return {"status": "missing_artifact", "artifact": legacy["artifact"]}

    target["artifact"].parent.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for key in ("artifact", "thresholds", "training"):
        if legacy[key].is_file():
            os.replace(legacy[key], target[key])
            moved.append(key)

    root = project_root(config)
    relative_artifact = target["artifact"].resolve().relative_to(root).as_posix()

    training: dict[str, Any] | None = None
    if target["training"].is_file():
        training = load_json(target["training"])
        validation = training.get("artifact_validation")
        if isinstance(validation, dict):
            validation["artifact"] = relative_artifact
        save_json(target["training"], training)

    if target["thresholds"].is_file():
        thresholds = load_json(target["thresholds"])
        thresholds["model_artifact"] = relative_artifact
        save_json(target["thresholds"], thresholds)

    payload = torch.load(target["artifact"], map_location="cpu", weights_only=False)
    if training is not None:
        payload["training"] = training
    else:
        validation = payload.get("training", {}).get("artifact_validation")
        if isinstance(validation, dict):
            validation["artifact"] = relative_artifact
    temporary = target["artifact"].with_name(target["artifact"].name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target["artifact"])
    del payload

    return {
        "status": "migrated",
        "moved": moved,
        "artifact": target["artifact"],
        "thresholds": target["thresholds"],
        "training": target["training"],
        "artifact_relative": relative_artifact,
    }
