"""Single-image inference for trained anomaly detectors."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .categories import (
    OutOfScopeCategoryError,
    UnknownCategoryError,
    ensure_in_scope,
)
from .config import get_device
from .preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageReadError,
    image_to_tensor,
    is_image_file,
    load_image_rgb,
)
from .training import (
    MODEL_NAMES,
    artifact_path,
    load_artifact,
    thresholds_path,
    trained_categories,
)
from .utils import load_json, save_json
from .visualization import anomaly_map_visuals, heatmap_rgb, overlay_heatmap

DISCLAIMER = (
    "Research prototype. Thresholds were calibrated on held-out normal training images "
    "of the selected MVTec category; do not use as a production quality decision without "
    "site-specific validation."
)


class InferenceError(ValueError):
    """Raised for invalid inputs or missing inference artifacts."""


def coerce_image(source: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """Validate an input and return it as an RGB PIL image."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise InferenceError(f"Input image does not exist: {path}")
        if not is_image_file(path):
            raise InferenceError(
                f"Unsupported image extension '{path.suffix}'. "
                "Use PNG, JPEG, BMP or TIFF."
            )
        try:
            image = load_image_rgb(path)
        except ImageReadError as exc:
            raise InferenceError(str(exc)) from exc
    elif isinstance(source, Image.Image):
        try:
            image = source.convert("RGB")
        except (OSError, ValueError) as exc:
            raise InferenceError(f"Cannot convert uploaded image to RGB: {exc}") from exc
    elif isinstance(source, np.ndarray):
        if source.dtype != np.uint8:
            raise InferenceError(f"numpy images must be uint8, got {source.dtype}")
        try:
            if source.ndim == 2:
                image = Image.fromarray(source, mode="L").convert("RGB")
            elif source.ndim == 3 and source.shape[2] == 3:
                image = Image.fromarray(source, mode="RGB")
            elif source.ndim == 3 and source.shape[2] == 4:
                image = Image.fromarray(source, mode="RGBA").convert("RGB")
            else:
                raise InferenceError(
                    f"Expected HxW, HxWx3 or HxWx4 uint8 array, got shape {source.shape}"
                )
        except (OSError, ValueError, TypeError) as exc:
            if isinstance(exc, InferenceError):
                raise
            raise InferenceError(f"Cannot interpret image array: {exc}") from exc
    else:
        raise InferenceError(
            f"Unsupported input type {type(source).__name__}; expected path, PIL image or array"
        )
    if image.width < 8 or image.height < 8:
        raise InferenceError(f"Image is too small to inspect: {image.size}")
    return image


@dataclass
class InspectionResult:
    model_name: str
    category: str
    score: float
    image_threshold: float
    pixel_threshold: float
    decision: str
    anomaly_map: np.ndarray
    normalised_map: np.ndarray
    normalised_threshold: float
    heatmap: np.ndarray
    overlay: np.ndarray
    predicted_mask: np.ndarray
    defect_area_fraction: float
    source: str
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "category": self.category,
            "source": self.source,
            "score": self.score,
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "score_margin": self.score - self.image_threshold,
            "decision": self.decision,
            "defect_area_fraction": self.defect_area_fraction,
            "original_size": list(self.original_size),
            "processed_size": list(self.processed_size),
            "latency_ms": self.latency_ms,
            "note": DISCLAIMER,
        }


class IndustrialInspector:
    """Loads one trained artifact and inspects individual images."""

    def __init__(
        self,
        config: dict[str, Any],
        model_name: str = "patchcore",
        category: str | None = None,
        device: str | torch.device | None = None,
        output_size: int | None = None,
    ) -> None:
        if model_name not in MODEL_NAMES:
            raise InferenceError(f"Unknown model '{model_name}'; choose from {MODEL_NAMES}")
        self.config = config
        self.model_name = model_name
        self.category = str(category or config["data"]["category"])
        try:
            ensure_in_scope(self.category, config)
        except OutOfScopeCategoryError as exc:
            raise InferenceError(str(exc)) from exc
        except UnknownCategoryError as exc:
            raise InferenceError(str(exc)) from exc
        self.device = torch.device(device) if device is not None else get_device(config)
        self.output_size = int(
            output_size if output_size is not None else config["evaluation"]["pixel_size"]
        )
        if self.output_size <= 0:
            raise InferenceError("output_size must be positive")

        available = trained_categories(config)
        threshold_file = thresholds_path(config, self.category, model_name)
        if not threshold_file.is_file():
            raise InferenceError(
                f"No calibrated thresholds for category '{self.category}' and model "
                f"'{model_name}' ({threshold_file}). Trained categories: "
                f"{', '.join(available) or 'none'}. Run scripts/04_train.py and "
                "scripts/05_calibrate.py for this category; no other category is used as a "
                "fallback."
            )
        threshold_record = load_json(threshold_file)
        if threshold_record.get("category") != self.category:
            raise InferenceError(
                f"Threshold file {threshold_file} belongs to category "
                f"'{threshold_record.get('category')}', not '{self.category}'"
            )
        if threshold_record.get("model") != model_name:
            raise InferenceError(
                f"Threshold file {threshold_file} belongs to model "
                f"'{threshold_record.get('model')}', not '{model_name}'"
            )
        self.thresholds = threshold_record
        self.image_threshold = float(threshold_record["image_threshold"])
        self.pixel_threshold = float(threshold_record["pixel_threshold"])
        if not np.isfinite([self.image_threshold, self.pixel_threshold]).all():
            raise InferenceError("Threshold file contains non-finite values")

        artifact = artifact_path(config, self.category, model_name)
        if not artifact.is_file():
            raise InferenceError(
                f"Missing model artifact: {artifact}. Run scripts/04_train.py for category "
                f"'{self.category}'; no other category is used as a fallback."
            )
        self.model, self.artifact_metadata = load_artifact(artifact, device=self.device)
        if self.artifact_metadata.get("model_name") != model_name:
            raise InferenceError(
                f"Artifact {artifact} contains '{self.artifact_metadata.get('model_name')}', "
                f"not '{model_name}'"
            )
        if self.artifact_metadata.get("category") != self.category:
            raise InferenceError(
                f"Artifact category '{self.artifact_metadata.get('category')}' does not match "
                f"requested category '{self.category}'"
            )
        self.image_size = int(getattr(self.model, "image_size", config["data"]["image_size"]))
        self._validate_preprocessing()
        self.artifact_path = artifact

    def _validate_preprocessing(self) -> None:
        preprocessing = self.artifact_metadata.get("preprocessing")
        if not isinstance(preprocessing, dict):
            raise InferenceError(
                f"Artifact for {self.category}/{self.model_name} has no preprocessing metadata"
            )
        if int(preprocessing.get("image_size", -1)) != self.image_size:
            raise InferenceError(
                f"Preprocessing mismatch for {self.category}/{self.model_name}: artifact image "
                f"size {preprocessing.get('image_size')} vs model {self.image_size}"
            )
        mean = np.asarray(preprocessing.get("mean"), dtype=np.float64)
        std = np.asarray(preprocessing.get("std"), dtype=np.float64)
        if mean.shape != (3,) or not np.allclose(mean, IMAGENET_MEAN, atol=1e-6):
            raise InferenceError(
                f"Preprocessing mismatch: artifact mean {preprocessing.get('mean')} does not "
                "match the implemented ImageNet normalisation"
            )
        if std.shape != (3,) or not np.allclose(std, IMAGENET_STD, atol=1e-6):
            raise InferenceError(
                f"Preprocessing mismatch: artifact std {preprocessing.get('std')} does not "
                "match the implemented ImageNet normalisation"
            )
        if preprocessing.get("resize") != "bilinear" or preprocessing.get("color_mode") != "RGB":
            raise InferenceError(
                "Preprocessing mismatch: artifact requires resize="
                f"{preprocessing.get('resize')}, colour={preprocessing.get('color_mode')}"
            )

    def describe(self) -> dict[str, Any]:
        training = self.artifact_metadata.get("training", {})
        return {
            "model": self.model_name,
            "category": self.category,
            "device": str(self.device),
            "image_size": self.image_size,
            "output_size": self.output_size,
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "artifact": self.artifact_path.as_posix(),
            "trained_at": training.get("trained_at"),
            "fit_images": training.get("fit_images"),
            "backbone": self.artifact_metadata.get("backbone", {}).get("name"),
            "threshold_calibration": {
                "val_images": self.thresholds.get("val_images"),
                "image_percentile": self.thresholds.get("image_percentile"),
                "pixel_percentile": self.thresholds.get("pixel_percentile"),
                "test_data_used": self.thresholds.get("test_data_used"),
            },
            "disclaimer": DISCLAIMER,
        }

    @torch.no_grad()
    def inspect(
        self, source: str | Path | Image.Image | np.ndarray, source_name: str | None = None
    ) -> InspectionResult:
        image = coerce_image(source)
        original_size = image.size
        if source_name is None:
            source_name = Path(source).name if isinstance(source, (str, Path)) else "upload"
        tensor = image_to_tensor(image, self.image_size).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        maps, scores = self.model.predict(tensor, out_size=self.output_size)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        anomaly_map = maps[0, 0].detach().cpu().float().numpy()
        score = float(scores[0].detach().cpu())
        if not np.isfinite(anomaly_map).all() or not np.isfinite(score):
            raise InferenceError("Model produced non-finite scores")
        resized = np.asarray(
            image.resize((self.output_size, self.output_size), Image.Resampling.BILINEAR)
        )
        normalised, normalised_threshold = anomaly_map_visuals(anomaly_map, self.pixel_threshold)
        predicted_mask = anomaly_map >= self.pixel_threshold
        return InspectionResult(
            model_name=self.model_name,
            category=self.category,
            score=score,
            image_threshold=self.image_threshold,
            pixel_threshold=self.pixel_threshold,
            decision="defect" if score >= self.image_threshold else "pass",
            anomaly_map=anomaly_map.astype(np.float32),
            normalised_map=normalised.astype(np.float32),
            normalised_threshold=normalised_threshold,
            heatmap=heatmap_rgb(normalised),
            overlay=overlay_heatmap(resized, normalised),
            predicted_mask=predicted_mask,
            defect_area_fraction=float(predicted_mask.mean()),
            source=source_name,
            original_size=(int(original_size[0]), int(original_size[1])),
            processed_size=(self.output_size, self.output_size),
            latency_ms=latency_ms,
        )


def _safe_stem(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "image"


def save_result(
    result: InspectionResult, output_dir: str | Path, stem: str | None = None
) -> dict[str, str]:
    """Write heatmap, overlay, predicted mask and JSON metadata for one inspection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(stem or Path(result.source).stem)
    outputs = {
        "heatmap": output_dir / f"{stem}_heatmap.png",
        "overlay": output_dir / f"{stem}_overlay.png",
        "mask": output_dir / f"{stem}_mask.png",
        "metadata": output_dir / f"{stem}_result.json",
    }
    Image.fromarray(result.heatmap, mode="RGB").save(outputs["heatmap"])
    Image.fromarray(result.overlay, mode="RGB").save(outputs["overlay"])
    Image.fromarray((result.predicted_mask.astype(np.uint8) * 255), mode="L").save(
        outputs["mask"]
    )
    save_json(outputs["metadata"], {**result.to_dict(), "outputs": {
        key: path.name for key, path in outputs.items() if key != "metadata"
    }})
    return {key: path.as_posix() for key, path in outputs.items()}
