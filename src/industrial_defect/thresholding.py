"""Normal-only anomaly threshold calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models import PaDiMModel, PatchCoreModel

LIMITATIONS = [
    "Thresholds are calibrated only on held-out normal training images; no test labels "
    "or test anomalies are used.",
    "The percentile fixes the expected false-positive rate on the calibration distribution, "
    "which may not match production drift.",
    "The pixel threshold is calibrated on anomaly maps resized to the evaluation resolution "
    "(512x512 by default), not the original 1024x1024 resolution.",
    "Normal-only calibration cannot guarantee a good defect-type balance; the operating "
    "point should be selected from business costs, not from test performance.",
]


@torch.no_grad()
def calibrate_thresholds(
    model: PatchCoreModel | PaDiMModel,
    loader: DataLoader,
    eval_size: int,
    image_percentile: float,
    pixel_percentile: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Compute image/pixel thresholds from normal validation scores only."""
    if not 0.0 < image_percentile < 100.0 or not 0.0 < pixel_percentile < 100.0:
        raise ValueError("Percentiles must lie strictly between 0 and 100")
    image_scores: list[np.ndarray] = []
    pixel_values: list[np.ndarray] = []
    for batch in loader:
        maps, scores = model.predict(batch["image"], out_size=eval_size)
        image_scores.append(scores.numpy().astype(np.float32))
        pixel_values.append(maps.flatten(1).numpy().astype(np.float32))
    scores_array = np.concatenate(image_scores)
    pixels_array = np.concatenate(pixel_values)
    image_threshold = float(np.percentile(scores_array, image_percentile, method="linear"))
    pixel_threshold = float(np.percentile(pixels_array, pixel_percentile, method="linear"))
    record = {
        "calibration_split": "normal-only held-out subset of train/good",
        "normal_only": True,
        "test_data_used": False,
        "val_images": int(scores_array.shape[0]),
        "val_pixel_values": int(pixels_array.size),
        "image_percentile": float(image_percentile),
        "pixel_percentile": float(pixel_percentile),
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "image_percentile_method": "numpy linear percentile of validation image scores",
        "pixel_percentile_method": "numpy linear percentile of validation pixel scores",
        "pixel_eval_size": int(eval_size),
        "val_image_score_stats": {
            "min": float(scores_array.min()),
            "mean": float(scores_array.mean()),
            "median": float(np.median(scores_array)),
            "std": float(scores_array.std(ddof=0)),
            "max": float(scores_array.max()),
        },
        "val_image_false_positive_rate_at_threshold": float(
            (scores_array >= image_threshold).mean()
        ),
        "val_pixel_false_positive_rate_at_threshold": float(
            (pixels_array >= pixel_threshold).mean()
        ),
        "limitations": LIMITATIONS,
    }
    return record, scores_array, pixels_array
