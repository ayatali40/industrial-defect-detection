"""Consistent, readable visualisations for EDA, evaluation and error analysis."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PIL import Image  # noqa: E402

COLORMAP = "inferno"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": False,
            "figure.autolayout": False,
        }
    )


def _load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _load_mask(path: str | Path | None) -> np.ndarray | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return (np.asarray(image.convert("L")) > 127).astype(np.float32)


def _resize(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a 2D float map with bilinear interpolation."""
    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR))


def _display_range(anomaly_map: np.ndarray, pixel_threshold: float) -> tuple[float, float]:
    """Per-image display range; never hides the calibrated threshold."""
    finite = anomaly_map[np.isfinite(anomaly_map)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(finite, 1.0))
    high_percentile = float(np.percentile(finite, 99.9))
    high = max(high_percentile, float(pixel_threshold) * 1.02, low + 1e-6)
    return low, high


def _normalise(anomaly_map: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((anomaly_map - low) / (high - low), 0.0, 1.0)


def _overlay(image: np.ndarray, normalised: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heat = (colormaps[COLORMAP](normalised)[..., :3] * 255.0).astype(np.float32)
    return ((1.0 - alpha) * image.astype(np.float32) + alpha * heat).clip(0, 255).astype(np.uint8)


def anomaly_map_visuals(
    anomaly_map: np.ndarray, pixel_threshold: float
) -> tuple[np.ndarray, float]:
    """Normalise a map for display and return (normalised_map, normalised_threshold).

    The display range always includes the calibrated pixel threshold so the
    overlay and contour are never hidden by a bright outlier.
    """
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    low, high = _display_range(anomaly_map, pixel_threshold)
    normalised = _normalise(anomaly_map, low, high)
    normalised_threshold = float(np.clip((pixel_threshold - low) / (high - low), 0.0, 1.0))
    return normalised, normalised_threshold


def overlay_heatmap(image: np.ndarray, normalised: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a normalised anomaly map over an RGB image."""
    return _overlay(image, normalised, alpha=alpha)


def heatmap_rgb(normalised: np.ndarray) -> np.ndarray:
    """Colourise a normalised anomaly map as an RGB uint8 image."""
    return (colormaps[COLORMAP](normalised)[..., :3] * 255.0).round().astype(np.uint8)


def _gt_prediction_overlay(
    image: np.ndarray, gt: np.ndarray | None, prediction: np.ndarray
) -> np.ndarray:
    base = (image.astype(np.float32) * 0.55 + 60.0).clip(0, 255)
    if gt is not None:
        base[gt > 0.5] = [0, 220, 0]
    only_pred = prediction & (gt <= 0.5 if gt is not None else np.ones_like(prediction, bool))
    base[only_pred] = [230, 30, 30]
    if gt is not None:
        both = prediction & (gt > 0.5)
        base[both] = [250, 220, 40]
    return base.astype(np.uint8)


def _save(figure: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def save_inspection_panel(
    image_path: str | Path,
    anomaly_map: np.ndarray,
    mask_path: str | Path | None,
    image_threshold: float,
    pixel_threshold: float,
    score: float,
    output_path: str | Path,
    title: str = "",
) -> Path:
    """Six-panel inspection figure: image, GT, heatmap, overlay, prediction, GT vs prediction."""
    _apply_style()
    image = _load_rgb(image_path)
    height, width = image.shape[:2]
    mask = _load_mask(mask_path)
    anomaly = _resize(np.asarray(anomaly_map, dtype=np.float32), (width, height))
    low, high = _display_range(anomaly, pixel_threshold)
    normalised = _normalise(anomaly, low, high)
    normalised_threshold = float(np.clip((pixel_threshold - low) / (high - low), 0.0, 1.0))
    prediction = anomaly >= pixel_threshold
    decision = "DEFECT" if score >= image_threshold else "PASS"

    figure, axes = plt.subplots(1, 6, figsize=(26, 4.6))
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    if mask is None:
        axes[1].text(0.5, 0.5, "No ground truth\n(inference)", ha="center", va="center")
        axes[1].imshow(np.zeros((height, width)), cmap="gray", vmin=0, vmax=1)
    else:
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground-truth mask")
    heat = axes[2].imshow(normalised, cmap=COLORMAP, vmin=0, vmax=1)
    axes[2].set_title("Anomaly heatmap (normalised)")
    figure.colorbar(heat, ax=axes[2], fraction=0.046, pad=0.03)
    axes[3].imshow(_overlay(image, normalised))
    if 0.0 < normalised_threshold < 1.0:
        axes[3].contour(normalised, levels=[normalised_threshold], colors="cyan", linewidths=1.2)
    axes[3].set_title("Overlay + pixel threshold contour")
    axes[4].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("Predicted defect region")
    axes[5].imshow(_gt_prediction_overlay(image, mask, prediction))
    axes[5].set_title("GT (green) vs prediction (red)")
    for axis in axes:
        axis.axis("off")
    header = f"{title} | score={score:.3f} | image threshold={image_threshold:.3f} | {decision}"
    figure.suptitle(header, fontsize=12)
    return _save(figure, output_path)


def save_contact_sheet(
    items: Iterable[dict[str, Any]],
    output_path: str | Path,
    pixel_threshold: float,
    title: str,
) -> Path:
    """Rows of original / heatmap / overlay / GT-vs-prediction for error categories."""
    _apply_style()
    items = list(items)
    if not items:
        raise ValueError("No items supplied for the contact sheet")
    figure, axes = plt.subplots(len(items), 4, figsize=(18, 3.0 * len(items)), squeeze=False)
    for row, item in enumerate(items):
        image = _load_rgb(item["image_path"])
        height, width = image.shape[:2]
        anomaly = _resize(np.asarray(item["anomaly_map"], dtype=np.float32), (width, height))
        low, high = _display_range(anomaly, pixel_threshold)
        normalised = _normalise(anomaly, low, high)
        prediction = anomaly >= pixel_threshold
        mask = _load_mask(item.get("mask_path"))
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(item.get("title", ""))
        axes[row, 1].imshow(normalised, cmap=COLORMAP, vmin=0, vmax=1)
        axes[row, 1].set_title("Heatmap")
        axes[row, 2].imshow(_overlay(image, normalised))
        axes[row, 2].set_title("Overlay")
        axes[row, 3].imshow(_gt_prediction_overlay(image, mask, prediction))
        axes[row, 3].set_title("GT (green) / prediction (red)")
        for axis in axes[row]:
            axis.axis("off")
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(figure, output_path)


def save_score_distribution(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_path: str | Path,
    title: str,
) -> Path:
    _apply_style()
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    normal = scores[labels == 0]
    anomalous = scores[labels == 1]
    bins = np.linspace(float(scores.min()), float(scores.max()), 35)
    axis.hist(normal, bins=bins, alpha=0.65, label=f"normal (n={len(normal)})", color="#2c7fb8")
    axis.hist(
        anomalous,
        bins=bins,
        alpha=0.65,
        label=f"anomalous (n={len(anomalous)})",
        color="#de2d26",
    )
    axis.axvline(threshold, color="black", linestyle="--", linewidth=1.4, label=f"threshold={threshold:.3f}")
    axis.set_xlabel("Image-level anomaly score")
    axis.set_ylabel("Number of images")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    return _save(figure, output_path)


def save_metric_comparison(results: dict[str, dict[str, Any]], output_path: str | Path) -> Path:
    """Grouped bars for image-level and pixel-level metrics of all evaluated models."""
    _apply_style()
    image_metrics = ["image_auroc", "image_average_precision", "image_f1"]
    pixel_metrics = [
        "pixel_auroc",
        "pixel_average_precision",
        "pixel_f1",
        "pro_auc",
    ]
    model_names = list(results)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for axis, metrics, title in (
        (axes[0], image_metrics, "Image-level metrics"),
        (axes[1], pixel_metrics, "Pixel-level metrics"),
    ):
        positions = np.arange(len(metrics))
        width = 0.8 / max(1, len(model_names))
        for index, model_name in enumerate(model_names):
            values = [float(results[model_name].get(name, 0.0)) for name in metrics]
            axis.bar(
                positions + index * width - (len(model_names) - 1) * width / 2,
                values,
                width,
                label=model_name,
                color=colors[index % len(colors)],
            )
            for x_position, value in zip(
                positions + index * width - (len(model_names) - 1) * width / 2, values, strict=True
            ):
                axis.text(x_position, value + 0.015, f"{value:.3f}", ha="center", fontsize=7)
        axis.set_xticks(positions)
        axis.set_xticklabels([name.replace("_", " ") for name in metrics], rotation=15)
        axis.set_ylim(0, 1.08)
        axis.set_title(title)
        axis.legend()
    figure.tight_layout()
    return _save(figure, output_path)


def save_dataset_grid(
    items: Iterable[dict[str, Any]],
    output_path: str | Path,
    title: str,
    columns: int = 4,
    show_masks: bool = False,
) -> Path:
    """Grid of dataset samples, optionally with green ground-truth contours."""
    _apply_style()
    items = list(items)
    rows = int(np.ceil(len(items) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.4 * rows), squeeze=False)
    for index, item in enumerate(items):
        row, column = divmod(index, columns)
        axis = axes[row, column]
        image = _load_rgb(item["image_path"])
        axis.imshow(image)
        if show_masks:
            mask = _load_mask(item.get("mask_path"))
            if mask is not None and mask.max() > 0:
                axis.contour(mask, levels=[0.5], colors="lime", linewidths=1.2)
        axis.set_title(item.get("title", Path(item["image_path"]).stem))
        axis.axis("off")
    for index in range(len(items), rows * columns):
        row, column = divmod(index, columns)
        axes[row, column].axis("off")
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(figure, output_path)
