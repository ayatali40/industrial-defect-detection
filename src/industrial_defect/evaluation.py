"""Image-level, pixel-level and per-region overlap metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score


def _as_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D map, got shape {array.shape}")
    return array


def _confusion(predictions: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    predictions = predictions.astype(bool)
    labels = labels.astype(bool)
    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    tn = int(np.sum(~predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def image_level_metrics(
    labels: list[int] | np.ndarray, scores: list[float] | np.ndarray, threshold: float
) -> dict[str, Any]:
    """Image-level AUROC/AP and thresholded P/R/F1 with an explicit confusion matrix."""
    y_true = np.asarray(labels).astype(int)
    y_score = np.asarray(scores, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError("labels and scores must have the same shape")
    n_normal = int(np.sum(y_true == 0))
    n_anomalous = int(np.sum(y_true == 1))
    if n_normal == 0 or n_anomalous == 0:
        raise ValueError("Image metrics require both normal and anomalous samples")
    auroc = float(roc_auc_score(y_true, y_score))
    average_precision = float(average_precision_score(y_true, y_score))
    predictions = y_score >= threshold
    confusion = _confusion(predictions, y_true)
    precision, recall, f1 = _prf(confusion["tp"], confusion["fp"], confusion["fn"])
    return {
        "image_auroc": auroc,
        "image_average_precision": average_precision,
        "image_precision": precision,
        "image_recall": recall,
        "image_f1": f1,
        "image_threshold": float(threshold),
        "n_samples": int(y_true.size),
        "n_normal": n_normal,
        "n_anomalous": n_anomalous,
        "image_confusion": confusion,
        "predicted_anomalous": int(predictions.sum()),
    }


def pixel_level_metrics(
    maps: list[np.ndarray], masks: list[np.ndarray], threshold: float
) -> dict[str, Any]:
    """Pooled pixel AUROC/AP and thresholded pixel P/R/F1 over the evaluated split."""
    if not maps:
        raise ValueError("No anomaly maps supplied")
    scores = np.concatenate([_as_2d(item).reshape(-1) for item in maps]).astype(np.float32)
    labels = (
        np.concatenate([_as_2d(item).reshape(-1) for item in masks]) > 0.5
    )
    n_positive = int(labels.sum())
    n_negative = int(labels.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("Pixel metrics require both positive and negative pixels")
    auroc = float(roc_auc_score(labels, scores))
    average_precision = float(average_precision_score(labels, scores))
    predictions = scores >= threshold
    confusion = _confusion(predictions, labels)
    precision, recall, f1 = _prf(confusion["tp"], confusion["fp"], confusion["fn"])
    return {
        "pixel_auroc": auroc,
        "pixel_average_precision": average_precision,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_threshold": float(threshold),
        "pixel_confusion": confusion,
        "positive_pixels": n_positive,
        "negative_pixels": n_negative,
        "positive_pixel_rate": n_positive / float(labels.size),
        "predicted_positive_pixels": int(predictions.sum()),
        "evaluated_images": len(maps),
    }


def _quantiles_from_sorted(sorted_values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    positions = quantiles * (len(sorted_values) - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(sorted_values) - 1)
    fraction = positions - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def pro_metric(
    anomalous_maps: list[np.ndarray],
    anomalous_masks: list[np.ndarray],
    normal_maps: list[np.ndarray],
    fpr_limit: float = 0.3,
    steps: int = 100,
) -> dict[str, Any]:
    """Area under the per-region overlap (PRO) curve up to a false-positive rate limit.

    For every connected ground-truth region, the per-region overlap is the
    fraction of its pixels whose anomaly score exceeds a threshold. Regions are
    weighted by area within each image and across the dataset. Thresholds are
    derived from the pixel-score quantiles of normal images so that the x-axis
    is the pixel-level false-positive rate. The score is the trapezoidal area
    under overlap(FPR) for FPR in [0, fpr_limit], normalised by fpr_limit.
    """
    if not anomalous_maps or not normal_maps:
        raise ValueError("PRO requires anomalous and normal anomaly maps")
    if not 0.0 < fpr_limit <= 1.0:
        raise ValueError("fpr_limit must be in (0, 1]")
    normal_scores = np.concatenate([_as_2d(item).reshape(-1) for item in normal_maps]).astype(
        np.float64
    )
    normal_scores.sort()
    fprs = np.linspace(0.0, fpr_limit, steps + 1)
    thresholds = _quantiles_from_sorted(normal_scores, 1.0 - fprs)
    detected_weighted = np.zeros_like(fprs)
    total_area = 0.0
    region_count = 0
    structure = np.ones((3, 3), dtype=np.uint8)
    for map_item, mask_item in zip(anomalous_maps, anomalous_masks, strict=True):
        anomaly_map = _as_2d(map_item).astype(np.float64)
        mask = _as_2d(mask_item) > 0.5
        labels, region_total = ndimage.label(mask, structure=structure)
        for region_id in range(1, region_total + 1):
            region = labels == region_id
            area = int(region.sum())
            if area == 0:
                continue
            region_scores = np.sort(anomaly_map[region])
            above = len(region_scores) - np.searchsorted(
                region_scores, thresholds, side="right"
            )
            detected_weighted += above
            total_area += area
            region_count += 1
    if total_area == 0:
        raise ValueError("No ground-truth regions found for PRO")
    overlap = detected_weighted / total_area
    area_under = float(np.trapz(overlap, fprs) / fpr_limit)
    return {
        "pro_auc": area_under,
        "pro_fpr_limit": float(fpr_limit),
        "pro_steps": int(steps),
        "pro_regions": int(region_count),
        "pro_gt_pixels": int(total_area),
        "pro_fpr": fprs.tolist(),
        "pro_overlap": overlap.tolist(),
    }
