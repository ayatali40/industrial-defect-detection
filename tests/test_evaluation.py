"""Image, pixel and PRO metric tests with hand-checkable cases."""

from __future__ import annotations

import numpy as np
import pytest

from industrial_defect.evaluation import (
    image_level_metrics,
    pixel_level_metrics,
    pro_metric,
)


def test_image_metrics_perfect_separation():
    metrics = image_level_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    assert metrics["image_auroc"] == pytest.approx(1.0)
    assert metrics["image_average_precision"] == pytest.approx(1.0)
    assert metrics["image_precision"] == pytest.approx(1.0)
    assert metrics["image_recall"] == pytest.approx(1.0)
    assert metrics["image_f1"] == pytest.approx(1.0)
    assert metrics["image_confusion"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}


def test_image_metrics_conservative_threshold():
    metrics = image_level_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=10.0)
    assert metrics["predicted_anomalous"] == 0
    assert metrics["image_precision"] == 0.0
    assert metrics["image_recall"] == 0.0
    assert metrics["image_f1"] == 0.0
    assert metrics["image_auroc"] == pytest.approx(1.0)


def test_image_metrics_require_both_classes():
    with pytest.raises(ValueError, match="both normal and anomalous"):
        image_level_metrics([0, 0], [0.1, 0.2], threshold=0.5)


def test_pixel_metrics_perfect_map():
    normal = np.zeros((2, 2), dtype=np.float32)
    anomaly = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    mask = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    metrics = pixel_level_metrics([normal, anomaly], [np.zeros_like(mask), mask], threshold=0.5)
    assert metrics["pixel_auroc"] == pytest.approx(1.0)
    assert metrics["pixel_average_precision"] == pytest.approx(1.0)
    assert metrics["pixel_precision"] == pytest.approx(1.0)
    assert metrics["pixel_recall"] == pytest.approx(1.0)
    assert metrics["pixel_f1"] == pytest.approx(1.0)
    assert metrics["positive_pixels"] == 2
    assert metrics["predicted_positive_pixels"] == 2


def test_pixel_metrics_reject_empty_masks():
    with pytest.raises(ValueError, match="both positive and negative"):
        pixel_level_metrics(
            [np.zeros((2, 2), dtype=np.float32)],
            [np.zeros((2, 2), dtype=np.float32)],
            threshold=0.5,
        )


def test_pro_perfect_detection():
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[1:3, 1:3] = 1.0
    anomaly = mask * 10.0
    normal = np.zeros((4, 4), dtype=np.float32)
    result = pro_metric([anomaly], [mask], [normal], fpr_limit=0.3, steps=20)
    assert result["pro_regions"] == 1
    assert result["pro_gt_pixels"] == 4
    assert result["pro_auc"] == pytest.approx(1.0)
    assert len(result["pro_fpr"]) == 21


def test_pro_partial_overlap_is_between_zero_and_one():
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    anomaly = np.zeros((8, 8), dtype=np.float32)
    anomaly[2:4, 2:6] = 0.9
    anomaly[4:6, 2:6] = 0.1
    normal = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    result = pro_metric([anomaly], [mask], [normal], fpr_limit=0.3, steps=20)
    assert 0.0 < result["pro_auc"] < 1.0


def test_pro_requires_inputs_and_valid_limit():
    with pytest.raises(ValueError, match="requires anomalous and normal"):
        pro_metric([], [], [])
    valid_mask = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="fpr_limit"):
        pro_metric([valid_mask], [valid_mask], [valid_mask], fpr_limit=0.0)
