"""Normal-only threshold calibration tests with a deterministic dummy model."""

from __future__ import annotations

import pytest
import torch

from industrial_defect.thresholding import calibrate_thresholds


class DummyModel:
    def predict(self, images, out_size=None):
        batch = len(images)
        base = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1, 1)
        maps = base + torch.arange(4, dtype=torch.float32).view(1, 1, 2, 2)
        scores = maps.flatten(1).amax(dim=1)
        return maps, scores


def _loader():
    return [
        {"image": torch.zeros(2, 3, 4, 4)},
        {"image": torch.zeros(1, 3, 4, 4)},
    ]


def test_calibration_uses_validation_scores_only():
    record, image_scores, pixel_values = calibrate_thresholds(
        DummyModel(), _loader(), eval_size=2, image_percentile=50.0, pixel_percentile=75.0
    )
    assert image_scores.tolist() == [3.0, 4.0, 3.0]
    assert pixel_values.size == 12
    assert record["image_threshold"] == pytest.approx(3.0)
    assert record["pixel_threshold"] == pytest.approx(3.0)
    assert record["val_images"] == 3
    assert record["normal_only"] is True
    assert record["test_data_used"] is False
    assert record["limitations"]
    assert record["val_image_false_positive_rate_at_threshold"] == pytest.approx(1.0)
    assert record["val_pixel_false_positive_rate_at_threshold"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("percentile", [0.0, 100.0, -1.0, 150.0])
def test_invalid_percentiles_raise(percentile):
    with pytest.raises(ValueError, match="Percentiles"):
        calibrate_thresholds(
            DummyModel(), _loader(), eval_size=2, image_percentile=percentile, pixel_percentile=50.0
        )
