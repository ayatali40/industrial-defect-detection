"""Visualisation helpers produce finite, readable outputs."""

from __future__ import annotations

import numpy as np
from PIL import Image

from industrial_defect.visualization import (
    anomaly_map_visuals,
    heatmap_rgb,
    overlay_heatmap,
    save_inspection_panel,
)


def test_anomaly_visuals_keep_threshold_visible():
    anomaly = np.zeros((16, 16), dtype=np.float32)
    normalised, normalised_threshold = anomaly_map_visuals(anomaly, pixel_threshold=4.0)
    assert normalised.shape == (16, 16)
    assert 0.0 <= normalised.min() <= normalised.max() <= 1.0
    assert 0.0 < normalised_threshold <= 1.0


def test_heatmap_and_overlay_shapes():
    anomaly = np.random.default_rng(0).random((12, 10)).astype(np.float32)
    normalised, _ = anomaly_map_visuals(anomaly, pixel_threshold=0.5)
    image = np.zeros((12, 10, 3), dtype=np.uint8)
    heatmap = heatmap_rgb(normalised)
    overlay = overlay_heatmap(image, normalised)
    assert heatmap.shape == (12, 10, 3) and heatmap.dtype == np.uint8
    assert overlay.shape == (12, 10, 3) and overlay.dtype == np.uint8
    assert overlay.max() > 0


def test_save_inspection_panel(tmp_path):
    rng = np.random.default_rng(0)
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray((rng.random((32, 32, 3)) * 255).astype(np.uint8)).save(image_path)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    Image.fromarray(mask).save(mask_path)
    anomaly = rng.random((32, 32)).astype(np.float32) * 5.0
    output = save_inspection_panel(
        image_path,
        anomaly,
        mask_path,
        image_threshold=3.0,
        pixel_threshold=2.5,
        score=3.5,
        output_path=tmp_path / "panel.png",
        title="unit test",
    )
    assert output.is_file()
    assert output.stat().st_size > 10_000
