"""Inference input validation, outputs and artifact consistency tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from industrial_defect.config import configured_path
from industrial_defect.inference import (
    IndustrialInspector,
    InferenceError,
    InspectionResult,
    coerce_image,
    save_result,
)


def test_coerce_image_rejects_missing_path(tmp_path):
    with pytest.raises(InferenceError, match="does not exist"):
        coerce_image(tmp_path / "missing.png")


def test_coerce_image_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(InferenceError, match="Unsupported image extension"):
        coerce_image(path)


def test_coerce_image_rejects_tiny_images():
    with pytest.raises(InferenceError, match="too small"):
        coerce_image(Image.new("RGB", (4, 4)))


def test_coerce_image_rejects_float_arrays():
    with pytest.raises(InferenceError, match="uint8"):
        coerce_image(np.zeros((8, 8, 3), dtype=np.float32))


def test_coerce_image_rejects_bad_array_shape():
    with pytest.raises(InferenceError, match="Expected HxW"):
        coerce_image(np.zeros((8, 8, 5), dtype=np.uint8))


def test_coerce_image_accepts_grayscale_and_rgba():
    assert coerce_image(np.zeros((8, 8), dtype=np.uint8)).mode == "RGB"
    assert coerce_image(np.zeros((8, 8, 4), dtype=np.uint8)).mode == "RGB"


def test_coerce_image_rejects_unknown_type():
    with pytest.raises(InferenceError, match="Unsupported input type"):
        coerce_image(42)


def _fake_result(source: str = "sample.png") -> InspectionResult:
    anomaly = np.linspace(0.0, 5.0, 64, dtype=np.float32).reshape(8, 8)
    normalised = anomaly / anomaly.max()
    return InspectionResult(
        model_name="patchcore",
        category="transistor",
        score=12.5,
        image_threshold=10.0,
        pixel_threshold=2.5,
        decision="defect",
        anomaly_map=anomaly,
        normalised_map=normalised,
        normalised_threshold=0.5,
        heatmap=(normalised[..., None].repeat(3, axis=2) * 255).astype(np.uint8),
        overlay=(normalised[..., None].repeat(3, axis=2) * 255).astype(np.uint8),
        predicted_mask=anomaly >= 2.5,
        defect_area_fraction=float((anomaly >= 2.5).mean()),
        source=source,
        original_size=(1024, 1024),
        processed_size=(8, 8),
        latency_ms=1.0,
    )


def test_save_result_writes_expected_files(tmp_path):
    result = _fake_result()
    outputs = save_result(result, tmp_path, stem="sample")
    assert set(outputs) == {"heatmap", "overlay", "mask", "metadata"}
    for path in outputs.values():
        assert Path(path).is_file()
    metadata = json.loads((tmp_path / "sample_result.json").read_text())
    assert metadata["decision"] == "defect"
    assert metadata["score"] == pytest.approx(12.5)
    assert metadata["outputs"]["mask"] == "sample_mask.png"
    assert "Research prototype" in metadata["note"]


def test_unknown_category_raises(config):
    with pytest.raises(InferenceError, match="Unknown MVTec AD category"):
        IndustrialInspector(config, "patchcore", "not_a_category")


def test_untrained_category_never_falls_back(tmp_config):
    with pytest.raises(InferenceError, match="No calibrated thresholds"):
        IndustrialInspector(tmp_config, "patchcore", "bottle")


def test_out_of_scope_category_raises(config):
    with pytest.raises(InferenceError, match="outside the frozen project scope"):
        IndustrialInspector(config, "patchcore", "carpet")


def test_unknown_model_raises(config):
    with pytest.raises(InferenceError, match="Unknown model"):
        IndustrialInspector(config, "yolo", "transistor")


@pytest.mark.integration
def test_inspector_matches_saved_prediction(config, project_root):
    inspector = IndustrialInspector(config, "patchcore", "transistor", device="cpu")
    result = inspector.inspect(project_root / "transistor" / "test" / "good" / "000.png")
    predictions = pd.read_csv(
        configured_path(config, "predictions_dir")
        / "transistor"
        / "patchcore_test_predictions.csv"
    )
    row = predictions[predictions.relative_path == "test/good/000.png"].iloc[0]
    assert result.score == pytest.approx(float(row.score), abs=1e-4)
    assert result.decision == ("defect" if row.predicted_label == 1 else "pass")
    assert result.anomaly_map.shape == (512, 512)
    assert result.overlay.shape == (512, 512, 3)
    assert result.heatmap.shape == (512, 512, 3)
    assert result.predicted_mask.dtype == bool
    second = inspector.inspect(project_root / "transistor" / "test" / "good" / "000.png")
    assert second.score == pytest.approx(result.score, abs=1e-6)
    assert "fit_images" in inspector.describe()
