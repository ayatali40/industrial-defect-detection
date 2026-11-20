"""Preprocessing and mask-alignment tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from industrial_defect.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageReadError,
    denormalize,
    image_to_tensor,
    load_image_rgb,
    load_mask_binary,
    resize_mask_binary,
)


def _save_png(path, array, mode):
    Image.fromarray(array, mode=mode).save(path)
    return path


def test_image_to_tensor_shape_and_normalisation(tmp_path):
    path = _save_png(tmp_path / "white.png", np.full((12, 10, 3), 255, np.uint8), "RGB")
    tensor = image_to_tensor(load_image_rgb(path), 16)
    assert tensor.shape == (3, 16, 16)
    assert tensor.dtype == torch.float32
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    assert torch.allclose(tensor, (torch.ones(3, 16, 16) - mean) / std, atol=1e-5)


def test_grayscale_is_converted_to_rgb(tmp_path):
    path = _save_png(tmp_path / "grey.png", np.zeros((8, 8), np.uint8), "L")
    image = load_image_rgb(path)
    assert image.mode == "RGB"
    assert image.size == (8, 8)


def test_unreadable_image_raises(tmp_path):
    path = tmp_path / "broken.png"
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(ImageReadError):
        load_image_rgb(path)


def test_image_size_must_be_positive(tmp_path):
    path = _save_png(tmp_path / "img.png", np.zeros((8, 8, 3), np.uint8), "RGB")
    with pytest.raises(ValueError, match="positive"):
        image_to_tensor(load_image_rgb(path), 0)


def test_mask_is_binary(tmp_path):
    array = np.array([[0, 127, 128], [255, 1, 200]], dtype=np.uint8)
    path = _save_png(tmp_path / "mask.png", array, "L")
    mask = load_mask_binary(path)
    assert mask.shape == (1, 2, 3)
    expected = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    assert torch.equal(mask[0], expected)


def test_mask_integer_downsample_preserves_thin_regions():
    mask = torch.zeros(1, 4, 4)
    mask[0, 0, 1] = 1.0
    resized = resize_mask_binary(mask, 2)
    assert resized.shape == (1, 2, 2)
    assert resized.sum().item() == 1.0
    assert resized[0, 0, 0].item() == 1.0


def test_mask_non_integer_resize_stays_binary():
    mask = torch.ones(1, 3, 3)
    resized = resize_mask_binary(mask, 2)
    assert resized.shape == (1, 2, 2)
    assert set(torch.unique(resized).tolist()) <= {0.0, 1.0}
    assert resized.sum().item() > 0


def test_resize_mask_rejects_bad_shape():
    with pytest.raises(ValueError, match="Expected mask shape"):
        resize_mask_binary(torch.zeros(3, 4), 2)


def test_denormalize_round_trip():
    array = np.array(
        [[[10, 80, 250], [255, 0, 128]], [[30, 200, 60], [1, 2, 3]]], dtype=np.uint8
    )
    normalised = (array.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN)) / np.array(
        IMAGENET_STD
    )
    tensor = torch.from_numpy(normalised).permute(2, 0, 1)
    restored = denormalize(tensor)
    assert restored.shape == (2, 2, 3)
    assert np.all(np.abs(restored.astype(int) - array.astype(int)) <= 1)
