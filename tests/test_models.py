"""Model, coreset and post-processing tests."""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader

from industrial_defect.dataset import MVTecDataset, discover_samples
from industrial_defect.models import FeatureExtractor, PaDiMModel, PatchCoreModel
from industrial_defect.models.common import gaussian_blur_map, upsample_maps
from industrial_defect.models.patchcore_model import greedy_coreset


def test_gaussian_blur_preserves_constant_maps():
    maps = torch.ones(2, 1, 16, 16)
    smoothed = gaussian_blur_map(maps, 2.0)
    assert torch.allclose(smoothed, maps, atol=1e-5)


def test_gaussian_blur_spreads_impulse_and_keeps_mass():
    maps = torch.zeros(1, 1, 31, 31)
    maps[0, 0, 15, 15] = 1.0
    smoothed = gaussian_blur_map(maps, 2.0)
    assert smoothed.max().item() < 1.0
    assert math.isclose(smoothed.sum().item(), 1.0, rel_tol=1e-4, abs_tol=1e-4)
    assert smoothed[0, 0, 15, 15] > smoothed[0, 0, 0, 0]


def test_gaussian_blur_sigma_zero_is_identity():
    maps = torch.rand(1, 1, 8, 8)
    assert gaussian_blur_map(maps, 0.0) is maps


def test_upsample_maps_shape():
    maps = torch.rand(2, 1, 8, 8)
    upsampled = upsample_maps(maps, 20)
    assert upsampled.shape == (2, 1, 20, 20)
    assert upsample_maps(upsampled, 20) is upsampled


def test_greedy_coreset_is_deterministic():
    features = torch.randn(60, 8, generator=torch.Generator().manual_seed(0))
    first = greedy_coreset(features, 10, seed=42)
    second = greedy_coreset(features, 10, seed=42)
    assert first.shape == (10, 8)
    assert torch.equal(first, second)


def test_greedy_coreset_budget_validation():
    features = torch.randn(5, 3)
    with pytest.raises(ValueError, match="positive"):
        greedy_coreset(features, 0, seed=1)
    assert greedy_coreset(features, 5, seed=1).shape == (5, 3)


def test_padim_rejects_too_many_features():
    extractor = FeatureExtractor(pretrained=False, layers=("layer2",))
    with pytest.raises(ValueError, match="exceeds backbone feature dimension"):
        PaDiMModel(extractor, n_features=600)


@pytest.mark.integration
def test_patchcore_and_padim_fit_predict(real_category_dir):
    train = discover_samples(real_category_dir, "train")[:6]
    test = discover_samples(real_category_dir, "test")[:2]
    loader = DataLoader(
        MVTecDataset(real_category_dir, "train", samples=train, image_size=64),
        batch_size=3,
    )
    extractor = FeatureExtractor(pretrained=True)
    patchcore = PatchCoreModel(
        extractor, image_size=64, projection_dim=32, smooth_sigma=1.0, seed=42
    )
    stats = patchcore.fit(loader, max_candidates=500, coreset_ratio=0.05)
    assert stats["memory_bank_size"] > 0
    assert stats["feature_grid"] == [8, 8]

    test_images = torch.stack(
        [
            MVTecDataset(real_category_dir, "test", samples=test, image_size=64)[index]["image"]
            for index in range(2)
        ]
    )
    maps, scores = patchcore.predict(test_images, out_size=32)
    assert maps.shape == (2, 1, 32, 32)
    assert scores.shape == (2,)
    assert torch.isfinite(maps).all() and torch.isfinite(scores).all()
    maps_again, scores_again = patchcore.predict(test_images, out_size=32)
    assert torch.allclose(scores, scores_again)

    padim = PaDiMModel(
        extractor, image_size=64, n_features=20, covariance_epsilon=0.01, smooth_sigma=1.0
    )
    padim_stats = padim.fit(loader)
    assert padim_stats["positions"] == 64
    padim_maps, padim_scores = padim.predict(test_images, out_size=32)
    assert padim_maps.shape == (2, 1, 32, 32)
    assert padim_scores.shape == (2,)
    assert torch.isfinite(padim_maps).all() and torch.isfinite(padim_scores).all()


def test_unfitted_models_raise_on_predict():
    extractor = FeatureExtractor(pretrained=False, layers=("layer2",))
    patchcore = PatchCoreModel(extractor, image_size=32, projection_dim=8)
    with pytest.raises(RuntimeError, match="fitted"):
        patchcore.predict(torch.zeros(1, 3, 32, 32))
    padim = PaDiMModel(extractor, image_size=32, n_features=8)
    with pytest.raises(RuntimeError, match="fitted"):
        padim.predict(torch.zeros(1, 3, 32, 32))


def test_extractor_metadata_reports_pretrained_flag():
    extractor = FeatureExtractor(pretrained=False, layers=("layer2",))
    metadata = extractor.metadata()
    assert metadata["pretrained"] is False
    assert metadata["output_dim"] == 512
