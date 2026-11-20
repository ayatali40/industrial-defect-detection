"""Dataset layout, normal-only split and manifest consistency tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from industrial_defect.dataset import (
    DatasetLayoutError,
    MVTecDataset,
    MVTecSample,
    assert_normal_only,
    build_normal_split,
    discover_samples,
    find_categories,
    get_or_create_normal_split,
    resolve_category_dir,
)


def _samples(count: int, label: int = 0) -> list[MVTecSample]:
    defect = "good" if label == 0 else "cut_lead"
    return [
        MVTecSample(
            path=Path(f"{defect}/{index:03d}.png"),
            relative_path=f"train/{defect}/{index:03d}.png",
            label=label,
            defect_type=defect,
            mask_path=None,
        )
        for index in range(count)
    ]


def test_build_normal_split_is_deterministic_and_disjoint():
    samples = _samples(10)
    first = build_normal_split(samples, 0.2, 42)
    second = build_normal_split(samples, 0.2, 42)
    assert first == second
    assert len(first["fit"]) == 8
    assert len(first["val"]) == 2
    assert set(first["fit"]).isdisjoint(first["val"])
    assert set(first["fit"]) | set(first["val"]) == {sample.relative_path for sample in samples}


def test_build_normal_split_rejects_anomalies():
    with pytest.raises(ValueError, match="only normal images"):
        build_normal_split(_samples(4) + _samples(1, label=1), 0.2, 42)


def test_build_normal_split_validates_fraction():
    with pytest.raises(ValueError, match="val_fraction"):
        build_normal_split(_samples(4), 0.0, 42)


def test_assert_normal_only_reports_paths():
    with pytest.raises(ValueError, match="train/cut_lead/000.png"):
        assert_normal_only(_samples(2) + _samples(1, label=1))


def test_resolve_category_dir_accepts_root_and_category(tmp_path):
    category = tmp_path / "data" / "transistor"
    (category / "train" / "good").mkdir(parents=True)
    (category / "test" / "good").mkdir(parents=True)
    assert find_categories(tmp_path / "data") == ["transistor"]
    assert resolve_category_dir(tmp_path / "data", "transistor") == category
    assert resolve_category_dir(category, "transistor") == category
    with pytest.raises(DatasetLayoutError):
        resolve_category_dir(category, "bottle")


def test_manifest_seed_mismatch_raises(config, real_category_dir, tmp_path):
    manifest = tmp_path / "split.json"
    first = get_or_create_normal_split(real_category_dir, manifest, 0.2, 42)
    assert len(first["fit"]) == 170
    assert len(first["val"]) == 43
    assert manifest.is_file()
    with pytest.raises(DatasetLayoutError, match="seed"):
        get_or_create_normal_split(real_category_dir, manifest, 0.2, 7)


@pytest.mark.integration
def test_real_dataset_counts(real_category_dir):
    train = discover_samples(real_category_dir, "train")
    test = discover_samples(real_category_dir, "test")
    assert len(train) == 213
    assert len(test) == 100
    assert all(sample.label == 0 for sample in train)
    anomalous = [sample for sample in test if sample.label == 1]
    assert len(anomalous) == 40
    assert all(sample.mask_path is not None and sample.mask_path.is_file() for sample in anomalous)
    assert {sample.defect_type for sample in anomalous} == {
        "bent_lead",
        "cut_lead",
        "damaged_case",
        "misplaced",
    }


@pytest.mark.integration
def test_dataset_returns_aligned_mask(real_category_dir):
    test_samples = discover_samples(real_category_dir, "test")
    anomalous = next(sample for sample in test_samples if sample.label == 1)
    normal = next(sample for sample in test_samples if sample.label == 0)
    dataset = MVTecDataset(
        real_category_dir, "test", samples=[anomalous, normal], image_size=64, load_masks=True, mask_size=32
    )
    item = dataset[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["mask"].shape == (1, 32, 32)
    assert item["has_mask"] is True
    assert item["label"].item() == 1
    normal_item = dataset[1]
    assert normal_item["has_mask"] is False
    assert torch.count_nonzero(normal_item["mask"]) == 0


@pytest.mark.integration
def test_dataset_without_masks_has_no_mask_key(real_category_dir):
    sample = discover_samples(real_category_dir, "train")[0]
    dataset = MVTecDataset(real_category_dir, "train", samples=[sample], image_size=32)
    item = dataset[0]
    assert "mask" not in item
    assert isinstance(item["original_size"], tuple)
