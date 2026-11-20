"""MVTec AD layout discovery, dataset access and deterministic normal-only splits."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import resolve_path
from .preprocessing import (
    image_to_tensor,
    is_image_file,
    load_image_rgb,
    load_mask_binary,
    resize_mask_binary,
)
from .utils import load_json, save_json


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the dataset root or category is missing."""


class DatasetLayoutError(RuntimeError):
    """Raised when the dataset does not follow the expected MVTec AD layout."""


@dataclass(frozen=True)
class MVTecSample:
    path: Path
    relative_path: str
    label: int
    defect_type: str
    mask_path: Path | None


def find_categories(data_root: str | Path) -> list[str]:
    root = Path(data_root)
    if not root.exists():
        raise DatasetNotFoundError(
            f"Dataset root does not exist: {root}. "
            "Download MVTec AD manually from https://www.mvtec.com/research-teaching/datasets/mvtec-ad "
            "and extract it (see data/raw/README.md)."
        )
    if not root.is_dir():
        raise DatasetNotFoundError(f"Dataset root is not a directory: {root}")
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "train").is_dir() and (child / "test").is_dir()
    )


def resolve_category_dir(data_root: str | Path, category: str) -> Path:
    """Accept either an MVTec root containing ``<category>/`` or the category folder itself."""
    root = Path(data_root)
    if not root.exists():
        raise DatasetNotFoundError(
            f"Dataset root does not exist: {root}. Download MVTec AD manually from the official "
            "source and follow data/raw/README.md."
        )
    candidate = root / category
    if candidate.is_dir() and (candidate / "train").is_dir() and (candidate / "test").is_dir():
        return candidate
    if (root / "train").is_dir() and (root / "test").is_dir():
        if root.name == category:
            return root
        raise DatasetLayoutError(
            f"data_root points to category folder '{root.name}' but configured category is "
            f"'{category}'"
        )
    available = find_categories(root)
    raise DatasetLayoutError(
        f"Category '{category}' not found under {root}. Available categories: {available or 'none'}"
    )


def list_defect_types(category_dir: str | Path) -> list[str]:
    test_dir = Path(category_dir) / "test"
    if not test_dir.is_dir():
        return []
    return sorted(child.name for child in test_dir.iterdir() if child.is_dir())


def discover_samples(category_dir: str | Path, split: str) -> list[MVTecSample]:
    """Discover samples for ``split`` in {'train', 'test'} without reading image content."""
    category_dir = Path(category_dir)
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    split_dir = category_dir / split
    if not split_dir.is_dir():
        raise DatasetLayoutError(f"Missing '{split}' directory in {category_dir}")
    samples: list[MVTecSample] = []
    for class_dir in sorted(child for child in split_dir.iterdir() if child.is_dir()):
        defect_type = class_dir.name
        label = 0 if defect_type == "good" else 1
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file() or not is_image_file(image_path):
                continue
            mask_path: Path | None = None
            if split == "test" and label == 1:
                expected = (
                    category_dir
                    / "ground_truth"
                    / defect_type
                    / f"{image_path.stem}_mask{image_path.suffix}"
                )
                mask_path = expected if expected.is_file() else None
            samples.append(
                MVTecSample(
                    path=image_path,
                    relative_path=image_path.relative_to(category_dir).as_posix(),
                    label=label,
                    defect_type=defect_type,
                    mask_path=mask_path,
                )
            )
    samples.sort(key=lambda sample: sample.relative_path)
    return samples


def assert_normal_only(samples: list[MVTecSample]) -> None:
    """Training may only see defect-free images."""
    anomalous = [sample.relative_path for sample in samples if sample.label != 0]
    if anomalous:
        raise ValueError(
            "Training data must contain only normal images, found anomalous samples: "
            + ", ".join(anomalous[:5])
        )


class MVTecDataset(Dataset):
    """Map-style dataset returning preprocessed images and optional aligned masks."""

    def __init__(
        self,
        category_dir: str | Path,
        split: str,
        samples: list[MVTecSample] | None = None,
        image_size: int = 224,
        load_masks: bool = False,
        mask_size: int | None = None,
    ) -> None:
        self.category_dir = Path(category_dir)
        self.split = split
        self.image_size = image_size
        self.load_masks = load_masks
        self.mask_size = mask_size if mask_size is not None else image_size
        self.samples = (
            list(samples) if samples is not None else discover_samples(self.category_dir, split)
        )
        if not self.samples:
            raise DatasetLayoutError(f"No images found for split '{split}' in {self.category_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = load_image_rgb(sample.path)
        original_size = image.size
        tensor = image_to_tensor(image, self.image_size)
        mask: torch.Tensor | None = None
        has_mask = False
        if self.load_masks:
            if sample.mask_path is not None and Path(sample.mask_path).is_file():
                mask = resize_mask_binary(load_mask_binary(sample.mask_path), self.mask_size)
                has_mask = True
            else:
                mask = torch.zeros(1, self.mask_size, self.mask_size, dtype=torch.float32)
        result: dict[str, object] = {
            "image": tensor,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "path": sample.path.as_posix(),
            "relative_path": sample.relative_path,
            "defect_type": sample.defect_type,
            "mask_path": sample.mask_path.as_posix() if sample.mask_path else "",
            "has_mask": has_mask,
            "original_size": original_size,
        }
        if self.load_masks:
            result["mask"] = mask
        return result

    def samples_for(self, relative_paths: list[str]) -> list[MVTecSample]:
        lookup = {sample.relative_path: sample for sample in self.samples}
        missing = [path for path in relative_paths if path not in lookup]
        if missing:
            raise DatasetLayoutError(
                "Split manifest references paths that are not in the dataset: "
                + ", ".join(missing[:5])
            )
        return [lookup[path] for path in relative_paths]


def build_normal_split(
    samples: list[MVTecSample], val_fraction: float, seed: int
) -> dict[str, list[str]]:
    """Deterministically split normal training images into fit and calibration subsets."""
    assert_normal_only(samples)
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if len(samples) < 2:
        raise ValueError("At least two normal images are required to create a fit/val split")
    ordered = sorted(sample.relative_path for sample in samples)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    val_count = max(1, round(len(ordered) * val_fraction))
    val_count = min(val_count, len(ordered) - 1)
    return {"fit": sorted(ordered[val_count:]), "val": sorted(ordered[:val_count])}


def get_or_create_normal_split(
    category_dir: str | Path,
    manifest_path: str | Path,
    val_fraction: float,
    seed: int,
) -> dict[str, list[MVTecSample]]:
    """Load the recorded split or create and persist a new one.

    Reusing the manifest guarantees that threshold calibration always uses the
    same held-out normal images, independent of retraining.
    """
    category_dir = Path(category_dir)
    manifest_path = Path(manifest_path)
    samples = discover_samples(category_dir, "train")
    assert_normal_only(samples)
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        if manifest.get("seed") != seed or float(manifest.get("val_fraction")) != float(
            val_fraction
        ):
            raise DatasetLayoutError(
                f"Existing split {manifest_path} was created with seed={manifest.get('seed')} "
                f"and val_fraction={manifest.get('val_fraction')}; configuration requests "
                f"seed={seed}, val_fraction={val_fraction}. Remove the manifest to regenerate."
            )
        fit = [sample for sample in samples if sample.relative_path in set(manifest["fit_paths"])]
        val = [sample for sample in samples if sample.relative_path in set(manifest["val_paths"])]
    else:
        split = build_normal_split(samples, val_fraction, seed)
        category = category_dir.name
        manifest = {
            "category": category,
            "seed": seed,
            "val_fraction": val_fraction,
            "fit_paths": split["fit"],
            "val_paths": split["val"],
            "counts": {"fit": len(split["fit"]), "val": len(split["val"])},
            "note": "Normal-only split. Test anomalies are never used for fitting or calibration.",
        }
        save_json(manifest_path, manifest)
        fit = [sample for sample in samples if sample.relative_path in set(split["fit"])]
        val = [sample for sample in samples if sample.relative_path in set(split["val"])]
    if len(fit) != len(manifest["fit_paths"]) or len(val) != len(manifest["val_paths"]):
        raise DatasetLayoutError(
            f"Split manifest {manifest_path} does not match the current dataset contents"
        )
    return {"fit": fit, "val": val}


def resolve_manifest_path(config: dict, category: str) -> Path:
    processed = resolve_path(config, config["paths"]["processed_dir"])
    return processed / "splits" / f"{category}.json"
