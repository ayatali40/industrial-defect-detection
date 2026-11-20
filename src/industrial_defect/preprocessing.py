"""Image and mask preprocessing.

The same functions are used for training features, calibration, evaluation and
inference. Only the spatial size is configurable; the normalisation is fixed so
that all stages see identical model inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageReadError(RuntimeError):
    """Raised when an image cannot be read or decoded."""


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def load_image_rgb(path: str | Path) -> Image.Image:
    """Load an image as RGB, regardless of the stored colour mode."""
    path = Path(path)
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ImageReadError(f"Cannot read image '{path}': {exc}") from exc


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """Resize (bilinear) and normalise a PIL image to a float32 CHW tensor."""
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def preprocess_image_file(path: str | Path, image_size: int) -> torch.Tensor:
    return image_to_tensor(load_image_rgb(path), image_size)


def load_mask_binary(path: str | Path) -> torch.Tensor:
    """Load a ground-truth mask as a binary float tensor with shape (1, H, W)."""
    path = Path(path)
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ImageReadError(f"Cannot read mask '{path}': {exc}") from exc
    binary = (array > 127).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0)


def resize_mask_binary(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Down/up-sample a binary mask while preserving thin defect regions.

    When the resize factor is an integer, max-pooling is used so that any
    defective pixel in a block keeps the block positive. Otherwise nearest
    neighbour is used and the result is re-binarised.
    """
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(f"Expected mask shape (1, H, W), got {tuple(mask.shape)}")
    height, width = mask.shape[-2:]
    if height == size and width == size:
        return mask.float()
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if height % size == 0 and width % size == 0:
        factor = height // size
        if factor > 1:
            return F.max_pool2d(mask.unsqueeze(0), kernel_size=factor).squeeze(0).float()
    resized = F.interpolate(mask.unsqueeze(0), size=(size, size), mode="nearest").squeeze(0)
    return (resized > 0.5).float()


def mask_size_for(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def image_metadata(path: str | Path) -> dict[str, object]:
    path = Path(path)
    with Image.open(path) as image:
        image.load()
        return {
            "path": path.as_posix(),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
        }


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor back to a displayable HWC uint8 array."""
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected CHW tensor with 3 channels, got {tuple(tensor.shape)}")
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    array = (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)
