"""Shared helpers for anomaly map post-processing."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_blur_map(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian smoothing of anomaly maps with shape ``(B, 1, H, W)``.

    ``sigma`` is expressed in pixels of the supplied map resolution. Call this
    after upsampling so that the blur radius is independent of the feature grid.
    """
    if sigma <= 0:
        return maps
    radius = max(1, int(round(4.0 * sigma)))
    coordinates = torch.arange(
        -radius, radius + 1, dtype=maps.dtype, device=maps.device
    )
    kernel = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    horizontal = F.pad(maps, (radius, radius, 0, 0), mode="replicate")
    smoothed = F.conv2d(horizontal, kernel.view(1, 1, 1, -1))
    vertical = F.pad(smoothed, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(vertical, kernel.view(1, 1, -1, 1))


def upsample_maps(maps: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinearly resize ``(B, 1, H, W)`` anomaly maps to ``size x size``."""
    if maps.shape[-1] == size and maps.shape[-2] == size:
        return maps
    return F.interpolate(maps, size=(size, size), mode="bilinear", align_corners=False)
