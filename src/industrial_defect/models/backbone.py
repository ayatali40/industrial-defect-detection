"""Pretrained ResNet feature extractor used by PatchCore and PaDiM."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

SUPPORTED_LAYERS = ("layer2", "layer3")
_LAYER_CHANNELS = {"layer2": 512, "layer3": 1024}


class FeatureExtractor(nn.Module):
    """Frozen ImageNet ResNet-50 producing a concatenated mid-level feature map.

    ``layer2`` (stride 8) and ``layer3`` (stride 16, bilinearly upsampled) are
    concatenated after optional 3x3 local neighbourhood averaging, following the
    PatchCore/PaDiM feature design.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        layers: Sequence[str] = ("layer2", "layer3"),
        local_pooling: bool = True,
    ) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError(f"Unsupported backbone '{backbone}'; only 'resnet50' is implemented")
        self.layers = tuple(layers)
        unknown = set(self.layers) - set(SUPPORTED_LAYERS)
        if unknown or not self.layers:
            raise ValueError(f"Unsupported layers {sorted(unknown)}; use {SUPPORTED_LAYERS}")
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.pretrained = bool(pretrained)
        try:
            model = resnet50(weights=weights)
        except Exception as exc:  # pragma: no cover - depends on local weight cache
            raise RuntimeError(
                "Could not load ImageNet ResNet-50 weights. The first run needs the pretrained "
                "weights to be downloadable (or present in the local torch cache). "
                f"Original error: {exc}"
            ) from exc
        self.stem = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
        )
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.local_pooling = local_pooling
        self._neighborhood = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @property
    def output_dim(self) -> int:
        return sum(_LAYER_CHANNELS[name] for name in self.layers)

    def train(self, mode: bool = True) -> FeatureExtractor:  # noqa: ARG002 - frozen extractor
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        layer_outputs: dict[str, torch.Tensor] = {}
        features = self.layer2(features)
        layer_outputs["layer2"] = features
        if "layer3" in self.layers:
            features = self.layer3(features)
            layer_outputs["layer3"] = features
        target_size = layer_outputs["layer2"].shape[-2:]
        merged: list[torch.Tensor] = []
        for name in self.layers:
            tensor = layer_outputs[name]
            if self.local_pooling:
                tensor = self._neighborhood(tensor)
            if tensor.shape[-2:] != target_size:
                tensor = F.interpolate(
                    tensor, size=target_size, mode="bilinear", align_corners=False
                )
            merged.append(tensor)
        return torch.cat(merged, dim=1)

    def metadata(self) -> dict[str, object]:
        return {
            "name": "resnet50",
            "pretrained": self.pretrained,
            "layers": list(self.layers),
            "local_pooling": self.local_pooling,
            "output_dim": self.output_dim,
        }
