"""PatchCore: memory-bank anomaly detection with greedy coreset subsampling."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import DataLoader

from .common import gaussian_blur_map, upsample_maps


def greedy_coreset(features: torch.Tensor, budget: int, seed: int) -> torch.Tensor:
    """Greedy k-center selection (O(budget * N) with incremental distances)."""
    features = features.float()
    count = len(features)
    if budget <= 0:
        raise ValueError(f"coreset budget must be positive, got {budget}")
    if budget >= count:
        return features.clone()
    generator = torch.Generator().manual_seed(seed)
    selected = torch.empty(budget, dtype=torch.long)
    selected[0] = int(torch.randint(count, (1,), generator=generator))
    sq_norms = (features * features).sum(dim=1)
    min_sq = (features - features[selected[0]]).pow(2).sum(dim=1)
    for index in range(1, budget):
        previous = int(selected[index - 1])
        distances = sq_norms + sq_norms[previous] - 2.0 * (features @ features[previous])
        min_sq = torch.minimum(min_sq, distances.clamp_min_(0.0))
        selected[index] = int(torch.argmax(min_sq))
    return features[selected].contiguous()


class PatchCoreModel:
    """PatchCore with random projection, coreset memory bank and kNN (k=1) scoring.

    Scoring is the maximum smoothed patch distance to the memory bank. The
    original score re-weighting term is intentionally omitted; this is
    documented as a simplification.
    """

    name = "patchcore"

    def __init__(
        self,
        extractor: torch.nn.Module,
        image_size: int = 224,
        projection_dim: int = 256,
        smooth_sigma: float = 4.0,
        seed: int = 42,
    ) -> None:
        self.extractor = extractor
        self.image_size = image_size
        self.projection_dim = projection_dim
        self.smooth_sigma = smooth_sigma
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        self.projection = torch.randn(
            extractor.output_dim, projection_dim, generator=generator
        ) / math.sqrt(projection_dim)
        self.bank: torch.Tensor | None = None
        self.device = torch.device("cpu")

    def to(self, device: torch.device | str) -> PatchCoreModel:
        self.device = torch.device(device)
        self.extractor.to(self.device)
        self.projection = self.projection.to(self.device)
        if self.bank is not None:
            self.bank = self.bank.to(self.device)
        return self

    def eval(self) -> PatchCoreModel:
        self.extractor.eval()
        return self

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bchw,cd->bdhw", features, self.projection)

    @torch.no_grad()
    def fit(
        self,
        loader: DataLoader,
        max_candidates: int = 15000,
        coreset_ratio: float = 0.01,
    ) -> dict[str, Any]:
        self.extractor.eval()
        chunks: list[torch.Tensor] = []
        total_patches = 0
        grid: tuple[int, int] | None = None
        for batch in loader:
            images = batch["image"] if isinstance(batch, dict) else batch
            features = self._project(self.extractor(images.to(self.device)))
            if grid is None:
                grid = (features.shape[-2], features.shape[-1])
            flat = features.flatten(2).transpose(1, 2).cpu()
            chunks.append(flat)
            total_patches += flat.shape[0] * flat.shape[1]
        candidates = torch.cat(chunks, dim=0).reshape(-1, self.projection_dim)
        del chunks
        if len(candidates) > max_candidates:
            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(len(candidates), generator=generator)[:max_candidates]
            candidates = candidates[indices]
        core_budget = min(len(candidates), max(1, int(round(coreset_ratio * total_patches))))
        self.bank = greedy_coreset(candidates, core_budget, self.seed).to(self.device)
        return {
            "total_patches_seen": total_patches,
            "candidate_patches": int(len(candidates)),
            "coreset_budget": core_budget,
            "memory_bank_size": int(self.bank.shape[0]),
            "projected_dim": int(self.projection_dim),
            "feature_grid": list(grid) if grid else None,
        }

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, out_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bank is None:
            raise RuntimeError("PatchCoreModel must be fitted before inference")
        features = self._project(self.extractor(images.to(self.device)))
        batch, _, height, width = features.shape
        flat = features.flatten(2).transpose(1, 2)
        minima: list[torch.Tensor] = []
        for chunk in torch.split(flat, 512, dim=1):
            distances = torch.cdist(chunk, self.bank)
            minima.append(distances.amin(dim=2))
        maps = torch.cat(minima, dim=1).view(batch, 1, height, width)
        target_size = int(out_size) if out_size is not None else int(self.image_size)
        maps = upsample_maps(maps, target_size)
        # Smooth at output resolution: a sigma expressed on the 28x28 feature
        # grid would become a very large blur after upsampling.
        maps = gaussian_blur_map(maps, self.smooth_sigma)
        scores = maps.flatten(1).amax(dim=1)
        return maps.cpu(), scores.cpu()

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self.bank is None:
            raise RuntimeError("Cannot export an unfitted PatchCoreModel")
        return {
            "projection": self.projection.detach().cpu().clone(),
            "bank": self.bank.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        projection = state["projection"]
        bank = state["bank"]
        if tuple(projection.shape) != tuple(self.projection.shape):
            raise ValueError(
                f"Projection shape mismatch: {tuple(projection.shape)} vs "
                f"{tuple(self.projection.shape)}"
            )
        if bank.ndim != 2 or bank.shape[1] != self.projection_dim:
            raise ValueError(f"Invalid memory bank shape: {tuple(bank.shape)}")
        self.projection = projection.float().to(self.device)
        self.bank = bank.float().to(self.device)

    def model_config(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "projection_dim": self.projection_dim,
            "smooth_sigma": self.smooth_sigma,
            "seed": self.seed,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            **self.model_config(),
            "scoring": "max smoothed kNN (k=1) patch distance to coreset memory bank",
            "memory_bank_size": int(self.bank.shape[0]) if self.bank is not None else 0,
        }
