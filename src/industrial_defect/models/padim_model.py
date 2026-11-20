"""PaDiM: per-patch-position Gaussian modelling with Mahalanobis scoring."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from .common import gaussian_blur_map, upsample_maps


class PaDiMModel:
    """PaDiM with random feature-dimension selection and shared positional Gaussians.

    A single multivariate Gaussian is fitted per spatial position over the
    concatenated layer2/layer3 features. Distances are standardised with the
    mean/std of training distances per position before smoothing.
    """

    name = "padim"

    def __init__(
        self,
        extractor: torch.nn.Module,
        image_size: int = 224,
        n_features: int = 150,
        covariance_epsilon: float = 0.01,
        smooth_sigma: float = 4.0,
        seed: int = 42,
    ) -> None:
        self.extractor = extractor
        self.image_size = image_size
        self.n_features = n_features
        self.covariance_epsilon = covariance_epsilon
        self.smooth_sigma = smooth_sigma
        self.seed = seed
        self.feature_dim = int(extractor.output_dim)
        if n_features > self.feature_dim:
            raise ValueError(
                f"n_features={n_features} exceeds backbone feature dimension {self.feature_dim}"
            )
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(self.feature_dim, generator=generator)[:n_features]
        self.feature_indices = indices.sort().values
        self.mean: torch.Tensor | None = None
        self.cov_inv: torch.Tensor | None = None
        self.train_mean: torch.Tensor | None = None
        self.train_std: torch.Tensor | None = None
        self.device = torch.device("cpu")

    def to(self, device: torch.device | str) -> PaDiMModel:
        self.device = torch.device(device)
        self.extractor.to(self.device)
        for name in ("mean", "cov_inv", "train_mean", "train_std"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(self.device))
        self.feature_indices = self.feature_indices.to(self.device)
        return self

    def eval(self) -> PaDiMModel:
        self.extractor.eval()
        return self

    def _select_features(self, features: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, D, H*W) with the fixed random feature subset."""
        flat = features.flatten(2)
        indices = self.feature_indices.to(flat.device)
        return flat[:, indices, :]

    @torch.no_grad()
    def fit(self, loader: DataLoader) -> dict[str, Any]:
        self.extractor.eval()
        collected: list[torch.Tensor] = []
        grid: tuple[int, int] | None = None
        for batch in loader:
            images = batch["image"] if isinstance(batch, dict) else batch
            features = self.extractor(images.to(self.device))
            if grid is None:
                grid = (features.shape[-2], features.shape[-1])
            selected = self._select_features(features)
            collected.append(selected.transpose(1, 2).cpu())
        data = torch.cat(collected, dim=0)  # (N, P, D)
        n_images, positions, dims = data.shape
        if n_images < 2:
            raise ValueError("PaDiM requires at least two normal training images")
        mean = data.mean(dim=0)
        centered = data - mean
        covariance = torch.einsum("npd,npe->pde", centered, centered) / (n_images - 1)
        identity = torch.eye(dims, dtype=covariance.dtype).expand_as(covariance)
        covariance = covariance + self.covariance_epsilon * identity
        # Batched torch.linalg.inv triggers an Intel MKL failure on this platform;
        # the covariance is symmetric positive definite, so invert via Cholesky.
        cov_inv = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
        self.mean = mean.to(self.device)
        self.cov_inv = cov_inv.to(self.device)

        distances = [self._mahalanobis(chunk.to(self.device)).cpu() for chunk in data.split(64)]
        train_distances = torch.cat(distances, dim=0)
        self.train_mean = train_distances.mean(dim=0).to(self.device)
        self.train_std = train_distances.std(dim=0).clamp_min(1e-6).to(self.device)
        del collected, data, centered, covariance
        return {
            "fit_images": n_images,
            "positions": positions,
            "n_features": dims,
            "feature_grid": list(grid) if grid else None,
            "covariance_epsilon": self.covariance_epsilon,
            "train_distance_mean": float(self.train_mean.mean().item()),
            "train_distance_std": float(self.train_std.mean().item()),
        }

    def _mahalanobis(self, features: torch.Tensor) -> torch.Tensor:
        """Mahalanobis distance per position, chunked to bound memory."""
        if self.mean is None or self.cov_inv is None:
            raise RuntimeError("PaDiMModel must be fitted before inference")
        difference = features - self.mean
        batch, positions, _ = difference.shape
        output = torch.empty(batch, positions, dtype=features.dtype, device=features.device)
        for start in range(0, positions, 64):
            end = min(start + 64, positions)
            block = difference[:, start:end, :]
            solved = torch.einsum("pde,bpe->bpd", self.cov_inv[start:end], block)
            output[:, start:end] = (block * solved).sum(dim=-1)
        return output.clamp_min_(0.0)

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, out_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train_mean is None or self.train_std is None:
            raise RuntimeError("PaDiMModel must be fitted before inference")
        features = self.extractor(images.to(self.device))
        batch, _, height, width = features.shape
        selected = self._select_features(features).transpose(1, 2)  # (B, P, D)
        distances = self._mahalanobis(selected)
        standardized = (distances - self.train_mean) / self.train_std
        maps = standardized.view(batch, 1, height, width)
        target_size = int(out_size) if out_size is not None else int(self.image_size)
        maps = upsample_maps(maps, target_size)
        # Smooth at output resolution; sigma is defined in output pixels.
        maps = gaussian_blur_map(maps, self.smooth_sigma)
        scores = maps.flatten(1).amax(dim=1)
        return maps.cpu(), scores.cpu()

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self.mean is None or self.cov_inv is None:
            raise RuntimeError("Cannot export an unfitted PaDiMModel")
        assert self.train_mean is not None and self.train_std is not None
        return {
            "mean": self.mean.detach().cpu().clone(),
            "cov_inv": self.cov_inv.detach().cpu().clone(),
            "train_mean": self.train_mean.detach().cpu().clone(),
            "train_std": self.train_std.detach().cpu().clone(),
            "feature_indices": self.feature_indices.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        mean = state["mean"]
        cov_inv = state["cov_inv"]
        if mean.ndim != 2 or mean.shape[1] != self.n_features:
            raise ValueError(f"Invalid PaDiM mean shape: {tuple(mean.shape)}")
        if cov_inv.shape != (mean.shape[0], self.n_features, self.n_features):
            raise ValueError(f"Invalid PaDiM covariance shape: {tuple(cov_inv.shape)}")
        self.mean = mean.float().to(self.device)
        self.cov_inv = cov_inv.float().to(self.device)
        self.train_mean = state["train_mean"].float().to(self.device)
        self.train_std = state["train_std"].float().to(self.device)
        self.feature_indices = state["feature_indices"].long().to(self.device)

    def model_config(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "n_features": self.n_features,
            "covariance_epsilon": self.covariance_epsilon,
            "smooth_sigma": self.smooth_sigma,
            "seed": self.seed,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            **self.model_config(),
            "scoring": "max smoothed, position-standardised Mahalanobis distance",
            "positions": int(self.mean.shape[0]) if self.mean is not None else 0,
        }
