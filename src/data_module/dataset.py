import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from typing import Optional, Sequence, Literal
from pathlib import Path

from src.config import DataConfig



class ToyRegressionDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        g = torch.Generator().manual_seed(cfg.seed)

        self.X = torch.randn(cfg.n_samples, 7, generator=g)

        w = torch.tensor([0.5, -1.2, 0.3, 0.8, -0.7, 1.5, 0.2])
        y = (
            (self.X @ w)
            + 0.5 * self.X[:, 0] ** 2
            - 0.3 * torch.sin(self.X[:, 1])
        )

        noise = cfg.noise_std * torch.randn(cfg.n_samples, generator=g)
        self.y = y + noise

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RealRegressionDataset(Dataset):
    """
    Loads preprocessed data from an .npz file with:
      - X: (N, D)
      - y: (N,) or (N, 1)
      - feat_names: (D,) optional
    """
    def __init__(self, npz_path: str):
        npz_path = Path(npz_path)
        if not npz_path.is_file():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        if "X" not in data or "y" not in data:
            raise ValueError("NPZ must contain 'X' and 'y' arrays.")

        X = data["X"]
        y = data["y"]

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        
        if y.ndim > 1:
            y = y.squeeze(-1)  # ensure (N,)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D after squeezing, got shape {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same length, got {X.shape[0]} vs {y.shape[0]}")

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

        self.feat_names = None
        if "feat_names" in data:
            self.feat_names = [str(f) for f in data["feat_names"]]

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NormalizedSubset(Dataset):
    """
    Wraps a base dataset and a set of indices, applies fixed normalization
    to X using provided stats. y is left unchanged.

    This is crucial for:
      - computing stats only on train
      - applying the same transform to val/test without leakage
    """
    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        method: Literal["minmax", "standardize"],
        x_min: Optional[torch.Tensor] = None,
        x_max: Optional[torch.Tensor] = None,
        x_mean: Optional[torch.Tensor] = None,
        x_std: Optional[torch.Tensor] = None,
    ):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.method = method

        # Store stats as 1D tensors
        self.x_min = x_min
        self.x_max = x_max
        self.x_mean = x_mean
        self.x_std = x_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        x, y = self.base_dataset[base_idx]  # assume x,y are tensors

        if self.method == "minmax":
            x = (x - self.x_min) / (self.x_max - self.x_min + 1e-12)
        elif self.method == "standardize":
            x = (x - self.x_mean) / (self.x_std + 1e-12)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        return x, y



def build_dataset(cfg: DataConfig) -> Dataset:
    if cfg.dataset == "toy":
        return ToyRegressionDataset(cfg)
    elif cfg.dataset == "real":
        if not cfg.npz_path:
            raise ValueError("DataConfig.npz_path must be set for dataset='real'.")
        return RealRegressionDataset(cfg.npz_path)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


def _compute_normalization_stats(
    dataset: Dataset,
    indices: Sequence[int],
    method: Literal["minmax", "standardize"] = "minmax",
):
    """
    Compute feature-wise normalization stats from a subset of indices.
    Assumes dataset[ i ][0] returns a 1D tensor of features.
    """
    if hasattr(dataset, "X"):
        # Fast path: use underlying tensor if exposed (Toy + Real datasets do)
        X = dataset.X[indices]  # (N_train, D)
    else:
        # Generic fallback: stack from __getitem__
        xs = [dataset[i][0] for i in indices]
        X = torch.stack(xs, dim=0)

    if method == "minmax":
        x_min = X.min(dim=0).values
        x_max = X.max(dim=0).values
        return {"method": method, "x_min": x_min, "x_max": x_max}
    elif method == "standardize":
        x_mean = X.mean(dim=0)
        x_std = X.std(dim=0)
        # avoid zero std
        x_std = torch.where(x_std == 0, torch.ones_like(x_std), x_std)
        return {"method": method, "x_mean": x_mean, "x_std": x_std}
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def make_train_val_loaders(
    cfg: DataConfig,
    normalize: bool = True,
    norm_method: str = "minmax",
    return_stats: bool = False,
):
    dataset = build_dataset(cfg)
    n_total = len(dataset)
    n_val = int(n_total * cfg.val_ratio)
    n_train = n_total - n_val

    train_subset, val_subset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    stats = None

    if normalize:
        train_indices = train_subset.indices
        val_indices = val_subset.indices

        stats = _compute_normalization_stats(dataset, train_indices, method=norm_method)

        if norm_method == "minmax":
            train_ds = NormalizedSubset(
                dataset, train_indices,
                method="minmax",
                x_min=stats["x_min"],
                x_max=stats["x_max"],
            )
            val_ds = NormalizedSubset(
                dataset, val_indices,
                method="minmax",
                x_min=stats["x_min"],
                x_max=stats["x_max"],
            )
        else:
            train_ds = NormalizedSubset(
                dataset, train_indices,
                method="standardize",
                x_mean=stats["x_mean"],
                x_std=stats["x_std"],
            )
            val_ds = NormalizedSubset(
                dataset, val_indices,
                method="standardize",
                x_mean=stats["x_mean"],
                x_std=stats["x_std"],
            )
    else:
        train_ds = train_subset
        val_ds = val_subset

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    if return_stats:
        return train_loader, val_loader, stats
    return train_loader, val_loader

def make_cv_loaders(
    dataset: Dataset,
    train_idx,
    val_idx,
    cfg: DataConfig,
    normalize: bool = True,
    norm_method: str = "minmax",
    return_stats: bool = False,
):
    stats = None

    if normalize:
        stats = _compute_normalization_stats(dataset, train_idx, method=norm_method)

        if norm_method == "minmax":
            train_ds = NormalizedSubset(
                dataset, train_idx,
                method="minmax",
                x_min=stats["x_min"],
                x_max=stats["x_max"],
            )
            val_ds = NormalizedSubset(
                dataset, val_idx,
                method="minmax",
                x_min=stats["x_min"],
                x_max=stats["x_max"],
            )
        else:
            train_ds = NormalizedSubset(
                dataset, train_idx,
                method="standardize",
                x_mean=stats["x_mean"],
                x_std=stats["x_std"],
            )
            val_ds = NormalizedSubset(
                dataset, val_idx,
                method="standardize",
                x_mean=stats["x_mean"],
                x_std=stats["x_std"],
            )
    else:
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    if return_stats:
        return train_loader, val_loader, stats
    return train_loader, val_loader
