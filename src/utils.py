import torch
import numpy as np
from pathlib import Path
from typing import Dict, Union

from src.config import get_experiment_dirs


def print_tensor(x, name):
    if isinstance(x, torch.Tensor):
        x = x.float()            # ensure float for mean/std
    elif isinstance(x, np.ndarray):
        x = x.astype(np.float32)
    else:
        raise ValueError(f"unsupported type of x: {type(x)} ") 
    
    print(f"{name:<15}: shape: {x.shape}, min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}, std: {x.std():.4f}")


def count_params(model: torch.nn.Module):
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_trainable} trainable / {n_total} total")


def save_normalization_stats(
    hparams_dir: Union[str, Path],
    stats: Dict,
    name: str = "scaler_stats",
) -> Path:
    """
    Save normalization statistics (min/max or mean/std) to:
        <exp_root>/hparams/<name>.npz

    Args:
        exp_root: experiment root directory (Path or str).
        stats: dict returned by _compute_normalization_stats, e.g.:
               {
                 "method": "minmax",
                 "x_min": torch.Tensor(D),
                 "x_max": torch.Tensor(D),
               }
               or
               {
                 "method": "standardize",
                 "x_mean": torch.Tensor(D),
                 "x_std": torch.Tensor(D),
               }
        name: base filename (without extension).

    Returns:
        Path to the saved .npz file.
    """
    hparams_dir = Path(hparams_dir)

    method = stats.get("method")
    if method not in ("minmax", "standardize"):
        raise ValueError(f"Unknown normalization method in stats: {method}")

    payload = {"method": method}

    if method == "minmax":
        payload["x_min"] = stats["x_min"].detach().cpu().numpy()
        payload["x_max"] = stats["x_max"].detach().cpu().numpy()
    else:  # standardize
        payload["x_mean"] = stats["x_mean"].detach().cpu().numpy()
        payload["x_std"] = stats["x_std"].detach().cpu().numpy()

    out_path = hparams_dir / f"{name}.npz"
    np.savez(out_path, **payload)

    print(f"[INFO] Saved normalization stats to: {out_path}")
    return out_path


def load_normalization_stats(path: Union[str, Path]) -> Dict:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    method = str(data["method"])
    stats = {"method": method}
    if method == "minmax":
        stats["x_min"] = torch.tensor(data["x_min"], dtype=torch.float32)
        stats["x_max"] = torch.tensor(data["x_max"], dtype=torch.float32)
    else:
        stats["x_mean"] = torch.tensor(data["x_mean"], dtype=torch.float32)
        stats["x_std"] = torch.tensor(data["x_std"], dtype=torch.float32)
    return stats
