import torch
import numpy as np
from pathlib import Path
from typing import Dict, Union, Any

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
    Save normalization statistics for X (and optionally Y) to:
        <hparams_dir>/<name>.npz

    Expected stats structure:
        stats = {
            "x": {
                "method": "minmax" or "standardize",
                "x_min": Tensor(D), "x_max": Tensor(D),       # if minmax
                "x_mean": Tensor(D), "x_std": Tensor(D),      # if standardize
            } or None,
            "y": {
                "method": "minmax" or "standardize",
                "y_min": float, "y_max": float,               # if minmax
                "y_mean": float, "y_std": float,              # if standardize
            } or None,
        }
    """
    hparams_dir = Path(hparams_dir)

    stats_x = stats.get("x", None)
    stats_y = stats.get("y", None)

    payload = {}

    # ---------------- X stats ----------------
    if stats_x is not None:
        method_x = stats_x.get("method")
        if method_x not in ("minmax", "standardize"):
            raise ValueError(f"Unknown X normalization method: {method_x}")
        payload["x_method"] = method_x

        if method_x == "minmax":
            payload["x_min"] = stats_x["x_min"].detach().cpu().numpy()
            payload["x_max"] = stats_x["x_max"].detach().cpu().numpy()
        else:
            payload["x_mean"] = stats_x["x_mean"].detach().cpu().numpy()
            payload["x_std"] = stats_x["x_std"].detach().cpu().numpy()
    else:
        payload["x_method"] = "none"

    # ---------------- Y stats ----------------
    if stats_y is not None:
        method_y = stats_y.get("method")
        if method_y not in ("minmax", "standardize"):
            raise ValueError(f"Unknown Y normalization method: {method_y}")
        payload["y_method"] = method_y
        payload["normalize_target"] = True

        if method_y == "minmax":
            payload["y_min"] = float(stats_y["y_min"])
            payload["y_max"] = float(stats_y["y_max"])
        else:
            payload["y_mean"] = float(stats_y["y_mean"])
            payload["y_std"] = float(stats_y["y_std"])
    else:
        payload["y_method"] = "none"
        payload["normalize_target"] = False

    out_path = hparams_dir / f"{name}.npz"
    np.savez(out_path, **payload)

    print(f"[INFO] Saved normalization stats to: {out_path}")
    return out_path


def load_normalization_stats(path: Union[str, Path]) -> Dict:
    """
    Load normalization stats saved by save_normalization_stats().

    Returns:
        {
          "x": {
             "method": "minmax"/"standardize",
             "x_min"/"x_max" or "x_mean"/"x_std",
          } or None,
          "y": {
             "method": "minmax"/"standardize",
             "y_min"/"y_max" or "y_mean"/"y_std",
          } or None,
        }
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    stats_x = None
    stats_y = None

    # ---------- X ----------
    x_method = str(data.get("x_method", "none"))
    if x_method in ("minmax", "standardize"):
        stats_x = {"method": x_method}
        if x_method == "minmax":
            stats_x["x_min"] = torch.tensor(data["x_min"], dtype=torch.float32)
            stats_x["x_max"] = torch.tensor(data["x_max"], dtype=torch.float32)
        else:
            stats_x["x_mean"] = torch.tensor(data["x_mean"], dtype=torch.float32)
            stats_x["x_std"] = torch.tensor(data["x_std"], dtype=torch.float32)

    # ---------- Y ----------
    normalize_target = bool(data.get("normalize_target", False))
    y_method = str(data.get("y_method", "none"))

    if normalize_target and y_method in ("minmax", "standardize"):
        stats_y = {"method": y_method}
        if y_method == "minmax":
            stats_y["y_min"] = float(data["y_min"])
            stats_y["y_max"] = float(data["y_max"])
        else:
            stats_y["y_mean"] = float(data["y_mean"])
            stats_y["y_std"] = float(data["y_std"])

    return {"x": stats_x, "y": stats_y}

