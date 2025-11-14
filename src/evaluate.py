import torch
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path
import numpy as np
from typing import Union

from src.utils import load_normalization_stats
from src.data_module.dataset import NormalizedSubset  # reuse your existing class


class NPZDataset(Dataset):
    """Minimal dataset wrapper for NPZ files."""
    def __init__(self, npz_path: Union[str, Path], sample_rate: float = 1.0):
        data = np.load(npz_path, allow_pickle=True)
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)
        if self.y.ndim > 1:
            self.y = self.y.squeeze(-1)
        self.feat_names = data.get("feat_names", None)

        if sample_rate < 1.0:
            n_samples = self.X.shape[0]
            n_sampled = int(n_samples * sample_rate)
            indices = np.random.choice(n_samples, n_sampled, replace=False)
            self.X = self.X[indices]
            self.y = self.y[indices]
            print(f"[INFO] Sampled {n_sampled} / {n_samples} from {npz_path}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@torch.no_grad()
def evaluate_on_npz(
    trainer,
    npz_path: Union[str, Path],
    scaler_stats_path: Union[str, Path],
    sample_rate: float = 0.1,
    batch_size: int = 64,
    norm_method: str = "minmax",
    return_raw_outputs: bool = True,
):
    """
    Evaluate a trained model on a .npz test dataset.

    Args:
        trainer: your Trainer instance (must have model, device, loss_fn, metrics, evaluate()).
        npz_path: path to the test npz (with 'X' and 'y').
        scaler_stats_path: path to saved normalization stats (.npz).
        sample_rate: optional subsampling ratio inside NPZDataset.
        batch_size: DataLoader batch size.
        norm_method: either 'minmax' or 'standardize' for X (must match training).
        return_raw_outputs: passed through to trainer.evaluate.
    """

    # 1. Load test dataset (raw, unnormalized X and y)
    test_ds_raw = NPZDataset(npz_path, sample_rate=sample_rate)

    # 2. Load normalization stats (computed on train set)
    stats_all = load_normalization_stats(scaler_stats_path)
    stats_x = stats_all.get("x", None)
    stats_y = stats_all.get("y", None)

    # X normalization: check method
    if stats_x is None:
        raise ValueError(
            "No X normalization stats found in scaler_stats. "
            "Did you save stats with stats['x']?"
        )

    method_x = stats_x["method"]
    assert method_x == norm_method, (
        f"X normalization method mismatch: got {method_x} vs {norm_method}"
    )

    # Y normalization: optional
    normalize_target = stats_y is not None
    target_norm_method = None
    y_min = y_max = y_mean = y_std = None

    if normalize_target:
        target_norm_method = stats_y["method"]
        if target_norm_method == "minmax":
            y_min = stats_y["y_min"]
            y_max = stats_y["y_max"]
        elif target_norm_method == "standardize":
            y_mean = stats_y["y_mean"]
            y_std = stats_y["y_std"]
        else:
            raise ValueError(f"Unknown Y normalization method: {target_norm_method}")

    # 3. Wrap with NormalizedSubset using saved stats for X (and optionally Y)
    all_indices = list(range(len(test_ds_raw)))

    if method_x == "minmax":
        test_ds = NormalizedSubset(
            test_ds_raw,
            indices=all_indices,
            method="minmax",
            x_min=stats_x["x_min"],
            x_max=stats_x["x_max"],
            # Y normalization options
            normalize_target=normalize_target,
            target_norm_method=target_norm_method or "minmax",
            y_min=y_min,
            y_max=y_max,
            y_mean=y_mean,
            y_std=y_std,
        )
    else:  # "standardize"
        test_ds = NormalizedSubset(
            test_ds_raw,
            indices=all_indices,
            method="standardize",
            x_mean=stats_x["x_mean"],
            x_std=stats_x["x_std"],
            # Y normalization options
            normalize_target=normalize_target,
            target_norm_method=target_norm_method or "minmax",
            y_min=y_min,
            y_max=y_max,
            y_mean=y_mean,
            y_std=y_std,
        )

    # 4. Build DataLoader
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 5. Evaluate using your existing trainer
    out_dict = trainer.evaluate(test_loader, return_raw_outputs=return_raw_outputs)
    avg_loss = out_dict["loss"]
    metrics_result = out_dict["metrics"]

    # 6. Log & return
    print("\nTest Evaluation")
    print(f"Loss: {avg_loss:.6f}")
    for name, val in metrics_result.items():
        print(f"{name:>8s}: {val:.6f}")

    # Note:
    # - If stats_y is not None, loss/metrics are computed in normalized target space.
    # - You can inverse-transform preds/targets outside this function using stats_y.

    return out_dict

