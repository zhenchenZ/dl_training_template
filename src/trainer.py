import os
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple

import torch

from src.config import TrainConfig, OptimConfig, CheckpointConfig, LoggingConfig, get_experiment_dirs
from src.metrics import DEFAULT_REGRESSION_METRICS


def build_optimizer(params, cfg: OptimConfig) -> torch.optim.Optimizer:
    name = cfg.name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.name}")


class Trainer:
    """
    Minimal trainer + checkpointing.

    Uses:
      - TrainConfig for loop behavior
      - CheckpointConfig for saving
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        train_cfg: TrainConfig,
        logging_cfg: LoggingConfig,
        checkpoint_cfg: Optional[CheckpointConfig] = None,
        metrics: Optional[Dict[str, Callable]] = None,
    ):
        self.train_cfg = train_cfg
        self.checkpoint_cfg = checkpoint_cfg
        self.logging_cfg = logging_cfg

        if train_cfg.device:
            self.device = train_cfg.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\n*** Using device: {self.device} ***\n")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or {}

        # directories
        _, self.ckpt_dir, _, _ = get_experiment_dirs(logging_cfg, create=True, subdirs=("checkpoints",))

        # checkpoint state
        self.best_value: Optional[float] = None


    def _run_epoch(self, dataloader, train: bool = True) -> Tuple[float, Dict[str, float]]:
        self.model.train(train)

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        # iteration to exhaust current epoch
        for xb, yb in dataloader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            if train:
                self.optimizer.zero_grad()
            preds = self.model(xb)
            loss = self.loss_fn(preds, yb)

            if train:
                loss.backward()
                self.optimizer.step()

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(preds.detach())
            all_targets.append(yb.detach())

        avg_loss = total_loss / max(total_samples, 1)

        # evaluation at the end of the epoch
        metric_vals = {}
        if self.metrics and all_preds:
            preds_cat = torch.cat(all_preds, dim=0)
            targets_cat = torch.cat(all_targets, dim=0)
            for name, fn in self.metrics.items():
                metric_vals[name] = fn(preds_cat, targets_cat)

        return avg_loss, metric_vals

    def _maybe_save_checkpoint(
        self,
        epoch: int,
        val_loss: Optional[float],
        val_metrics: Dict[str, float],
    ):
        if self.checkpoint_cfg is None:
            return

        monitor = self.checkpoint_cfg.monitor
        mode = self.checkpoint_cfg.mode

        # Decide which value to monitor
        if monitor == "val_loss":
            value = val_loss
        else:
            # expect monitor like "val_r2"
            value = val_metrics.get(monitor, None)

        if value is None:
            return

        # Save last if requested
        if self.checkpoint_cfg.save_last:
            path = os.path.join(self.ckpt_dir, "last.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                },
                path,
            )

        # Save best if requested
        if not self.checkpoint_cfg.save_best:
            return

        if self.best_value is None:
            is_better = True
        elif mode == "min":
            is_better = value < self.best_value
        elif mode == "max":
            is_better = value > self.best_value
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if is_better:
            self.best_value = value
            path = os.path.join(self.ckpt_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "monitor": monitor,
                    "best_value": self.best_value,
                },
                path,
            )

    def fit(self, train_loader, val_loader=None):
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }
        
        for epoch in range(1, self.train_cfg.num_epochs + 1):
            train_loss, train_metrics = self._run_epoch(train_loader, train=True)
            history["train_loss"].append(train_loss)
            history["train_metrics"].append(train_metrics)

            if val_loader is not None:
                val_loss, val_metrics = self._run_epoch(val_loader, train=False)
                history["val_loss"].append(val_loss)
                history["val_metrics"].append(val_metrics)

                self._maybe_save_checkpoint(epoch, val_loss, val_metrics)
            else:
                val_loss, val_metrics = None, {}

            # logging
            if epoch % self.train_cfg.log_interval == 0:
                msg = f"[Epoch {epoch}] train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | val_loss={val_loss:.4f}"
                if train_metrics:
                    msg += " | " + ", ".join(
                        f"train_{k}={v:.4f}" for k, v in train_metrics.items()
                    )
                if val_metrics:
                    msg += " | " + ", ".join(
                        f"val_{k}={v:.4f}" for k, v in val_metrics.items()
                    )
                print(msg)

        return history

    @torch.no_grad()
    def evaluate(self, dataloader, return_raw_outputs=False):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        for xb, yb in dataloader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            preds = self.model(xb)
            loss = self.loss_fn(preds, yb)

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(preds.detach())
            all_targets.append(yb.detach())

        avg_loss = total_loss / max(total_samples, 1)

        metrics_result = {}
        if self.metrics and all_preds:
            preds_cat = torch.cat(all_preds, dim=0)
            targets_cat = torch.cat(all_targets, dim=0)
            for name, fn in self.metrics.items():
                metrics_result[name] = fn(preds_cat, targets_cat)

        out = {
            "loss": avg_loss,
            "metrics": metrics_result,
        }
        if return_raw_outputs:
            out.update({
                "preds": torch.cat(all_preds, dim=0),
                "targets": torch.cat(all_targets, dim=0),
            })
            return out
        else:
            return out
