import numpy as np
import copy
import torch
from pathlib import Path

from src.config import Config, get_experiment_dirs
from src.models import build_mlp
from src.metrics import DEFAULT_REGRESSION_METRICS
from src.trainer import Trainer, build_optimizer
from src.data_module.dataset import build_dataset, make_train_val_loaders, make_cv_loaders
from src.report import save_and_plot_cv_summary, save_and_plot_cv_folds
from src.utils import save_normalization_stats
from src.visu import plot_history

"""
Usage of this module:

from config import load_config
from experiment import run_experiment


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    result = run_experiment(cfg)
    print("\n=== Experiment result ===")
    print(result)


if __name__ == "__main__":
    main()


"""
def run_single_experiment(cfg: Config):
    """
    Standard train/val run using cfg.data.val_ratio.
    """
    train_loader, val_loader, norm_stats = make_train_val_loaders(
        cfg.data,
        normalize=True,
        norm_method="minmax",
        return_stats=True,
    )
    _, _, _, hparams_dir = get_experiment_dirs(cfg.logging, subdirs=("hparams",))
    if norm_stats is not None:
        save_normalization_stats(hparams_dir, norm_stats, name="scaler_stats")

    model = build_mlp(cfg.model)
    optimizer = build_optimizer(model.parameters(), cfg.optim)
    loss_fn = torch.nn.MSELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_cfg=cfg.train,
        logging_cfg=cfg.logging,
        checkpoint_cfg=cfg.checkpoint,
        metrics=DEFAULT_REGRESSION_METRICS,
    )

    history = trainer.fit(train_loader, val_loader)
    val_loss, val_metrics = trainer.evaluate(val_loader)

    return {
        "history": history,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
    }


def run_cv_experiment(cfg: Config):
    """
    K-fold cross-validation.

    - Uses the SAME dataset instance for all folds
    - For each fold, creates a fresh model/optimizer/trainer
    - Returns per-fold metrics + mean/std
    """
    n_splits = cfg.cv.n_splits
    assert n_splits >= 2, "run_cv_experiment requires cv.n_splits >= 2"
    base_exp_name = cfg.logging.experiment_name

    dataset = build_dataset(cfg.data)
    n_samples = len(dataset)
    indices = np.arange(n_samples)

    if cfg.cv.shuffle:
        rng = np.random.default_rng(cfg.cv.seed)
        rng.shuffle(indices)

    folds = np.array_split(indices, n_splits)

    fold_results = []

    for fold_idx in range(n_splits):
        cfg = copy.deepcopy(cfg)  # avoid overwriting base config
        val_idx = folds[fold_idx]
        train_idx = np.concatenate(
            [folds[i] for i in range(n_splits) if i != fold_idx]
        )

        train_loader, val_loader, norm_stats = make_cv_loaders(
            dataset,
            train_idx,
            val_idx,
            cfg.data,
            normalize=True,
            norm_method="minmax",
            return_stats=True,
        )
        _, _, _, hparams_dir = get_experiment_dirs(cfg.logging, subdirs=("hparams",))
        if norm_stats is not None:
            save_normalization_stats(hparams_dir, norm_stats, name="scaler_stats")

        model = build_mlp(cfg.model)
        optimizer = build_optimizer(model.parameters(), cfg.optim)
        loss_fn = torch.nn.MSELoss()

        # ========= overwrite experiment name to include fold =========
        fold_name = f"{base_exp_name}/folds/fold_{fold_idx:03d}"
        cfg.logging.experiment_name = fold_name
        # fold_exp_root, _, _, _ = get_experiment_dirs(cfg.logging)
        # =============================================================

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_cfg=cfg.train,
            logging_cfg=cfg.logging,
            checkpoint_cfg=cfg.checkpoint,
            metrics=DEFAULT_REGRESSION_METRICS,
        )

        print(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")
        trainer.fit(train_loader, val_loader)
        val_loss, val_metrics = trainer.evaluate(val_loader)

        print(f"Fold {fold_idx + 1} val_loss={val_loss:.4f} | "
              + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

        fold_results.append(
            {
                "fold": fold_idx,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }
        )

    # Aggregate metrics
    metric_names = fold_results[0]["val_metrics"].keys()
    summary = {"val_loss": {}, "metrics": {}}

    # loss
    losses = [fr["val_loss"] for fr in fold_results]
    summary["val_loss"]["mean"] = float(np.mean(losses))
    summary["val_loss"]["std"] = float(np.std(losses))

    # each metric
    for m in metric_names:
        vals = [fr["val_metrics"][m] for fr in fold_results]
        summary["metrics"][m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    return {
        "folds": fold_results,
        "summary": summary,
    }


def run_experiment(cfg: Config):
    """
    Entry point:
    - If cv.n_splits == 1 -> single train/val
    - If cv.n_splits >= 2 -> k-fold CV
    """
    if cfg.cv.n_splits <= 1:
        # run single train/val experiment
        result = run_single_experiment(cfg)

        # save the history plot
        print("\nFinal Validation Loss:", result["val_loss"])
        print("Final Validation Metrics:", result["val_metrics"])
        _, _, logs_dir, _ = get_experiment_dirs(cfg.logging, create=True, subdirs=("logs",))
        fig = plot_history(result["history"], savepath=Path(logs_dir) / "training_history.png")

    else:
        result =  run_cv_experiment(cfg)

        # save per-fold logging and plots
        exp_root, *_ = get_experiment_dirs(cfg.logging, create=False)
        save_and_plot_cv_folds(result["folds"], exp_root) # result["folds"] is List[Dict]
        
        # save and plot CV summary
        save_and_plot_cv_summary(result["summary"], exp_root) # result["summary"] is Dict
    
    return result