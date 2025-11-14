import os
import argparse
from pathlib import Path
from time import time

from src.config import load_config, get_experiment_dirs
from src.sweep_config import load_sweep_config
from src.sweep import random_search
from src.visu import plot_histories_overlay
from src.report import save_sweep_best_summary, save_and_plot_sweep_trials_metrics


if __name__ == "__main__":
    """
    Example usage:
python hyper_param_sweep.py --base_config config/base_config_logistic.yaml --sweep_config config/sweep_config_logistic.yaml --exp_name 001-sweep_experiment-logistic
    """
    t0_glb = time()

    parser = argparse.ArgumentParser(description="Hyperparameter Sweep")
    parser.add_argument("-b", "--base_config", type=str, default="config/base_config.yaml", help="Path to base config file")
    parser.add_argument("-s", "--sweep_config", type=str, default="config/sweep_config.yaml", help="Path to sweep config file")
    parser.add_argument("--exp_name", type=str, default=None, help="Optional experiment name to override the one in config")
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    if args.exp_name is not None:
        base_cfg.logging.experiment_name = args.exp_name
    sweep_cfg = load_sweep_config(args.sweep_config)
    do_cv_in_sweep = base_cfg.cv.n_splits > 1

    # ============================================
    # Run Random Search
    # ============================================
    result = random_search(
        base_cfg=base_cfg,
        sweep_cfg=sweep_cfg
    )

    # ============================================
    # Save Summary of Best Trial
    # ============================================
    exp_root, *_ = get_experiment_dirs(base_cfg.logging)
    save_sweep_best_summary(result, monitor_metric=sweep_cfg.monitor, exp_root=exp_root)

    # ============================================
    # Plot Histories Overlay (not available in Cross-Validation)
    # ============================================
    if not do_cv_in_sweep:
        histories = [
            result['trials'][i]['result']['history']
            for i in range(len(result['trials']))
        ]
        
        _, _, logs_dir, _ = get_experiment_dirs(base_cfg.logging, create=False)   
        fig = plot_histories_overlay(
            histories,
            labels=[f"cfg {i+1}" for i in range(len(histories))],
            metrics_to_plot=None,             # or ["rmse", "r2"]
            suptitle="Comparison of configs",
            n_cols=3,
            savepath=Path(logs_dir) / "sweep_histories.png",
        )

    # ============================================
    # Plot comparison of different trials' best val metrics
    # ============================================
    if not do_cv_in_sweep:
        trials_metrics = [
            result['trials'][i]['result']['val_metrics']
            for i in range(len(result['trials']))
        ]
    save_and_plot_sweep_trials_metrics(trials_metrics, exp_root, monitor_metric=sweep_cfg.monitor)

    t1_glb = time()
    print(f"\nDONE. Total sweep time: {t1_glb - t0_glb:.1f} seconds")
    