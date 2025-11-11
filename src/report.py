import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import numpy as np
import yaml
import matplotlib.pyplot as plt


def save_and_plot_cv_summary(
    summary: Dict[str, Any],
    exp_root: Union[str, Path],
    filename: str = "cv_sweep_summary.txt",
    show: bool = False,
):
    """
    Given a CV summary dict and an experiment root dir:

    1. Build an easy-to-read text summary.
    2. Print it.
    3. Save it to `exp_root / filename`.
    4. Plot mean ± std for val_loss and metrics:
         - One subplot for loss-like metrics (val_loss, mse, mae, rmse, etc.).
         - One subplot for r2 (if present), with [0, 1]-ish range.

    Expected summary format:
    {
      "val_loss": {"mean": float, "std": float},
      "metrics": {
        "<name>": {"mean": float, "std": float},
        ...
      }
    }
    """
    exp_root = Path(exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)
    out_path = exp_root / filename

    # ---------- 1 & 2. Build and print summary text ----------
    val_loss_mean = summary["val_loss"]["mean"]
    val_loss_std = summary["val_loss"]["std"]

    lines = []
    lines.append("==================================")
    lines.append("Cross-Validation / Sweep Summary")
    lines.append("==================================")
    lines.append(f"val_loss: mean = {val_loss_mean:.6f}, std = {val_loss_std:.6f}")
    lines.append("")
    lines.append("Metrics:")
    metrics = summary.get("metrics", {})
    for name, stats in metrics.items():
        m = stats.get("mean", None)
        s = stats.get("std", None)
        if m is not None and s is not None:
            lines.append(f"  {name:>6}: mean = {m:.6f}, std = {s:.6f}")
        else:
            lines.append(f"  {name:>6}: {json.dumps(stats)}")

    summary_str = "\n".join(lines)

    print("\n" + summary_str)

    # ---------- 3. Save to file ----------
    out_path.write_text(summary_str, encoding="utf-8")
    print(f"\n[INFO] Summary saved to: {out_path}")

    # ---------- 4. Plot mean ± std ----------
    # Collect stats: treat val_loss as a metric named "val_loss"
    all_stats = {"val_loss": summary["val_loss"]}
    all_stats.update(metrics)

    # Separate r2 (scale ~[0,1]) from others (loss-like)
    loss_like_names = []
    loss_like_means = []
    loss_like_stds = []

    r2_mean = None
    r2_std = None

    for name, stats in all_stats.items():
        mean = float(stats["mean"])
        std = float(stats["std"])

        if name.lower() == "r2":
            r2_mean = mean
            r2_std = std
        else:
            loss_like_names.append(name)
            loss_like_means.append(mean)
            loss_like_stds.append(std)

    # Decide subplot layout
    has_r2 = r2_mean is not None
    n_rows = 2 if has_r2 else 1

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(8, 4 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    # ---- 4a. Loss-like metrics subplot ----
    ax0 = axes[0]
    if loss_like_names:
        x = np.arange(len(loss_like_names))
        ax0.bar(
            x,
            loss_like_means,
            yerr=loss_like_stds,
            capsize=5,
        )
        ax0.set_xticks(x)
        ax0.set_xticklabels(loss_like_names, rotation=15)
        ax0.set_ylabel("Value")
        ax0.set_title("Mean ± std of val_loss and metrics")

        # Nice y-limits: [min(mean-std), max(mean+std)] with a bit of padding
        lows = [m - s for m, s in zip(loss_like_means, loss_like_stds)]
        highs = [m + s for m, s in zip(loss_like_means, loss_like_stds)]
        y_min = min(lows)
        y_max = max(highs)
        if y_min == y_max:
            # Degenerate case: expand a little
            y_min -= 0.1 * abs(y_min) if y_min != 0 else -0.1
            y_max += 0.1 * abs(y_max) if y_max != 0 else 0.1
        pad = 0.1 * (y_max - y_min)
        ax0.set_ylim(y_min - pad, y_max + pad)
        ax0.grid(True, axis="y", alpha=0.3)

    else:
        ax0.text(0.5, 0.5, "No loss-like metrics to plot", ha="center", va="center")
        ax0.axis("off")

    # ---- 4b. r2 subplot (if present) ----
    if has_r2:
        ax1 = axes[1]
        x = np.array([0])
        ax1.bar(
            x,
            [r2_mean],
            yerr=[r2_std],
            capsize=5,
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(["r2"])
        ax1.set_ylabel("r2")
        ax1.set_title("Mean ± std of r2")

        # r2 is usually in [0,1]; we adapt bounds but keep it nice
        lo = r2_mean - 3 * r2_std
        hi = r2_mean + 3 * r2_std
        lo = max(0.0, lo)
        hi = min(1.0, hi if hi > lo else r2_mean + 0.05)

        # Ensure a bit of padding
        pad = max(0.02, 0.1 * (hi - lo)) if hi > lo else 0.05
        ax1.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))

        ax1.grid(True, axis="y", alpha=0.3)

    if show:
        plt.show()
    else:
        # Save figure next to summary if not showing
        fig_path = exp_root / "cv_sweep_summary.png"
        fig.savefig(fig_path, dpi=150)
        print(f"[INFO] Summary plot saved to: {fig_path}")

    return summary_str


def save_and_plot_cv_folds(
    folds: List[Dict[str, Any]],
    exp_root: Union[str, Path],
    json_filename: str = "cv_folds_results.json",
    fig_prefix: str = "cv_folds_",  # each metric: <fig_prefix><name>.png
    show: bool = False,
):
    """
    Handle per-fold CV logs:

    1. Save folds info to a JSON file in exp_root/logs/.
    2. For each metric (and val_loss), create a separate bar plot over folds.

    Args:
        folds: list of fold result dicts:
            [
              {
                "fold": int,
                "val_loss": float,
                "val_metrics": {
                  "mse": float,
                  "mae": float,
                  "rmse": float,
                  "r2": float,
                  ...
                }
              },
              ...
            ]
        exp_root: experiment root directory (Path or str).
        json_filename: name for the saved JSON file.
        fig_prefix: prefix for each figure file name.
        show: if True, show each figure; otherwise only save and close.
    """
    exp_root = Path(exp_root)
    logs_dir = exp_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not folds:
        raise ValueError("No folds provided to save_and_plot_cv_folds().")

    # ---------------- 1. Save JSON ----------------
    json_path = logs_dir / json_filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"folds": folds}, f, indent=2)
    print(f"\n[INFO] CV folds results saved to: {json_path}")

    # ---------------- 2. Prepare meta info ----------------
    fold_ids = [fold.get("fold", i) for i, fold in enumerate(folds)]
    n_folds = len(fold_ids)

    # Collect metric names from first fold
    first_metrics = folds[0].get("val_metrics", {})
    metric_names = sorted(first_metrics.keys())

    # Treat val_loss as a metric as well
    all_items = ["val_loss"] + metric_names

    # ---------------- 3. Plot one figure per item ----------------
    for item in all_items:
        # Extract values for this item across folds
        values = []
        for fold in folds:
            if item == "val_loss":
                v = fold.get("val_loss", np.nan)
            else:
                v = fold.get("val_metrics", {}).get(item, np.nan)
            values.append(float(v))

        values = np.array(values, dtype=float)

        fig, ax = plt.subplots(figsize=(5, 3.2))

        x = np.arange(n_folds)
        ax.bar(x, values, width=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([f"F{fid}" for fid in fold_ids])
        ax.set_title(f"{item} per fold")
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", alpha=0.3)

        # ---- Smart y-limits ----
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))

        if item.lower() == "r2":
            # r2 in [0,1]: zoom around observed range with padding
            lo = max(0.0, vmin - 0.05)
            hi = min(1.0, vmax + 0.05)
            if hi - lo < 0.1:
                mid = 0.5 * (lo + hi)
                lo = max(0.0, mid - 0.05)
                hi = min(1.0, mid + 0.05)
            ax.set_ylim(lo, hi)
        else:
            if vmin == vmax:
                # degenerate: expand a bit
                delta = 0.1 * abs(vmin) if vmin != 0 else 0.1
                vmin -= delta
                vmax += delta
            pad = 0.1 * (vmax - vmin)
            ax.set_ylim(vmin - pad, vmax + pad)

        fig.tight_layout()

        # Sanitize metric name for filename
        safe_name = str(item).replace(" ", "_").replace("/", "_")
        fig_path = logs_dir / f"{fig_prefix}{safe_name}.png"
        fig.savefig(fig_path, dpi=150)
        print(f"[INFO] Saved {item} plot to: {fig_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def save_sweep_best_summary(
        result: dict, monitor_metric: str, exp_root: Union[str, Path], file_name: str = "sweep_best_summary.txt"
    ) -> str:
    """
    Summarize and save the best trial info after random_search.

    Creates a file:
        runs/<experiment_name>/sweep_best_summary.txt

    Args:
        result: dict returned by random_search(), with keys:
            - best_trial_idx
            - best_value: best objective value of the monitored metric
            - best_cfg: best Config
            - best_result: result dict from run_experiment()
        
        monitor_metric: str, name of the monitored metric (e.g. "val_loss")
        exp_root: str or Path, root dir of the experiment (runs/<experiment_name>)
    """
    best_trial_idx = result.get("best_trial_idx")
    best_value = result.get("best_value")
    best_cfg = result.get("best_cfg")
    best_result = result.get("best_result")

    # Build the summary text
    lines = [
        "==================================",
        "Best Trial Summary",
        "==================================",
        f"Trial Index: {best_trial_idx}",
        f"Best {monitor_metric} value: {best_value}",
        f"Best val loss: {best_result.get('val_loss')}",
        "Best val metrics:",
        json.dumps(best_result.get("val_metrics", {}), indent=2),
        "\nBest Config:\n",
        yaml.safe_dump(best_cfg.to_dict(), sort_keys=False, default_flow_style=False),
        "==================================",
    ]
    summary_str = "\n".join(lines)

    # Print to stdout
    print("\n" + summary_str)

    # Save to ./runs/<exp>/sweep_best_summary.txt
    Path(exp_root).mkdir(parents=True, exist_ok=True)
    out_path = Path(exp_root) / file_name
    out_path.write_text(summary_str, encoding="utf-8")

    print(f"\n[INFO] Best sweep summary saved to: {out_path}")

    return summary_str


def save_and_plot_sweep_trials_metrics(
    trials_metrics: List[Dict[str, Any]],
    exp_root: Union[str, Path],
    monitor_metric: str,
    show: bool = False,
    fig_prefix: str = "sweep_metrics_",
):
    """
    For a list of per-trial metrics dicts, create separate bar plots per metric.

    Args:
        trials_metrics: list of dicts, one per trial, e.g.
                        [
                          {"val_loss": 0.1, "r2": 0.98},
                          {"val_loss": 0.12, "r2": 0.975},
                          ...
                        ]
        exp_root: experiment root directory.
        monitor_metric: name of the metric used for model selection (its plot
                        is saved directly under exp_root).
        show: if True, display each figure; otherwise only save & close.
        fig_prefix: prefix for generated filenames.

    Behavior:
        - For each metric key found in trials_metrics:
            - Create a bar plot: x-axis = trials, y-axis = metric value.
            - If metric == monitor_metric:
                save to: exp_root / f"{fig_prefix}{metric}.png"
              else:
                save to: exp_root/logs / f"{fig_prefix}{metric}.png"
    """
    if not trials_metrics:
        raise ValueError("trials_metrics is empty.")

    exp_root = Path(exp_root)
    logs_dir = exp_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Collect all metric names across trials
    metric_names = set()
    for m in trials_metrics:
        metric_names.update(m.keys())
    metric_names = sorted(metric_names)

    n_trials = len(trials_metrics)
    trial_labels = [f"T{i}" for i in range(n_trials)]

    for metric in metric_names:
        # Gather values for this metric across trials
        values = []
        for m in trials_metrics:
            v = m.get(metric, np.nan)
            values.append(float(v))
        values = np.array(values, dtype=float)

        # Create bar plot
        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(n_trials)
        ax.bar(x, values, width=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(trial_labels)
        ax.set_title(f"{metric} per trial")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.3)

        # Smart y-limits
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))

        if metric.lower() == "r2":
            lo = max(0.0, vmin - 0.05)
            hi = min(1.0, vmax + 0.05)
            if hi - lo < 0.1:
                mid = 0.5 * (lo + hi)
                lo = max(0.0, mid - 0.05)
                hi = min(1.0, mid + 0.05)
            ax.set_ylim(lo, hi)
        else:
            if vmin == vmax:
                delta = 0.1 * abs(vmin) if vmin != 0 else 0.1
                vmin -= delta
                vmax += delta
            pad = 0.1 * (vmax - vmin)
            ax.set_ylim(vmin - pad, vmax + pad)

        fig.tight_layout()

        # Decide save location
        safe_name = str(metric).replace(" ", "_").replace("/", "_")
        if metric == monitor_metric:
            out_dir = exp_root
        else:
            out_dir = logs_dir

        out_path = out_dir / f"{fig_prefix}{safe_name}.png"
        fig.savefig(out_path, dpi=150)
        print(f"[INFO] Saved {metric} plot to: {out_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)