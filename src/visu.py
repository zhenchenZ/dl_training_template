import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_history(
    history: Dict,
    metrics_to_plot: Optional[List[str]] = None,
    figsize=(6, 5),
    sharex=True,
    suptitle: Optional[str] = "Training history",
    savepath: Optional[str] = None,
    n_cols: int = 3,
):
    """
    Plot training/validation losses and metrics from a history dict.

    Expected history format:
      - history["train_loss"]: list[float], length = n_steps (usually epochs)
      - history["val_loss"]:   list[float] or empty/missing
      - history["train_metrics"]: list[dict], each dict: {metric_name: value}
      - history["val_metrics"]:   list[dict], same structure as train_metrics

    For metrics:
      - Automatically collects all metric names from train/val dicts,
        unless `metrics_to_plot` is explicitly provided.
      - For each metric, creates one subplot:
          * train_metric vs step
          * val_metric vs step (if available)

    Args:
        history: dict with keys as described above.
        metrics_to_plot: optional list of metric names to restrict plotting.
        figsize: figure size for matplotlib.
        sharex: whether to share x-axis across subplots.
        suptitle: overall title (set None to disable).
        savepath: if provided, saves the figure to this path.
    """
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_metrics_hist = history.get("train_metrics", [])
    val_metrics_hist = history.get("val_metrics", [])

    n_steps = len(train_loss)
    x = np.arange(1, n_steps + 1)

    # ---- Collect metric names ----
    metric_names = set()

    for d in train_metrics_hist:
        metric_names.update(d.keys())
    for d in val_metrics_hist:
        metric_names.update(d.keys())

    if metrics_to_plot is not None:
        metric_names = [m for m in metric_names if m in metrics_to_plot]
    else:
        metric_names = sorted(metric_names)

    # One subplot for loss + one per metric
    n_plots = 1 + len(metric_names)
    if n_plots == 0:
        raise ValueError("Nothing to plot: history appears to be empty.")

    n_rows = (len(metric_names) + 1 + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize[0]*n_cols, figsize[1]*n_rows),
        sharex=sharex if n_plots > 1 else False,
    )

    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    ax_idx = 0

    # ---- Plot loss ----
    ax_loss = axes[ax_idx]
    ax_idx += 1

    if train_loss:
        ax_loss.plot(x, train_loss, label="train_loss")
    if val_loss:
        # Ensure val length compatibility
        xv = np.arange(1, len(val_loss) + 1)
        ax_loss.plot(xv, val_loss, label="val_loss", linestyle="--")

    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    # ---- Plot each metric ----
    for metric_name in metric_names:
        ax = axes[ax_idx]
        ax_idx += 1

        # Collect per-step values with NaN fallback
        train_vals = [
            (m.get(metric_name) if m is not None else np.nan)
            for m in train_metrics_hist
        ]
        val_vals = [
            (m.get(metric_name) if m is not None else np.nan)
            for m in val_metrics_hist
        ]

        train_vals = np.array(train_vals, dtype=float) if train_vals else None
        val_vals = np.array(val_vals, dtype=float) if val_vals else None

        if train_vals is not None and len(train_vals) > 0:
            ax.plot(
                np.arange(1, len(train_vals) + 1),
                train_vals,
                label=f"train_{metric_name}",
            )

        if val_vals is not None and len(val_vals) > 0 and not np.all(np.isnan(val_vals)):
            ax.plot(
                np.arange(1, len(val_vals) + 1),
                val_vals,
                label=f"val_{metric_name}",
                linestyle="--",
            )

        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Step (epoch)")

    if suptitle:
        fig.suptitle(suptitle)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Training history plot saved to: {savepath}")
    else:
        return fig


def plot_histories_overlay(
    histories: List[Dict],
    labels: Optional[List[str]] = None,
    metrics_to_plot: Optional[List[str]] = None,
    base_index: int = 0,
    figsize=(6, 5),
    n_cols: int = 3,
    sharex: bool = True,
    suptitle: Optional[str] = "Training history (comparison)",
    savepath: Optional[str] = None,
):
    """
    Overlay multiple training histories (e.g. CV folds or hyperparam sweeps)
    on the same subplot layout created by the existing plot_history().

    This function:
      1) calls your plot_history() once on histories[base_index]
      2) reuses its axes (Loss + metrics)
      3) draws additional runs' curves on the same axes for comparison

    Args:
        histories: list of history dicts (same format as for plot_history).
        labels: optional list of run labels (len == len(histories)).
                If None, defaults to ["run_0", "run_1", ...].
        metrics_to_plot: subset of metric names to plot (passed through).
        base_index: which history to use as the base layout.
        figsize, n_cols, sharex, suptitle, savepath:
            forwarded to the underlying plot_history for the base run.
    """
    if not histories:
        raise ValueError("No histories provided.")

    n_runs = len(histories)

    if labels is None:
        labels = [f"run_{i}" for i in range(n_runs)]
    if len(labels) != n_runs:
        raise ValueError("len(labels) must match len(histories).")

    if not (0 <= base_index < n_runs):
        raise ValueError("base_index out of range.")

    # 1) Create base figure using your existing function (unchanged)
    base_history = histories[base_index]
    fig = plot_history(
        history=base_history,
        metrics_to_plot=metrics_to_plot,
        figsize=figsize,
        sharex=sharex,
        suptitle=suptitle,
        savepath=None,
        n_cols=n_cols,
    )

    # 2) Build a mapping: axis title -> axis
    axes_by_title = {ax.get_title(): ax for ax in fig.axes}

    # 3) Relabel base run's lines to include its label (optional but clearer)
    base_label = labels[base_index]
    for ax in fig.axes:
        new_lines = []
        for line in ax.get_lines():
            old_label = line.get_label()
            # Skip matplotlib internals like "_nolegend_"
            if old_label and not old_label.startswith("_"):
                line.set_label(f"{base_label}_{old_label}")
            new_lines.append(line)
        if new_lines:
            ax.legend()

    # 4) Overlay other runs
    for i, (hist, run_label) in enumerate(zip(histories, labels)):
        if i == base_index:
            continue  # already plotted

        train_loss = hist.get("train_loss", [])
        val_loss = hist.get("val_loss", [])
        train_metrics_hist = hist.get("train_metrics", [])
        val_metrics_hist = hist.get("val_metrics", [])

        # ---- Loss axis ----
        loss_ax = axes_by_title.get("Loss", None)
        if loss_ax is not None and train_loss:
            x_train = np.arange(1, len(train_loss) + 1)
            loss_ax.plot(
                x_train,
                train_loss,
                label=f"{run_label}_train_loss",
                alpha=0.9,
            )

            if val_loss:
                x_val = np.arange(1, len(val_loss) + 1)
                loss_ax.plot(
                    x_val,
                    val_loss,
                    label=f"{run_label}_val_loss",
                    linestyle="--",
                    alpha=0.9,
                )

            loss_ax.legend()
            loss_ax.grid(True)

        # ---- Metrics axes ----
        # Collect all metric names present in this run
        metric_names = set()
        for d in train_metrics_hist:
            if d:
                metric_names.update(d.keys())
        for d in val_metrics_hist:
            if d:
                metric_names.update(d.keys())

        # Optionally restrict to provided metrics_to_plot
        if metrics_to_plot is not None:
            metric_names = [m for m in metric_names if m in metrics_to_plot]

        for metric_name in metric_names:
            ax = axes_by_title.get(metric_name, None)
            if ax is None:
                # This metric wasn't present in base history; skip (keeps impl simple)
                continue

            train_vals = [
                (m.get(metric_name) if m is not None else np.nan)
                for m in train_metrics_hist
            ]
            val_vals = [
                (m.get(metric_name) if m is not None else np.nan)
                for m in val_metrics_hist
            ]

            train_vals = np.array(train_vals, dtype=float) if train_vals else None
            val_vals = np.array(val_vals, dtype=float) if val_vals else None

            if train_vals is not None and len(train_vals) > 0:
                ax.plot(
                    np.arange(1, len(train_vals) + 1),
                    train_vals,
                    label=f"{run_label}_train_{metric_name}",
                    alpha=0.9,
                )

            if val_vals is not None and len(val_vals) > 0 and not np.all(np.isnan(val_vals)):
                ax.plot(
                    np.arange(1, len(val_vals) + 1),
                    val_vals,
                    label=f"{run_label}_val_{metric_name}",
                    linestyle="--",
                    alpha=0.9,
                )

            ax.legend()
            ax.grid(True)

    # 5) Final layout / save
    if suptitle:
        fig.suptitle(suptitle)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


def plot_pred_vs_true(y_true, y_pred, title="Prediction vs True", savepath=None, **kwargs):
    figsize = kwargs.get("figsize", (6,6))
    s = kwargs.get("s", 20)
    alpha = kwargs.get("alpha", 0.6)

    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=alpha, s=s)
    minv, maxv = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([minv, maxv], [minv, maxv], "r--", lw=2, label="Ideal")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_residuals(y_true, y_pred, title="Residuals", savepath=None):
    residuals = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, alpha=0.6, s=20)
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals (True - Predicted)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()