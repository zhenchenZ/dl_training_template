import itertools
import os
import math
import random
from pathlib import Path
import json
import yaml
from copy import deepcopy
from typing import Dict, Any, List, Union, Optional

import numpy as np

from src.config import Config, get_experiment_dirs
from src.sweep_config import SweepConfig
from src.experiment import run_experiment


def _log_jsonl(log_path: str, record: Dict[str, Any]):
    """
    Append a single JSON record as one line to log_path.
    Creates the file if it doesn't exist.
    """
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def _to_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x))
    except (TypeError, ValueError):
        raise TypeError(f"Expected a float-like value, got {x!r} ({type(x)})")


def _to_int(x: Any) -> int:
    if isinstance(x, int):
        return x
    try:
        return int(str(x))
    except (TypeError, ValueError):
        raise TypeError(f"Expected an int-like value, got {x!r} ({type(x)})")


def set_by_path(cfg: Config, key: str, value: Any):
    """
    Set cfg.attr.subattr = value given a dotted path like "model.hidden_dims".

    For example:
        set_by_path(cfg, "optim.lr", 0.001) will set cfg.optim.lr = 0.001
    """
    parts = key.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p) 
    # after loop, obj is the parent of the final attribute, e.g. cfg.optim
    setattr(obj, parts[-1], value)


def get_from_path(obj: Any, key: str):
    parts = key.split(".")
    for p in parts:
        obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
    return obj

def sample_value(spec, rng: random.Random):
    """
    Sample a value from a search-space spec.

    Supported:
      - list/tuple -> random choice
      - {"type": "uniform", "low": ..., "high": ...}
      - {"type": "int", "low": ..., "high": ...}
      - {"type": "log_uniform", "low": ..., "high": ...}
      - callable(rng) -> custom
    """
    # list/tuple of choices
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            raise ValueError("Choice list for search space is empty.")
        return rng.choice(spec)

    # callable: full control
    if callable(spec):
        return spec(rng)

    if isinstance(spec, dict):
        t = spec.get("type")

        if t == "uniform":
            low = _to_float(spec["low"])
            high = _to_float(spec["high"])
            return rng.uniform(low, high)

        if t == "int":
            low = _to_int(spec["low"])
            high = _to_int(spec["high"])
            # inclusive bounds (like randint)
            return rng.randint(low, high)

        if t == "log_uniform":
            low = _to_float(spec["low"])
            high = _to_float(spec["high"])
            if low <= 0 or high <= 0:
                raise ValueError(f"log_uniform requires low/high > 0, got low={low}, high={high}")
            log_low = math.log(low)
            log_high = math.log(high)
            return math.exp(rng.uniform(log_low, log_high))

    raise ValueError(f"Unsupported search space spec: {spec!r}")


def get_objective(result: Dict[str, Any], monitor: str = "val_loss") -> float:
    """
    we make this robust to both: 
        - no CV: {"val_loss": float, "val_metrics": {...}} 
        - CV: {"summary": {"val_loss": {"mean": ...}, "metrics": {...}}}

    Extract a scalar objective from run_experiment(result).
    - monitor="val_loss": use val_loss (or mean CV val_loss)
    - monitor="<metric_name>": e.g. "rmse", "r2" (uses CV mean if available)
    """
    # CV case
    if "summary" in result:
        summary = result["summary"]
        if monitor == "val_loss":
            return float(summary["val_loss"]["mean"])
        else:
            # monitor is metric name like "rmse"
            m = summary["metrics"].get(monitor)
            if m is None:
                raise KeyError(f"Metric {monitor} not found in CV summary")
            return float(m["mean"])

    # Single-split case
    if monitor == "val_loss":
        return float(result["val_loss"])

    val_metrics = result.get("val_metrics", {})
    if monitor not in val_metrics:
        raise KeyError(f"Metric {monitor} not found in val_metrics")
    return float(val_metrics[monitor])


def random_search(
    base_cfg: Config,
    sweep_cfg: SweepConfig
):
    """
    Random search over hyperparameters.

    Layout:

    runs/
      <base_experiment_name>/
        config.yaml                  # base config (you save this in main)
        logs/
          random_search_logs.jsonl   # sweep-level summary (this function)
        hparams/
          best_trial.yaml            # best trial full config
        trials/
          trial_000/
            config.yaml
            checkpoints/
            logs/
            hparams/
          trial_001/
            ...

    Each trial gets its own experiment subfolder via cfg.logging.experiment_name.
    """
    # unpack sweep_cfg
    search_space = getattr(sweep_cfg, "search_space", {})
    n_trials = getattr(sweep_cfg, "n_trials", 20)
    monitor = getattr(sweep_cfg, "monitor", "val_loss")
    mode = getattr(sweep_cfg, "mode", "min")
    seed  = getattr(sweep_cfg, "seed", 0)
    log_full_result = getattr(sweep_cfg, "log_full_result", False)

    rng = random.Random(seed)
    trials: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Base experiment dirs (for sweep-level artifacts)
    # ------------------------------------------------------------------
    _, _, base_logs_dir, base_hparams_dir = get_experiment_dirs(base_cfg.logging, create=True, subdirs=("logs", "hparams"))

    # Sweep-level log file
    log_path = os.path.join(base_logs_dir, "random_search_logs.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)  # clear previous sweep logs

    best_value = None
    best_cfg = None
    best_result = None
    best_trial_idx = None

    base_exp_name = base_cfg.logging.experiment_name

    for i in range(n_trials):
        # Start from base config
        cfg = deepcopy(base_cfg)

        # ------------------------------------------------------------------
        # Sample hyperparameters into cfg
        # ------------------------------------------------------------------
        sampled_params = {}
        for key, spec in search_space.items():
            val = sample_value(spec, rng)
            set_by_path(cfg, key, val) # e.g. key = "optim.lr" -> cfg.optim.lr = val
            sampled_params[key] = val

        # ------------------------------------------------------------------
        # Trial-specific experiment name & dirs
        # We encode trials as sub-experiments:
        #   <base_exp_name>/trials/trial_000
        # This keeps run_experiment + Trainer logic unchanged.
        # ------------------------------------------------------------------
        trial_name = f"{base_exp_name}/trials/trial_{i:03d}"
        cfg.logging.experiment_name = trial_name

        trial_exp_root, _, _, _ = get_experiment_dirs(cfg.logging)

        # Snapshot full config for this trial
        trial_config_path = os.path.join(trial_exp_root, "config.yaml")
        with open(trial_config_path, "w", encoding="utf-8") as f:
            f.write(cfg.to_yaml())

        print(f"\n=== Random search trial {i+1}/{n_trials} ===")
        print(f"Experiment dir: {trial_exp_root}")
        print("Sampled params:", sampled_params)

        # ------------------------------------------------------------------
        # Run experiment for this trial
        # ------------------------------------------------------------------
        result = run_experiment(cfg)
        obj = get_objective(result, monitor=monitor)
        print(f"Objective ({monitor}) = {obj:.6f}")

        # ------------------------------------------------------------------
        # Store trial info in-memory
        # ------------------------------------------------------------------
        trial_info = {
            "trial": i,
            "experiment_dir": str(trial_exp_root),
            "params": sampled_params,
            "objective": obj,
        }
        if log_full_result:
            trial_info["result"] = result

        # always keep full in-memory result for convenient downstream analysis
        trials.append({**trial_info, "result": result})

        # ------------------------------------------------------------------
        # Append lightweight record to global sweep log
        # ------------------------------------------------------------------
        to_log = dict(trial_info)
        if not log_full_result and "result" in to_log:
            to_log.pop("result")
        _log_jsonl(log_path, to_log)

        # ------------------------------------------------------------------
        # Track best
        # ------------------------------------------------------------------
        if best_value is None:
            is_better = True
        elif mode == "min":
            is_better = obj < best_value
        elif mode == "max":
            is_better = obj > best_value
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if is_better:
            best_value = obj
            best_cfg = deepcopy(cfg)
            best_result = result
            best_trial_idx = i

            # Save best trial config at sweep level
            best_cfg_path = os.path.join(base_hparams_dir, "best_trial.yaml")
            with open(best_cfg_path, "w", encoding="utf-8") as f:
                f.write(best_cfg.to_yaml())

            print(f"\n-> New best (trial {i}, {monitor}={best_value:.6f})")
            print(f"   Best config saved to: {best_cfg_path}\n")

    return {
        "best_value": best_value,
        "best_cfg": best_cfg,
        "best_result": best_result,
        "best_trial_idx": best_trial_idx,
        "trials": trials,
        "sweep_log_path": log_path,
    }