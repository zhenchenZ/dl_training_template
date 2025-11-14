# Deep Learning Training

Minimal but structured template for **deep learning experiments on tabular data** (regression / logistic regression) using PyTorch.

The repo focuses on:

* A clean **config-driven** pipeline (YAML)
* **Standalone training**
* **Hyperparameter sweeping** (random search)
* **Cross-validation** using the best hyperparameters

Jupyter notebooks in this repo are mostly for debugging / exploration; the *canonical* entry points are the Python scripts.

---

## 1. Installation

Clone the repo:

```bash
git clone https://github.com/zhenchenZ/dl_training_template.git
cd dl_training_template
```

Create a Python environment (example with `conda`):

```bash
conda create -n dl_template python=3.10
conda activate dl_template
```

Install dependencies (minimal set):

```bash
pip install torch numpy matplotlib
```

(Optionally add anything else you use in notebooks.)

---

## 2. Data preparation

The training code expects **preprocessed data** stored in an `.npz` file:

```text
data/preprocessed/cleaned_x_y_hot.npz
```

The file should contain at least:

* `X`: shape `(N, D)` — features (e.g. weather variables)
* `y`: shape `(N,)` or `(N, 1)` — target values (e.g. heating power)
* `feat_names` (optional): shape `(D,)` — list of feature names

You can change the path in the config (see below), but the format must be the same.

---

## 3. Configuration files

All experiment hyperparameters live in `config/*.yaml`.
Key files:

* `config/base_config.yaml` – baseline MLP regression
* `config/base_config_logistic.yaml` – logistic-regression style model on normalized targets
* `config/sweep_config.yaml` – hyperparameter sweep over MLP model & optim params
* `config/sweep_config_logistic.yaml` – sweep for logistic-regression config

### 3.1. Base config structure

Example: `config/base_config.yaml`

```yaml
model:
  name: "mlp"          # "mlp" or "logistic_regression"
  input_dim: 6
  hidden_dims: [256]
  activation: "relu"
  dropout: 0.0

data:
  dataset: "real"      # "real" uses NPZ; "toy" uses synthetic
  npz_path: "data/preprocessed/cleaned_x_y_hot.npz"
  batch_size: 1024
  seed: 42
  normalize_x: true
  x_norm_method: "minmax"    # "minmax" or "standardize"
  normalize_y: false         # true only if you want to normalize targets
  y_norm_method: ""
  return_stats: true
  val_ratio: 0.2

optim:
  name: "adam"
  lr: 0.009
  weight_decay: 0.0000136

train:
  num_epochs: 40
  log_interval: 5
  device: "cuda"  # or "cpu" or null for auto

logging:
  root_dir: "./runs"
  experiment_name: "real_dataset_experiment"

checkpoint:
  save_best: true
  save_last: false
  monitor: "val_loss"
  mode: "min"

cv:
  n_splits: 5      # 1 => normal train/val; >=2 => K-fold CV
  shuffle: true
  seed: 42
```

You can duplicate this file and tweak it to define new experiments.

---

## 4. Experiment folder structure

The helper `get_experiment_dirs` creates the following structure for each experiment name: ([GitHub][2])

```text
runs/
  <experiment_name>/
    config.yaml              # snapshot of config (for train.py)
    checkpoints/             # model checkpoints
    logs/                    # training logs & plots
    hparams/                 # sweep configs, scaler stats, best_trial.yaml
    trials/                  # (created by hyper_param_sweep.py)
      trial_000/
        config.yaml
        ...
      trial_001/
        ...
    folds/                   # (created by CV)
      fold_000/
      fold_001/
      ...
```

---

## 5. Standalone training

Entry point: **`train.py`**

Usage:

```bash
python train.py \
  --config config/base_config.yaml \
  --exp_name 000-mlp_baseline
```

Arguments:

* `--config` (`-c`): path to a base config YAML
* `--exp_name`: optional override for `logging.experiment_name` in the config

What happens:

1. Config is loaded via `src.config.load_config`.
2. The config is printed in a nice YAML format (`Config.print()`).
3. A snapshot of the config is saved to:

   ```text
   runs/<exp_name>/hparams/config.yaml
   ```
4. `run_experiment(cfg)` is called:

   * If `cv.n_splits <= 1`: single train/val split (`run_single_experiment`)
   * If `cv.n_splits >= 2`: K-fold CV (`run_cv_experiment`) ([GitHub][4])
5. For non-CV runs, a training history plot is saved to:

   ```text
   runs/<exp_name>/logs/training_history.png
   ```
6. Best checkpoint(s) are saved to:

   ```text
   runs/<exp_name>/checkpoints/
   ```

### Example: MLP regression

```bash
python train.py \
  --config config/base_config.yaml \
  --exp_name 000-mlp_regression
```

### Example: logistic-regression-style model

```bash
python train.py \
  --config config/base_config_logistic.yaml \
  --exp_name 000-logistic_regression
```

---

## 6. Hyperparameter sweeping (random search)

Entry point: **`hyper_param_sweep.py`**

Usage example (MLP):

```bash
python hyper_param_sweep.py \
  --base_config config/base_config.yaml \
  --sweep_config config/sweep_config.yaml \
  --exp_name 001-sweep_mlp
```

Usage example (logistic):

```bash
python hyper_param_sweep.py \
  --base_config config/base_config_logistic.yaml \
  --sweep_config config/sweep_config_logistic.yaml \
  --exp_name 001-sweep_logistic
```

### 6.1. Sweep config

`config/sweep_config.yaml` looks like:

```yaml
sweep:
  n_trials: 50
  monitor: "r2"        # objective metric
  mode: "max"          # "min" or "max"
  seed: 42
  log_full_result: false

  search_space:
    model.hidden_dims:
      - [64, 64]
      - [128, 128]
      - [128, 128, 128]
      - [64, 64, 64, 64]
    model.dropout:
      - 0.0
      - 0.2
      - 0.5
    optim.lr:
      type: "log_uniform"
      low: 1e-4
      high: 1e-2
    optim.weight_decay:
      type: "log_uniform"
      low: 1e-6
      high: 1e-3
```

Supported search space specs: ([GitHub][6])

* A **list** of values → uniform random choice
* `{"type": "uniform", "low": ..., "high": ...}`
* `{"type": "int", "low": ..., "high": ...}` (inclusive)
* `{"type": "log_uniform", "low": ..., "high": ...}`

### 6.2. What hyper_param_sweep.py does

For each trial:

1. Start from `base_config` (`Config` object).
2. Sample a set of hyperparameters from `search_space` and write them into the config (`set_by_path`).
3. Construct a trial-specific experiment name:

   ```text
   <exp_name>/trials/trial_000
   <exp_name>/trials/trial_001
   ...
   ```
4. Save the full config for that trial:

   ```text
   runs/<exp_name>/trials/trial_000/config.yaml
   ```
5. Call `run_experiment(cfg)` (which may itself do CV depending on `cv.n_splits`).
6. Compute an objective value from the result (monitor `"val_loss"` or a metric like `"r2"`). ([GitHub][6])
7. Log a lightweight JSONL record to:

   ```text
   runs/<exp_name>/logs/random_search_logs.jsonl
   ```

When the sweep finishes:

* The **best config** is saved to:

  ```text
  runs/<exp_name>/hparams/best_trial.yaml
  ```
* A human-readable summary of the best trial is saved to:

  ```text
  runs/<exp_name>/sweep_best_summary.txt
  ```
* If `cv.n_splits == 1`, additional plots are created:

  * Overlaid training histories:

    ```text
    runs/<exp_name>/logs/sweep_histories.png
    ```
  * Bar plots of per-trial metrics:

    ````text
    runs/<exp_name>/sweep_metrics_<metric>.png      # monitored metric
    runs/<exp_name>/logs/sweep_metrics_<metric>.png # other metrics
    ``` :contentReference[oaicite:7]{index=7}
    ````

---

## 7. Cross-validation with best hyperparameters

Typical workflow:

1. **Run a sweep** with `cv.n_splits = 1` in your base config to find good hyperparameters.
2. Inspect:

   ```text
   runs/<exp_name>/hparams/best_trial.yaml
   ```
3. Create a new config file, e.g. `config/base_config_best.yaml`, by copying `base_config.yaml` and overwriting:

   * `model.*`, `optim.*` with values from `best_trial.yaml`
   * `cv.n_splits` to e.g. `5` for 5-fold CV
   * optionally, a new `logging.experiment_name`, e.g. `real_dataset_experiment_cv`

Example:

```yaml
# config/base_config_best.yaml
model:
  name: "mlp"
  input_dim: 6
  hidden_dims: [128, 128]
  activation: "relu"
  dropout: 0.2

# data, optim, train, logging, checkpoint...
cv:
  n_splits: 5
  shuffle: true
  seed: 42
```

4. Run cross-validation with `train.py`:

```bash
python train.py \
  --config config/base_config_best.yaml \
  --exp_name 002-mlp_cv_best
```

### 7.1. CV outputs

When `cv.n_splits >= 2`, `run_experiment` calls `run_cv_experiment`. ([GitHub][4])

You get:

1. **Per-fold logs & plots** (via `save_and_plot_cv_folds`):

   ```text
   runs/<exp_name>/logs/cv_folds_results.json
   runs/<exp_name>/logs/cv_folds_val_loss.png
   runs/<exp_name>/logs/cv_folds_mse.png
   runs/<exp_name>/logs/cv_folds_rmse.png
   runs/<exp_name>/logs/cv_folds_r2.png
   ...
   ```

2. **CV summary** (mean ± std across folds) (via `save_and_plot_cv_summary`):

   ```text
   runs/<exp_name>/cv_sweep_summary.txt
   runs/<exp_name>/cv_sweep_summary.png
   ```

The text summary includes mean/std for `val_loss` and all metrics; the PNG shows bar plots with error bars.

---

## 8. Metrics

Regression metrics are defined in `src/metrics.py`, and documented in `METRICS.md`. Typical ones include:

* `mse` – mean squared error
* `mae` – mean absolute error
* `rmse` – root mean squared error
* `r2` – coefficient of determination

These are used both during training (`Trainer`) and for sweep/CV reporting.

---

## 9. Quick recipes

### 9.1. One-shot training with current base config

```bash
python train.py \
  --config config/base_config.yaml \
  --exp_name 000-mlp_baseline
```

### 9.2. Hyperparameter sweep on MLP

```bash
python hyper_param_sweep.py \
  --base_config config/base_config.yaml \
  --sweep_config config/sweep_config.yaml \
  --exp_name 001-mlp_sweep
```

Then inspect:

* `runs/001-mlp_sweep/hparams/best_trial.yaml`
* `runs/001-mlp_sweep/sweep_best_summary.txt`
* `runs/001-mlp_sweep/sweep_metrics_*.png`

### 9.3. Cross-validation using best hyperparameters

1. Create `config/base_config_best.yaml` from `best_trial.yaml`.
2. Set `cv.n_splits > 1`.
3. Run:

```bash
python train.py \
  --config config/base_config_best.yaml \
  --exp_name 002-mlp_cv_best
```

---