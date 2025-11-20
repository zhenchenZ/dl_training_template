import argparse
from pathlib import Path
from time import time
from src.config import load_config, get_experiment_dirs
from src.experiment import run_experiment
from src.visu import plot_history

if __name__ == "__main__":
    """
    Example usage:
python train.py --config config/base_config_logistic.yaml --exp_name 002-cv_on_best_trial_logistic
    """
    t_start = time()

    # Config
    parser = argparse.ArgumentParser(description="Train a model based on the provided configuration.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--exp_name", type=str, default=None, help="Optional experiment name to override the one in config") 
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.exp_name is not None:
        cfg.logging.experiment_name = args.exp_name
    print("\nStarting training with the following configuration:")
    cfg.print()

    # Save the configuration used for training
    _, _, _, hparams_dir = get_experiment_dirs(cfg.logging, create=True, subdirs=("hparams",))             
    config_snapshot_path = Path(hparams_dir) / "config.yaml"
    with open(config_snapshot_path, "w") as f:
        f.write(cfg.to_yaml())
    print(f"\nConfiguration saved to: {config_snapshot_path}")

    # Run the experiment
    _ = run_experiment(cfg)

    t_end = time()
    print(f"\nTraining completed in {t_end - t_start:.2f} seconds.")



