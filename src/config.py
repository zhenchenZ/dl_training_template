from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    name: str = "mlp"  # "mlp" or "logistic_regression"
    input_dim: int = 7
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"
    dropout: float = 0.0
    output_dim: int = 1


# in config.py
@dataclass
class DataConfig:
    dataset: str = "real"          # or "real"
    npz_path: str = ""            # used when dataset == "real"
    batch_size: int = 64
    seed: int = 42
    normalize_x: bool = True
    x_norm_method: str = "minmax"
    normalize_y: bool = True
    y_norm_method: str = "minmax"
    return_stats: bool = True
    val_ratio: float = 0.2
    # n_samples: int = 3000
    # noise_std: float = 0.1

@dataclass
class OptimConfig:
    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9


@dataclass
class TrainConfig:
    num_epochs: int = 30
    log_interval: int = 5
    device: Optional[str] = None  # "cuda", "cpu", or None = auto


@dataclass
class LoggingConfig:
    # Root folder for all experiments
    root_dir: str = "./runs"
    # Name of this experiment (subfolder under root_dir)
    experiment_name: str = "mlp_regression_exp1"
    # Subdirectories inside the experiment folder
    subdir_checkpoints: str = "checkpoints"
    subdir_logs: str = "logs"
    subdir_hparams: str = "hparams"


@dataclass
class CheckpointConfig:
    save_best: bool = True
    save_last: bool = True
    monitor: str = "val_loss"  # "val_loss" or e.g. "val_r2"
    mode: str = "min"          # "min" or "max"


@dataclass
class CVConfig:
    n_splits: int = 1          # 1 => no CV, just a normal train/val split
    shuffle: bool = True
    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    cv: CVConfig = field(default_factory=CVConfig)

    def to_dict(self) -> dict:
        """Recursively convert to a plain Python dict."""
        return asdict(self)

    def to_yaml(self) -> str:
        """Return a pretty YAML representation."""
        return yaml.safe_dump(
            self.to_dict(),
            sort_keys=False,           # preserve section order
            default_flow_style=False,  # use block style (multi-line)
            allow_unicode=True,
        )

    def print(self):
        """Print the config nicely formatted as YAML."""
        print("\n===== Current Configuration =====")
        print(self.to_yaml())
        print("=================================\n")
    
    def to_dict(self) -> dict:
        """Recursively convert to a plain Python dict."""
        return asdict(self)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        optim=OptimConfig(**raw.get("optim", {})),
        train=TrainConfig(**raw.get("train", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        checkpoint=CheckpointConfig(**raw.get("checkpoint", {})),
        cv=CVConfig(**raw.get("cv", {})),
    )


def get_experiment_dirs(
    logging_cfg: LoggingConfig,
    create: bool = True,
    subdirs: Tuple[str, ...] = (),
):
    """
    Returns (exp_root, ckpt_dir, logs_dir, hparams_dir).

    Args:
        logging_cfg: LoggingConfig instance.
        create: if True, create the directories that exist in `subdirs`.
        subdirs: optional tuple of subdirectory names to create under exp_root.

    Notes:
        - Always returns the Path objects (even if not created).
        - Only creates folders explicitly listed in `subdirs` when create=True.
    """
    exp_root = Path(logging_cfg.root_dir) / logging_cfg.experiment_name
    ckpt_dir = exp_root / logging_cfg.subdir_checkpoints
    logs_dir = exp_root / logging_cfg.subdir_logs
    hparams_dir = exp_root / logging_cfg.subdir_hparams

    if create:
        exp_root.mkdir(parents=True, exist_ok=True)
        for name, path in {
            "checkpoints": ckpt_dir,
            "logs": logs_dir,
            "hparams": hparams_dir,
        }.items():
            if name in subdirs:
                path.mkdir(parents=True, exist_ok=True)

    return exp_root, ckpt_dir, logs_dir, hparams_dir