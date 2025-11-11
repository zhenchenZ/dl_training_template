from dataclasses import dataclass, field
from typing import Any, Dict, Union
from pathlib import Path
import yaml


@dataclass
class SweepConfig:
    n_trials: int = 20
    monitor: str = "val_loss"
    mode: str = "min"          # "min" or "max"
    seed: int = 0
    log_full_result: bool = False
    search_space: Dict[str, Any] = field(default_factory=dict)


def load_sweep_config(path: Union[str, Path]) -> SweepConfig:
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if "sweep" in raw:
        s = raw["sweep"]
    else:
        # allow using the file without top-level "sweep:"
        s = raw

    return SweepConfig(
        n_trials=int(s.get("n_trials", 20)),
        monitor=str(s.get("monitor", "val_loss")),
        mode=str(s.get("mode", "min")),
        seed=int(s.get("seed", 0)),
        log_full_result=bool(s.get("log_full_result", False)),
        search_space=dict(s.get("search_space", {})),
    )
