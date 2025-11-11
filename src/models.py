import torch
import torch.nn as nn
from src.config import ModelConfig


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation: {name}")


class MLPRegressor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        layers = []
        in_dim = cfg.input_dim
        act_cls = get_activation(cfg.activation)

        if cfg.hidden_dims:
            for h in cfg.hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(act_cls())
                if cfg.dropout > 0.0:
                    layers.append(nn.Dropout(cfg.dropout))
                in_dim = h
            layers.append(nn.Linear(in_dim, cfg.output_dim))
        else:
            layers.append(nn.Linear(in_dim, cfg.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_mlp(cfg: ModelConfig) -> MLPRegressor:
    return MLPRegressor(cfg)
