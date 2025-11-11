import torch


@torch.no_grad()
def mse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean((preds - targets) ** 2).item()


@torch.no_grad()
def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean(torch.abs(preds - targets)).item()


@torch.no_grad()
def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


@torch.no_grad()
def r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    var = torch.var(targets)
    if var.item() == 0.0:
        return 0.0
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return (1.0 - ss_res / ss_tot).item()


DEFAULT_REGRESSION_METRICS = {
    "mse": mse,
    "mae": mae,
    "rmse": rmse,
    "r2": r2_score,
}
