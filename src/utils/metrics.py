"""
Metrics tracking for training and evaluation.
"""

from typing import Dict, List, Optional
import numpy as np


class MetricsTracker:
    """Track and store metrics during training."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}
        self.history = {}
    
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Update metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/epoch number
        """
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)
            self.metrics[name] = value
    
    def get_metric(self, name: str) -> float:
        """Get current value of a metric."""
        return self.metrics.get(name, 0.0)
    
    def get_history(self, name: str) -> List[float]:
        """Get history of a metric."""
        return self.history.get(name, [])
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all current metrics."""
        return self.metrics.copy()
    
    def get_all_history(self) -> Dict[str, List[float]]:
        """Get all metric histories."""
        return self.history.copy()
    
    def compute_statistics(self, name: str) -> Dict[str, float]:
        """
        Compute statistics for a metric.
        
        Args:
            name: Metric name
        
        Returns:
            Dictionary with mean, std, min, max
        """
        history = self.get_history(name)
        if not history:
            return {}
        
        return {
            'mean': np.mean(history),
            'std': np.std(history),
            'min': np.min(history),
            'max': np.max(history),
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.history = {}
    
    def __repr__(self) -> str:
        return f"MetricsTracker(metrics={self.metrics})"


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self, name: str):
        """
        Args:
            name: Name of the meter
        """
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


def compute_accuracy(predictions, targets) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
    
    Returns:
        Accuracy as a percentage
    """
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def compute_top_k_accuracy(output, target, k: int = 5) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        output: Model output logits
        target: Ground truth labels
        k: Top k predictions to consider
    
    Returns:
        Top-k accuracy as a percentage
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return 100.0 * correct_k.item() / batch_size


import torch


def compute_loss_batch(model, criterion, data, target, device):
    """
    Compute loss for a batch.
    
    Args:
        model: Neural network model
        criterion: Loss function
        data: Input data
        target: Target labels
        device: Device to use
    
    Returns:
        Tuple of (loss, predictions)
    """
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)
    _, predictions = torch.max(output, 1)
    return loss, predictions
