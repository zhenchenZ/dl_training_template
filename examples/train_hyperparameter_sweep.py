"""
Example of hyperparameter sweeping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import BaseDataset, create_data_loaders, split_data
from src.models import create_model
from src.training import Trainer, HyperparameterSweeper
from src.config import Config


def generate_dummy_data(n_samples=1000, input_dim=784, n_classes=10):
    """Generate dummy data."""
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


def model_fn(config):
    """Model factory function."""
    return create_model(config.get('model_type', 'mlp'), config)


def train_fn(model, train_loader, val_loader, config):
    """Training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config
    )
    
    return trainer.train(train_loader, val_loader)


def main():
    """Main function."""
    print("Hyperparameter Sweeping Example")
    print("="*50)
    
    # Generate data
    X, y = generate_dummy_data(n_samples=800)
    X_train, y_train, X_val, y_val = split_data(X, y, train_ratio=0.8)
    
    train_dataset = BaseDataset(X_train, y_train)
    val_dataset = BaseDataset(X_val, y_val)
    
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, batch_size=32
    )
    
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.001, 0.0001],
        'dropout_rate': [0.3, 0.5],
        'hidden_dims': [[128, 64], [256, 128]],
    }
    
    # Create sweeper
    sweeper = HyperparameterSweeper(
        param_grid=param_grid,
        output_dir='sweep_results'
    )
    
    # Base configuration
    base_config = {
        'model_type': 'mlp',
        'input_dim': 784,
        'output_dim': 10,
        'epochs': 3,  # Reduce for demo
        'weight_decay': 1e-5,
        'gradient_clip': 1.0,
        'log_dir': 'logs/sweep',
        'checkpoint_dir': 'checkpoints/sweep',
        'save_freq': 1,
    }
    
    # Run sweep
    results = sweeper.run_sweep(
        model_fn=model_fn,
        train_fn=train_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        base_config=base_config
    )
    
    # Get best configuration
    best_config = sweeper.get_best_config(metric='val_loss', mode='min')
    
    print("\n" + "="*50)
    print("Hyperparameter Sweep Complete!")
    print(f"Best Configuration: {best_config}")
    print("="*50)


if __name__ == '__main__':
    main()
