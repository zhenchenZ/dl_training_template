"""
Basic training example using the DL training template.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import BaseDataset, create_data_loaders, split_data
from models import create_model
from training import Trainer
from evaluation import evaluate_model
from config import Config


def generate_dummy_data(n_samples=1000, input_dim=784, n_classes=10):
    """Generate dummy data for demonstration."""
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


def main():
    """Main training function."""
    # Load configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'default_config.yaml'
    config = Config.from_yaml(str(config_path))
    
    print("Configuration loaded:")
    print(config)
    
    # Set device
    device = torch.device(config.get('device') if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Generate dummy data
    print("\nGenerating dummy data...")
    X, y = generate_dummy_data(
        n_samples=1000,
        input_dim=config.get('model.input_dim', 784),
        n_classes=config.get('model.output_dim', 10)
    )
    
    # Split data
    X_train, y_train, X_val, y_val = split_data(
        X, y,
        train_ratio=config.get('data.train_ratio', 0.8),
        random_state=config.get('data.random_state', 42)
    )
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    # Create datasets
    train_dataset = BaseDataset(X_train, y_train)
    val_dataset = BaseDataset(X_val, y_val)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=config.get('data.batch_size', 32),
        num_workers=config.get('data.num_workers', 4)
    )
    
    # Create model
    print("\nCreating model...")
    model_config = config.to_dict()['model']
    model = create_model(
        model_type=config.get('model.type', 'mlp'),
        config=model_config
    )
    print(f"Model created with {model.get_num_parameters():,} parameters")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    optimizer_name = config.get('training.optimizer', 'adam')
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('training.learning_rate', 0.001),
            weight_decay=config.get('training.weight_decay', 1e-5)
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.get('training.learning_rate', 0.001),
            momentum=0.9,
            weight_decay=config.get('training.weight_decay', 1e-5)
        )
    
    # Create trainer
    training_config = {
        **config.to_dict()['training'],
        **config.to_dict()['logging']
    }
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=training_config
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        save_plots=True,
        output_dir='evaluation_results'
    )
    
    print("\nTraining completed successfully!")
    print(f"Final train accuracy: {history['train_accuracy'][-1]:.2f}%")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.2f}%")


if __name__ == '__main__':
    main()
