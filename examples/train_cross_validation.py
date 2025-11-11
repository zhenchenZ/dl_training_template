"""
Example of cross-validation training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import BaseDataset
from src.models import create_model
from src.training import Trainer, CrossValidator
from src.config import Config


def generate_dummy_data(n_samples=1000, input_dim=784, n_classes=10):
    """Generate dummy data."""
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


def train_function(model, train_loader, val_loader, config):
    """Training function for cross-validation."""
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
    print("Cross-Validation Training Example")
    print("="*50)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'default_config.yaml'
    config = Config.from_yaml(str(config_path))
    
    # Generate data
    X, y = generate_dummy_data(n_samples=500)
    dataset = BaseDataset(X, y)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create cross-validator
    cv = CrossValidator(n_splits=5, stratified=True, random_state=42)
    
    # Model factory function
    def model_fn():
        model_config = config.to_dict()['model']
        return create_model(config.get('model.type'), model_config)
    
    # Run cross-validation
    training_config = {
        **config.to_dict()['training'],
        **config.to_dict()['logging'],
        **config.to_dict()['data'],
        'epochs': 5,  # Reduce epochs for demo
    }
    
    results = cv.evaluate(
        model_fn=model_fn,
        dataset=dataset,
        train_fn=train_function,
        config=training_config
    )
    
    print("\n" + "="*50)
    print("Cross-Validation Complete!")
    print(f"Average Validation Accuracy: {results['avg_val_acc']:.2f}% ± {results['std_val_acc']:.2f}%")
    print(f"Average Validation Loss: {results['avg_val_loss']:.4f} ± {results['std_val_loss']:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()
