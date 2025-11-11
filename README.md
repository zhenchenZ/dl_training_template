# Deep Learning Training Template

A minimal yet comprehensive deep learning training template that enables training and evaluation with all essential features for production-ready ML workflows.

## Features

### 🔧 Core Functionality
- **Data Processing Modules**: Flexible dataset classes, data loaders, preprocessing, and augmentation utilities
- **Model Architecture Definition**: Base model classes and example implementations (MLP, CNN, ResNet blocks)
- **Training Dynamics Tracking**: Comprehensive metrics tracking and monitoring
- **Logging System**: TensorBoard integration and custom logging utilities
- **Checkpointing**: Automatic model checkpointing with best model saving
- **Evaluation Framework**: Detailed model evaluation with multiple metrics and visualization

### 🚀 Advanced Features
- **Hyperparameter Sweeping**: Grid search and random search for optimization
- **Cross-Validation**: K-fold and stratified k-fold cross-validation support
- **Configuration Management**: YAML-based configuration system
- **Early Stopping**: Automatic training termination based on validation metrics
- **Learning Rate Scheduling**: Adaptive learning rate adjustment
- **Gradient Clipping**: Prevent exploding gradients

## Installation

```bash
# Clone the repository
git clone https://github.com/zhenchenZ/dl_training_template.git
cd dl_training_template

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Start

### Basic Training Example

```python
import torch
from src.data import BaseDataset, create_data_loaders
from src.models import create_model
from src.training import Trainer
from src.config import Config

# Load configuration
config = Config.from_yaml('configs/default_config.yaml')

# Create datasets and loaders
train_dataset = BaseDataset(X_train, y_train)
val_dataset = BaseDataset(X_val, y_val)
train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)

# Create model
model = create_model('mlp', config.to_dict()['model'])

# Setup training
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = Trainer(model, optimizer, criterion, device, config.to_dict()['training'])

# Train
history = trainer.train(train_loader, val_loader)
```

### Run Example Scripts

```bash
# Basic training
python examples/train_basic.py

# Cross-validation
python examples/train_cross_validation.py

# Hyperparameter sweeping
python examples/train_hyperparameter_sweep.py
```

## Project Structure

```
dl_training_template/
├── src/
│   ├── data/               # Data processing modules
│   │   ├── dataloader.py   # Dataset classes and data loaders
│   │   └── preprocessing.py # Preprocessing utilities
│   ├── models/             # Model architectures
│   │   └── base_model.py   # Base model classes and implementations
│   ├── training/           # Training utilities
│   │   ├── trainer.py      # Main trainer class
│   │   ├── cross_validation.py      # Cross-validation
│   │   └── hyperparameter_sweep.py  # Hyperparameter tuning
│   ├── evaluation/         # Evaluation tools
│   │   └── evaluator.py    # Model evaluation and metrics
│   ├── utils/              # Utility functions
│   │   ├── metrics.py      # Metrics tracking
│   │   ├── checkpointing.py # Checkpoint management
│   │   └── logging_utils.py # Logging utilities
│   └── config/             # Configuration management
│       └── config.py       # Config class
├── configs/                # Configuration files
│   ├── default_config.yaml # Default configuration
│   └── cnn_config.yaml     # CNN model configuration
├── examples/               # Example training scripts
│   ├── train_basic.py      # Basic training example
│   ├── train_cross_validation.py    # Cross-validation example
│   └── train_hyperparameter_sweep.py # Hyperparameter sweep example
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## Configuration

The template uses YAML files for configuration. Example configuration:

```yaml
# Data settings
data:
  batch_size: 32
  num_workers: 4
  train_ratio: 0.8
  normalize: true
  augmentation: false

# Model settings
model:
  type: mlp
  input_dim: 784
  output_dim: 10
  hidden_dims: [256, 128]
  dropout_rate: 0.5

# Training settings
training:
  epochs: 10
  learning_rate: 0.001
  weight_decay: 0.00001
  gradient_clip: 1.0
  optimizer: adam
  use_scheduler: true
  early_stopping: true
  patience: 10
```

## Key Components

### Data Module (`src/data/`)

- **BaseDataset**: Custom dataset class for PyTorch
- **DataProcessor**: Data preprocessing and augmentation
- **create_data_loaders()**: Convenient data loader creation
- **split_data()**: Train/validation splitting

### Models Module (`src/models/`)

- **BaseModel**: Abstract base class for all models
- **SimpleMLP**: Multi-layer perceptron implementation
- **SimpleCNN**: Convolutional neural network
- **ResidualBlock**: Building block for ResNet-like architectures
- **create_model()**: Factory function for model creation

### Training Module (`src/training/`)

- **Trainer**: Main training class with full training loop
  - Automatic logging to TensorBoard
  - Checkpoint saving
  - Early stopping
  - Learning rate scheduling
  - Gradient clipping
- **CrossValidator**: K-fold cross-validation
- **HyperparameterSweeper**: Grid search optimization
- **RandomSearch**: Random hyperparameter search

### Evaluation Module (`src/evaluation/`)

- **ModelEvaluator**: Comprehensive model evaluation
  - Accuracy, precision, recall, F1-score
  - Confusion matrix
  - ROC curves (for binary classification)
  - Classification reports
- **evaluate_model()**: Convenience function for quick evaluation

### Utils Module (`src/utils/`)

- **MetricsTracker**: Track and store training metrics
- **CheckpointManager**: Manage model checkpoints
- **TrainingLogger**: Custom logging for training progress
- **AverageMeter**: Compute running averages

### Config Module (`src/config/`)

- **Config**: Configuration management class
  - Load from YAML files
  - Save to YAML files
  - Nested configuration access
  - Dynamic updates

## Usage Examples

### 1. Custom Model Training

```python
from src.models import BaseModel
import torch.nn as nn

class CustomModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
```

### 2. Cross-Validation

```python
from src.training import CrossValidator

cv = CrossValidator(n_splits=5, stratified=True)
results = cv.evaluate(
    model_fn=lambda: create_model('mlp', config),
    dataset=dataset,
    train_fn=train_function,
    config=config
)
print(f"Average accuracy: {results['avg_val_acc']:.2f}%")
```

### 3. Hyperparameter Sweeping

```python
from src.training import HyperparameterSweeper

param_grid = {
    'learning_rate': [0.001, 0.0001, 0.00001],
    'dropout_rate': [0.3, 0.5, 0.7],
    'hidden_dims': [[128, 64], [256, 128], [512, 256]]
}

sweeper = HyperparameterSweeper(param_grid)
results = sweeper.run_sweep(model_fn, train_fn, train_loader, val_loader, base_config)
best_config = sweeper.get_best_config(metric='val_acc', mode='max')
```

### 4. Model Evaluation

```python
from src.evaluation import evaluate_model

results = evaluate_model(
    model=trained_model,
    data_loader=test_loader,
    device=device,
    save_plots=True,
    output_dir='evaluation_results'
)
```

## Monitoring Training

The template automatically logs metrics to TensorBoard:

```bash
tensorboard --logdir=logs
```

Visit `http://localhost:6006` to view:
- Training and validation loss curves
- Accuracy metrics
- Learning rate changes
- Model graphs

## Checkpointing

Checkpoints are automatically saved during training:
- Periodic checkpoints every N epochs (configurable)
- Best model based on validation loss
- Final model after training completes

Load a checkpoint:

```python
from src.utils import CheckpointManager

checkpoint_manager = CheckpointManager('checkpoints')
checkpoint = checkpoint_manager.load_checkpoint('best_model.pth', device)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- tensorboard >= 2.13.0
- pyyaml >= 6.0
- tqdm >= 4.65.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this template in your research, please cite:

```bibtex
@software{dl_training_template,
  author = {zhenchenZ},
  title = {Deep Learning Training Template},
  year = {2024},
  url = {https://github.com/zhenchenZ/dl_training_template}
}
```

## Acknowledgments

This template incorporates best practices from various deep learning frameworks and research projects.