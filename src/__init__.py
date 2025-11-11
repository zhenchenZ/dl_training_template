"""
Deep Learning Training Template

A minimal yet comprehensive deep learning training template.
"""

__version__ = '0.1.0'

from .data import (
    BaseDataset,
    DataProcessor,
    create_data_loaders,
    split_data,
    Preprocessor,
)

from .models import (
    BaseModel,
    SimpleMLP,
    SimpleCNN,
    create_model,
)

from .training import (
    Trainer,
    CrossValidator,
    HyperparameterSweeper,
    RandomSearch,
)

from .evaluation import (
    ModelEvaluator,
    evaluate_model,
)

from .utils import (
    MetricsTracker,
    CheckpointManager,
    TrainingLogger,
)

from .config import Config

__all__ = [
    # Data
    'BaseDataset',
    'DataProcessor',
    'create_data_loaders',
    'split_data',
    'Preprocessor',
    # Models
    'BaseModel',
    'SimpleMLP',
    'SimpleCNN',
    'create_model',
    # Training
    'Trainer',
    'CrossValidator',
    'HyperparameterSweeper',
    'RandomSearch',
    # Evaluation
    'ModelEvaluator',
    'evaluate_model',
    # Utils
    'MetricsTracker',
    'CheckpointManager',
    'TrainingLogger',
    # Config
    'Config',
]
