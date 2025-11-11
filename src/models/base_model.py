"""
Base model architecture classes.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.input_dim = config.get('input_dim', 784)
        self.output_dim = config.get('output_dim', 10)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SimpleMLP(BaseModel):
    """Simple Multi-Layer Perceptron model."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Model configuration with 'input_dim', 'hidden_dims', 'output_dim'
        """
        super().__init__(config)
        
        hidden_dims = config.get('hidden_dims', [256, 128])
        dropout_rate = config.get('dropout_rate', 0.5)
        
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class SimpleCNN(BaseModel):
    """Simple Convolutional Neural Network."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Model configuration with 'input_channels', 'output_dim'
        """
        super().__init__(config)
        
        input_channels = config.get('input_channels', 1)
        num_filters = config.get('num_filters', [32, 64, 128])
        kernel_size = config.get('kernel_size', 3)
        dropout_rate = config.get('dropout_rate', 0.5)
        
        # Convolutional layers
        conv_layers = []
        in_channels = input_channels
        
        for out_channels in num_filters:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_rate * 0.5)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate the flattened size after convolutions
        # This is a placeholder; actual size depends on input dimensions
        self.flatten_size = config.get('flatten_size', 2048)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, self.output_dim)
        )
        
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for ResNet-like architectures."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


def create_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('mlp', 'cnn')
        config: Model configuration dictionary
    
    Returns:
        Instantiated model
    """
    models = {
        'mlp': SimpleMLP,
        'cnn': SimpleCNN,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](config)
