"""
Basic tests for models module.
"""

import unittest
import torch
from src.models import SimpleMLP, SimpleCNN, create_model, BaseModel


class TestModelsModule(unittest.TestCase):
    """Test model creation and functionality."""
    
    def test_simple_mlp(self):
        """Test SimpleMLP model."""
        config = {
            'input_dim': 784,
            'output_dim': 10,
            'hidden_dims': [256, 128],
            'dropout_rate': 0.5
        }
        
        model = SimpleMLP(config)
        self.assertIsInstance(model, BaseModel)
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 784)
        output = model(x)
        
        self.assertEqual(output.shape, (batch_size, 10))
    
    def test_simple_cnn(self):
        """Test SimpleCNN model."""
        config = {
            'input_channels': 1,
            'output_dim': 10,
            'num_filters': [32, 64],
            'kernel_size': 3,
            'dropout_rate': 0.3,
            'flatten_size': 3136  # 64 * 7 * 7 after two pooling layers on 28x28 input
        }
        
        model = SimpleCNN(config)
        self.assertIsInstance(model, BaseModel)
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)
        output = model(x)
        
        self.assertEqual(output.shape, (batch_size, 10))
    
    def test_create_model(self):
        """Test model factory function."""
        config = {
            'input_dim': 784,
            'output_dim': 10,
            'hidden_dims': [128, 64],
        }
        
        model = create_model('mlp', config)
        self.assertIsInstance(model, SimpleMLP)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            create_model('invalid_type', config)
    
    def test_get_num_parameters(self):
        """Test parameter counting."""
        config = {
            'input_dim': 10,
            'output_dim': 2,
            'hidden_dims': [5],
            'dropout_rate': 0.0
        }
        
        model = SimpleMLP(config)
        num_params = model.get_num_parameters()
        
        # Should have parameters from layers
        self.assertGreater(num_params, 0)


if __name__ == '__main__':
    unittest.main()
