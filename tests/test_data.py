"""
Basic tests for data module.
"""

import unittest
import numpy as np
import torch
from src.data import BaseDataset, DataProcessor, split_data, Preprocessor


class TestDataModule(unittest.TestCase):
    """Test data processing functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.X = np.random.randn(100, 10).astype(np.float32)
        self.y = np.random.randint(0, 2, 100)
    
    def test_base_dataset(self):
        """Test BaseDataset creation."""
        dataset = BaseDataset(self.X, self.y)
        self.assertEqual(len(dataset), 100)
        
        # Test __getitem__
        sample, label = dataset[0]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
    
    def test_split_data(self):
        """Test data splitting."""
        X_train, y_train, X_val, y_val = split_data(
            self.X, self.y, train_ratio=0.8, random_state=42
        )
        
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_val), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_val), 20)
    
    def test_preprocessor(self):
        """Test Preprocessor."""
        preprocessor = Preprocessor(method='standard')
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(self.X)
        self.assertEqual(X_transformed.shape, self.X.shape)
        
        # Check that mean is close to 0 and std close to 1
        self.assertAlmostEqual(np.mean(X_transformed), 0, places=5)
        self.assertAlmostEqual(np.std(X_transformed), 1, places=5)
    
    def test_data_processor(self):
        """Test DataProcessor."""
        config = {
            'normalize': True,
            'augmentation': False,
            'mean': [0.5],
            'std': [0.5]
        }
        processor = DataProcessor(config)
        
        # Test normalization
        data = np.array([1.0, 2.0, 3.0])
        normalized = processor.normalize(data)
        expected = (data - 0.5) / 0.5
        np.testing.assert_array_almost_equal(normalized, expected)


if __name__ == '__main__':
    unittest.main()
