"""
Basic tests for utils module.
"""

import unittest
import torch
import os
import tempfile
import shutil
from pathlib import Path
from src.utils import MetricsTracker, CheckpointManager, AverageMeter


class TestUtilsModule(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_metrics_tracker(self):
        """Test MetricsTracker."""
        tracker = MetricsTracker()
        
        # Update metrics
        tracker.update({'loss': 0.5, 'accuracy': 0.8})
        self.assertEqual(tracker.get_metric('loss'), 0.5)
        self.assertEqual(tracker.get_metric('accuracy'), 0.8)
        
        # Update again
        tracker.update({'loss': 0.3, 'accuracy': 0.85})
        self.assertEqual(len(tracker.get_history('loss')), 2)
        
        # Test statistics
        stats = tracker.compute_statistics('loss')
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
    
    def test_average_meter(self):
        """Test AverageMeter."""
        meter = AverageMeter('test')
        
        meter.update(1.0)
        self.assertEqual(meter.avg, 1.0)
        
        meter.update(2.0)
        self.assertEqual(meter.avg, 1.5)
        
        meter.reset()
        self.assertEqual(meter.avg, 0)
    
    def test_checkpoint_manager(self):
        """Test CheckpointManager."""
        checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        manager = CheckpointManager(checkpoint_dir)
        
        # Test saving checkpoint
        state = {
            'epoch': 0,
            'model_state_dict': {},
            'loss': 0.5
        }
        manager.save_checkpoint(state, 'test_checkpoint.pth')
        
        # Check that file exists
        checkpoint_path = Path(checkpoint_dir) / 'test_checkpoint.pth'
        self.assertTrue(checkpoint_path.exists())
        
        # Test loading checkpoint
        loaded_state = manager.load_checkpoint('test_checkpoint.pth')
        self.assertEqual(loaded_state['epoch'], 0)
        self.assertEqual(loaded_state['loss'], 0.5)
        
        # Test listing checkpoints
        checkpoints = manager.list_checkpoints()
        self.assertIn('test_checkpoint.pth', checkpoints)


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_config_creation(self):
        """Test Config creation."""
        from src.config import Config
        
        config = Config()
        self.assertIsNotNone(config.to_dict())
        
        # Test default config has required keys
        self.assertIn('data', config.to_dict())
        self.assertIn('model', config.to_dict())
        self.assertIn('training', config.to_dict())
    
    def test_config_get_set(self):
        """Test Config get/set methods."""
        from src.config import Config
        
        config = Config()
        
        # Test get
        batch_size = config.get('data.batch_size')
        self.assertIsNotNone(batch_size)
        
        # Test set
        config.set('data.batch_size', 64)
        self.assertEqual(config.get('data.batch_size'), 64)
    
    def test_config_yaml(self):
        """Test Config YAML save/load."""
        from src.config import Config
        
        config = Config()
        yaml_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Save to YAML
        config.to_yaml(yaml_path)
        self.assertTrue(os.path.exists(yaml_path))
        
        # Load from YAML
        loaded_config = Config.from_yaml(yaml_path)
        self.assertEqual(
            config.get('data.batch_size'),
            loaded_config.get('data.batch_size')
        )


if __name__ == '__main__':
    unittest.main()
