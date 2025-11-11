"""
Checkpointing utilities for saving and loading model states.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        filename: str,
        is_best: bool = False
    ):
        """
        Save checkpoint.
        
        Args:
            state: Dictionary containing model state and metadata
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        filepath = self.checkpoint_dir / filename
        torch.save(state, filepath)
        print(f"Checkpoint saved to {filepath}")
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(state, best_path)
            print(f"Best model saved to {best_path}")
    
    def load_checkpoint(
        self,
        filename: str,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            filename: Checkpoint filename
            device: Device to load checkpoint to
        
        Returns:
            Checkpoint state dictionary
        """
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        if device is None:
            checkpoint = torch.load(filepath)
        else:
            checkpoint = torch.load(filepath, map_location=device)
        
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob('*.pth'))
        return [cp.name for cp in checkpoints]
    
    def delete_checkpoint(self, filename: str):
        """Delete a checkpoint."""
        filepath = self.checkpoint_dir / filename
        if filepath.exists():
            filepath.unlink()
            print(f"Deleted checkpoint: {filepath}")
        else:
            print(f"Checkpoint not found: {filepath}")
    
    def keep_last_n_checkpoints(self, n: int = 5):
        """
        Keep only the last n checkpoints.
        
        Args:
            n: Number of checkpoints to keep
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pth'),
            key=lambda x: x.stat().st_mtime
        )
        
        if len(checkpoints) > n:
            for checkpoint in checkpoints[:-n]:
                checkpoint.unlink()
                print(f"Deleted old checkpoint: {checkpoint.name}")


def save_model_architecture(model, filepath: str):
    """
    Save model architecture to a text file.
    
    Args:
        model: PyTorch model
        filepath: Path to save architecture description
    """
    with open(filepath, 'w') as f:
        f.write(str(model))
        f.write('\n\n')
        f.write(f"Total parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
    
    print(f"Model architecture saved to {filepath}")


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save configuration
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from {filepath}")
    return config
