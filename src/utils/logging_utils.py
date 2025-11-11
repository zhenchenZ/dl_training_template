"""
Logging utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = 'training',
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger for training.
    
    Args:
        name: Logger name
        log_file: Optional file to write logs to
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """Custom logger for training progress."""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Args:
            log_file: Optional file to write logs to
        """
        self.logger = setup_logger(log_file=log_file)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None
    ):
        """Log epoch results."""
        msg = f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
        
        if val_loss is not None and val_acc is not None:
            msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        
        self.logger.info(msg)
    
    def log_batch(self, batch_idx: int, loss: float, accuracy: float):
        """Log batch results."""
        self.logger.debug(f"Batch {batch_idx}: Loss: {loss:.4f}, Acc: {accuracy:.2f}%")
    
    def log_model_info(self, model):
        """Log model information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {model.__class__.__name__}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def log_training_start(self, config):
        """Log training start."""
        self.logger.info("="*50)
        self.logger.info("Starting training with configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("="*50)
    
    def log_training_end(self, best_acc: float, best_loss: float):
        """Log training end."""
        self.logger.info("="*50)
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation accuracy: {best_acc:.2f}%")
        self.logger.info(f"Best validation loss: {best_loss:.4f}")
        self.logger.info("="*50)
    
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
