"""
Trainer class for model training with tracking, logging, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import os
from pathlib import Path

from ..utils.metrics import MetricsTracker
from ..utils.checkpointing import CheckpointManager


class Trainer:
    """Main trainer class for deep learning models."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Args:
            model: Neural network model
            optimizer: Optimizer for training
            criterion: Loss function
            device: Device to train on
            config: Training configuration
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        
        # Training settings
        self.epochs = config.get('epochs', 10)
        self.gradient_clip = config.get('gradient_clip', None)
        
        # Logging and checkpointing
        self.log_dir = config.get('log_dir', 'logs')
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        self.save_freq = config.get('save_freq', 1)
        
        # Initialize components
        self.metrics_tracker = MetricsTracker()
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Learning rate scheduler (optional)
        self.scheduler = None
        if config.get('use_scheduler', False):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config.get('scheduler_patience', 5),
                factor=config.get('scheduler_factor', 0.5)
            )
        
        # Early stopping
        self.early_stopping = config.get('early_stopping', False)
        self.patience = config.get('patience', 10)
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Current epoch and step
        self.current_epoch = 0
        self.global_step = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Logging
            if batch_idx % 10 == 0:
                self.writer.add_scalar(
                    'Train/BatchLoss',
                    loss.item(),
                    self.global_step
                )
            
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        print(f"Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Log training metrics
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
            
            print(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%")
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                
                # Log validation metrics
                self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
                
                print(f"Epoch {epoch}: Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.2f}%")
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    self.scheduler.step(val_metrics['loss'])
                
                # Early stopping
                if self.early_stopping:
                    if val_metrics['loss'] < self.best_loss:
                        self.best_loss = val_metrics['loss']
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            print(f"Early stopping triggered after {epoch + 1} epochs")
                            break
            
            # Checkpointing
            if (epoch + 1) % self.save_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'] if val_loader else None,
                }
                self.checkpoint_manager.save_checkpoint(
                    checkpoint,
                    f"checkpoint_epoch_{epoch}.pth"
                )
        
        # Save final model
        self.checkpoint_manager.save_checkpoint(
            {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            "final_model.pth"
        )
        
        self.writer.close()
        return history
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
