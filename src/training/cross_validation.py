"""
Cross-validation utilities for model evaluation.
"""

import torch
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, Any, List, Tuple, Callable
from torch.utils.data import DataLoader, Subset

from ..data.dataloader import BaseDataset


class CrossValidator:
    """K-Fold cross-validation for deep learning models."""
    
    def __init__(
        self,
        n_splits: int = 5,
        stratified: bool = False,
        random_state: int = 42
    ):
        """
        Args:
            n_splits: Number of folds
            stratified: Whether to use stratified k-fold
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        
        if stratified:
            self.kfold = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )
        else:
            self.kfold = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )
    
    def split_dataset(
        self,
        dataset: BaseDataset
    ) -> List[Tuple[Subset, Subset]]:
        """
        Split dataset into k folds.
        
        Args:
            dataset: Dataset to split
        
        Returns:
            List of (train_subset, val_subset) tuples
        """
        splits = []
        indices = np.arange(len(dataset))
        
        # Get labels for stratified split
        if isinstance(self.kfold, StratifiedKFold):
            labels = [dataset[i][1].item() for i in indices]
            fold_generator = self.kfold.split(indices, labels)
        else:
            fold_generator = self.kfold.split(indices)
        
        for train_idx, val_idx in fold_generator:
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            splits.append((train_subset, val_subset))
        
        return splits
    
    def evaluate(
        self,
        model_fn: Callable,
        dataset: BaseDataset,
        train_fn: Callable,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.
        
        Args:
            model_fn: Function to create a new model instance
            dataset: Complete dataset
            train_fn: Training function that takes (model, train_loader, val_loader, config)
            config: Training configuration
        
        Returns:
            Dictionary with cross-validation results
        """
        splits = self.split_dataset(dataset)
        fold_results = []
        
        print(f"Starting {self.n_splits}-fold cross-validation...")
        
        for fold, (train_subset, val_subset) in enumerate(splits):
            print(f"\nFold {fold + 1}/{self.n_splits}")
            
            # Create data loaders for this fold
            train_loader = DataLoader(
                train_subset,
                batch_size=config.get('batch_size', 32),
                shuffle=True,
                num_workers=config.get('num_workers', 4)
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=config.get('batch_size', 32),
                shuffle=False,
                num_workers=config.get('num_workers', 4)
            )
            
            # Create new model for this fold
            model = model_fn()
            
            # Train and evaluate
            history = train_fn(model, train_loader, val_loader, config)
            
            fold_results.append({
                'fold': fold,
                'history': history,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_train_acc': history['train_accuracy'][-1],
                'final_val_acc': history['val_accuracy'][-1],
            })
        
        # Aggregate results
        avg_val_loss = np.mean([r['final_val_loss'] for r in fold_results])
        std_val_loss = np.std([r['final_val_loss'] for r in fold_results])
        avg_val_acc = np.mean([r['final_val_acc'] for r in fold_results])
        std_val_acc = np.std([r['final_val_acc'] for r in fold_results])
        
        results = {
            'fold_results': fold_results,
            'avg_val_loss': avg_val_loss,
            'std_val_loss': std_val_loss,
            'avg_val_acc': avg_val_acc,
            'std_val_acc': std_val_acc,
        }
        
        print(f"\nCross-Validation Results:")
        print(f"Average Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
        print(f"Average Val Accuracy: {avg_val_acc:.2f}% ± {std_val_acc:.2f}%")
        
        return results
