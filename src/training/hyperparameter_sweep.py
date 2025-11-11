"""
Hyperparameter sweeping utilities.
"""

import torch
import itertools
from typing import Dict, Any, List, Callable
import json
import os
from pathlib import Path


class HyperparameterSweeper:
    """Grid search for hyperparameter tuning."""
    
    def __init__(self, param_grid: Dict[str, List[Any]], output_dir: str = 'sweep_results'):
        """
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            output_dir: Directory to save sweep results
        """
        self.param_grid = param_grid
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all possible configurations from parameter grid."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        configs = []
        for combination in itertools.product(*values):
            config = dict(zip(keys, combination))
            configs.append(config)
        
        return configs
    
    def run_sweep(
        self,
        model_fn: Callable,
        train_fn: Callable,
        train_loader: Any,
        val_loader: Any,
        base_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Run hyperparameter sweep.
        
        Args:
            model_fn: Function to create model
            train_fn: Training function
            train_loader: Training data loader
            val_loader: Validation data loader
            base_config: Base configuration to merge with sweep parameters
        
        Returns:
            List of results for each configuration
        """
        configs = self.generate_configs()
        print(f"Running hyperparameter sweep with {len(configs)} configurations...")
        
        for idx, sweep_params in enumerate(configs):
            print(f"\n{'='*50}")
            print(f"Configuration {idx + 1}/{len(configs)}")
            print(f"Parameters: {sweep_params}")
            print(f"{'='*50}")
            
            # Merge sweep parameters with base config
            config = {**base_config, **sweep_params}
            
            # Create model
            model = model_fn(config)
            
            # Train
            history = train_fn(model, train_loader, val_loader, config)
            
            # Store results
            result = {
                'config_idx': idx,
                'parameters': sweep_params,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_train_acc': history['train_accuracy'][-1],
                'final_val_acc': history['val_accuracy'][-1],
                'history': history,
            }
            
            self.results.append(result)
            
            # Save intermediate results
            self._save_results()
        
        # Find best configuration
        best_result = min(self.results, key=lambda x: x['final_val_loss'])
        print(f"\n{'='*50}")
        print("Best Configuration:")
        print(f"Parameters: {best_result['parameters']}")
        print(f"Val Loss: {best_result['final_val_loss']:.4f}")
        print(f"Val Accuracy: {best_result['final_val_acc']:.2f}%")
        print(f"{'='*50}")
        
        return self.results
    
    def _save_results(self):
        """Save sweep results to file."""
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            ser_result = {
                'config_idx': result['config_idx'],
                'parameters': result['parameters'],
                'final_train_loss': result['final_train_loss'],
                'final_val_loss': result['final_val_loss'],
                'final_train_acc': result['final_train_acc'],
                'final_val_acc': result['final_val_acc'],
            }
            serializable_results.append(ser_result)
        
        output_file = self.output_dir / 'sweep_results.json'
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def get_best_config(self, metric: str = 'val_loss', mode: str = 'min') -> Dict[str, Any]:
        """
        Get the best configuration based on a metric.
        
        Args:
            metric: Metric to optimize ('val_loss' or 'val_acc')
            mode: 'min' for minimization, 'max' for maximization
        
        Returns:
            Best parameter configuration
        """
        if not self.results:
            raise ValueError("No results available. Run sweep first.")
        
        metric_key = f'final_{metric}'
        
        if mode == 'min':
            best_result = min(self.results, key=lambda x: x[metric_key])
        else:
            best_result = max(self.results, key=lambda x: x[metric_key])
        
        return best_result['parameters']


class RandomSearch:
    """Random search for hyperparameter tuning."""
    
    def __init__(
        self,
        param_distributions: Dict[str, Callable],
        n_iter: int = 10,
        output_dir: str = 'random_search_results'
    ):
        """
        Args:
            param_distributions: Dictionary mapping parameter names to sampling functions
            n_iter: Number of random configurations to try
            output_dir: Directory to save results
        """
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def sample_config(self) -> Dict[str, Any]:
        """Sample a random configuration."""
        return {name: sampler() for name, sampler in self.param_distributions.items()}
    
    def run_search(
        self,
        model_fn: Callable,
        train_fn: Callable,
        train_loader: Any,
        val_loader: Any,
        base_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run random search."""
        print(f"Running random search with {self.n_iter} iterations...")
        
        for idx in range(self.n_iter):
            print(f"\n{'='*50}")
            print(f"Iteration {idx + 1}/{self.n_iter}")
            
            # Sample configuration
            sampled_params = self.sample_config()
            print(f"Parameters: {sampled_params}")
            print(f"{'='*50}")
            
            # Merge with base config
            config = {**base_config, **sampled_params}
            
            # Create model and train
            model = model_fn(config)
            history = train_fn(model, train_loader, val_loader, config)
            
            # Store results
            result = {
                'iteration': idx,
                'parameters': sampled_params,
                'final_val_loss': history['val_loss'][-1],
                'final_val_acc': history['val_accuracy'][-1],
            }
            self.results.append(result)
        
        return self.results
