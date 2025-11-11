"""
Evaluation utilities for trained models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model: Trained model to evaluate
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, data_loader: DataLoader) -> tuple:
        """
        Get predictions for a dataset.
        
        Args:
            data_loader: Data loader for the dataset
        
        Returns:
            Tuple of (predictions, true_labels, probabilities)
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                
                # Get probabilities
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the model.
        
        Args:
            data_loader: Data loader for evaluation
        
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions, labels, probabilities = self.predict(data_loader)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(labels, predictions)
        
        # Classification report
        class_report = classification_report(labels, predictions, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities,
        }
        
        # ROC AUC for binary classification
        if len(np.unique(labels)) == 2:
            try:
                roc_auc = roc_auc_score(labels, probabilities[:, 1])
                results['roc_auc'] = roc_auc
            except Exception as e:
                print(f"Could not compute ROC AUC: {e}")
        
        return results
    
    def plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            conf_matrix: Confusion matrix
            class_names: Names of classes
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names if class_names else 'auto',
            yticklabels=class_names if class_names else 'auto'
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve for binary classification.
        
        Args:
            labels: True labels
            probabilities: Predicted probabilities
            save_path: Path to save the plot
        """
        if len(np.unique(labels)) != 2:
            print("ROC curve only available for binary classification")
            return
        
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        roc_auc = roc_auc_score(labels, probabilities[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_evaluation_report(self, results: Dict[str, Any]):
        """Print formatted evaluation report."""
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1 Score:  {results['f1_score']:.4f}")
        
        if 'roc_auc' in results:
            print(f"ROC AUC:   {results['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
        print("="*50 + "\n")


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    save_plots: bool = False,
    output_dir: str = 'evaluation_results'
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use
        save_plots: Whether to save evaluation plots
        output_dir: Directory to save plots
    
    Returns:
        Evaluation results
    """
    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate(data_loader)
    evaluator.print_evaluation_report(results)
    
    if save_plots:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            results['confusion_matrix'],
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )
        
        # Plot ROC curve if binary classification
        if 'roc_auc' in results:
            evaluator.plot_roc_curve(
                results['labels'],
                results['probabilities'],
                save_path=os.path.join(output_dir, 'roc_curve.png')
            )
    
    return results
