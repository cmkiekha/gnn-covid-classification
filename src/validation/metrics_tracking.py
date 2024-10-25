# src/validation/metrics_tracking.py
import logging
import numpy as np
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)

class MetricsTracker:
    """
    Track and store metrics during training and evaluation.
    """
    
    def __init__(self):
        """Initialize metrics containers."""
        self.training_metrics = {
            'generator_loss': [],
            'critic_loss': [],
            'gradient_penalty': [],
            'wasserstein_distance': []
        }
        self.generation_metrics = {
            'ks_statistics': [],
            'mmd_scores': [],
            'correlation_preservation': []
        }
        self.fold_metrics = {}
        
    def update_training_metrics(self, metrics_dict: Dict[str, float]):
        """
        Update training phase metrics.
        
        Args:
            metrics_dict: Dictionary of metric names and values
        """
        if not isinstance(metrics_dict, dict):
            logger.error("Metrics must be provided as a dictionary")
            return
            
        for key, value in metrics_dict.items():
            if key in self.training_metrics:
                self.training_metrics[key].append(value)
                
    def update_generation_metrics(self, metrics_dict: Dict[str, float]):
        """
        Update generation quality metrics.
        
        Args:
            metrics_dict: Dictionary of metric names and values
        """
        if not isinstance(metrics_dict, dict):
            logger.error("Metrics must be provided as a dictionary")
            return
            
        for key, value in metrics_dict.items():
            if key in self.generation_metrics:
                self.generation_metrics[key].append(value)
                
    def update_fold_metrics(self, fold_number: int, metrics: Dict[str, float]):
        """
        Track metrics for each cross-validation fold.
        
        Args:
            fold_number: Current fold number
            metrics: Dictionary of metrics for this fold
        """
        if fold_number not in self.fold_metrics:
            self.fold_metrics[fold_number] = {}
        self.fold_metrics[fold_number].update(metrics)
        
    def get_cross_validation_summary(self) -> Dict[str, float]:
        """
        Compute summary statistics across all folds.
        
        Returns:
            Dictionary containing mean and std for each metric
        """
        summary = {}
        metrics_to_summarize = [
            'generator_loss', 
            'critic_loss', 
            'ks_statistics'
        ]
        
        for metric in metrics_to_summarize:
            values = [
                fold.get(metric, []) 
                for fold in self.fold_metrics.values()
            ]
            if values:
                summary[f'mean_{metric}'] = float(np.mean(values))
                summary[f'std_{metric}'] = float(np.std(values))
                
        return summary