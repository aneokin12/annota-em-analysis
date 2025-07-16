import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def calculate_f1_score(predictions, ground_truth):
    """
    Calculate F1 score between predictions and ground truth.
    
    Args:
        predictions: 1D array of predicted labels (0 or 1)
        ground_truth: 1D array of true labels (0 or 1)
    
    Returns:
        f1: F1 score (harmonic mean of precision and recall)
    """
    return f1_score(ground_truth, predictions, zero_division=0)

def calculate_precision(predictions, ground_truth):
    """
    Calculate precision between predictions and ground truth.
    
    Args:
        predictions: 1D array of predicted labels (0 or 1)
        ground_truth: 1D array of true labels (0 or 1)
    
    Returns:
        precision: Precision score
    """
    return precision_score(ground_truth, predictions, zero_division=0)

def calculate_recall(predictions, ground_truth):
    """
    Calculate recall between predictions and ground truth.
    
    Args:
        predictions: 1D array of predicted labels (0 or 1)
        ground_truth: 1D array of true labels (0 or 1)
    
    Returns:
        recall: Recall score
    """
    return recall_score(ground_truth, predictions, zero_division=0)

def calculate_accuracy(predictions, ground_truth):
    """
    Calculate accuracy between predictions and ground truth.
    
    Args:
        predictions: 1D array of predicted labels (0 or 1)
        ground_truth: 1D array of true labels (0 or 1)
    
    Returns:
        accuracy: Accuracy score
    """
    return accuracy_score(ground_truth, predictions)

def calculate_all_metrics(predictions, ground_truth):
    """
    Calculate all metrics (F1, precision, recall, accuracy) at once.
    
    Args:
        predictions: 1D array of predicted labels (0 or 1)
        ground_truth: 1D array of true labels (0 or 1)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    return {
        'f1': calculate_f1_score(predictions, ground_truth),
        'precision': calculate_precision(predictions, ground_truth),
        'recall': calculate_recall(predictions, ground_truth),
        'accuracy': calculate_accuracy(predictions, ground_truth)
    }

def print_metrics_summary(predictions, ground_truth, round_num=None):
    """
    Print a summary of all metrics.
    
    Args:
        predictions: 1D array of predicted labels (0 or 1)
        ground_truth: 1D array of true labels (0 or 1)
        round_num: Optional round number for labeling
    """
    metrics = calculate_all_metrics(predictions, ground_truth)
    
    prefix = f"Round {round_num}: " if round_num is not None else ""
    print(f"{prefix}F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, "
          f"Recall={metrics['recall']:.3f}, Accuracy={metrics['accuracy']:.3f}")
    
    return metrics 