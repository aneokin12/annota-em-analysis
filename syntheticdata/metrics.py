import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score



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

# --- Custom metrics for 0=NOT_RELEVANT, 1=RELEVANT, 2=REQUIRED ---
def calculate_custom_metrics(predictions, ground_truth, binary_schema=False):
    """
    Custom metrics for DS-EM predictions vs expert labels with 0=NOT_RELEVANT, 1=RELEVANT, 2=REQUIRED.
    Args:
        predictions: 1D array of DS-EM predictions (0=not relevant, 1=relevant)
        ground_truth: 1D array of expert labels (0=not relevant, 1=relevant, 2=required)
        binary_schema: If True, treat 2 as 1 for metrics calculations (standard binary schema)
    Returns:
        dict: precision, negative_precision, recall, negative_recall, positive_f1, negative_f1, macro_f1
    """
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    
    # If binary_schema is True, treat 2 as 1 for metrics calculations
    if binary_schema:
        # Create a copy of ground_truth where 2 is treated as 1
        gt_for_metrics = np.where(ground_truth == 2, 1, ground_truth)
    else:
        gt_for_metrics = ground_truth
    
    # print("preds",predictions)
    # print("gt",ground_truth)
    # Precision: Of all predicted relevant, what fraction are labeled relevant or required?
    pred_relevant = predictions == 1
    if binary_schema:
        # In binary schema, treat 2 as 1, so relevant is just 1
        gt_relevant_or_required = gt_for_metrics == 1
    else:
        # Original logic: relevant or required (1 or 2)
        gt_relevant_or_required = np.isin(ground_truth, [1, 2])
    
    if np.sum(pred_relevant) > 0:
        precision = np.sum(gt_relevant_or_required & pred_relevant) / np.sum(pred_relevant)
    else:
        precision = np.nan

    # Negative Precision: Of all predicted not relevant, what fraction are labeled not relevant or relevant?
    pred_not_relevant = predictions == 0
    if binary_schema:
        # In binary schema, not required is just 0
        gt_not_required = gt_for_metrics == 0
    else:
        # Original logic: not required (0 or 1)
        gt_not_required = np.isin(ground_truth, [0, 1])
    
    if np.sum(pred_not_relevant) > 0:
        negative_precision = np.sum(gt_not_required & pred_not_relevant) / np.sum(pred_not_relevant)
    else:
        negative_precision = np.nan

    # Recall: Of all required, what fraction are predicted relevant?
    if binary_schema:
        # In binary schema, required is just 1
        gt_required = gt_for_metrics == 1
    else:
        # Original logic: required is 2
        gt_required = np.isin(ground_truth, [2])
    
    if np.sum(gt_required) > 0:
        recall = np.sum(predictions[gt_required] == 1) / np.sum(gt_required)
    else:
        recall = np.nan

    # Negative Recall: Of all not relevant, what fraction are predicted not relevant?
    gt_not_relevant = ground_truth == 0
    if np.sum(gt_not_relevant) > 0:
        negative_recall = np.sum(predictions[gt_not_relevant] == 0) / np.sum(gt_not_relevant)
    else:
        negative_recall = np.nan

    # Calculate F1 scores
    # Positive F1: harmonic mean of precision and recall
    if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
        positive_f1 = 2 * (precision * recall) / (precision + recall)
    else:
        positive_f1 = np.nan

    # Negative F1: harmonic mean of negative precision and negative recall
    if not np.isnan(negative_precision) and not np.isnan(negative_recall) and (negative_precision + negative_recall) > 0:
        negative_f1 = 2 * (negative_precision * negative_recall) / (negative_precision + negative_recall)
    else:
        negative_f1 = np.nan

    # Macro-F1: unweighted mean of positive and negative F1
    f1_scores = []
    if not np.isnan(positive_f1):
        f1_scores.append(positive_f1)
    if not np.isnan(negative_f1):
        f1_scores.append(negative_f1)
    
    if len(f1_scores) > 0:
        macro_f1 = np.mean(f1_scores)
    else:
        macro_f1 = np.nan

    return {
        'precision': precision,
        'negative_precision': negative_precision,
        'recall': recall,
        'negative_recall': negative_recall,
        'positive_f1': positive_f1,
        'negative_f1': negative_f1,
        'macro_f1': macro_f1
    }

def print_custom_metrics_summary(predictions, ground_truth, round_num=None, binary_schema=False):
    metrics = calculate_custom_metrics(predictions, ground_truth, binary_schema)
    prefix = f"Round {round_num}: " if round_num is not None else ""
    # print(f"{prefix}Precision={metrics['precision']:.3f}, NegPrecision={metrics['negative_precision']:.3f}, "
        #   f"Recall={metrics['recall']:.3f}, NegRecall={metrics['negative_recall']:.3f}, "
        #   f"PosF1={metrics['positive_f1']:.3f}, NegF1={metrics['negative_f1']:.3f}, MacroF1={metrics['macro_f1']:.3f}")
    return metrics 