import numpy as np
import matplotlib.pyplot as plt



def positive_precision(y_true, y_pred):
    """
    
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # -----------------------------
    # 1) Positive Precision
    # -----------------------------
    pred_1_indices = np.where(y_pred == 1)[0]
    if len(pred_1_indices) == 0:
        pos_precision = 0.0
    else:
        correct_pos = np.sum(np.isin(y_true[pred_1_indices], [1, 2]))
        pos_precision = correct_pos / len(pred_1_indices)


    return pos_precision


def negative_precision(y_true, y_pred):
    # -----------------------------
    # 2) Negative Precision
    # -----------------------------
    pred_0_indices = np.where(y_pred == 0)[0]
    if len(pred_0_indices) == 0:
        neg_precision = 0.0
    else:
        correct_neg = np.sum(np.isin(y_true[pred_0_indices], [0, 1]))
        neg_precision = correct_neg / len(pred_0_indices)
    return neg_precision

def positive_recall(y_true, y_pred):
    # -----------------------------
    # 3) Positive Recall
    # -----------------------------
    req_indices = np.where(y_true == 2)[0]
    if len(req_indices) == 0:
        pos_recall = 0.0
    else:
        predicted_as_1 = np.sum(y_pred[req_indices] == 1)
        pos_recall = predicted_as_1 / len(req_indices)
    return pos_recall


def negative_recall(y_true, y_pred):
    # -----------------------------
    # 4) Negative Recall
    # -----------------------------
    not_rel_indices = np.where(y_true == 0)[0]
    if len(not_rel_indices) == 0:
        neg_recall = 0.0
    else:
        predicted_as_0 = np.sum(y_pred[not_rel_indices] == 0)
        neg_recall = predicted_as_0 / len(not_rel_indices)

    return neg_recall


def accuracy(y_true, y_pred):
    # -----------------------------
    # Custom Accuracy
    # -----------------------------
    correct_count = 0
    for i in range(len(y_true)):
        if y_true[i] == 2:
            if y_pred[i] == 1:
                correct_count += 1
        elif y_true[i] == 1:
            if y_pred[i] == 1:
                correct_count += 1
        else:  # y_true[i] == 0
            if y_pred[i] == 0:
                correct_count += 1
    custom_accuracy = correct_count / len(y_true) if len(y_true) > 0 else 0.0

    return custom_accuracy



def macro_f1(y_true, y_pred):
    """
    Calculate the macro F1 score based on positive and negative precision and recall.
    
    Parameters:
    y_true (array-like): Ground truth labels where:
           0 = not relevant
           1 = relevant (but not required)
           2 = required
    y_pred (array-like): Predicted labels where:
           0 = not relevant
           1 = relevant
    
    Returns:
    float: The macro F1 score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate positive precision and recall
    pos_precision = positive_precision(y_true, y_pred)
    pos_recall = positive_recall(y_true, y_pred)
    
    # Calculate negative precision and recall
    neg_precision = negative_precision(y_true, y_pred)
    neg_recall = negative_recall(y_true, y_pred)
    
    # Calculate positive F1
    if pos_precision == 0 or pos_recall == 0:
        pos_f1 = 0.0
    else:
        pos_f1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
    
    # Calculate negative F1
    if neg_precision == 0 or neg_recall == 0:
        neg_f1 = 0.0
    else:
        neg_f1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
    
    # Calculate macro F1 (unweighted mean of positive and negative F1)
    macro_f1_score = (pos_f1 + neg_f1) / 2
    
    return macro_f1_score


