import numpy as np

# ---------- helper masks ----------

def _to_int(a):
    return np.asarray(a, dtype=int)

def _safe_div(num, den):
    return 0.0 if den == 0 else num / den

# ---------- metrics ----------

def positive_precision(y_true, y_pred):
    """P(y_true in {1,2} | y_pred == 1)"""
    y_true, y_pred = _to_int(y_true), _to_int(y_pred)
    pred_pos_mask = (y_pred == 1)
    correct = np.isin(y_true[pred_pos_mask], [1, 2]).sum()
    return _safe_div(correct, pred_pos_mask.sum())

def positive_recall(y_true, y_pred):
    """P(y_pred == 1 | y_true == 2)"""
    y_true, y_pred = _to_int(y_true), _to_int(y_pred)
    req_mask = (y_true == 2)
    caught = (y_pred[req_mask] == 1).sum()
    return _safe_div(caught, req_mask.sum())

def negative_precision(y_true, y_pred):
    """P(y_true in {0,1} | y_pred == 0)"""
    y_true, y_pred = _to_int(y_true), _to_int(y_pred)
    pred_neg_mask = (y_pred == 0)
    correct = np.isin(y_true[pred_neg_mask], [0, 1]).sum()
    return _safe_div(correct, pred_neg_mask.sum())

def negative_recall(y_true, y_pred):
    """P(y_pred == 0 | y_true == 0)"""
    y_true, y_pred = _to_int(y_true), _to_int(y_pred)
    not_rel_mask = (y_true == 0)
    caught = (y_pred[not_rel_mask] == 0).sum()
    return _safe_div(caught, not_rel_mask.sum())

def macro_f1(y_true, y_pred):
    pp, pr = positive_precision(y_true, y_pred), positive_recall(y_true, y_pred)
    np_, nr = negative_precision(y_true, y_pred), negative_recall(y_true, y_pred)

    pos_f1 = _safe_div(2 * pp * pr, pp + pr) if pp and pr else 0.0
    neg_f1 = _safe_div(2 * np_ * nr, np_ + nr) if np_ and nr else 0.0

    return (pos_f1 + neg_f1) / 2

def accuracy(y_true, y_pred):
    """
    Custom accuracy: a prediction of 1 is counted correct for y_true 1 **or** 2.
    """
    y_true, y_pred = _to_int(y_true), _to_int(y_pred)
    correct = ((y_true == 0) & (y_pred == 0) |          # true negatives
               (y_true != 0) & (y_pred == 1)).sum()     # both 1 and 2 count
    return correct / len(y_true) if len(y_true) else 0.0
