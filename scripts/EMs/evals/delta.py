import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Ground Truth and Required Mask Utilities
# ---------------------------------------------------------
def update_required(relevant, required):
    for i in range(len(relevant)):
        if relevant[i] == 1 and required[i] == 1:
            relevant[i] = 2
    return relevant


def filter_invalid_indices(gt, valids):
    """
    Given a ground truth list and corresponding predictions,
    remove entries where the ground truth is -1.
    """
    # valid_indices = [i for i, val in enumerate(gt) if val != -1]
    filtered_gt = [gt[i] for i in valids]
    print(filtered_gt)
    return filtered_gt

def custom_precision_recall(y_true, y_pred):
    """
    Computes the custom precision, recall, and (optionally) accuracy under the scheme:
      - Ground truth labels: 0 (not relevant), 1 (relevant), 2 (required)
      - Predictions: 0 (not relevant), 1 (relevant)
    Returns:
      pos_precision, neg_precision, pos_recall, neg_recall, custom_accuracy
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

    # -----------------------------
    # 2) Negative Precision
    # -----------------------------
    pred_0_indices = np.where(y_pred == 0)[0]
    if len(pred_0_indices) == 0:
        neg_precision = 0.0
    else:
        correct_neg = np.sum(np.isin(y_true[pred_0_indices], [0, 1]))
        neg_precision = correct_neg / len(pred_0_indices)

    # -----------------------------
    # 3) Positive Recall
    # -----------------------------
    req_indices = np.where(y_true == 2)[0]
    if len(req_indices) == 0:
        pos_recall = 0.0
    else:
        predicted_as_1 = np.sum(y_pred[req_indices] == 1)
        pos_recall = predicted_as_1 / len(req_indices)

    # -----------------------------
    # 4) Negative Recall
    # -----------------------------
    not_rel_indices = np.where(y_true == 0)[0]
    if len(not_rel_indices) == 0:
        neg_recall = 0.0
    else:
        predicted_as_0 = np.sum(y_pred[not_rel_indices] == 0)
        neg_recall = predicted_as_0 / len(not_rel_indices)

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

    return pos_precision, neg_precision, pos_recall, neg_recall, custom_accuracy


def custom_positive_f1(y_true, y_pred):
    """
    Computes the custom positive precision, positive recall, and F1 score given:
      - y_true in {0, 1, 2} where 2 = required
      - y_pred in {0, 1}
    Returns:
      A dictionary with keys 'precision', 'recall', and 'f1'
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    pred_1_indices = np.where(y_pred == 1)[0]
    if len(pred_1_indices) == 0:
        pos_precision = 0.0
    else:
        correct_pos = np.sum(np.isin(y_true[pred_1_indices], [1, 2]))
        pos_precision = correct_pos / len(pred_1_indices)

    true_required_indices = np.where(y_true == 2)[0]
    if len(true_required_indices) == 0:
        pos_recall = 0.0
    else:
        predicted_as_relevant = np.sum(y_pred[true_required_indices] == 1)
        pos_recall = predicted_as_relevant / len(true_required_indices)

    if (pos_precision + pos_recall) == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * pos_precision * pos_recall / (pos_precision + pos_recall)

    return {'precision': pos_precision, 'recall': pos_recall, 'f1': f1}

# ---------------------------------------------------------
# Evaluation Over Time
# ---------------------------------------------------------
def evaluate_time_series_predictions(pred_tensor, ground_truth_datasets):
    """
    Evaluate a time series of predictions against multiple ground truth datasets.

    Parameters:
      pred_tensor: numpy array of shape (n_time_steps, total_length)
      ground_truth_datasets: list of dictionaries, each with:
          - 'name': identifier for the dataset
          - 'start': starting index (inclusive) in the prediction tensor
          - 'end': ending index (exclusive) in the prediction tensor
          - 'labels': list of ground truth labels (may include -1 for invalid entries)

    Returns:
      results: a dictionary keyed by dataset name. For each dataset, two lists of metrics
               are stored (one for custom_precision_recall and one for custom_positive_f1)
               corresponding to each time step.
    """
    results = {}
    n_time_steps = pred_tensor.shape[0]
    print(n_time_steps)

    for dataset in ground_truth_datasets:
        name = dataset['name']
        start = dataset['start']
        end = dataset['end']
        gt_full = dataset['labels']
        valids = dataset['valids']

        # Initialize storage for metrics for this dataset.
        results[name] = {
            'custom_precision_recall': [],
            'custom_positive_f1': []
        }

        # Loop over each time step.
        for t in range(n_time_steps):
            # Get the prediction slice for this dataset at time step t.
            # print(pred_tensor[t])
            preds_slice = pred_tensor[t][start:end]

            # Filter out any invalid indices where ground truth == -1.
            gt_filtered = filter_invalid_indices(gt_full, valids)
            #print(len(preds_slice))
            #print(len(gt_filtered))
            # Evaluate using the provided custom functions.
            pr, neg_pr, pr_rec, neg_rec, cust_acc = custom_precision_recall(gt_filtered, preds_slice)
            pos_f1_metrics = custom_positive_f1(gt_filtered, preds_slice)

            # Store the metrics for this time step.
            results[name]['custom_precision_recall'].append({
                'time': t,
                'pos_precision': pr,
                'neg_precision': neg_pr,
                'pos_recall': pr_rec,
                'neg_recall': neg_rec,
                'custom_accuracy': cust_acc
            })
            results[name]['custom_positive_f1'].append({
                'time': t,
                'precision': pos_f1_metrics['precision'],
                'recall': pos_f1_metrics['recall'],
                'f1': pos_f1_metrics['f1']
            })

    return results

# ---------------------------------------------------------
# Delta Computation and Plotting
# ---------------------------------------------------------
def compute_delta_metrics(results1, results2):
    """
    Given two results dictionaries (from evaluate_time_series_predictions),
    compute the delta (results2 - results1) for each metric at each time step.
    """
    delta_results = {}
    for dataset in results1:
        delta_results[dataset] = {
            'custom_precision_recall': [],
            'custom_positive_f1': []
        }
        n_time_steps = len(results1[dataset]['custom_precision_recall'])
        for t in range(n_time_steps):
            r1_pr = results1[dataset]['custom_precision_recall'][t]
            
        
            r2_pr = results2[dataset]['custom_precision_recall'][t]
            print('original pos precision', r1_pr['pos_precision'])
            print('original neg precision', r1_pr['neg_precision'])
            print('original pos recall', r1_pr['pos_recall'])
            print('original neg recall', r1_pr['neg_recall'])
            print('original accuracy', r1_pr['custom_accuracy'])
            
            print('remodel pos precision', r2_pr['pos_precision'])
            print('remodel neg precision', r2_pr['neg_precision'])
            print('remodel pos recall', r2_pr['pos_recall'])
            print('remodel neg recall', r2_pr['neg_recall'])
            print('remodel accuracy', r2_pr['custom_accuracy'])
            
            delta_pr = {
                'time': t,
                'pos_precision': r2_pr['pos_precision'] - r1_pr['pos_precision'],
                'neg_precision': r2_pr['neg_precision'] - r1_pr['neg_precision'],
                'pos_recall': r2_pr['pos_recall'] - r1_pr['pos_recall'],
                'neg_recall': r2_pr['neg_recall'] - r1_pr['neg_recall'],
                'custom_accuracy': r2_pr['custom_accuracy'] - r1_pr['custom_accuracy']
            }
            r1_f1 = results1[dataset]['custom_positive_f1'][t]
            r2_f1 = results2[dataset]['custom_positive_f1'][t]
            delta_f1 = {
                'time': t,
                'precision': r2_f1['precision'] - r1_f1['precision'],
                'recall': r2_f1['recall'] - r1_f1['recall'],
                'f1': r2_f1['f1'] - r1_f1['f1']
            }
            delta_results[dataset]['custom_precision_recall'].append(delta_pr)
            delta_results[dataset]['custom_positive_f1'].append(delta_f1)
    return delta_results

def plot_delta_results(delta_results):
    """
    For each dataset, plot the delta metrics (over time) for both the custom precision/recall/accuracy
    and the custom F1 score.
    """
    for dataset, metrics in delta_results.items():
        time_steps = [entry['time'] for entry in metrics['custom_precision_recall']]
        pos_precision = [entry['pos_precision'] for entry in metrics['custom_precision_recall']]
        neg_precision = [entry['neg_precision'] for entry in metrics['custom_precision_recall']]
        pos_recall    = [entry['pos_recall'] for entry in metrics['custom_precision_recall']]
        neg_recall    = [entry['neg_recall'] for entry in metrics['custom_precision_recall']]
        cust_acc      = [entry['custom_accuracy'] for entry in metrics['custom_precision_recall']]
        f1_scores     = [entry['f1'] for entry in metrics['custom_positive_f1']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, pos_precision, label="Delta Positive Precision")
        plt.plot(time_steps, neg_precision, label="Delta Negative Precision")
        plt.plot(time_steps, pos_recall, label="Delta Positive Recall")
        plt.plot(time_steps, neg_recall, label="Delta Negative Recall")
        plt.plot(time_steps, cust_acc, label="Delta Custom Accuracy")
        plt.plot(time_steps, f1_scores, label="Delta F1 Score")
        plt.xlabel("Time Step")
        plt.ylabel("Delta Metric Value")
        plt.title(f"{dataset.capitalize()} - Delta Metrics Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

# ---------------------------------------------------------
# Main Section: Load Predictions, Define Ground Truth, and Compute Deltas
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load the two npy files with simulation labels.
    # Adjust file names as needed.
    preds_file1 = np.load("sim_labels.npy", allow_pickle=True)
    preds_file2 = np.load("remodeled3_sim_labels.npy", allow_pickle=True)
    
    # Convert to 2D arrays (time_steps x prediction_length)
    preds_tensor1 = np.vstack(preds_file1)
    preds_tensor2 = np.vstack(preds_file2)
    
    # -------------------------------
    # Define Ground Truth Arrays and Masks
    # -------------------------------
    # (These arrays are provided examples; ensure they match your actual data.)
    ay = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 0, 0, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, -1, -1, -1, -1, 0, 0, 0, 0, 1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1]
    ayvalid_indices = [i for i, val in enumerate(ay) if val != -1]
    mc = [-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, 1, 1, 1, 1, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 1, 1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 1, 1, 1, 1, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0, 1, 0, 1, 0, -1, -1, -1, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, -1, -1, -1, 0, 0, 0, 1, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, 1, 1, 1, 0, -1, -1, -1, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0]
    mcvalid_indices = [i for i, val in enumerate(mc) if val != -1]
    sm = [-1, -1, -1, -1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 1, 1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1]
    smvalid_indices = [i for i, val in enumerate(sm) if val != -1]
    po = [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 1, 1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 0, -1, -1, -1, 0, 1, 1, 1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 1, 1, 1, 1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1]
    povalid_indices = [i for i, val in enumerate(po) if val != -1]
    
    # Define "required" masks (example arrays – adjust as needed)
    porterreq = [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ayersreq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    carthyreq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    smithreq = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # If your actual masks are longer, include the full arrays.
    
    # Update ground truth arrays with required markings
    ayers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    nathan = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jane = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    emma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    ayers = update_required(ayers, ayersreq)
    nathan = update_required(nathan, porterreq)
    jane = update_required(jane, smithreq)
    emma = update_required(emma, carthyreq)
    
    # -----------------------------------------------------
    # Define Ground Truth Datasets
    # For each dataset, you must specify the slice of the prediction vector (start, end),
    # the corresponding ground truth labels, and valid indices.
    # Adjust the start/end indices as appropriate.
    # -----------------------------------------------------
    ground_truth_datasets = [
        {
            'name': 'porter',
            'start': 0,
            'end': 171,
            'valids': povalid_indices,
            'labels': nathan  # Adjust so that length equals end-start (here 105; adjust as needed)
        },
        {
            'name': 'ayers',
            'start': 171,
            'end': 483,
            'valids': ayvalid_indices,
            'labels': ayers
        },
        {
            'name': 'mccarthy',
            'start': 483,
            'end': 643,
            'valids': mcvalid_indices,
            'labels': emma
        },
        {
            'name': 'smith',
            'start': 643,
            'end': 796,
            'valids': smvalid_indices,
            'labels': jane
        }
    ]
    
    # -----------------------------------------------------
    # Compute metrics for each npy file
    # -----------------------------------------------------
    results_file1 = evaluate_time_series_predictions(preds_tensor1, ground_truth_datasets)
    results_file2 = evaluate_time_series_predictions(preds_tensor2, ground_truth_datasets)
    
    # Compute delta metrics: (results_file2 - results_file1)
    delta_results = compute_delta_metrics(results_file1, results_file2)
    
    plot_delta_results(delta_results)
