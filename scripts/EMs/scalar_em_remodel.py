import numpy as np
import warnings
from datetime import datetime, timezone

# ----- Provided Helper Functions -----
def C_step(T):
    """
    Collapse (hard–assign) the soft label matrix T.
    For each task (row), set the maximum probability entry to 1 and all others to 0.
    """
    I, J = T.shape
    collapsed_probabilities = np.zeros_like(T)
    collapsed_probabilities[np.arange(I), T.argmax(1)] = 1
    return collapsed_probabilities

def smooth(data, method, coeff):
    """
    Smooth the data using the specified method.
    
    Parameters:
      - data: the input array (e.g., a confusion tensor).
      - method: a string specifying the smoothing method ("Laplace" or "None").
      - coeff: the smoothing coefficient.
    
    Returns:
      - The smoothed data.
    """
    if method == "Laplace":
        return (data + coeff) / (1 + 2 * coeff)
    elif method == "None":
        return data
    else:
        print("Unknown Method " + method + ", returning unchanged data by default.")
        return data

# ----- Dawid–Skene EM Algorithm -----
def dawid_skene(
    N,
    max_iter=1000,
    collapse=C_step,
    check_convergence=True,
    tol=0.001,
    smoothing_method="Laplace",
    C=True,
    coeff=0.01,
):
    """
    Run the Dawid–Skene EM algorithm on an annotation tensor N of shape (I, J, K).
    
    Returns a 1D array of hard labels (length I), computed as np.argmax(T, axis=1).
    """
    N = N.astype(np.float64)
    I, J, K = np.shape(N)
    
    # Majority Vote Initialization
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            K_sum = np.sum(N, axis=2)
            T = K_sum / (np.sum(K_sum, axis=1).reshape(-1, 1))
        except Exception as e:
            print("Error during majority vote initialization:", e)
            raise

    if C:
        T = C_step(T)
    
    # EM Algorithm
    for it in range(max_iter):
        p = np.mean(T, axis=0).reshape(-1, 1)
        confusion = np.tensordot(T, N, axes=([0, 0]))
        confusion = smooth(confusion, smoothing_method, coeff)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            confusion /= np.repeat(np.sum(confusion, axis=1), J, axis=0).reshape(J, J, K)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_conf = np.log(confusion)
            num = np.tensordot(N, log_conf, axes=((2, 1), (2, 1)))
            denom = np.repeat(np.log(np.matmul(np.exp(num), p.reshape(-1, 1))), J, axis=1)
            log_like = np.log(p.reshape(1, -1)) + num - denom
            T_new = np.exp(log_like).astype("float64")
        if C:
            T_new = C_step(T_new)
        if check_convergence and np.mean(np.abs(T_new - T)) < tol:
            T = T_new
            break
        else:
            T = T_new

    return np.argmax(T, axis=1)

# ----- Main Script -----
if __name__ == '__main__':
    # Set NumPy print options to show full arrays.
    np.set_printoptions(threshold=np.inf)
    
    # Load timeline.npy.
    # Expected: dictionary with keys = timestamps (datetime objects),
    # and values = annotation tensors of shape (I, J, K)
    try:
        timeline = np.load("timeline.npy", allow_pickle=True).item()
    except Exception as e:
        print("Error loading timeline.npy:", e)
        exit(1)
    
    print("Loaded timeline.npy with type:", type(timeline))
    
    # Sort the timestamps in time order.
    sorted_timestamps = sorted(timeline.keys())
    print("\nTimestamps in timeline (sorted):")
    print(sorted_timestamps)
    
    final_results = {}
    sim_labels_list = []  # List to store 1D prediction arrays in time order

    # Initialize persistent mask and previous raw annotation tensor.
    persistent_mask = None  # Will be a boolean array matching the shape of a valid annotation tensor.
    prev_raw_ann = None
    
    for timestamp in sorted_timestamps:
        # Load raw annotation tensor.
        raw_ann = timeline[timestamp]
        # Check if the number of worker nodes is at least 112.
        if raw_ann.shape[2] != 112:
            print(f"Timestamp {timestamp} skipped: only {raw_ann.shape[2]} worker nodes (<112).")
            continue
        
        print(f"Timestamp {timestamp}: Number of worker nodes = {raw_ann.shape[2]}")
        
        # Convert to float64 for processing.
        raw_ann = raw_ann.astype(np.float64)
        
        # Initialize persistent_mask for the first valid tensor.
        if persistent_mask is None:
            persistent_mask = np.zeros_like(raw_ann, dtype=bool)
            processed_ann = raw_ann.copy()  # No changes for the first timestamp.
        else:
            # Compare current raw tensor to previous raw tensor.
            diff = raw_ann != prev_raw_ann
            # Update the persistent mask: once a cell changes, it remains flagged.
            persistent_mask = persistent_mask | diff
            # Create a processed tensor where any flagged cell is set to 1.5.
            processed_ann = raw_ann.copy()
            processed_ann[persistent_mask] = 1.5
        
        # Save the current raw tensor for the next iteration comparison.
        prev_raw_ann = raw_ann.copy()
        
        # Run Dawid–Skene using the processed annotation tensor.
        preds = dawid_skene(
            processed_ann,
            max_iter=1000,
            tol=0.001,
            smoothing_method="Laplace",
            coeff=0.01,
        )
        final_results[timestamp] = preds
        sim_labels_list.append(preds)
    
    # Create a 2D array of predictions.
    sim_labels = np.array(sim_labels_list, dtype=object)
    print("\nSimulated Labels (2D object array):")
    print(sim_labels)
    
    # Save the simulated labels array to sim_labels.npy.
    np.save("remodeled3_sim_labels.npy", sim_labels)
    print("\nSimulated labels saved to 'sim_labels.npy'.")
    
    # Save final results dictionary as well.
    np.save("final_predictions.npy", final_results)
    print("Final predictions saved to 'final_predictions.npy'.")
