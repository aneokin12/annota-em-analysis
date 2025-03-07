import numpy as np
import warnings

# ======== DS-EM Functions with debug ========

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

def dawid_skene(
    N,
    max_iter=1000,
    collapse=C_step,
    check_convergence=True,
    tol=0.001,
    smoothing_method="Laplace",
    C=True,
    coeff=0.01,
    debug=False
):
    """
    Run the Dawid–Skene EM algorithm on an annotation tensor N of shape (I, J, K).
    
    Returns a 1D array of hard labels (length I), computed as np.argmax(T, axis=1).

    Parameters
    ----------
    debug : bool
        If True, prints intermediate debug information (class priors, partial confusion, etc.).
    """
    N = N.astype(np.float64)
    I, J, K = np.shape(N)
    
    # Majority Vote Initialization
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            K_sum = np.sum(N, axis=2)  # shape (I, J)
            # T[i,j] = fraction of workers that chose label j for task i
            T = K_sum / (np.sum(K_sum, axis=1, keepdims=True))
        except Exception as e:
            print("Error during majority vote initialization:", e)
            raise

    if debug:
        print(f"Tensor: {N}")
        print(f"[DS-EM] Initial majority-vote T shape: {T.shape}")
        # If T is large, you might only want to print a small slice:
        print(f"[DS-EM] T (first 5 tasks):\n{T[:5]}")

    if C:
        T = collapse(T)
        if debug:
            print(f"[DS-EM] After collapse, T (first 5 tasks):\n{T[:5]}")

    # EM Algorithm
    for iteration in range(max_iter):
        # M-step: estimate class prior p and confusion
        p = np.mean(T, axis=0).reshape(-1, 1)  # shape (J,1)
        confusion = np.tensordot(T, N, axes=([0, 0]))  # shape (J,J,K)
        confusion = smooth(confusion, smoothing_method, coeff)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            confusion /= np.repeat(np.sum(confusion, axis=1), J, axis=0).reshape(J, J, K)
        
        if debug and iteration < 5:  # only print for first few iterations to reduce spam
            print(f"[DS-EM Iter {iteration}] p (class prior): {p.ravel()}")
            # confusion is shape (J,J,K). Printing all might be huge:
            # let's print confusion for first 2 workers only, if K >= 2
            kshow = min(K, 2)
            for kk in range(kshow):
                print(f"[DS-EM Iter {iteration}] Confusion matrix for worker {kk}:")
                print(confusion[:, :, kk])
        
        # E-step: compute new T by log-likelihood
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_conf = np.log(confusion)  # shape (J,J,K)
            num = np.tensordot(N, log_conf, axes=((2,1),(2,1)))  # shape (I,J)
            denom = np.repeat(np.log(np.matmul(np.exp(num), p)), J, axis=1)
            log_like = np.log(p.T) + num - denom
            T_new = np.exp(log_like)

        if C:
            T_new = C_step(T_new)

        # Check convergence
        delta = np.mean(np.abs(T_new - T))
        if debug and iteration < 5:
            print(f"[DS-EM Iter {iteration}] T diff: {delta:.6f}; T_new stats => min={T_new.min()}, max={T_new.max()}")
        if check_convergence and delta < tol:
            if debug:
                print(f"[DS-EM] Converged at iteration {iteration} with mean abs diff={delta:.6f}")
            T = T_new
            break
        else:
            T = T_new

    final_labels = np.argmax(T, axis=1)
    if debug:
        print("[DS-EM] Final label distribution:", np.bincount(final_labels))
    return final_labels


# ======== Extra-Worker Construction Functions ========

def unify_tensors(N_prev, N_curr):
    """
    Zero-pad to unify shapes of N_prev, N_curr.
    Returns N_prev_aligned, N_curr_aligned of shape (I_new, J, K_new).
    """
    I1, J1, K1 = N_prev.shape
    I2, J2, K2 = N_curr.shape
    if J1 != J2:
        raise ValueError(f"Mismatch in #labels: {J1} vs {J2}")

    I_new = max(I1, I2)
    K_new = max(K1, K2)

    N_prev_aligned = np.zeros((I_new, J1, K_new), dtype=N_prev.dtype)
    N_curr_aligned = np.zeros((I_new, J1, K_new), dtype=N_curr.dtype)

    # Copy old data
    N_prev_aligned[:I1, :, :K1] = N_prev
    # Copy new data
    N_curr_aligned[:I2, :, :K2] = N_curr

    return N_prev_aligned, N_curr_aligned

def build_extended_tensor(N_prev, N_curr, debug=False):
    """
    Build an extended tensor capturing the "edit" from N_prev -> N_curr.
    If N_prev is None, just replicate N_curr in first half, zero in second half.
    Otherwise:
      - unify shapes => N_prev_aligned, N_curr_aligned
      - extended shape => (I_new, J_new, 2*K_new)
      - first half = N_curr_aligned
      - second half = only the tasks that changed for each overlapping worker
    """
    if N_prev is None:
        I_c, J_c, K_c = N_curr.shape
        N_extended = np.zeros((I_c, J_c, 2*K_c), dtype=N_curr.dtype)
        N_extended[:, :, :K_c] = N_curr
        if debug:
            print("[build_extended_tensor] No previous data => extended shape =", N_extended.shape)
        return N_extended

    N_prev_aligned, N_curr_aligned = unify_tensors(N_prev, N_curr)
    I_new, J_new, K_new = N_prev_aligned.shape

    N_extended = np.zeros((I_new, J_new, 2*K_new), dtype=N_curr_aligned.dtype)
    # Put current in the first half
    N_extended[:, :, :K_new] = N_curr_aligned

    # Overlap region for tasks & workers
    I_overlap = min(N_prev.shape[0], N_curr.shape[0])
    K_overlap = min(N_prev.shape[2], N_curr.shape[2])

    changes_count = 0
    for k in range(K_overlap):
        changed_indices = np.where(
            (N_prev_aligned[:I_overlap, :, k] != N_curr_aligned[:I_overlap, :, k]).any(axis=1)
        )[0]
        changes_count += len(changed_indices)
        for i in changed_indices:
            N_extended[i, :, k+K_new] = N_curr_aligned[i, :, k]

    if debug:
        print(f"[build_extended_tensor] Extended shape = {N_extended.shape}, found {changes_count} changes total.")
    return N_extended


# ======== Main Script ========
if __name__ == '__main__':
    # Set NumPy print options (to see more of large arrays if needed).
    np.set_printoptions(threshold=100, edgeitems=3, linewidth=120)

    # Load timeline.npy
    try:
        timeline = np.load("timeline.npy", allow_pickle=True).item()
    except Exception as e:
        print("Error loading timeline.npy:", e)
        exit(1)
    
    print("Loaded timeline.npy with type:", type(timeline))
    
    # Sort timestamps
    sorted_timestamps = sorted(timeline.keys())
    print("\nTimestamps in timeline (sorted):")
    print(sorted_timestamps)
    
    final_results = {}
    sim_labels_list = []

    prev_used = None  # Store the last 112-worker array used
    earlystopper = 0
    earlydebug = True
    for timestamp in sorted_timestamps:
        if earlydebug and earlystopper > 10: break
        ann_array = timeline[timestamp]
        # Check if # worker nodes == 112
        if ann_array.shape[2] != 112:
            print(f"[MAIN] Timestamp {timestamp} skipped: only {ann_array.shape[2]} worker nodes (<112).")
            continue

        # If we get here, shape[2] = 112
        print(f"\n[MAIN] Timestamp {timestamp}: #Workers = {ann_array.shape[2]}, shape={ann_array.shape}")

        # Build extended array from prev_used => ann_array
        N_extended = build_extended_tensor(prev_used, ann_array, debug=True)

        # Optionally: print a small slice of N_extended for debugging
        # (Only do this if your data is not huge, or you'll get a huge print!)
        # e.g. show the first 2 tasks, first 2 labels, first 2 workers in each half:
        if N_extended.shape[0] > 2:
            print("[MAIN] N_extended sample (first 2 tasks, all labels, first 4 'workers'):\n",
                  N_extended[:2, :, :4])

        # Run DS-EM on extended
        preds = dawid_skene(N_extended, 
                            max_iter=20,  # you can shorten or lengthen for debugging
                            tol=0.001,
                            smoothing_method="Laplace",
                            coeff=0.01,
                            debug=True)  # <== Turn on debug for DS-EM

        final_results[timestamp] = preds
        sim_labels_list.append(preds)
        prev_used = ann_array
        earlystopper += 1

    # Create a 2D array of predictions (one row per timestamp)
    sim_labels = np.array(sim_labels_list, dtype=object)
    print("\n[MAIN] Simulated Labels (2D object array):")
    print(sim_labels)
    
    # Save the simulated labels array
    np.save("extraworkerpreds.npy", sim_labels)
    print("\n[MAIN] Simulated labels saved to 'sim_labels.npy'.")

    # Save the final results dictionary
    # np.save("final_predictions.npy", final_results)
    # print("[MAIN] Final predictions saved to 'final_predictions.npy'.")
