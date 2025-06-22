import numpy as np
import warnings
from datetime import datetime, timezone
# from utils import createTensorDump

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
    smoothing="Laplace",
    C=True,
):

    N = N.astype(np.float64)
    I, J, K = np.shape(N)

    # Majority Vote initialization
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            K_sum = np.sum(N, axis=2)
            T = K_sum / (np.sum(K_sum, axis=1).reshape(-1, 1))
        except:
            print("JKSums:", np.sum(K_sum, axis=1).reshape(-1, 1))
            assert False

    # Collapses the probabilities to a single estimated label per task
    if C:
        T = collapse(T)

    # EM Algorithm

    for i in range(max_iter):

        # Maximization Step:

        # Computes Prior Probability of each class
        p = np.mean(T, axis=0).reshape(-1, 1)

        # Unnormalized Confusion Tensor (J X J X K)
        confusion = np.tensordot(T, N, axes=([0, 0]))

        # Smooths the unnormalized confusion to avoid divide-by-zero when normalizing.
        confusion = smooth(confusion, smoothing, 0.01)

        # Normalizes the Confusion Tensor
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            confusion /= np.repeat(np.sum(confusion, axis=1), J, axis=0).reshape(
                J, J, K
            )

        # Expectation Step:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Compute Log-Likelihood
            log_conf = np.log(confusion)
            num = np.tensordot(N, log_conf, axes=((2, 1), (2, 1)))
            denom = np.repeat(
                np.log(np.matmul(np.exp(num), p.reshape(-1, 1))), J, axis=1
            )
            log_like = np.log(p.reshape(1, -1)) + num - denom

        # Translate Log-Likelihoods to Label Probabilities
        T_new = np.exp(log_like).astype("float64")

        if C:
            T_new = collapse(T_new)

        if check_convergence and np.mean(np.abs(T_new - T)) < tol:
            T = T_new
            break
        else:
            T = T_new

    # Return Best-Estimated Labels
    return np.argmax(T, axis=1)