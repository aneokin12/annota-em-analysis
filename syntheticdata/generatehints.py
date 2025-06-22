# generatehints.py

import os
import numpy as np
from model import dawid_skene
import sampler


def generate_golds(N, positive_rate=0.30, seed=123):
    """
    Given an input tensor N of shape (I,2,K), produce a synthetic "ground truth"
    vector y_true of length I, with a Bernoulli(positive_rate) draw for each row.
    Returns a 1‐D numpy array of 0/1 of length I.
    """
    rng = np.random.default_rng(seed)
    I, two, K = N.shape
    assert two == 2, "N must be shape (I,2,K)"
    y_true = rng.binomial(1, positive_rate, size=I)
    return y_true


def correct_action(hint_t, y_label):
    """
    Given:
      - hint_t ∈ {0,1}: 0 = NOT_RELEVANT hint, 1 = MISSED hint
      - y_label ∈ {0,1}: 1 means “this row truly is relevant,” 0 means “not relevant.”
    Return the TRUE/“ideal” click on a hint:
      - If hint_t == 0, the hint is “this line is NOT relevant”:
          ideal = 'AGREE' if y_label == 0 else 'DISAGREE'
      - If hint_t == 1, the hint is “this line should have been marked relevant”:
          ideal = 'AGREE' if y_label == 1 else 'DISAGREE'
    """
    if hint_t == 0:       # NOT_RELEVANT hint
        return "AGREE" if (y_label == 0) else "DISAGREE"
    else:                 # MISSED (relevant) hint
        return "AGREE" if (y_label == 1) else "DISAGREE"


def apply_synthetic_round(N, line_map, y_true,
                          sample_fn,
                          response_rate=0.30,
                          random_seed=None):
    """
    Perform exactly one “synthetic round” of re‐sampling existing votes in N.
    - N         : numpy array shape (I, 2, K).  N[i,0,k]==1 means annotator k clicked DISAGREE on row i.
                   N[i,1,k]==1 means annotator k clicked AGREE on row i.  If both zero, no vote.
    - line_map  : dict mapping row‐index i → (line_number, hint_type).  hint_type ∈ {0,1}.
                   We only use hint_type here.
    - y_true    : length‐I array of {0,1} (ground truth).
    - sample_fn : a function returning a length‐4 confusion vector q_vec = [p_TN,p_FP,p_FN,p_TP].
                   Example:  sample_fn = lambda: sampler.sample_student_global(sampler.alpha_global)
    - response_rate : fraction (0.0–1.0) of *existing votes* to re‐sample (flip or keep).
    - random_seed: optional int to seed numpy RNG so results are reproducible.
    Returns a brand‐new copy N_new of shape (I,2,K) with some votes flipped.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    I, two, K = N.shape
    assert two == 2, "N must have shape (I,2,K)"

    # 1) Gather all existing (i,j,k) triples where a vote is present:
    existing_votes = []
    for i in range(I):
        ks0 = np.where(N[i, 0, :] == 1)[0]  # indices k who clicked DISAGREE
        ks1 = np.where(N[i, 1, :] == 1)[0]  # indices k who clicked AGREE
        for k in ks0:
            existing_votes.append((i, 0, k))
        for k in ks1:
            existing_votes.append((i, 1, k))

    # 2) How many to re‐sample?
    n_votes_total = len(existing_votes)
    n_to_change   = int(np.floor(response_rate * n_votes_total))

    # 3) Pick a random subset of those to re‐sample
    chosen_indices = np.random.choice(n_votes_total,
                                      size=n_to_change,
                                      replace=False)
    chosen_pairs = [existing_votes[idx] for idx in chosen_indices]

    # 4) Copy N → N_new
    N_new = N.copy()

    # 5) For each chosen (i,j_old,k), roll a new click via a synthetic student
    flips = 0
    for (i, j_old, k) in chosen_pairs:
        # (a) Look up hint_type from line_map[i]
        _, hint_t = line_map[i]
        y_label   = y_true[i]
        ideal_click = correct_action(hint_t, y_label)

        # (b) Sample one confusion‐vector q_vec = [p_TN,p_FP,p_FN,p_TP]
        q_vec = sample_fn()

        # (c) Use sampler.simulate_response to get either 'AGREE' or 'DISAGREE'
        #     Pass `ideal_click == 'AGREE'` as boolean true_is_relevant
        new_click = sampler.simulate_response(ideal_click == "AGREE", q_vec)

        # (d) Convert new_click → j_new (0 for DISAGREE, 1 for AGREE)
        j_new = 1 if new_click == "AGREE" else 0

        # (e) If j_new differs from j_old, flip bits in N_new
        if j_new != j_old:
            N_new[i, j_old, k] = 0
            N_new[i, j_new, k] = 1
            flips += 1
        # If j_new == j_old, leave that vote unchanged

    print(f"Synthetic round: flipped {flips} out of {n_to_change} sampled votes.")
    return N_new


# ─────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    here = os.path.dirname(os.path.realpath(__file__))

    # 1) Load the original hint‐vote tensor (I,2,K)
    tensor_path = os.path.join(here, "hint_tensor.npy")
    N_orig = np.load(tensor_path)       # shape: (I,2,K)
    I, two, K = N_orig.shape
    assert two == 2

    # 2) Load the line_map that you saved earlier
    #    (should be a dict or array of length I mapping i → (line_no, hint_type)).
    lm_path = os.path.join(here, "line_map.npy")
    line_map = np.load(lm_path, allow_pickle=True).item()
    assert len(line_map) == I

    # 3) Generate a synthetic “gold” vector y_true of length I
    y_true = generate_golds(N_orig, positive_rate=0.30, seed=123)
    assert len(y_true) == I

    # 4) Choose which sampler to use.  For example:
    #    – Global Dirichlet student:  sample_fn = lambda: sampler.sample_student_global(sampler.alpha_global)
    #    – Bootstrapped real student: sample_fn = lambda: sampler.sample_student_bootstrap(sampler.df_probs)
    #    – Hierarchical:              sample_fn = lambda: sampler.sample_student_hier(sampler.alpha_global, kappa=0.5)
    sample_fn = lambda: sampler.sample_student_global(sampler.alpha_global)

    # 5) Perform one synthetic “round” of vote‐flipping at 30% re‐sampling rate
    N_synth = apply_synthetic_round(
        N=N_orig,
        line_map=line_map,
        y_true=y_true,
        sample_fn=sample_fn,
        response_rate=0.30,
        random_seed=42
    )

    # 6) Re‐run Dawid–Skene EM on the new tensor
    print("Running Dawid–Skene EM on synthetic tensor …")
    em_preds = dawid_skene(N_synth)
    print("→ EM returned", em_preds.shape, "hard labels.")

    # 7) Save outputs if desired
    out_tensor_path = os.path.join(here, "hint_tensor_synth.npy")
    np.save(out_tensor_path, N_synth)
    print("Saved synthetic tensor to", out_tensor_path)

    out_em_path = os.path.join(here, "em_preds_synth.npy")
    np.save(out_em_path, em_preds)
    print("Saved new EM predictions (length I) to", out_em_path)
