# sampler.py

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────
# 1) Load the per‐student confusion counts from disk
#    The file "student_confusions.npy" should be a dict of the form:
#      { user_id: (TN_count, FP_count, FN_count, TP_count), … }
#    where each 4‐tuple is (TN, FP, FN, TP).
# ────────────────────────────────────────────────────────────────

CONFPATH = "/Users/saminthachandrasiri/Annota/annota-v2-cloud/functions/py-functions/student_confusions.npy"    # adjust path if needed

# Load the dictionary { uid: (TN,FP,FN,TP), … }
conf_dict = np.load(CONFPATH, allow_pickle=True).item()

# Build a DataFrame from it, indexed by uid
df_counts = (
    pd.DataFrame
      .from_dict(conf_dict, orient="index",
                 columns=["TN","FP","FN","TP"])
      .fillna(0)              # in case some student has missing entries
      .astype(int)
)



alpha_global = df_counts.sum(axis=0).values + 1
# alpha_global is a length‐4 vector [α_TN, α_FP, α_FN, α_TP].

# For “bootstrap” sampling, we also keep the empirical per‐student
# confusion proportions:
df_probs = df_counts.div(df_counts.sum(axis=1), axis=0).fillna(0)
# df_probs.loc[uid] is a length‐4 list [p_TN, p_FP, p_FN, p_TP] for that student.

# ────────────────────────────────────────────────────────────────
# 3) Define sampling functions
# ────────────────────────────────────────────────────────────────

def sample_student_global(alpha):
    """
    Option A – one global Dirichlet. Returns a length‐4 probability vector [p_TN,p_FP,p_FN,p_TP],
    drawn from Dirichlet(alpha).
    """
    return np.random.default_rng().dirichlet(alpha)

def sample_student_bootstrap(df_probs):
    """
    Option B – resample a real annotator’s empirical proportions.
    Returns one row from df_probs (a length‐4 vector summing to 1).
    """
    return df_probs.sample(1, replace=True).values.flatten()

def sample_student_hier(alpha0, kappa=1.0):
    """
    Option C – hierarchical: first draw α_i ~ Gamma(shape=alpha0, scale=1/kappa)
    component‐wise, then draw q_i ~ Dirichlet(α_i).
    Returns a length‐4 q_i.
    """
    # Draw a length‐4 “α_i” by sampling each component from Gamma(shape=α0_j, scale=1/kappa)
    alpha_i = np.random.default_rng().gamma(shape=alpha0, scale=1.0/kappa)
    # Then draw Dirichlet(alpha_i)
    return np.random.default_rng().dirichlet(alpha_i)

def simulate_response(true_is_relevant, q_vec):
    """
    Given:
      - true_is_relevant: Boolean (True means the correct action is “AGREE”).
      - q_vec: length‐4 confusion vector [p_TN, p_FP, p_FN, p_TP], summing to 1.
        Indices:
          q_vec[0] = P(click DISAGREE | true is NOT relevant)      = TN
          q_vec[1] = P(click AGREE  | true is NOT relevant)      = FP
          q_vec[2] = P(click DISAGREE | true is relevant)        = FN
          q_vec[3] = P(click AGREE  | true is relevant)          = TP
    Returns: the string 'AGREE' or 'DISAGREE', rolled according to q_vec.
    """
    rng = np.random.default_rng()
    if true_is_relevant:
        # “true_is_relevant => correct action = AGREE => use TP prob for AGREE”
        p_agree = q_vec[3]
    else:
        # “true_is_not_relevant => correct action = DISAGREE => AGREE only if FP”
        p_agree = q_vec[1]
    return "AGREE" if rng.random() < p_agree else "DISAGREE"


# ────────────────────────────────────────────────────────────────
# 4) Expose objects for import
# ────────────────────────────────────────────────────────────────

__all__ = [
    "df_counts",
    "df_probs",
    "alpha_global",
    "sample_student_global",
    "sample_student_bootstrap",
    "sample_student_hier",
    "simulate_response",
]
