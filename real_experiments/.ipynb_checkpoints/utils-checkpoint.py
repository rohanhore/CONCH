
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

def compute_p_values(
    logits,
    class_before: int,
    class_after: int,
    nperm: int = 400,
    random_state: int | None = None,
    log_prior_odds: float = 0.0,
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
):
    """
    For each t in {1,...,n-1}:
      - Shuffle the left block [0..t-1] and right block [t..n-1] independently,
      - Recompute the (MLE-normalized) relative-likelihood score at t,
      - Compute LEFT-tailed p-values: ( #{score_perm(t) <= score_obs(t)} + 1 ) / (nperm + 1).
    """
    logits = np.asarray(logits, dtype=float)
    n = logits.shape[0]

    # per-t independent seeds (reproducible)
    ss = np.random.SeedSequence(random_state)
    seeds = [int(cs.generate_state(1)[0]) for cs in ss.spawn(n - 1)]

    # score at a single t from logits
    def score_from_logits_at_t(arr, t):
        # cumulative log-odds favoring "before" vs "after", centered by max (MLE normalization)
        rel = np.cumsum(arr[:, class_before] - arr[:, class_after] - float(log_prior_odds))
        return float(rel[t - 1] - rel.max())

    # worker for one t
    def pvalue_single_t(t, seed):
        rng = np.random.default_rng(seed)
        s_obs = score_from_logits_at_t(logits, t)

        idx_left  = np.arange(t, dtype=int)
        idx_right = np.arange(t, n, dtype=int)

        count = 0
        for _ in range(nperm):
            pi_left  = rng.permutation(idx_left)
            pi_right = rng.permutation(idx_right)
            pi = np.concatenate([pi_left, pi_right])
            s_perm = score_from_logits_at_t(logits[pi], t)
            if s_perm <= s_obs:
                count += 1

        return t, (count + 1) / (nperm + 1)

    # ensure usable backend with nested worker
    effective_backend = "threading" if backend == "loky" else backend

    iterator = tqdm(range(1, n), total=n - 1, leave=True, desc="computing CONCH p-values")
    results = Parallel(n_jobs=n_jobs, backend=effective_backend, verbose=verbose)(
        delayed(pvalue_single_t)(t, seeds[t - 1]) for t in iterator
    )

    pvals = np.empty(n - 1, dtype=float)
    for t, p in results:
        pvals[t - 1] = float(p)
    return pvals

def confidence_set(p_values, alpha: float):
    """Return {t : p(t) > alpha} using 1-indexed t."""
    return [t + 1 for t, p in enumerate(p_values) if p > alpha]


# def compute_sequential_cifar100_scores_fast(preds, probs):
#     """
#     preds: (n,) numpy int labels (0..99)
#     probs: (n,100) torch tensor of probabilities
#     Returns left_scores, right_scores arrays as in your MNIST code but computed O(n^2) instead of O(n^3).
#     """
#     n = len(preds)
#     # One-hot for cumulative counts
#     K = probs.shape[1]
#     preds_t = torch.tensor(preds, dtype=torch.long)
#     oh = F.one_hot(preds_t, num_classes=K).cumsum(dim=0)              # (n, K), prefix counts inclusive
#     total_counts = oh[-1]                                             # counts over [0..n-1]
#     suf = (total_counts - F.pad(oh[:-1], (0,0,1,0), value=0))         # suffix counts at index i: counts over [i..n-1]

#     # Prefix/suffix modes
#     prefix_mode = oh.argmax(dim=1).numpy()                            # baseline class for left side at r
#     suffix_mode = suf.argmax(dim=1).numpy()                           # baseline class for right side at r

#     # Reference per t
#     # For left_scores[t, r], reference is mode over [t+1..]  => suffix_mode[t+1]
#     # For right_scores[t, r], reference is mode over [..t]   => prefix_mode[t]
#     ref_right = np.empty(n-1, dtype=int)  # reference for left_scores
#     ref_left  = np.empty(n-1, dtype=int)  # reference for right_scores
#     for t in range(n-1):
#         ref_right[t] = suffix_mode[t+1]
#         ref_left[t]  = prefix_mode[t]

#     # Now form left/right score matrices
#     left_scores  = np.zeros((n-1, n), dtype=np.float64)
#     right_scores = np.zeros((n-1, n), dtype=np.float64)

#     # Pull numpy view for speeds
#     probs_np = probs

#     # LEFT: r in [0..t]
#     for t in range(n-1):
#         ref_c = ref_right[t]
#         # vectorized for r=0..t:
#         r_idx = np.arange(0, t+1)
#         base_cls = prefix_mode[r_idx]
#         num = probs_np[r_idx, base_cls]
#         den = probs_np[r_idx, ref_c] + 1e-10
#         left_scores[t, r_idx] = num / den

#         # RIGHT: r in [t+1..n-1]
#         ref_c2 = ref_left[t]
#         r_idx2 = np.arange(t+1, n)
#         base_cls2 = suffix_mode[r_idx2]
#         num2 = probs_np[r_idx2, base_cls2]
#         den2 = probs_np[r_idx2, ref_c2] + 1e-10
#         right_scores[t, r_idx2] = num2 / den2

#     return left_scores, right_scores, prefix_mode, suffix_mode
    
# def conformal_pvalues_from_scores(left_scores, right_scores):
#     n = left_scores.shape[1]
#     pvals = np.zeros(n-1, dtype=np.float64)
#     for t in tqdm(range(n-1), desc="KS p-values"):
#         p_curr = np.zeros(n, dtype=np.float64)

#         # Left segment r = 0..t: randomized rank within its own prefix
#         arr = left_scores[t, :t+1]
#         # compute ranks efficiently without Python loops
#         # For each r, we need rank among arr[:r+1]
#         # We'll just loop r (O(n^2) total over t) which is fine for n~800
#         for r in range(t+1):
#             a = arr[:r+1]
#             v = arr[r]
#             rank = np.sum(v < a) + rng.uniform(0,1) * np.sum(v == a)
#             p_curr[r] = rank / (r+1)

#         # Right segment r = t+1..n-1
#         arr2 = right_scores[t, t+1:]
#         L = n - (t+1)
#         for j, r in enumerate(range(t+1, n)):
#             a = arr2[j:]  # segment [r..end]
#             v = right_scores[t, r]
#             rank = np.sum(v < a) + rng.uniform(0,1) * np.sum(v == a)
#             p_curr[r] = rank / (n - r)

#         # KS on both sides then combine (same as your code)
#         p_left  = ks_1samp(p_curr[:t+1],  uniform.cdf, method="exact")[1]
#         p_right = ks_1samp(p_curr[t+1:], uniform.cdf, method="exact")[1]
#         pvals[t] = 1 - (1 - min(p_left, p_right))**2
#     return pvals



