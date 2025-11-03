
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


