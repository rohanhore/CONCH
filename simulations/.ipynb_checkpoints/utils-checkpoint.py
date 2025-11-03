import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

def compute_p_values(
    x,
    score_fn,                   
    nperm: int = 100,
    random_state: int | None = None,
    n_jobs: int = -1,
    backend: str = "loky",       
    verbose: int = 0,
):
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    
    # per-t independent seeds (reproducible)
    ss = np.random.SeedSequence(random_state)
    seeds = [int(cs.generate_state(1)[0]) for cs in ss.spawn(n - 1)]

    # nested worker 
    def pvalue_single_t(t, seed):
        rng = np.random.default_rng(seed)
        s_obs = score_fn(x, t)
        idx_before = np.arange(t)
        idx_after = np.arange(t, n)
        count = 0
        for _ in range(nperm):
            pi = np.empty(n, dtype=int)
            pi[:t] = rng.permutation(idx_before)
            pi[t:] = rng.permutation(idx_after)
            if score_fn(x[pi], t) <= s_obs:
                count += 1
        return t, (count + 1) / (nperm + 1)

    # ensure usable backend with nested worker
    effective_backend = "threading" if backend == "loky" else backend

    iterator = range(1, n)
    iterator = tqdm(iterator, total=n - 1, leave=True, desc="computing CONCH p-values")

    results = Parallel(n_jobs=n_jobs, backend=effective_backend, verbose=0)(
        delayed(pvalue_single_t)(t, seeds[t - 1]) for t in iterator
    )

    pvals = np.empty(n - 1, dtype=float)
    for t, p in results:
        pvals[t - 1] = float(p)
    return pvals

def confidence_set(p_values, alpha):
    return [t+1 for t, p in enumerate(p_values) if p > alpha] 

