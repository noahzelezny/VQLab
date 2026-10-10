"""k-means for codebook fitting: seeding, assignment, Lloyd updates.

Every fitter's k-means lives here. They are NOT one algorithm with knobs:
each fitter's numerics are pinned (fits are compared byte-for-byte, and
tests/test_kmeans_golden.py holds each one to its pre-consolidation output),
so the variants keep their own arithmetic and are named for how they work.

    in-RAM numpy (additive VQ, additive_codebooks.py)
        kmeanspp_init(X, K, rng)          exact k-means++ over all of X
        assign(X, C, chunk, prefer_mx)    argmin; MLX above 32768 rows
        assign_mx_np(X, C, chunk)         the MLX half of assign()
        lloyd(X, C, iters, rng)           vectorised M-step, dead centroids revived

    streamed host data, MLX distances (geo_build.py, fit_additive.py)
        kmeanspp_init_pooled(X, k, rng)   ++ over a 2^18-row sample
        assign_chunked(X, C)              argmin in K-scaled chunks
        lloyd_weighted(X, C, w, iters, K, rng)  weighted float64 M-step

    resident MLX arrays (fit_moe.py, fit_dense_vq.py, fit_ple.py)
        kmeanspp_init_mx(X, k, cap, W)    ++ on a subsample, mx.random draws
        kmeans(X, k, iters, init, W)      Lloyd with a scatter-add update
        assign_mx(X, C, chunk)            argmin -> uint32 codes

MLX is imported lazily where it is used, so the numpy path runs without it.
"""
from __future__ import annotations

import numpy as np

# Rows per chunk in assign_chunked / lloyd_weighted before the K cap.
CHUNK = 1 << 18


def _mx():
    import mlx.core as mx
    return mx


def _have_mx() -> bool:
    try:
        _mx()
        return True
    except Exception:                      # pragma: no cover
        return False


# --------------------------------------------------------- in-RAM numpy
def kmeanspp_init(X, K, rng):
    """Exact k-means++ over every row of X (numpy, float32 centroids)."""
    n = X.shape[0]
    C = np.empty((K, X.shape[1]), np.float32)
    C[0] = X[rng.integers(n)]
    d2 = ((X - C[0]) ** 2).sum(1)
    for i in range(1, K):
        p = d2 / max(d2.sum(), 1e-12)
        C[i] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((X - C[i]) ** 2).sum(1))
    return C


def assign_mx_np(X, C, chunk=1 << 18):
    """argmin_k ||x - c_k||^2 on the GPU for numpy X, C; int32 codes. Same
    result as assign()'s numpy path, ~20x faster at 100M+ rows."""
    mx = _mx()
    Cm = mx.array(np.ascontiguousarray(C, dtype=np.float32))
    cn = mx.sum(Cm * Cm, axis=1)
    out = np.empty(X.shape[0], np.int32)
    for i in range(0, X.shape[0], chunk):
        xb = mx.array(np.ascontiguousarray(X[i:i + chunk], dtype=np.float32))
        d = (mx.sum(xb * xb, axis=1)[:, None] - 2.0 * (xb @ Cm.T)
             + cn[None, :])
        a = mx.argmin(d, axis=1)
        mx.eval(a)
        out[i:i + chunk] = np.array(a)
        mx.clear_cache()
    return out


def assign(X, C, chunk=1 << 16, prefer_mx=True):
    """Nearest centroid per row (int32). Above 32768 rows it runs on MLX
    when available (prefer_mx); otherwise numpy in `chunk`-row blocks."""
    if prefer_mx and X.shape[0] > 1 << 15 and _have_mx():
        return assign_mx_np(X, C)
    out = np.empty(X.shape[0], np.int32)
    cn = (C * C).sum(1)
    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk]
        d = (xb * xb).sum(1)[:, None] - 2.0 * (xb @ C.T) + cn[None, :]
        out[i:i + chunk] = np.argmin(d, axis=1)
    return out


def lloyd(X, C, iters, rng):
    """Lloyd with a vectorised M-step -- a K-long python loop over 100M-row
    masks is slower than the assignment it follows. Updates C in place."""
    K, D = C.shape
    for _ in range(iters):
        a = assign(X, C)
        cnt = np.bincount(a, minlength=K).astype(np.float32)
        sums = np.zeros((K, D), np.float32)
        for d in range(D):                     # D is 2/4/8, not K
            sums[:, d] = np.bincount(a, weights=X[:, d], minlength=K)
        live = cnt > 0
        C[live] = sums[live] / cnt[live, None]
        if (~live).any():                      # revive dead centroids
            C[~live] = X[rng.integers(X.shape[0], size=int((~live).sum()))]
    return C


# ----------------------------------------- streamed host data, MLX distances
def sq_dists(X, C):
    """[n, K] squared distances (MLX)."""
    mx = _mx()
    return (mx.sum(X * X, 1, keepdims=True) - 2 * (X @ C.T)
            + mx.sum(C * C, 1)[None, :])


def chunk_rows(K, chunk=CHUNK):
    """Rows per distance step. The matrix is rows x K fp32; at K16384 the
    default chunk is a 17 GB intermediate (K2048: 2 GB). Capped at ~4 GB so
    large codebooks fit under the memory limit without changing K<=4096."""
    return min(chunk, max(1 << 12, (1 << 30) // K))


def assign_chunked(Xs, C):
    """Nearest centroid per row of host array Xs against MLX C (numpy codes)."""
    mx = _mx()
    out, ch = [], chunk_rows(C.shape[0])
    for i in range(0, Xs.shape[0], ch):
        a = mx.argmin(sq_dists(mx.array(Xs[i:i + ch]), C), axis=1)
        mx.eval(a)
        out.append(np.array(a))
    return np.concatenate(out)


def lloyd_weighted(Xs, C, w, iters, K, rng):
    """Weighted Lloyd: centroid = sum(w x) / sum(w), accumulated in float64
    on the host; a dead centroid is reseeded from a random row. Returns an
    MLX float32 codebook."""
    mx = _mx()
    D = Xs.shape[1]
    for _ in range(iters):
        num = np.zeros((K, D), dtype=np.float64)
        den = np.zeros(K, dtype=np.float64)
        ch = chunk_rows(K)
        for i in range(0, Xs.shape[0], ch):
            xb, wb = Xs[i:i + ch], w[i:i + ch]
            a = mx.argmin(sq_dists(mx.array(xb), C), axis=1)
            mx.eval(a)
            a = np.array(a)
            np.add.at(num, a, xb * wb[:, None])
            np.add.at(den, a, wb)
        dead = den == 0
        if dead.any():
            num[dead] = Xs[rng.choice(Xs.shape[0], int(dead.sum()))]
            den[dead] = 1
        C = mx.array((num / den[:, None]).astype(np.float32))
    return C


def kmeanspp_init_pooled(Xf, k, rng, pool=1 << 18):
    """k-means++ over a `pool`-row sample of host array Xf (numpy rng for
    the draws, MLX for the distances). Returns numpy [k, D]."""
    mx = _mx()
    sample = Xf[rng.choice(Xf.shape[0], min(pool, Xf.shape[0]), replace=False)]
    P = mx.array(sample)
    out = [sample[rng.integers(sample.shape[0])]]
    dmin = None
    for _ in range(k - 1):
        d = mx.sum((P - mx.array(out[-1])[None, :]) ** 2, axis=1)
        dmin = d if dmin is None else mx.minimum(dmin, d)
        mx.eval(dmin)
        p = np.array(dmin, dtype=np.float64)
        p /= p.sum()
        out.append(sample[rng.choice(sample.shape[0], p=p)])
    return np.stack(out)


# --------------------------------------------------------- resident MLX
def kmeanspp_init_mx(X, k, cap=200_000, W=None):
    """k-means++ seeding: each new centre is drawn with probability
    proportional to its squared distance from the nearest chosen centre.

    WHY. Uniform-random seeding (the original) draws k points from the raw
    density, so dense regions get many near-duplicate centres and sparse
    regions get none — wasted codebook and, at worst, a collapsed fit (E44:
    tail30 L26 down_proj hit relerr 1.0 from a bad draw). ++ spreads the
    seeds and is the standard fix.

    Textbook ++ is k sequential passes over X; at k=2048 and |X|=2M that is
    ~4e9 distance evals. So seed on a SUBSAMPLE (cap) and hand the result to
    full k-means over all of X — the usual scalable compromise, and the
    refinement iterations see every point regardless.
    """
    mx = _mx()
    n = X.shape[0]
    if n > cap:
        sel = mx.random.randint(0, n, (cap,))
        X = X[sel]
        if W is not None:
            W = W[sel]
        n = cap
    first = int(mx.random.randint(0, n, (1,)).item())
    picks = [X[first]]
    d2 = mx.sum((X - picks[0]) ** 2, axis=1)
    for _ in range(k - 1):
        # With W, seeds are drawn proportional to w*d2 rather than d2, so the
        # SEEDING obeys the same objective as the refinement below. Seeding
        # unweighted and refining weighted starts every tail centre in the
        # bulk and asks 20 Lloyd steps to walk it out.
        pd2 = d2 if W is None else d2 * W
        tot = mx.sum(pd2)
        if float(tot.item()) <= 1e-12:          # X has <k distinct points
            picks.append(X[int(mx.random.randint(0, n, (1,)).item())])
            continue
        cdf = mx.cumsum(pd2 / tot)
        r = mx.random.uniform(shape=(1,))
        j = min(int(mx.sum(cdf < r).item()), n - 1)
        c = X[j]
        picks.append(c)
        d2 = mx.minimum(d2, mx.sum((X - c) ** 2, axis=1))
    return mx.stack(picks)


def kmeans(X, k, iters, init="kmeans++", W=None):
    """W=None reproduces the unweighted fit exactly. W is a per-row weight;
    ASSIGNMENT is unaffected (argmin of a distance does not see a positive
    scalar), only the centroid update and the ++ seeding are."""
    mx = _mx()
    step = max(50_000, int(5e8 / k))
    n = X.shape[0]
    # SEEDING IS DELIBERATELY UNWEIGHTED. Weighting BOTH the ++ seeding and the
    # centroid update compounds instead of composing: ++ already spreads seeds
    # by squared distance, so multiplying by mag^p puts nearly every seed in
    # the extreme tail and starves the bulk of centroids. Measured 2026-08-21
    # (E106), L00 down_proj K256 d4, mean held-out relerr:
    #     random init  p=0 0.2035  p=4 0.1685   (weighting helps)
    #     ++ seeding   p=0 0.1177  p=4 0.4869   (weighting destroys it)
    # and the real 397B fit aborted at 0.7111. The tail weighting belongs on
    # the UPDATE only; ++ keeps its own unweighted objective.
    C = (kmeanspp_init_mx(X, k) if init == "kmeans++"
         else X[mx.random.randint(0, n, (k,))])
    for _ in range(iters):
        cn = mx.sum(C * C, axis=1)
        parts = []
        for s0 in range(0, n, step):
            xb = X[s0:s0 + step]
            parts.append(mx.argmin(mx.sum(xb * xb, axis=1, keepdims=True)
                                   - 2 * (xb @ C.T) + cn[None, :], axis=1))
            mx.eval(parts[-1])
        a = mx.concatenate(parts)
        # SCATTER-ADD, not one-hot. The one-hot form built an [rows, k] fp32
        # matrix: 2e6*k*4 bytes = 65.5 GB at k=8192 (over Metal's 62.6 GB cap,
        # which is what killed every K8192 run; it reads as "[metal::malloc]
        # Attempting to allocate 65536000000 bytes", not as a command-buffer
        # timeout). Chunking it small enough to fit made it 33x more
        # iterations. scatter-add is O(n) in the assignment instead of
        # O(n*k), so cost stops depending on k entirely.
        # Weighted centroid = sum(w*x)/sum(w). Same two scatter-adds, so the
        # O(n) cost and the k-independence of the update are unchanged.
        # The GPU scatter-add's float accumulation order is not fixed, so
        # two runs may differ in the last bits (deterministic on mx.cpu).
        oh_sum = mx.zeros((k, X.shape[1]))
        cnt = mx.zeros((k,))
        oh_sum = oh_sum.at[a].add(X if W is None else X * W[:, None])
        cnt = cnt.at[a].add(mx.ones((n,), dtype=mx.float32) if W is None else W)
        mx.eval(oh_sum, cnt)
        C = mx.where(cnt[:, None] > 0,
                     oh_sum / mx.maximum(cnt[:, None], 1.0), C)
        mx.eval(C)
    return C


def assign_mx(X, C, chunk=None):
    """Nearest centroid per row of MLX X against MLX C, as uint32 codes.

    CHUNK MUST SCALE WITH K. The [chunk, K] fp32 distance matrix is
    chunk*K*4 bytes: at a fixed 2,000,000 rows that is 524 GB at K=65536,
    against Metal's 62.6 GB single-buffer cap. chunk=None budgets by K,
    floored at 50k rows so small-K fits keep large chunks."""
    mx = _mx()
    if chunk is None:
        chunk = max(50_000, int(5e8 / C.shape[0]))
    outs = []
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk]
        d = (x * x).sum(1, keepdims=True) - 2 * x @ C.T + (C * C).sum(1)
        outs.append(mx.argmin(d, axis=1).astype(mx.uint32))
        mx.eval(outs[-1])
    return mx.concatenate(outs)
