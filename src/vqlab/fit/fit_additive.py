"""Additive (residual) VQ: fit M small codebooks whose SUM replaces one large one.

WHY. The fused Metal kernels cache the codebook in threadgroup memory, and
Apple Silicon's limit is a hard 32768 bytes per threadgroup (verified on an
M3 Ultra: maxThreadgroupMemoryLength = 32768, Apple9). The runtime's own rule
is `K * 2 * D + 3 * 4096 > 32768` -> fall off the fast arm onto the
device-memory codebook. So:

    d8-K16384   1.75 b/w   256 KB table   device-cb (the shipped 2.1 floor)
    d4-K8192    3.25 b/w    64 KB table   device-cb
    d4-K2048    2.75 b/w    16 KB table   threadgroup, but 28 KB with the
                                          kernel's other 12 KB -- near the cap,
                                          so occupancy already suffers
    2 x d8-K128 1.75 b/w     4 KB table   threadgroup, with room to spare

Two 128-entry codebooks summed reach 128*128 = 16384 distinct points at the
same 14 bits per 8-dim subvector -- the SAME RATE as d8-K16384 -- from 1/64th
the table. That is the AQLM idea, and it is already Phase 2 of
docs/KL-TUNED-CODEBOOKS-PLAN.md.

WHAT IT COSTS. The 16384 reachable points are no longer freely placed: they
are the Minkowski sum C1 + C2, a structured subset of the space a free
16384-entry fit could occupy. Whether that constraint costs output quality is
an empirical question and NOT one weight-space relerr can answer -- law I.6
says relerr does not rank quality ACROSS GEOMETRIES, which is exactly this
comparison. It has to be KL on the assembled model.

HOW THIS IS TESTED WITHOUT A KERNEL. `--materialize` writes the 16384 sums
out as an ORDINARY d8-K16384 codebook. The artifact is then a normal VQ
artifact -- same format, same existing kernel, same bytes on disk -- whose
entries happen to be constrained. That measures the quality question exactly
and defers all kernel work until the answer is known. The speed win is not
measured this way and cannot be: it needs the real two-table kernel.

Fitting is residual: fit C1 on the data, then C2 on what C1 leaves behind,
then alternate re-assignment so the pair is chosen jointly rather than
greedily (greedy-only is what makes naive residual VQ underperform).
"""
import numpy as np

try:                                   # MLX for the hot loop; numpy is the
    import mlx.core as mx              # reference and the fallback
    _HAVE_MX = True
except Exception:                      # pragma: no cover
    _HAVE_MX = False


def _assign_mx(X, C, chunk=1 << 18):
    """argmin_k ||x - c_k||^2 on the GPU. Same result as _assign, ~20x faster.

    The additive fit assigns 100M+ subvectors against 128 centroids per stage
    per iteration; in numpy that is the whole cost of the job.
    """
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


def _kmeanspp(X, K, rng):
    n = X.shape[0]
    C = np.empty((K, X.shape[1]), np.float32)
    C[0] = X[rng.integers(n)]
    d2 = ((X - C[0]) ** 2).sum(1)
    for i in range(1, K):
        p = d2 / max(d2.sum(), 1e-12)
        C[i] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((X - C[i]) ** 2).sum(1))
    return C


def _assign(X, C, chunk=1 << 16, prefer_mx=True):
    if prefer_mx and _HAVE_MX and X.shape[0] > 1 << 15:
        return _assign_mx(X, C)
    out = np.empty(X.shape[0], np.int32)
    cn = (C * C).sum(1)
    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk]
        d = (xb * xb).sum(1)[:, None] - 2.0 * (xb @ C.T) + cn[None, :]
        out[i:i + chunk] = np.argmin(d, axis=1)
    return out


def _lloyd(X, C, iters, rng):
    """Lloyd with a vectorised M-step -- a K-long python loop over 100M-row
    masks is slower than the assignment it follows."""
    K, D = C.shape
    for _ in range(iters):
        a = _assign(X, C)
        cnt = np.bincount(a, minlength=K).astype(np.float32)
        sums = np.zeros((K, D), np.float32)
        for d in range(D):                     # D is 2/4/8, not K
            sums[:, d] = np.bincount(a, weights=X[:, d], minlength=K)
        live = cnt > 0
        C[live] = sums[live] / cnt[live, None]
        if (~live).any():                      # revive dead centroids
            C[~live] = X[rng.integers(X.shape[0], size=int((~live).sum()))]
    return C


def fit_additive(X, K, M=2, iters=12, rng=None):
    """Fit M codebooks of K entries whose SUM approximates X.

    Returns (list of M [K, D] codebooks, codes [N, M] int32).
    """
    rng = rng or np.random.default_rng(13)
    X = np.ascontiguousarray(X, dtype=np.float32)
    books, codes = [], np.zeros((X.shape[0], M), np.int32)

    R = X.copy()                               # residual
    for m in range(M):
        C = _lloyd(R, _kmeanspp(R, K, rng), 8, rng)
        books.append(C)
        codes[:, m] = _assign(R, C)
        R = R - C[codes[:, m]]

    # JOINT REFINEMENT. Greedy residual fitting picks each stage's code
    # without regard to what later stages can fix, and that is most of the gap
    # between naive residual VQ and AQLM. Re-choose one stage at a time
    # against the others' current contribution, then refit that stage.
    for _ in range(iters):
        for m in range(M):
            other = np.zeros_like(X)
            for j, C in enumerate(books):
                if j != m:
                    other += C[codes[:, j]]
            T = X - other                      # what stage m must explain
            codes[:, m] = _assign(T, books[m])
            books[m] = _lloyd(T, books[m], 2, rng)
            codes[:, m] = _assign(T, books[m])
    return books, codes


def materialize(books):
    """Collapse M codebooks into the single K**M codebook of all sums.

    The point of this is TESTABILITY: the result is an ordinary codebook that
    the existing kernel serves unchanged, so the quality of the additive
    constraint can be measured before any kernel exists. Index order is
    row-major over stages, i.e. idx = c0 * K1 + c1.
    """
    out = books[0]
    for C in books[1:]:
        out = (out[:, None, :] + C[None, :, :]).reshape(-1, C.shape[1])
    return out


def codes_to_flat(codes, K):
    """Stage codes -> the flat index into materialize()'s codebook."""
    flat = np.zeros(codes.shape[0], np.int64)
    for m in range(codes.shape[1]):
        flat = flat * K + codes[:, m]
    return flat
