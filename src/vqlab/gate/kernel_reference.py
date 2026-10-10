"""The ground truth the kernel acceptance checks score a fused VQ expert
kernel against: reconstruct each routed expert's W from codes, codebook and
group scales, then matmul in float32. Shared by kernel_accept.py (repo
runtime) and bundle_accept.py (the runtime lifted from an artifact)."""
from __future__ import annotations

import mlx.core as mx


def reference(x, eidx, codes, codebook, scales, D, G):
    """[T, OUT] float32 outputs for tokens x [T, IN] routed to experts eidx
    [T]; codes [E, OUT, NSUB] (unpacked), codebook [K, D], scales
    [E, OUT, IN // G]."""
    E, OUT, NSUB = codes.shape
    IN = NSUB * D
    outs = []
    for t in range(x.shape[0]):
        e = int(eidx[t])
        cb = codebook.astype(mx.float32)
        w = cb[codes[e].reshape(-1).astype(mx.int32)].reshape(OUT, NSUB, D)
        w = w.reshape(OUT, IN // G, G) * scales[e].astype(mx.float32)[:, :, None]
        outs.append((x[t].astype(mx.float32)[None, :] @ w.reshape(OUT, IN).T)[0])
    return mx.stack(outs)
