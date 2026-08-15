#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""GPTQ-style error-compensated rounding into MLX's affine grid (E34).

Standard GPTQ (Frantar et al.) with lazy block updates, adapted to MLX's
group-64 affine format: per (row, group) scale/bias computed from the
CURRENT (compensated) weights when the column loop reaches the group,
exactly the min/max grid mx.quantize would build — so the output packs
into the stock format with zero size change. Solve is per weight matrix
[out, in] with Hessian H = X^T X (+ mean-diag damping) from calibration
activations.

quantize_gptq(W, H) -> (q_idx uint8 [out,in], scales [out,G], biases [out,G])
pack_mlx(q_idx)     -> uint32 packed like mx.quantize's w
"""
import numpy as np
import ml_dtypes

GROUP = 64
BITS = 2
LEVELS = (1 << BITS) - 1  # 3
BLOCK = 128


def _grid(Wg):
    """per-row affine grid over one group's CURRENT weights — mirrors
    mx.quantize: scale = (max-min)/levels, bias = min. Snapped to bf16
    BEFORE any rounding decision, because bf16 is what the artifact stores
    — the solver must optimize against the grid the model will decode."""
    lo = Wg.min(axis=1)
    hi = Wg.max(axis=1)
    scale = (hi - lo) / LEVELS
    scale[scale == 0] = 1e-8
    scale = scale.astype(ml_dtypes.bfloat16).astype(np.float64)
    lo = lo.astype(ml_dtypes.bfloat16).astype(np.float64)
    scale[scale == 0] = 1e-8
    return scale, lo


def quantize_gptq(W, H, damp=0.01):
    W = W.astype(np.float64).copy()
    out_dim, in_dim = W.shape
    G = in_dim // GROUP
    Hd = H.astype(np.float64).copy()
    dead = np.diag(Hd) == 0
    Hd[dead, dead] = 1.0
    W[:, dead] = 0
    Hd += np.eye(in_dim) * (damp * np.mean(np.diag(Hd)))
    # Hinv via Cholesky of inverse (upper), as in reference GPTQ
    Hinv = np.linalg.cholesky(np.linalg.inv(Hd), upper=True)

    q_idx = np.zeros((out_dim, in_dim), dtype=np.uint8)
    scales = np.zeros((out_dim, G), dtype=np.float32)
    biases = np.zeros((out_dim, G), dtype=np.float32)

    for b0 in range(0, in_dim, BLOCK):
        b1 = min(b0 + BLOCK, in_dim)
        Wb = W[:, b0:b1].copy()
        Eb = np.zeros_like(Wb)
        Hb = Hinv[b0:b1, b0:b1]
        for j in range(b1 - b0):
            col = b0 + j
            g = col // GROUP
            if col % GROUP == 0:
                # grid from current compensated weights of this group
                s, z = _grid(W[:, col:col + GROUP])
                scales[:, g] = s
                biases[:, g] = z
            w = Wb[:, j]
            q = np.clip(np.round((w - biases[:, g]) / scales[:, g]),
                        0, LEVELS)
            q_idx[:, col] = q.astype(np.uint8)
            dq = q * scales[:, g] + biases[:, g]
            err = (w - dq) / Hb[j, j]
            if j + 1 < b1 - b0:
                Wb[:, j + 1:] -= np.outer(err, Hb[j, j + 1:])
            Eb[:, j] = err
        if b1 < in_dim:
            W[:, b1:] -= Eb @ Hinv[b0:b1, b1:]
        W[:, b0:b1] = Wb
    return q_idx, scales, biases


def quantize_rtn(W):
    """plain round-to-nearest into the same grid (mx.quantize equivalent)"""
    out_dim, in_dim = W.shape
    G = in_dim // GROUP
    q_idx = np.zeros((out_dim, in_dim), dtype=np.uint8)
    scales = np.zeros((out_dim, G), dtype=np.float32)
    biases = np.zeros((out_dim, G), dtype=np.float32)
    for g in range(G):
        Wg = W[:, g * GROUP:(g + 1) * GROUP]
        s, z = _grid(Wg)
        scales[:, g] = s
        biases[:, g] = z
        q_idx[:, g * GROUP:(g + 1) * GROUP] = np.clip(
            np.round((Wg - z[:, None]) / s[:, None]), 0, LEVELS
        ).astype(np.uint8)
    return q_idx, scales, biases


def dequant(q_idx, scales, biases):
    return (q_idx.astype(np.float32)
            * np.repeat(scales, GROUP, axis=1)
            + np.repeat(biases, GROUP, axis=1))


def pack_mlx(q_idx):
    """pack 2-bit indices -> uint32 words, 16 per word, little-endian
    (layout verified against mx.quantize/dequantize roundtrip)"""
    out_dim, in_dim = q_idx.shape
    v = q_idx.astype(np.uint32).reshape(out_dim, in_dim // 16, 16)
    shifts = (2 * np.arange(16, dtype=np.uint32))[None, None, :]
    return (v << shifts).sum(axis=2, dtype=np.uint32)


if __name__ == "__main__":
    # synthetic validation: anisotropic activations, output-error metric
    rng = np.random.default_rng(0)
    in_dim, out_dim, n = 2048, 1024, 4096
    # correlated activations (heavy anisotropy, like real hiddens)
    A = rng.standard_normal((in_dim, in_dim)) / np.sqrt(in_dim)
    X = rng.standard_normal((n, in_dim)) @ (np.eye(in_dim) + 2 * A)
    W = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.02
    H = X.T @ X
    Y = X @ W.T

    for name, (qi, s, b) in (
            ("rtn ", quantize_rtn(W)),
            ("gptq", quantize_gptq(W, H))):
        Wq = dequant(qi, s, b)
        oerr = np.linalg.norm(X @ Wq.T - Y) / np.linalg.norm(Y)
        werr = np.linalg.norm(Wq - W) / np.linalg.norm(W)
        print(f"{name}  weight_relerr {werr:.4f}  OUTPUT_relerr {oerr:.4f}")
