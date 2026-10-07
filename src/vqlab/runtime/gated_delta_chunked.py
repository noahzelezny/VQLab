"""Chunked (WY / UT-transform) gated delta rule for PREFILL (R3-3).

Drop-in for ``mlx_lm.models.gated_delta.gated_delta_update``: same
signature, same returns (y in q.dtype, final state in the state's dtype).
Decode (T < VQ_GDN_CHUNK_MIN_T) and vectorized gating (a.ndim == 4) go to
the stock sequential Metal kernel unchanged.

Recurrence (per head, S: [Dv, Dk]):
    S_t = g_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T,   y_t = S_t q_t
Within a chunk of C steps with in-chunk cumulative decay G_i = prod_{j<=i} g_j:
    A   = strict_lower(b_i (k_i.k_j) G_i/G_j)          T = (I + A)^-1
    W   = T (b G K),  U = T (b V)                      (UT transform)
    V'  = U - W S0^T
    y   = G (Q S0^T) + tril(Q K^T G_i/G_j) V'
    S_C = G_C S0 + (V' * G_C/G)^T K
(I + A)^-1 for nilpotent A is prod_m (I + (-A)^(2^m)): log2(C) batched
matmul pairs. Masked steps become g=1, b=0, y=0 -- the kernel's semantics
(state untouched, output zero). All arithmetic in fp32; log g is computed
directly (no exp-then-log underflow).
NOT bit-exact with the sequential kernel: different summation order.
"""
from __future__ import annotations

import math
import os

import mlx.core as mx
import mlx.nn as nn

CHUNK = int(os.environ.get("VQ_GDN_CHUNK", "64"))
MIN_T = int(os.environ.get("VQ_GDN_CHUNK_MIN_T", "256"))


def _stock():
    from mlx_lm.models import gated_delta as gd
    return gd


BASE = 16


def _inv_unit_lower(A):
    """(I + A)^-1 for strictly-lower A [..., n, n], n a power of two.

    Block recursion (both diagonal halves stacked and solved together):
    inv([[L11,0],[L21,L22]]) = [[T11,0],[-T22 L21 T11, T22]], with forward
    substitution at n <= BASE. Every intermediate is a block of the true
    inverse, so nothing grows past it. (The Neumann doubling product
    prod(I + (-A)^(2^m)) is algebraically identical but forms A^32, whose
    entries reach ~C(63,32) on real, repetitive keys: NaN in fp32. Measured.)
    """
    n = A.shape[-1]
    if n <= BASE:
        rows = [mx.broadcast_to(mx.eye(n, dtype=A.dtype)[0], A.shape[:-2] + (n,))]
        for i in range(1, n):
            prev = mx.stack(rows, axis=-2)                       # [..., i, n]
            r = -(A[..., i:i + 1, :i] @ prev)[..., 0, :]
            rows.append(r + mx.eye(n, dtype=A.dtype)[i])
        return mx.stack(rows, axis=-2)
    h = n // 2
    D = _inv_unit_lower(mx.stack([A[..., :h, :h], A[..., h:, h:]], axis=0))
    T11, T22 = D[0], D[1]
    T21 = -(T22 @ (A[..., h:, :h] @ T11))
    top = mx.concatenate([T11, mx.zeros_like(T11)], axis=-1)
    bot = mx.concatenate([T21, T22], axis=-1)
    return mx.concatenate([top, bot], axis=-2)


def _intra(q, k, v, lg, beta, C):
    """Batched per-chunk precompute. Inputs [B,H,N,C,*] fp32, lg = log g."""
    cum = mx.cumsum(lg, axis=-1)                       # [B,H,N,C]
    diff = cum[..., :, None] - cum[..., None, :]       # log G_i/G_j
    tri_incl = mx.tril(mx.ones((C, C), dtype=mx.bool_))
    tri_strict = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=-1)
    decay = mx.where(tri_incl, mx.exp(mx.where(tri_incl, diff, 0.0)), 0.0)
    kk = k @ k.swapaxes(-1, -2)
    A = mx.where(tri_strict, kk * decay * beta[..., :, None], 0.0)
    Tm = _inv_unit_lower(A)
    G = mx.exp(cum)
    W = Tm @ (k * (beta * G)[..., None])
    U = Tm @ (v * beta[..., None])
    Aqk = (q @ k.swapaxes(-1, -2)) * decay
    # G_C / G_j, for the end-of-chunk state update
    kd = k * mx.exp(cum[..., -1:] - cum)[..., None]
    return G, W, U, Aqk, kd


def _chunk_scan(q, G, W, U, Aqk, kd, state):
    """Sequential over chunks. q,W,kd: [B,H,N,C,Dk]; U: [B,H,N,C,Dv]."""
    N = q.shape[2]
    qG = q * G[..., None]
    gC = G[..., -1]                                    # [B,H,N]
    ys = []
    S = state                                          # [B,H,Dv,Dk]
    for n in range(N):
        St = S.swapaxes(-1, -2)                        # [B,H,Dk,Dv]
        vp = U[:, :, n] - W[:, :, n] @ St
        ys.append(qG[:, :, n] @ St + Aqk[:, :, n] @ vp)
        S = S * gC[:, :, n, None, None] + vp.swapaxes(-1, -2) @ kd[:, :, n]
    return mx.stack(ys, axis=2), S


def gated_delta_chunked(q, k, v, lg, beta, state, mask=None, chunk=None):
    """Core. q,k [B,T,Hk,Dk]; v [B,T,Hv,Dv]; lg = log g, beta [B,T,Hv]."""
    C = chunk or CHUNK
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    f32 = mx.float32
    q, k, v = q.astype(f32), k.astype(f32), v.astype(f32)
    lg, beta = lg.astype(f32), beta.astype(f32)
    if mask is not None:
        m = mask.astype(mx.bool_)[..., None]           # [B,T,1]
        lg = mx.where(m, lg, 0.0)
        beta = mx.where(m, beta, 0.0)
    if (r := Hv // Hk) > 1:
        q = mx.repeat(q, r, -2)
        k = mx.repeat(k, r, -2)
    pad = (-T) % C
    if pad:
        z = lambda x: mx.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        q, k, v, lg, beta = z(q), z(k), z(v), z(lg), z(beta)
    N = (T + pad) // C
    sh = lambda x: x.transpose(0, 2, 1, 3).reshape(B, Hv, N, C, x.shape[-1])
    sh3 = lambda x: x.transpose(0, 2, 1).reshape(B, Hv, N, C)
    q, k, v, lg, beta = sh(q), sh(k), sh(v), sh3(lg), sh3(beta)
    G, W, U, Aqk, kd = _intra(q, k, v, lg, beta, C)
    y, S = _chunk_scan(q, G, W, U, Aqk, kd, state.astype(f32))
    y = y.reshape(B, Hv, N * C, Dv)[:, :, :T].transpose(0, 2, 1, 3)
    if mask is not None:
        y = mx.where(mask.astype(mx.bool_)[..., None, None], y, 0.0)
    return y, S


def log_g(A_log, a, dt_bias):
    return -mx.exp(A_log.astype(mx.float32)) * nn.softplus(
        a.astype(mx.float32) + dt_bias.astype(mx.float32))


def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
                       use_kernel=True):
    gd = _stock()
    T = q.shape[1]
    if a.ndim == 4 or T < MIN_T:
        return gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state,
                                     mask, use_kernel)
    B, _, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    st_dtype = state.dtype
    beta = mx.sigmoid(b.astype(mx.float32))
    y, S = gated_delta_chunked(q, k, v, log_g(A_log, a, dt_bias), beta,
                               state, mask)
    return y.astype(q.dtype), S.astype(st_dtype)
