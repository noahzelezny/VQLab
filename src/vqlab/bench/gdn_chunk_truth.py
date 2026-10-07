"""gdn-chunk-truth: chunked vs sequential gated-delta scan against an fp64 reference.

R3-3 gate 1 (F138 rule: a divergence is disagreement, never error -- so
every path is measured against an EXACT float64 recurrence fed the same
rounded inputs the kernels get). Paths: the stock sequential Metal kernel,
the stock fp32 ops path (gated_delta_ops), and the chunked WY path
(vqlab.runtime.gated_delta_chunked). Also a per-T microbench of the scan
alone (kernel vs chunked), median of --reps.

Inputs: random (qwen3_5-shaped: Hk=16 Hv=32 Dk=Dv=128, L2-normed q/k) or
REAL activations captured from layer --layer of --art during one prefill.

    PYTHONPATH=src python -m vqlab.bench.gdn_chunk_truth --T 512 2048
    PYTHONPATH=src python -m vqlab.bench.gdn_chunk_truth --art <pin> --T 2048
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np


def ref64(q, k, v, a, b, A_log, dt_bias, state):
    """Exact sequential recurrence in float64. Arrays numpy; q,k [T,Hk,Dk]."""
    T, Hk, Dk = q.shape
    Hv, Dv = v.shape[1:]
    r = Hv // Hk
    q = np.repeat(q, r, 1); k = np.repeat(k, r, 1)
    x = a + dt_bias
    sp = np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))
    g = np.exp(-np.exp(A_log) * sp)                       # [T,Hv]
    beta = 1 / (1 + np.exp(-b))
    S = state.copy()                                      # [Hv,Dv,Dk]
    ys = np.empty((T, Hv, Dv))
    for t in range(T):
        S *= g[t][:, None, None]
        kv = np.einsum("hvd,hd->hv", S, k[t])
        delta = (v[t] - kv) * beta[t][:, None]
        S += delta[:, :, None] * k[t][:, None, :]
        ys[t] = np.einsum("hvd,hd->hv", S, q[t])
    return ys, S


def _random(T, dtype, seed=1234):
    rng = np.random.default_rng(seed)
    Hk, Hv, D = 16, 32, 128
    def nrm(x):
        return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)
    q = nrm(rng.standard_normal((1, T, Hk, D))) * D ** -0.5
    k = nrm(rng.standard_normal((1, T, Hk, D)))
    v = rng.standard_normal((1, T, Hv, D))
    a = rng.standard_normal((1, T, Hv))
    b = rng.standard_normal((1, T, Hv))
    A_log = np.log(rng.uniform(1, 16, Hv))
    dt = np.ones(Hv)
    c = lambda x: mx.array(x.astype(np.float32)).astype(dtype)
    return (c(q), c(k), c(v), c(a), c(b), mx.array(A_log.astype(np.float32)),
            mx.array(dt.astype(np.float32)))


def _capture(art, T, layer):
    import sys
    from mlx_lm import load
    model, tok = load(art)
    from mlx_lm.models import gated_delta as gd
    got, count = {}, [0]
    orig = gd.gated_delta_update

    def hook(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        if count[0] == layer:
            got["x"] = [mx.array(t) for t in (q, k, v, a, b, A_log, dt_bias)]
            mx.eval(got["x"])
        count[0] += 1
        return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)

    for name, mod in list(sys.modules.items()):
        if mod is not None and getattr(mod, "gated_delta_update", None) is not None \
                and name != "mlx_lm.models.gated_delta" and not name.startswith("vqlab."):
            mod.gated_delta_update = hook
    word = ("the quick brown fox jumps over the lazy dog while considering "
            "distributed inference ")
    ids = tok.encode(word * (T // 13 + 2))[:T]
    from mlx_lm.models.cache import make_prompt_cache
    mx.eval(model(mx.array([ids]), cache=make_prompt_cache(model)))
    if "x" not in got:
        raise SystemExit(f"captured nothing: {count[0]} GDN calls, --layer {layer}")
    x = got["x"]
    del model
    return x


def _err(y, ref):
    y = np.asarray(y.astype(mx.float32), dtype=np.float64)
    d = np.abs(y - ref)
    return d.max(), d.max() / np.abs(ref).max(), np.sqrt((d ** 2).mean() / (ref ** 2).mean())


def _time(fn, reps):
    mx.eval(fn()); ts = []
    for _ in range(reps):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--T", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--art", default=None, help="capture REAL activations")
    ap.add_argument("--layer", type=int, default=0, help="GDN layer index to capture")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), nargs="+",
                    default=["float32", "bfloat16"])
    ap.add_argument("--chunks", type=int, nargs="+", default=[64])
    ap.add_argument("--no-ops", action="store_true", help="skip gated_delta_ops")
    ap.add_argument("--no-ref", action="store_true", help="timing only")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    from mlx_lm.models import gated_delta as gd
    from vqlab.runtime import gated_delta_chunked as gc
    print("src  dtype     T     path         max_abs     max_rel     rms_rel  "
          "| state max_rel  rms_rel | scan ms")
    for T in a.T:
        real = _capture(a.art, T, a.layer) if a.art else None
        for dt in a.dtype:
            dtype = getattr(mx, dt)
            x = [t.astype(dtype) for t in real[:5]] + real[5:] if real else _random(T, dtype)
            q, k, v, aa, bb, A_log, dtb = x
            B, _, Hk, Dk = q.shape; Hv, Dv = v.shape[-2:]
            s0 = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
            if not a.no_ref:
                n64 = lambda t: np.asarray(t.astype(mx.float32), dtype=np.float64)
                yr, Sr = ref64(n64(q[0]), n64(k[0]), n64(v[0]), n64(aa[0]), n64(bb[0]),
                               n64(A_log), n64(dtb), np.zeros((Hv, Dv, Dk)))
            paths = {"kernel": lambda: gd.gated_delta_update(q, k, v, aa, bb, A_log, dtb, s0)}
            if not a.no_ops:
                paths["ops-fp32"] = lambda: gd.gated_delta_update(
                    q, k, v, aa, bb, A_log, dtb, s0, use_kernel=False)
            for C in a.chunks:
                def ch(C=C):
                    beta = mx.sigmoid(bb.astype(mx.float32))
                    y, S = gc.gated_delta_chunked(q, k, v, gc.log_g(A_log, aa, dtb),
                                                  beta, s0, chunk=C)
                    return y.astype(q.dtype), S
                paths[f"chunk{C}"] = ch
            for name, fn in paths.items():
                y, S = fn(); mx.eval(y, S)
                ms = _time(fn, a.reps) * 1e3 if name != "ops-fp32" else float("nan")
                if a.no_ref:
                    print(f"{'real' if real else 'rand'} {dt:9s} {T:5d} {name:10s} "
                          f"{'':>35s} | {'':>22s} | {ms:8.2f}", flush=True)
                    continue
                ea = _err(y[0], yr); es = _err(S[0], Sr)
                print(f"{'real' if real else 'rand'} {dt:9s} {T:5d} {name:10s} "
                      f"{ea[0]:10.3e} {ea[1]:10.3e} {ea[2]:10.3e} | {es[1]:10.3e} "
                      f"{es[2]:10.3e} | {ms:8.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
