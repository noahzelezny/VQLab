#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""E34d: how fast does GPTQ decision-noise fall with calibration size?

E34c showed disagreement (25.7%) EXCEEDS signal (19.6%) at ~2.4k tok/expert.
Before buying an 8x run we need the SLOPE: measure agreement between two
disjoint subsets at several sizes, fit log-log, extrapolate to the token
counts an 8x / 32x corpus would give. If the exponent says we need 100x,
the 8x run is futile and vector quant is the honest next move.
"""
import gc
import numpy as np
import mlx.core as mx
from gptq_solver import quantize_gptq, quantize_rtn

mx.set_cache_limit(8 << 30)
SRC = "/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-35B-A3B"
ACT = "/Volumes/Thunderbay SSD/Exo Models/rotlab-35B-base-struct6"
LAYER = 12
SIZES = (256, 512, 1024, 2048)
N_EXPERTS = 10
SEQ, CHUNK = 4096, 4

from mlx_lm.utils import load
from mlx_lm.models import switch_layers
with mx.stream(mx.cpu):
    model, tok, _ = load(ACT, lazy=True, return_config=True)
    wmodel, _, _ = load(SRC, lazy=True, return_config=True)

def _core(m):
    c = m
    for n in ("language_model", "model"):
        while hasattr(c, n):
            c = getattr(c, n)
    return c
core, wcore = _core(model), _core(wmodel)

ids = tok.encode(open("calib_corpus.txt", errors="replace").read())
n_seq = len(ids) // SEQ
toks = mx.array(ids[: n_seq * SEQ]).reshape(n_seq, SEQ)

cap = {"x": [], "i": []}
orig = switch_layers.SwitchGLU.__call__
def spy(self, x, indices, *a, **k):
    cap["x"].append(np.array(x.reshape(-1, x.shape[-1]).astype(mx.float32)))
    cap["i"].append(np.array(indices.reshape(-1, indices.shape[-1])))
    return orig(self, x, indices, *a, **k)

with mx.stream(mx.cpu):
    mx.eval(core.embed_tokens.parameters())
hs = [core.embed_tokens(toks[s:s+CHUNK]) for s in range(0, n_seq, CHUNK)]
mx.eval(hs)
for li in range(LAYER + 1):
    blk = core.layers[li]
    mask = None if blk.is_linear else "causal"
    with mx.stream(mx.cpu):
        mx.eval(blk.parameters())
    blk.eval()
    if li == LAYER:
        switch_layers.SwitchGLU.__call__ = spy
    hs = [blk(h, mask=mask, cache=None) for h in hs]
    mx.eval(hs)
    if li == LAYER:
        switch_layers.SwitchGLU.__call__ = orig
    core.layers[li] = None
    del blk
    gc.collect(); mx.clear_cache()

X = np.concatenate(cap["x"]); I = np.concatenate(cap["i"])
wblk = wcore.layers[LAYER]
with mx.stream(mx.cpu):
    mx.eval(wblk.parameters())
sm = wblk.mlp.switch_mlp
Wg = np.array(sm.gate_proj.weight.astype(mx.float32))
Wu = np.array(sm.up_proj.weight.astype(mx.float32))

rng = np.random.default_rng(0)
print(f"layer {LAYER}; disagreement between two DISJOINT subsets\n")
curve = []
for n in SIZES:
    dis, sig = [], []
    for e in range(0, 256, 256 // N_EXPERTS):
        rows = X[(I == e).any(axis=1)]
        if rows.shape[0] < 2 * n:
            continue
        idx = rng.permutation(rows.shape[0])
        A, B = rows[idx[:n]], rows[idx[n:2*n]]
        W = np.concatenate([Wg[e], Wu[e]], axis=0)
        qa, _, _ = quantize_gptq(W, A.T @ A)
        qb, _, _ = quantize_gptq(W, B.T @ B)
        qr, _, _ = quantize_rtn(W)
        dis.append(float((qa != qb).mean()))
        sig.append(float((qa != qr).mean()))
    if not dis:
        continue
    curve.append((n, np.mean(dis), np.mean(sig)))
    print(f"  {n:5d} tok/subset  disagreement {np.mean(dis)*100:5.2f}%   "
          f"signal(vs RTN) {np.mean(sig)*100:5.2f}%   "
          f"ratio {np.mean(dis)/np.mean(sig):.2f}")

# The DIAGNOSTIC IS THE RATIO, not either column alone, and NOT an
# extrapolation: the fitted slope on disagreement comes out POSITIVE (both
# columns grow with N), so a log-log extrapolation yields impossible >100%
# values. It was printed once and is deliberately removed — do not restore.
#
# Why both grow: at small N the Hessian is near-singular, damping dominates,
# and GPTQ ~= RTN (little signal, little noise). As N grows the Hessian
# conditions up and the solver moves more weights off RTN. A FLAT ratio
# therefore means every extra decision it earns is one an independent sample
# would have made differently -> no SNR improvement from more data.
ratios = [c[1] / c[2] for c in curve]
print(f"\nratio disagreement/signal across {curve[0][0]}-{curve[-1][0]} "
      f"tok/subset: {np.mean(ratios):.2f} +/- {np.std(ratios):.2f}")
print("FLAT ratio  -> noise scales WITH signal; more data will NOT help.")
print("FALLING     -> data is the lever; buy a bigger corpus.")
