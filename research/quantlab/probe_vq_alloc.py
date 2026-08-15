#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Per-projection (d,K) marginal-return curves — 397B, one layer, weight space.

The three expert projections are EQUAL THIRDS of expert mass but quantize very
differently (P3: down_proj VQ gains 60%, gate_up 15%). Uniform K is therefore
almost certainly a misallocation. This measures relerr vs bits for each
projection so an allocation can be CHOSEN rather than guessed.

Ranking only — weight relerr is not the objective (E34). Confirm the winner
end-to-end before believing it.
"""
import json, math, mlx.core as mx
mx.set_cache_limit(6 << 30)
S = '/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-397B-A17B-bf16'
idx = json.load(open(S + '/model.safetensors.index.json'))['weight_map']
G, LAYER, NEXP = 64, 20, 48

def kmeans(X, k, iters=18):
    C = X[mx.random.randint(0, X.shape[0], (k,))]
    xn = mx.sum(X * X, axis=1, keepdims=True)
    for _ in range(iters):
        cn = mx.sum(C * C, axis=1)
        a = mx.argmin(xn - 2 * (X @ C.T) + cn[None, :], axis=1); mx.eval(a)
        oh = (a[:, None] == mx.arange(k)[None, :]).astype(mx.float32)
        cnt = mx.sum(oh, axis=0)
        C = mx.where(cnt[:, None] > 0, (oh.T @ X) / mx.maximum(cnt[:, None], 1.0), C)
        mx.eval(C)
    return C

def relerr(W, d, k):
    e, o, i = W.shape
    Wg = W.reshape(-1, i // G, G)
    sc = mx.maximum(mx.max(mx.abs(Wg), axis=2, keepdims=True), 1e-8)
    sc = sc.astype(mx.bfloat16).astype(mx.float32)
    sub = (Wg / sc).reshape(-1, d); mx.eval(sub)
    samp = sub[mx.random.randint(0, sub.shape[0], (min(1_500_000, sub.shape[0]),))]
    samp = mx.where(mx.isnan(samp), 0.0, samp)      # guard: nan seeds poison k-means
    C = kmeans(samp, k)
    C = mx.where(mx.isnan(C), 0.0, C)
    cn = mx.sum(C * C, axis=1); parts = []
    # chunk must scale INVERSELY with K: the distance matrix is
    # [chunk, K] fp32 and Metal caps a single buffer at ~62 GB.
    step = max(100_000, int(6e9 / k))
    for c in range(0, sub.shape[0], step):
        xb = sub[c:c + step]
        a = mx.argmin(mx.sum(xb * xb, axis=1, keepdims=True) - 2 * (xb @ C.T) + cn[None, :], axis=1)
        parts.append(C[a]); mx.eval(parts[-1])
    R = (mx.concatenate(parts, axis=0).reshape(-1, i // G, G) * sc).reshape(e, o, i)
    return float(mx.linalg.norm(R - W) / mx.linalg.norm(W))

gk = f"model.language_model.layers.{LAYER}.mlp.experts.gate_up_proj"
dk = f"model.language_model.layers.{LAYER}.mlp.experts.down_proj"
GU = mx.load(S + '/' + idx[gk])[gk][:NEXP]
DN = mx.load(S + '/' + idx[dk])[dk][:NEXP]
mid = GU.shape[1] // 2
tens = {"gate": GU[:, :mid, :], "up": GU[:, mid:, :], "down": DN}

print(f"layer {LAYER}, {NEXP} experts, weight-space relerr (lower=better)\n")
print(f"{'setting':16s} {'bpw':>5s} " + "".join(f"{n:>9s}" for n in tens))
for d, k in ((4, 256), (4, 1024), (4, 2048), (4, 4096), (8, 4096), (8, 65536), (2, 16)):
    bpw = math.log2(k) / d + 16.0 / G
    row = {}
    for n, T in tens.items():
        row[n] = relerr(T.astype(mx.float32), d, k)
    print(f"d={d} K={k:<7d}  {bpw:5.2f} " + "".join(f"{row[n]:9.4f}" for n in tens), flush=True)
# RTN control
print()
for n, T in tens.items():
    wq, s2, b2 = mx.quantize(T, group_size=64, bits=2)
    W = T.astype(mx.float32)
    e = float(mx.linalg.norm(mx.dequantize(wq, s2, b2, group_size=64, bits=2).astype(mx.float32) - W) / mx.linalg.norm(W))
    print(f"RTN 2-bit (2.50 bpw)  {n:>6s}: {e:.4f}")
