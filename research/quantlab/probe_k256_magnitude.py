#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Why does the K256 refit fit BETTER and score WORSE? (E101 open question)

HYPOTHESIS. k-means minimizes AVERAGE distortion. When centroids are scarce
the objective trades the tail away: centroids pack into the dense middle of
the weight distribution and abandon rare large-magnitude weights, because
covering them costs more average error than it saves. Large weights dominate
the output. At K=2048/K=8192 there are enough centroids for both, so a better
fit is a better model. At K=256 they compete.

PREDICTION, which is what makes this falsifiable: bucket weights by |w|. The
refit should WIN on the low-|w| buckets (that is where its lower mean relerr
comes from) and LOSE on the high-|w| buckets. If instead it wins or ties
everywhere, the hypothesis is dead and the cause is elsewhere.

Reconstruction mirrors verify_artifact.py exactly (cpu-stream source read
included) so the numbers are comparable to the gate's.
"""
import json, pathlib, sys
import mlx.core as mx
import numpy as np

E_ROOT = pathlib.Path("/Volumes/Thunderbay SSD/Exo Models")
SRC = E_ROOT / "Qwen--Qwen3.5-397B-A17B-bf16"
ARMS = {"shipped2.4": E_ROOT / "TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw",
        "refit":      E_ROOT / "rotlab--397B-flatk256-refit-packed"}
G = 64
NEXP = 4          # experts per tensor to sample; full E is 512 and unnecessary
LAYERS = [10, 30, 50]
PROJ = "down_proj"
QS = [0, 50, 90, 99, 99.9, 100]   # |w| percentile buckets

src_map = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
_src_cache = {}


def src_tensor(li):
    name = f"model.language_model.layers.{li}.mlp.experts.down_proj"
    sh = src_map[name]
    if sh not in _src_cache:
        _src_cache.clear()
        _src_cache[sh] = mx.load(str(SRC / sh))
    return _src_cache[sh][name]


def recon(art, li, nexp):
    idx = json.load(open(art / "model.safetensors.index.json"))["weight_map"]
    cfg = json.load(open(art / "config.json"))["vq_modules"]
    mod = f"language_model.model.layers.{li}.mlp.switch_mlp.{PROJ}"
    meta = cfg[mod]
    data = mx.load(str(art / idx[mod + ".codes"]))
    codes = data[mod + ".codes"][:nexp]
    cb = data[mod + ".codebook"].astype(mx.float32)
    sc = data[mod + ".vq_scales"][:nexp].astype(mx.float32)
    d, in_d = meta["dim"], meta["in"]
    nsub = in_d // d
    if codes.dtype == mx.uint32:
        import vq_pack
        codes = mx.array(vq_pack.unpack(np.array(codes), nsub,
                                        meta["pack_bits"]).astype(np.uint32))
    e, out_d = codes.shape[0], codes.shape[1]
    w = cb[codes.reshape(-1)].reshape(e, out_d, nsub * d)
    w = (w.reshape(e, out_d, in_d // G, G) * sc[..., None]).reshape(e, out_d, in_d)
    return w


print(f"{'layer':>5} {'bucket |w| pct':>16} {'shipped2.4':>12} {'refit':>12} {'refit-shipped':>14}")
print("-" * 64)
agg = {}
for li in LAYERS:
    with mx.stream(mx.cpu):
        T = src_tensor(li)[:NEXP].astype(mx.float32)
        mx.eval(T)
    a = np.array(T).ravel()
    errs = {}
    for name, art in ARMS.items():
        w = recon(art, li, NEXP)
        mx.eval(w)
        errs[name] = np.array(w).ravel() - a
        del w
        mx.clear_cache()
    mag = np.abs(a)
    edges = np.percentile(mag, QS)
    for i in range(len(QS) - 1):
        m = (mag >= edges[i]) & (mag < edges[i + 1] if i + 1 < len(QS) - 1
                                 else mag <= edges[i + 1])
        if m.sum() == 0:
            continue
        # RMS error within the bucket, normalized by RMS weight in the bucket
        vals = {}
        for name in ARMS:
            vals[name] = float(np.sqrt((errs[name][m] ** 2).mean())
                               / max(np.sqrt((a[m] ** 2).mean()), 1e-12))
        lbl = f"{QS[i]:g}-{QS[i+1]:g}"
        delta = vals["refit"] - vals["shipped2.4"]
        agg.setdefault(lbl, []).append(delta)
        print(f"{li:5d} {lbl:>16} {vals['shipped2.4']:12.5f} {vals['refit']:12.5f} "
              f"{delta:+14.5f} {'REFIT WORSE' if delta > 0 else ''}")
    del T
    mx.clear_cache()

print("\nmean delta by bucket (positive = refit worse):")
for lbl, ds in agg.items():
    print(f"  |w| {lbl:>10} pct   {np.mean(ds):+.5f}")
