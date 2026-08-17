#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""VQ-PF1 probe #4: routing skew and the padded-GEMM tax.

probe_block_breakdown.py: _prefill costs 261 ms/call inside the REAL block
but 61 ms/call in probe_vq_prefill_regime.py, which routed tokens with a
UNIFORM rng. The only difference is the expert histogram. _prefill pads
every expert in a chunk to `cap = counts[eids].max()`, so its FLOPs scale
with ne*cap, not with the token count -- a skewed router pays the max, not
the mean, for every expert in the chunk.

This captures the REAL router distribution off a real block and reports the
pad ratio, then times _prefill under real vs uniform routing side by side.
"""
import argparse
import time

import mlx.core as mx
import numpy as np


def bench(fn, warmup=1, reps=3):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / reps


def pad_report(e_sorted, E, chunk):
    counts = np.bincount(e_sorted, minlength=E)
    touched = np.nonzero(counts)[0]
    real = padded = 0
    caps = []
    for c0 in range(0, len(touched), chunk):
        eids = touched[c0:c0 + chunk]
        cap = int(counts[eids].max())
        caps.append(cap)
        real += int(counts[eids].sum())
        padded += len(eids) * cap
    return dict(touched=len(touched), mean=counts[touched].mean(),
                maxc=int(counts.max()), caps=caps,
                real_rows=real, padded_rows=padded, ratio=padded / max(real, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Volumes/Thunderbay SSD/Exo Models/"
                                       "Qwen3.5-397B-A17B-VQ-2.4bpw")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--batch-seqs", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=36)
    args = ap.parse_args()

    mx.set_cache_limit(8 << 30)
    from mlx_lm.utils import load
    with mx.stream(mx.cpu):
        model, _, _ = load(args.model, lazy=True, return_config=True)
    core = model
    for name in ("language_model", "model"):
        while hasattr(core, name):
            core = getattr(core, name)
    blk = core.layers[args.layer]
    with mx.stream(mx.cpu):
        mx.eval(blk.parameters())
    blk.eval()

    # capture the real (token, expert) assignment this block's router emits
    captured = {}
    inst = None
    for _, m in blk.named_modules():
        if type(m).__name__ == "VQSwitchLinear":
            inst = m
            break
    cls = type(inst)
    orig = cls.__call__

    def spy(self, x, indices, sorted_indices=False):
        captured.setdefault("idx", np.array(indices.flatten(), copy=True))
        captured.setdefault("sorted", sorted_indices)
        return orig(self, x, indices, sorted_indices)

    cls.__call__ = spy
    hidden = blk.input_layernorm.weight.shape[-1]
    h = mx.random.normal((args.batch_seqs, args.seq_len, hidden)).astype(mx.float16)
    mx.eval(h)
    mx.eval(blk(h, mask="causal", cache=None))
    cls.__call__ = orig

    g = vq = None
    for _, m in blk.named_modules():
        if type(m).__name__ == "VQSwitchLinear":
            vq = m
            break
    E = vq.num_experts
    IN, OUT = vq.input_dims, vq.output_dims
    ns = cls.__call__.__globals__
    chunk = ns["_DECODE_CHUNK"] or ns["_default_decode_chunk"]()

    idx = captured["idx"]
    e_real = np.sort(idx)
    rng = np.random.default_rng(3)
    e_unif = np.sort(rng.integers(0, E, idx.shape).astype(idx.dtype))

    print(f"layer {args.layer}  bucket [{args.batch_seqs},{args.seq_len}]  "
          f"E={E} IN={IN} OUT={OUT}  N={idx.size}  _DECODE_CHUNK={chunk}")
    print(f"  router handed indices already sorted: {captured['sorted']}")
    for name, es in (("REAL router", e_real), ("uniform rng", e_unif)):
        r = pad_report(es, E, chunk)
        print(f"\n  {name}: touched {r['touched']}/{E}  "
              f"mean {r['mean']:.0f} rows/expert  max {r['maxc']}")
        print(f"    per-chunk caps: {r['caps']}")
        print(f"    padded rows {r['padded_rows']} vs real {r['real_rows']}  "
              f"=> {r['ratio']:.2f}x FLOPs")

    xf = mx.random.normal((idx.size, IN)).astype(mx.float16)
    mx.eval(xf)
    for name, es in (("REAL router", e_real), ("uniform rng", e_unif)):
        t = bench(lambda e=es: ns["_prefill"](
            xf, e, vq["codes"], vq["codebook"], vq["vq_scales"],
            pack_bits=vq.pack_bits, in_features=IN))
        print(f"  _prefill under {name}: {t*1e3:8.1f} ms")


if __name__ == "__main__":
    main()
