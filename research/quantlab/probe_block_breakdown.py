#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""VQ-PF1 probe #3: where inside a real VQ block does the time go?

probe_block_prefill.py: VQ block 1135 ms/bucket vs spicy 295 ms/bucket at
the scorer's shape. probe_vq_prefill_regime.py: the isolated expert GEMM is
at PARITY (61 ms/projection, x3 = 185 ms). So ~950 ms/bucket lives OUTSIDE
the expert matmul. This instruments VQSwitchLinear.__call__ stage by stage
on the real block to name it.
"""
import argparse
import time

import mlx.core as mx
import numpy as np

STATS = {}


def _t(key, fn):
    mx.synchronize()
    t0 = time.perf_counter()
    r = fn()
    mx.eval(r)
    mx.synchronize()
    STATS[key] = STATS.get(key, 0.0) + time.perf_counter() - t0
    return r


def instrument(VQSwitchLinear, vq_switch):
    def __call__(self, x, indices, sorted_indices=False):
        IN, OUT = self.input_dims, self.output_dims
        idx_flat = indices.flatten()
        N = idx_flat.size
        STATS["_N"] = N
        STATS["_calls"] = STATS.get("_calls", 0) + 1
        STATS["_sorted_in"] = STATS.get("_sorted_in", 0) + int(bool(sorted_indices))
        xf = _t("broadcast+reshape",
                lambda: mx.broadcast_to(x, (*indices.shape, 1, IN)).reshape(N, IN))
        if xf.dtype != mx.float16:
            xf = _t("astype fp16", lambda: xf.astype(mx.float16))
        pb = self.pack_bits
        if N <= vq_switch.VQ_FUSED_MAX_N:
            y = _t("fused", lambda: vq_switch._fused(
                xf, idx_flat.astype(mx.uint32), self["codes"],
                self["codebook"], self["vq_scales"], pack_bits=pb))
        else:
            mx.synchronize()
            t0 = time.perf_counter()
            idx_np = np.array(idx_flat, copy=False)
            STATS["idx->host sync"] = STATS.get("idx->host sync", 0.0) + \
                time.perf_counter() - t0
            if not sorted_indices:
                t0 = time.perf_counter()
                order = np.argsort(idx_np, kind="stable")
                inv = np.argsort(order, kind="stable")
                STATS["np.argsort x2"] = STATS.get("np.argsort x2", 0.0) + \
                    time.perf_counter() - t0
                xs = _t("gather rows (sort)",
                        lambda: xf[mx.array(order.astype(np.uint32))])
                y = _t("prefill", lambda: vq_switch._prefill(
                    xs, idx_np[order], self["codes"], self["codebook"],
                    self["vq_scales"], pack_bits=pb, in_features=IN))
                y = _t("gather rows (unsort)",
                       lambda: y[mx.array(inv.astype(np.uint32))])
            else:
                y = _t("prefill", lambda: vq_switch._prefill(
                    xf, idx_np, self["codes"], self["codebook"],
                    self["vq_scales"], pack_bits=pb, in_features=IN))
        return _t("reshape out",
                  lambda: y.astype(x.dtype).reshape(*indices.shape, 1, OUT))

    VQSwitchLinear.__call__ = __call__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Volumes/Thunderbay SSD/Exo Models/"
                                       "Qwen3.5-397B-A17B-VQ-2.4bpw")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--batch-seqs", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=36)
    ap.add_argument("--buckets", type=int, default=3)
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

    import sys
    # the artifact's model.py defines VQSwitchLinear in ITS OWN module; find
    # the live class off an actual instance so we patch what is really used.
    inst = None
    for _, m in blk.named_modules():
        if type(m).__name__ == "VQSwitchLinear":
            inst = m
            break
    assert inst is not None, "no VQSwitchLinear in this block"
    cls = type(inst)
    # mlx_lm execs the checkpoint's model.py under a synthetic name that is
    # not always registered in sys.modules; the class's globals ARE the
    # module namespace (_prefill/_fused/VQ_FUSED_MAX_N live there).
    vqmod = sys.modules.get(cls.__module__)
    if vqmod is None:
        vqmod = type("ns", (), cls.__call__.__globals__)
    print(f"patching {cls.__module__}.VQSwitchLinear")
    instrument(cls, vqmod)

    hidden = blk.input_layernorm.weight.shape[-1]
    B, L = args.batch_seqs, args.seq_len
    totals = []
    for i in range(args.buckets):
        STATS.clear()
        h = mx.random.normal((B, L, hidden)).astype(mx.float16)
        mx.eval(h)
        mx.synchronize()
        t0 = time.perf_counter()
        out = blk(h, mask="causal", cache=None)
        mx.eval(out)
        mx.synchronize()
        tot = time.perf_counter() - t0
        totals.append((tot, dict(STATS)))
        del h, out
        mx.clear_cache()

    tot, st = totals[-1]
    print(f"\nbucket [{B},{L}] = {B*L} tokens   BLOCK TOTAL {tot*1e3:.1f} ms")
    print(f"  VQSwitchLinear calls: {st.get('_calls')}  "
          f"N per call: {st.get('_N')}  "
          f"already-sorted calls: {st.get('_sorted_in')}")
    acct = 0.0
    for k, v in sorted(((k, v) for k, v in st.items() if not k.startswith("_")),
                       key=lambda kv: -kv[1]):
        acct += v
        print(f"    {k:24s} {v*1e3:9.1f} ms  {100*v/tot:5.1f}%")
    print(f"    {'--- accounted':24s} {acct*1e3:9.1f} ms  {100*acct/tot:5.1f}%")
    print(f"    {'--- rest of block':24s} {(tot-acct)*1e3:9.1f} ms  "
          f"{100*(tot-acct)/tot:5.1f}%")


if __name__ == "__main__":
    main()
