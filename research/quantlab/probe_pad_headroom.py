#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""VQ-PF1 probe #5: how much of the padded-GEMM tax is recoverable WITHOUT
touching the on-disk layout?

probe_routing_pad.py named the cause: _prefill pads every expert in a chunk
to that chunk's max token count, and real routing is skewed 8.7x (max 1574
vs mean 180), so at _DECODE_CHUNK=128 the GEMM does 5.80x the necessary
FLOPs. Two remedies cost nothing on disk:

  (a) smaller chunks     -- cap is a per-chunk max, so fewer experts per
                            chunk means a tighter cap, at the price of more
                            decode-kernel launches.
  (b) count-sorted chunks -- chunk experts by SIMILAR token count instead of
                            by expert id, so the max within a chunk is close
                            to its mean. Pure host-side reordering of which
                            experts share a GEMM; codes/weights untouched.

This measures pad ratio and wall time for the cross product, against the
real router histogram captured from a real block.
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


def pad_ratio(counts, touched, chunk):
    real = padded = 0
    for c0 in range(0, len(touched), chunk):
        eids = touched[c0:c0 + chunk]
        cap = int(counts[eids].max())
        real += int(counts[eids].sum())
        padded += len(eids) * cap
    return padded / max(real, 1)


def prefill_sorted_chunks(xf, idx_sorted_np, codes, codebook, scales,
                          in_features, chunk, ns):
    """(b): identical math to vq_switch._prefill, but experts are grouped
    into chunks by token count so `cap` tracks the chunk mean."""
    E = codes.shape[0]
    OUT = codes.shape[1]
    counts = np.bincount(idx_sorted_np, minlength=E)
    touched = np.nonzero(counts)[0]
    touched = touched[np.argsort(counts[touched], kind="stable")]
    starts = np.zeros(E + 1, np.int64)
    starts[1:] = np.cumsum(counts)
    ys = []
    for c0 in range(0, len(touched), chunk):
        eids = touched[c0:c0 + chunk]
        ne = len(eids)
        cap = int(counts[eids].max())
        gmap = np.zeros((ne, cap), np.uint32)
        vmask = np.zeros((ne, cap), bool)
        for i, e in enumerate(eids):
            c = counts[e]
            gmap[i, :c] = np.arange(starts[e], starts[e] + c, dtype=np.uint32)
            vmask[i, :c] = True
        w = ns["_decode_chunk"](codes, codebook, scales,
                                mx.array(eids.astype(np.uint32)),
                                in_features=in_features)
        xp = xf[mx.array(gmap.reshape(-1))].reshape(ne, cap, -1)
        yp = xp @ mx.swapaxes(w, 1, 2)
        flat_valid = np.nonzero(vmask.reshape(-1))[0].astype(np.uint32)
        ys.append(yp.reshape(ne * cap, OUT)[mx.array(flat_valid)])
        mx.eval(ys[-1])
        del w, xp, yp
    return mx.concatenate(ys, axis=0)


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

    inst = None
    for _, m in blk.named_modules():
        if type(m).__name__ == "VQSwitchLinear":
            inst = m
            break
    cls = type(inst)
    ns = cls.__call__.__globals__
    captured = {}
    orig = cls.__call__

    def spy(self, x, indices, sorted_indices=False):
        captured.setdefault("idx", np.array(indices.flatten(), copy=True))
        return orig(self, x, indices, sorted_indices)

    cls.__call__ = spy
    hidden = blk.input_layernorm.weight.shape[-1]
    h = mx.random.normal((args.batch_seqs, args.seq_len, hidden)).astype(mx.float16)
    mx.eval(h)
    mx.eval(blk(h, mask="causal", cache=None))
    cls.__call__ = orig

    vq = inst
    E, IN, OUT = vq.num_experts, vq.input_dims, vq.output_dims
    e_sorted = np.sort(captured["idx"])
    counts = np.bincount(e_sorted, minlength=E)
    touched_id = np.nonzero(counts)[0]
    touched_ct = touched_id[np.argsort(counts[touched_id], kind="stable")]

    xf = mx.random.normal((e_sorted.size, IN)).astype(mx.float16)
    mx.eval(xf)
    print(f"layer {args.layer}  N={e_sorted.size}  E={E} IN={IN} OUT={OUT}  "
          f"mean {counts[touched_id].mean():.0f} max {counts.max()} rows/expert")
    print(f"\n{'chunk':>6} | {'pad(id)':>8} {'time(id)':>10} | "
          f"{'pad(count)':>10} {'time(count)':>12}")
    for chunk in (8, 16, 32, 64, 128, 256, 512):
        ns["_DECODE_CHUNK"] = chunk
        p_id = pad_ratio(counts, touched_id, chunk)
        p_ct = pad_ratio(counts, touched_ct, chunk)
        t_id = bench(lambda: ns["_prefill"](
            xf, e_sorted, vq["codes"], vq["codebook"], vq["vq_scales"],
            pack_bits=vq.pack_bits, in_features=IN))
        t_ct = bench(lambda c=chunk: prefill_sorted_chunks(
            xf, e_sorted, vq["codes"], vq["codebook"], vq["vq_scales"],
            IN, c, ns))
        print(f"{chunk:6d} | {p_id:7.2f}x {t_id*1e3:9.1f}m | "
              f"{p_ct:9.2f}x {t_ct*1e3:11.1f}m")


if __name__ == "__main__":
    main()
