#!/usr/bin/env python
"""MoE PREFILL ATTRIBUTION on a real artifact, real prompt, real router.

The 2026-09-05 swarm produced 12 proposals that all hinge on one unmeasured
question: where do the ~120 s of a 9k-token MoE prefill actually go?
Candidates: host-side chunk-loop work (numpy gmap/vmask building + blocking
mx.eval sync per chunk), expert weight decode (_decode_chunk), the padded
batched GEMM, or none of the above (attention / harness). The salvaged
compute-frame worker's FLOP math says GEMM alone is seconds, not minutes.

Method: load the artifact through its OWN bundled model.py (the runtime a
downloader gets), monkeypatch the loaded module's _prefill via a VQ class's
__globals__ (mlx-lm loads model_file as an unregistered module; the class's
functions close over its dict — TWO-RUNTIMES.md), and wrap each phase with
mx.synchronize() fences. Fencing serializes host and device, so phase sums
OVERSTATE pipelined wall time — run once unfenced for the honest end-to-end
number, once fenced for the split. Router histograms are np.bincount of the
REAL eidx per call (VQ-PF1: synthetic routers invalidate everything).

Run (repo .venv, M3 Ultra, exo quiet):
  ../.venv/bin/python scripts/probe_moe_prefill_attrib.py --tokens 9000
"""
import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np

ART = os.path.expanduser(
    "~/.exo/models/TheDrainFlorist--Qwen3.6-35B-A3B-VQ-4.6bpw")

STATS = {
    "calls": 0, "host_s": 0.0, "decode_s": 0.0, "gather_s": 0.0,
    "gemm_eval_s": 0.0, "unscramble_s": 0.0,
    "pad_ratios": [], "skews": [], "touched": [], "rows": [],
}


def _instrument(mod_globals, fenced):
    """Replace _prefill with a phase-timed copy of the shipped logic."""
    _decode_chunk = mod_globals["_decode_chunk"]
    _default_decode_chunk = mod_globals["_default_decode_chunk"]

    def sync():
        if fenced:
            mx.synchronize()

    def prefill(xf, idx_sorted_np, codes, codebook, scales, pack_bits=0,
                in_features=None):
        t_call = time.perf_counter()
        chunk = mod_globals["_DECODE_CHUNK"] or _default_decode_chunk()
        mod_globals["_DECODE_CHUNK"] = chunk
        E, OUT, _ = codes.shape
        counts = np.bincount(idx_sorted_np, minlength=E)
        touched = np.nonzero(counts)[0]
        STATS["touched"].append(len(touched))
        STATS["rows"].append(int(counts.sum()))
        STATS["skews"].append(float(counts[touched].max() /
                                    max(1.0, counts[touched].mean())))
        touched = touched[np.argsort(counts[touched], kind="stable")]
        starts = np.zeros(E + 1, np.int64)
        starts[1:] = np.cumsum(counts)
        ys, row_ids = [], []
        pad_num = pad_den = 0
        for c0 in range(0, len(touched), chunk):
            t0 = time.perf_counter()
            eids = touched[c0:c0 + chunk]
            ne = len(eids)
            cap = int(counts[eids].max())
            pad_num += ne * cap
            pad_den += int(counts[eids].sum())
            gmap = np.zeros((ne, cap), np.uint32)
            vmask = np.zeros((ne, cap), bool)
            for i, e in enumerate(eids):
                c = counts[e]
                gmap[i, :c] = np.arange(starts[e], starts[e] + c,
                                        dtype=np.uint32)
                vmask[i, :c] = True
            sync()
            t1 = time.perf_counter()
            w = _decode_chunk(codes, codebook, scales,
                              mx.array(eids.astype(np.uint32)),
                              pack_bits=pack_bits, in_features=in_features)
            if fenced:
                mx.eval(w)
            t2 = time.perf_counter()
            xp = xf[mx.array(gmap.reshape(-1))].reshape(ne, cap, -1)
            if fenced:
                mx.eval(xp)
            t3 = time.perf_counter()
            yp = xp @ mx.swapaxes(w, 1, 2)
            flat_valid = np.nonzero(vmask.reshape(-1))[0].astype(np.uint32)
            ys.append(yp.reshape(ne * cap, OUT)[mx.array(flat_valid)])
            row_ids.append(gmap.reshape(-1)[flat_valid])
            mx.eval(ys[-1])          # shipped code evals per chunk too
            t4 = time.perf_counter()
            del w, xp, yp
            STATS["host_s"] += t1 - t0
            STATS["decode_s"] += t2 - t1
            STATS["gather_s"] += t3 - t2
            STATS["gemm_eval_s"] += t4 - t3
        t5 = time.perf_counter()
        y = mx.concatenate(ys, axis=0)
        rid = np.concatenate(row_ids)
        inv = np.empty(rid.shape[0], np.uint32)
        inv[rid] = np.arange(rid.shape[0], dtype=np.uint32)
        y = y[mx.array(inv)]
        if fenced:
            mx.eval(y)
        STATS["unscramble_s"] += time.perf_counter() - t5
        STATS["pad_ratios"].append(pad_num / max(1, pad_den))
        STATS["calls"] += 1
        return y

    mod_globals["_prefill"] = prefill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=ART)
    ap.add_argument("--tokens", type=int, default=9000)
    ap.add_argument("--fenced", action="store_true",
                    help="phase-fence _prefill (attribution mode)")
    a = ap.parse_args()

    from mlx_lm import load
    t0 = time.perf_counter()
    model, tok = load(a.art, lazy=False)
    mx.eval(model.parameters())
    print(f"loaded in {time.perf_counter()-t0:.1f}s, "
          f"resident {mx.get_active_memory()/2**30:.1f} GiB")

    # find a VQSwitchLinear instance -> its class globals = bundled module
    mod_globals = None
    for _, m in model.named_modules():
        if type(m).__name__ == "VQSwitchLinear":
            mod_globals = type(m).__call__.__globals__
            break
    if mod_globals is None:
        raise SystemExit("no VQSwitchLinear found — not a MoE VQ artifact")
    _instrument(mod_globals, a.fenced)
    print(f"instrumented _prefill (fenced={a.fenced})")

    text = open(os.path.join(a.art, "README.md")).read()
    ids = tok.encode(text)
    while len(ids) < a.tokens:
        ids = ids + ids
    ids = ids[:a.tokens]
    print(f"prompt: {len(ids)} tokens")

    from mlx_lm import generate
    t0 = time.perf_counter()
    generate(model, tok, prompt=tok.decode(ids), max_tokens=1)
    wall = time.perf_counter() - t0
    print(f"\nprefill+1tok wall: {wall:.1f}s  "
          f"({a.tokens/wall:.1f} tok/s)  peak {mx.get_peak_memory()/2**30:.1f} GiB")

    s = STATS
    print(f"\n_prefill calls: {s['calls']}")
    if s["calls"]:
        acc = s["host_s"] + s["decode_s"] + s["gather_s"] + s["gemm_eval_s"] \
            + s["unscramble_s"]
        print(f"router: touched/call median {int(np.median(s['touched']))} "
              f"of E, rows/call median {int(np.median(s['rows']))}, "
              f"skew max/mean median {np.median(s['skews']):.1f}x, "
              f"pad ratio median {np.median(s['pad_ratios']):.3f}")
        for k in ("host_s", "decode_s", "gather_s", "gemm_eval_s",
                  "unscramble_s"):
            print(f"  {k:14s} {s[k]:8.2f}s  {100*s[k]/max(acc,1e-9):5.1f}% "
                  f"of instrumented")
        print(f"  {'sum':14s} {acc:8.2f}s  vs wall {wall:.1f}s "
              f"(gap = attention/embed/other + pipelining)")
    json_path = "logs/probe_moe_prefill_attrib.json" if os.path.isdir("logs") \
        else "/tmp/probe_moe_prefill_attrib.json"
    json.dump({k: (v if not isinstance(v, list) else v) for k, v in s.items()},
              open(json_path, "w"), default=float, indent=1)
    print(f"raw stats -> {json_path}")


if __name__ == "__main__":
    main()
