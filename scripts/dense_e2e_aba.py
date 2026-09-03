#!/usr/bin/env python
"""END-TO-END for the dense-kernel arc: ONE load of the 27B 3.9bpw, both
questions answered from it (2026-09-03).

  1. DECODE A-B-A: devx off vs on, arms flipped IN-PROCESS (arc 5 protocol:
     throwaway + old-arm warm, then A,B,A,B,A,B), greedy, generated text
     compared byte-for-byte.
  2. PREFILL PEAK vs RESIDENT: a real 2048-token prompt, with the forced
     per-linear eval off (pre-arc) and on (shipped), reporting peak, active
     and prefill tok/s.

Runs against a SHADOW BUNDLE: a scratch dir of symlinks to the artifact's
safetensors + a model.py rebuilt from THIS worktree's vq_switch.py and
vq_dense.py. The on-disk artifact is never written to (a release is in
flight) and its own model.py snapshot -- which predates this arc -- is not
what gets loaded.

Run:  PYTHONPATH=src python scripts/dense_e2e_aba.py
"""
import argparse
import json
import os
import pathlib
import shutil
import statistics
import sys
import time

import mlx.core as mx

GiB = float(1 << 30)
ART_DEFAULT = os.path.expanduser(
    "~/.exo/models/TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw")
PROMPT = ("Write a detailed technical explanation of how vector "
          "quantization compresses neural network weights, covering "
          "codebooks, residuals and rate-distortion tradeoffs. ")


# Files this script WRITES in the shadow dir. They must never be symlinked
# to the artifact, because open(path, "w") on a symlink writes THROUGH it to
# the target -- which is how an earlier version of this function silently
# reformatted the artifact's own config.json (content unchanged, indentation
# lost) while claiming to be read-only. Named explicitly, and asserted below
# to be real files rather than links before either is opened for writing.
_WRITES = ("model.py", "config.json")


def shadow(art, dst, src_dir):
    """Symlink the artifact READ-ONLY, rebuild the files we generate."""
    dst = pathlib.Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    art = pathlib.Path(art)
    for p in art.iterdir():
        if p.name in _WRITES or p.is_dir():
            continue
        if ".pre-" in p.name or ".pre_" in p.name:
            continue
        (dst / p.name).symlink_to(p)
    for name in _WRITES:
        t = dst / name
        assert not t.is_symlink(), (
            f"{name} is a symlink into the artifact; writing it would "
            f"modify the artifact")
    src_dir = pathlib.Path(src_dir)
    sys.path.insert(0, str(src_dir))
    SHIM = __import__("dense_shim").SHIM
    model_py = ((src_dir / "vq_switch.py").read_text()
                + (src_dir / "vq_dense.py").read_text() + SHIM)
    compile(model_py, "model.py", "exec")
    (dst / "model.py").write_text(model_py)
    cfg = json.load(open(art / "config.json"))
    cfg["model_file"] = "model.py"
    json.dump(cfg, open(dst / "config.json", "w"))
    return str(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=ART_DEFAULT)
    ap.add_argument("--shadow", default="/tmp/vqlab-dense-arc-shadow")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--prefill", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    if not os.path.isdir(a.art):
        sys.exit(f"artifact not readable: {a.art}")

    here = pathlib.Path(__file__).resolve().parents[1] / "src" / "vqlab"
    path = shadow(a.art, a.shadow, here)
    print(f"shadow bundle: {path}\n")

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    t0 = time.perf_counter()
    model, tok = load(path)
    mx.eval(model.parameters())
    load_s = time.perf_counter() - t0
    resident = mx.get_active_memory() / GiB
    print(f"loaded in {load_s:.1f}s   resident {resident:.3f} GiB")

    # The bundle is its own module, loaded from model_file by importlib and
    # NOT registered in sys.modules -- so reach its globals through the
    # model class itself. That dict IS the module namespace, so writing to
    # it flips the arm the loaded kernels actually read.
    M = type(model).__init__.__globals__
    assert "_DENSE_DEVX" in M, "shadow bundle predates this arc"

    class _Arms:
        def __getattr__(self, k):
            return M[k]

        def __setattr__(self, k, v):
            M[k] = v

    arms = _Arms()
    print(f"bundle arms: devx={arms._DENSE_DEVX} ss={arms._DENSE_SS} "
          f"eval={arms._DENSE_DECODE_EVAL} "
          f"tile_mb={arms._DENSE_DECODE_TILE_MB} "
          f"spec={arms._SPEC_KERNELS}\n")

    # ---------- 2. PREFILL PEAK (done first: it needs a clean peak) --------
    print("### PREFILL PEAK vs RESIDENT, prompt = "
          f"{a.prefill} tokens")
    ids = mx.array(
        (tok.encode(PROMPT * 200)[:a.prefill],), dtype=mx.uint32)
    print(f"{'eval':>6s} {'peak':>9s} {'peak-resident':>14s} "
          f"{'prefill_s':>10s} {'tok/s':>9s}")
    pf = {}
    for ev in (False, True):
        arms._DENSE_DECODE_EVAL = ev
        model(ids[:, :16])                       # warm
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        out = model(ids)
        mx.eval(out)
        dt = time.perf_counter() - t0
        peak = mx.get_peak_memory() / GiB
        pf[ev] = (peak, dt, out)
        print(f"{str(ev):>6s} {peak:8.3f}G {peak - resident:13.3f}G "
              f"{dt:10.3f} {a.prefill/dt:9.1f}")
        del out
        mx.clear_cache()
    same = bool(mx.array_equal(pf[False][2].view(mx.uint16),
                               pf[True][2].view(mx.uint16)))
    print(f"prefill logits bit-identical across the eval arms: {same}")
    if not same:
        sys.exit("PREFILL BITS MOVED -- the forced eval must be free")
    arms._DENSE_DECODE_EVAL = True
    del pf
    mx.clear_cache()

    # ---------- 1. DECODE A-B-A -------------------------------------------
    print(f"\n### DECODE A-B-A, greedy, {a.tokens} tokens")
    sampler = make_sampler(temp=0.0)

    def gen():
        mx.synchronize()
        t0 = time.perf_counter()
        txt = generate(model, tok, PROMPT, max_tokens=a.tokens,
                       sampler=sampler, verbose=False)
        return a.tokens / (time.perf_counter() - t0), txt

    arms._DENSE_DEVX = False
    gen()                                        # throwaway + old-arm warm
    gen()
    res = {False: [], True: []}
    txts = {}
    for _ in range(a.rounds):
        for arm in (False, True):
            arms._DENSE_DEVX = arm
            r, txt = gen()
            res[arm].append(r)
            txts.setdefault(arm, txt)
    for arm in (False, True):
        v = res[arm]
        print(f"  devx={str(arm):5s}  " + " / ".join(f"{x:.3f}" for x in v)
              + f"   median {statistics.median(v):.3f} tok/s")
    off, on = statistics.median(res[False]), statistics.median(res[True])
    print(f"  speedup {on/off:.3f}x")
    print(f"  generated texts identical byte-for-byte: "
          f"{txts[False] == txts[True]}")
    if txts[False] != txts[True]:
        sys.exit("TEXTS DIVERGED -- devx is supposed to be bit-identical")
    print("\nOK")


if __name__ == "__main__":
    main()
