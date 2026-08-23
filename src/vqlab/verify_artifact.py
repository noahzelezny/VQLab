#!/usr/bin/env python3
"""Independently verify a VQ artifact against its bf16 source.

WHY. The tail30 collapse (E44) shipped relerr-1.0 tensors that only the fit
LOG knew about — and log-based auditing trusts what the fitter said at fit
time, not what is actually in the files. This decodes every VQ tensor FROM
THE ARTIFACT (packed or not) and measures relerr against the bf16 source
directly, so it catches all of: fit collapse, packing faults, shard-write
faults, and stale shards from a resumed fit. It shares NO reconstruction
code with the fitter beyond vq_pack.unpack (which has its own round-trip
test) — a bug in the fitter's math cannot hide itself here.

    ./verify_artifact.py --artifact <dir> --src <bf16 dir> --family qwen3_5_mlx
    ./verify_artifact.py ... --threshold 0.35     # exit 1 on any breach

Reads codes (+ pack_bits from config when uint32), codebook, vq_scales, and
rebuilds  W ~= codebook[codes] * scales  group-wise, mirroring the fitter's
normalize() contract: scales are fp16 max-abs per group of G along `in`.

Reports per-tensor relerr, the worst offenders, and per-projection means.
Healthy references (08-18): d4k2048 ~0.19, d2k2048 ~0.032, worst legitimate
tensor (L00) ~0.215. Anything near 1.0 is a destroyed weight.
"""
import argparse
import json
import math
import pathlib
import sys

import mlx.core as mx
import numpy as np

import vq_pack


sys.path.insert(0, str(pathlib.Path(__file__).parent))
from families import FAMILY  # shared registry (families.py)

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True)
ap.add_argument("--src", required=True, help="bf16 source dir")
ap.add_argument("--family", required=True, choices=sorted(FAMILY))
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--threshold", type=float, default=None,
                help="absolute relerr bar. GEOMETRY-SPECIFIC — healthy d4K128 "
                     "on the 397B lands ~0.46, so 0.35 cries wolf there. "
                     "Prefer --outlier, which needs no per-geometry tuning.")
ap.add_argument("--outlier", type=float, default=None, metavar="MULT",
                help="flag any tensor whose relerr exceeds MULT x the "
                     "artifact's OWN median (try 3.0). Corruption is an "
                     "outlier against its peers — tail30's dead tensor read "
                     "1.0000 beside peers at 0.032 — not a fixed bar. This is "
                     "the gate that survives across geometries.")
ap.add_argument("--limit", type=int, default=None,
                help="verify only the first N tensors (smoke mode)")
args = ap.parse_args()

ART, SRC = pathlib.Path(args.artifact), pathlib.Path(args.src)
FAM = FAMILY[args.family]
G = args.group

cfg = json.load(open(ART / "config.json"))
vq_modules = cfg.get("vq_modules", {})
if not vq_modules:
    # DENSE artifacts (build_dense_vq.py / build_e4b_vq.py) record their
    # modules under "vq_linear", not "vq_modules" — the runtime shim reads
    # vq_linear/vq_embed. Without this the gate exits "not a VQ artifact" on a
    # perfectly good dense artifact, which is the same shape as the 35B family
    # miss on 08-21: the gate could not read the artifact and that was mistaken
    # for the artifact being unreadable.
    vq_modules = cfg.get("vq_linear", {})
if not vq_modules:
    sys.exit("config.json has neither vq_modules nor vq_linear — not a VQ artifact?")

art_map = json.load(open(ART / "model.safetensors.index.json"))["weight_map"]
src_map = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]

_src_cache = {}


def src_tensor(li, proj):
    key_t, sub = FAM["proj"][proj]
    name = FAM["src_key"].format(li=li, key=key_t)
    sh = src_map[name]
    if sh not in _src_cache:
        _src_cache.clear()                      # one src shard resident
        _src_cache[sh] = mx.load(str(SRC / sh))
    T = _src_cache[sh][name]
    if sub is not None:                          # fused gate_up in HF layout
        half = T.shape[1] // 2
        T = T[:, half * sub:half * (sub + 1), :]
    if T.ndim == 2:                              # dense family (e4b): E=1
        T = T[None]
    return T


def layer_of(name):
    return int(name.split("layers.")[1].split(".")[0])


rows = []
_art_cache = {}
mods = sorted(vq_modules, key=lambda m: (art_map[m + ".codes"], m))
if args.limit:
    mods = mods[:args.limit]
for mod in mods:
    meta = vq_modules[mod]
    sh = art_map[mod + ".codes"]
    if sh not in _art_cache:
        _art_cache.clear()
        _art_cache[sh] = mx.load(str(ART / sh))
    data = _art_cache[sh]
    codes = data[mod + ".codes"]
    cb = data[mod + ".codebook"].astype(mx.float32)
    scales = data[mod + ".vq_scales"].astype(mx.float32)
    d = meta["dim"]
    in_d = meta["in"]
    nsub = in_d // d
    if codes.dtype == mx.uint32:                 # packed — unpack first
        codes = mx.array(vq_pack.unpack(np.array(codes), nsub,
                                        meta["pack_bits"]).astype(np.uint32))
    if codes.ndim == 2:
        # DENSE artifacts store codes/scales as 2D [OUT, NSUB]; the expert
        # format this loop speaks is 3D [E, OUT, NSUB]. build_dense_vq.py
        # squeezes them deliberately (a dense module wants 2D, and 2D keeps
        # the venv's expert-shaped VQ hook off them). Add the E=1 axis back
        # here, mirroring what the source path already does for 2D tensors.
        codes = codes[None]
        if scales.ndim == 2:
            scales = scales[None]
    E, out_d = codes.shape[0], codes.shape[1]

    li = layer_of(mod)
    proj = mod.rsplit(".", 1)[1]
    # Materialize the source ON THE CPU STREAM before any GPU math. Five
    # Metal-watchdog kills (kIOGPUCommandBufferCallbackErrorTimeout) at
    # DIFFERENT layers — including a ~100 MB chunk that cannot time out on
    # compute — localized the fault: the lazy shard read of the 751G bf16
    # source stalls on disk INSIDE a GPU command buffer, and the watchdog
    # kills the wait. The CPU stream has no watchdog; once the bytes are in
    # unified memory, the per-chunk GPU cast/diff below is microseconds.
    # The stream is bound at OP-CREATION time, not eval time — wrapping only
    # the eval left the load/slice ops on the GPU stream, and run 6 still
    # died at mx.eval(T) (further along: disk-stall timing is intermittent).
    # Create the load+slice UNDER the cpu stream so the whole read chain is
    # watchdog-free.
    with mx.stream(mx.cpu):
        T = src_tensor(li, proj)
        mx.eval(T)

    # W_hat = codebook[codes] * scale, group-wise along `in`
    num = den = 0.0
    CH = max(1, 8 // max(1, E // 64))            # experts per chunk
    for s in range(0, E, CH):
        c = codes[s:s + CH]
        w = cb[c.reshape(-1)].reshape(c.shape[0], out_d, nsub * d)
        sc = scales[s:s + CH]                     # [e, out, in/G]
        w = (w.reshape(c.shape[0], out_d, in_d // G, G)
             * sc[..., None]).reshape(c.shape[0], out_d, in_d)
        Tc = T[s:s + CH].astype(mx.float32)
        mx.eval(Tc)
        diff = w - Tc
        num += float(mx.sum(diff * diff).item())
        den += float(mx.sum(Tc * Tc).item())
        del Tc
        del w, diff
        mx.clear_cache()
    relerr = math.sqrt(num / max(den, 1e-12))
    rows.append((relerr, li, proj))
    print(f"    L{li:02d} {proj:10s} relerr {relerr:.4f}", flush=True)

rows.sort(reverse=True)
print(f"\nverified {len(rows)} tensors from the ARTIFACT (not the fit log)")
print("worst 5:")
for r, li, proj in rows[:5]:
    print(f"    L{li:02d} {proj:10s} {r:.4f}")
by_proj = {}
for r, li, proj in rows:
    by_proj.setdefault(proj, []).append(r)
for proj, v in sorted(by_proj.items()):
    print(f"mean {proj:10s} {sum(v) / len(v):.4f}  (n={len(v)})")

fail = False
if args.outlier is not None:
    vals = sorted(r for r, _, _ in rows)
    med = vals[len(vals) // 2]
    bar = med * args.outlier
    bad = [(r, li, p) for r, li, p in rows if r > bar]
    print(f"\noutlier gate: median {med:.4f} x{args.outlier} -> bar {bar:.4f}")
    if bad:
        print(f"FAIL: {len(bad)} tensors are outliers against their own peers")
        for r, li, p in bad[:8]:
            print(f"    L{li:02d} {p:10s} {r:.4f}  ({r / med:.1f}x median)")
        fail = True
    else:
        print(f"PASS: no tensor exceeds {args.outlier}x the artifact median")

if args.threshold is not None:
    bad = [(r, li, p) for r, li, p in rows if r > args.threshold]
    if bad:
        print(f"\nFAIL: {len(bad)} tensors above {args.threshold}")
        fail = True
    else:
        print(f"\nPASS: all tensors <= {args.threshold}")

if fail:
    sys.exit(1)
