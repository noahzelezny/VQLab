"""vqlab geo-build — rebuild an artifact with a NEW PER-LAYER GEOMETRY MAP.

The diff-style builder behind every mixed-geometry (v2) candidate: modules
named in the GEOMAP are refit from the bf16 teacher at their new (d, K);
every other module keeps its EXACT shipped bytes (symlinked, not copied).
That is what makes whole-artifact allocation probes cheap enough to replace
proxy scores entirely — the rule that produced the only trustworthy results
of the 2026-09 geometry campaign (docs/GEOMETRY-CAMPAIGN.md).

    vqlab geo-build --artifact <shipped dir> --teacher <bf16 dir> \
        --family qwen4_exp --geomap map.json --out <dir> [--reuse DIR ...]

GEOMAP is {module_name: {"dim": d, "k": K}}. Module names are runtime names
(e.g. model.layers.30.mlp.switch_mlp.gate_proj); the teacher key is derived
per family from families.FAMILY.

THREE THINGS THIS ENFORCES, each of which cost a real run to learn:

1. **Exact packing.** The pack format stores ceil(nsub/32)*bits words per
   row, so a geometry whose nsub is not a multiple of 32 silently pads. The
   shipped Flash-2.1 carried 46 such modules and 1.69 GB of literal zero
   words (F96/F97). A ragged (d, K) is REFUSED here with the arithmetic in
   the message, not discovered later by an audit.
2. **pack_bits in config.** The loader derives the codes shape from
   `pack_bits`; omitting it on a changed module makes the artifact fail to
   load with a shape error that names the wrong cause.
3. **A fit is named for its MODULE AND ITS GEOMETRY** (`<module>.d4-K256.
   safetensors`). Naming it after the module alone let two rungs' different
   geometries share one filename with different bytes, which makes an archive
   un-deduplicable and a cross-rung reuse silently wrong. Legacy unstamped
   parts are still accepted on the shape check below.
4. **Fit reuse is verified by CODEBOOK SHAPE *and* MODULE SHAPE, never by
   filename.** A part named for the right module at the wrong geometry loads
   silently and poisons the build. The codebook shape alone only pins (K, d),
   which a fit from ANOTHER MODEL at the same geometry also satisfies -- the
   2026-09-15 Flash pool holds 240 such d4-K2048 parts. The shipped artifact
   carries every module at its old geometry, and (experts, out) survive a
   change of K or d, so its codes give a free exact identity check.

Fit recipe: k-means++ init, Lloyd, then scale<->codebook alternation with
per-group least-squares scales (F80/F81 — the max-abs scale heuristic was
the one never-optimized component of the original recipe).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import FAMILY
import provenance

GSZ = 64
CHUNK = 1 << 18
NFITG = 1 << 17
ITERS_LLOYD = 20
ITERS_ALT = 12
EXT = ".safetensors"


def part_name(module, d, k, tag=None):
    """Fit filename: MODULE, GEOMETRY, and optionally a content tag.

    Three levels of identity, each of which was learned the hard way on
    2026-09-15:

    * MODULE alone (the original scheme) collides across rungs -- two
      geometries produced `model.layers.28...gate_proj` with identical names
      and sizes and different bytes. The original scheme survived that only
      because the enclosing *_parts dir named the fit run; flatten the two
      levels into one and the run identity is lost.
    * + GEOMETRY still collides, because fitting is STOCHASTIC (k-means++
      seeding): 138 (module, d, K) slots in the 2026-09 Flash pool hold two
      DIFFERENT valid codebooks from two different runs.
    * + a short CONTENT TAG is unique. Pass `tag` when pooling fits from
      several runs into one directory (an archive); omit it inside a single
      build's parts dir, where the dir already names the run.
    """
    stem = f"{module}.d{int(d)}-K{int(k)}"
    return f"{stem}.{tag}{EXT}" if tag else f"{stem}{EXT}"


def parse_part(fname):
    """(module, d, K) from a part filename; (module, None, None) if legacy.

    Tolerates an optional trailing content tag written by `part_name(tag=...)`.
    """
    base = fname[:-len(EXT)] if fname.endswith(EXT) else fname
    for _ in range(2):                      # strip an optional content tag
        head, _, tail = base.rpartition(".")
        if head and tail.startswith("d") and "-K" in tail:
            dpart, _, kpart = tail.partition("-K")
            if dpart[1:].isdigit() and kpart.isdigit():
                return head, int(dpart[1:]), int(kpart)
        if not head:
            break
        base = head
    return (fname[:-len(EXT)] if fname.endswith(EXT) else fname), None, None


def _log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def words_per_row(nsub: int, bits: int) -> int:
    return math.ceil(nsub / 32) * bits


def check_exact(name: str, IN: int, D: int, K: int) -> int:
    """Refuse ragged geometries up front (F97). Returns nsub."""
    nsub = IN // D
    if nsub % 32:
        ideal = math.ceil(nsub * math.ceil(math.log2(K)) / 32)
        got = words_per_row(nsub, math.ceil(math.log2(K)))
        raise SystemExit(
            f"REFUSING {name}: d{D} on IN={IN} gives nsub={nsub}, which is not "
            f"a multiple of 32. The packer charges whole 32-subvector blocks, "
            f"so each row would store {got} words where {ideal} carry the "
            f"information — {100*(got-ideal)/ideal:.0f}% dead bytes. "
            f"Pick a d that divides IN/32 (d4 -> nsub={IN//4}, d2 -> {IN//2}).")
    return nsub


def pack(codes: np.ndarray, bits: int) -> np.ndarray:
    E_, OUT_, nsub = codes.shape
    res = np.zeros((E_, OUT_, words_per_row(nsub, bits)), dtype=np.uint64)
    for j in range(nsub):
        off = (j & 31) * bits
        w = (j >> 5) * bits + (off >> 5)
        sh = off & 31
        v = codes[:, :, j].astype(np.uint64)
        res[:, :, w] |= (v << sh) & 0xFFFFFFFF
        if sh + bits > 32:
            res[:, :, w + 1] |= v >> (32 - sh)
    return res.astype(np.uint32)


def teacher_weight(teacher: str, index: dict, family: str, name: str) -> mx.array:
    """[E, OUT, IN] fp32 from the bf16 teacher, per the family's key map."""
    spec = FAMILY[family]
    li = name.split("layers.")[1].split(".")[0]
    proj = name.rsplit(".", 1)[1]
    src_name, half = spec["proj"][proj]
    key = spec["src_key"].format(li=li, key=src_name, e=0)
    if key not in index:                      # some families prefix differently
        alt = key.replace("model.language_model.", "model.model.")
        key = alt if alt in index else key
    # A STREAM BINDS AT OP-CREATION, NOT AT EVAL (quantlab IV.1). The load and
    # the gate_up half-slice have to be CREATED inside the CPU-stream block or
    # they stay bound to the GPU stream, and the eval below -- however it is
    # wrapped -- still pays the read inside a command buffer. That is a
    # watchdog kill, and it only shows up once the teacher is slow enough to
    # exceed the timeout: these same reads passed for months with the bf16
    # teacher on the SSD and died the first time it was read off the archive
    # HDD (GPU Timeout in teacher_weight, 2026-09-16).
    with mx.stream(mx.cpu):
        W = mx.load(os.path.join(teacher, index[key]))[key]
        if half is not None:
            h = W.shape[1] // 2
            W = W[:, :h, :] if half == 0 else W[:, h:, :]
        W = W.astype(mx.float32)
        mx.eval(W)
    return W


def _dists(X, C):
    return (mx.sum(X * X, 1, keepdims=True) - 2 * (X @ C.T)
            + mx.sum(C * C, 1)[None, :])


def _chunk(K):
    # The distance matrix is chunk x K fp32; at K16384 the default CHUNK is a
    # 17 GB intermediate per step (K2048: 2 GB). Cap it at ~4 GB so large
    # codebooks fit under the memory limit without changing K<=4096 fits.
    return min(CHUNK, max(1 << 12, (1 << 30) // K))


def _assign(Xs, C):
    out, ch = [], _chunk(C.shape[0])
    for i in range(0, Xs.shape[0], ch):
        a = mx.argmin(_dists(mx.array(Xs[i:i + ch]), C), axis=1)
        mx.eval(a)
        out.append(np.array(a))
    return np.concatenate(out)


def _lloyd(Xs, C, w, iters, K, rng):
    D = Xs.shape[1]
    for _ in range(iters):
        num = np.zeros((K, D), dtype=np.float64)
        den = np.zeros(K, dtype=np.float64)
        ch = _chunk(K)
        for i in range(0, Xs.shape[0], ch):
            xb, wb = Xs[i:i + ch], w[i:i + ch]
            a = mx.argmin(_dists(mx.array(xb), C), axis=1)
            mx.eval(a)
            a = np.array(a)
            np.add.at(num, a, xb * wb[:, None])
            np.add.at(den, a, wb)
        dead = den == 0
        if dead.any():
            num[dead] = Xs[rng.choice(Xs.shape[0], int(dead.sum()))]
            den[dead] = 1
        C = mx.array((num / den[:, None]).astype(np.float32))
    return C


def _kmeanspp(Xf, k, rng):
    pool = Xf[rng.choice(Xf.shape[0], min(1 << 18, Xf.shape[0]), replace=False)]
    P = mx.array(pool)
    out = [pool[rng.integers(pool.shape[0])]]
    dmin = None
    for _ in range(k - 1):
        d = mx.sum((P - mx.array(out[-1])[None, :]) ** 2, axis=1)
        dmin = d if dmin is None else mx.minimum(dmin, d)
        mx.eval(dmin)
        p = np.array(dmin, dtype=np.float64)
        p /= p.sum()
        out.append(pool[rng.choice(pool.shape[0], p=p)])
    return np.stack(out)


def fit_module(W, D, K, rng, tail_pow=0.0, plain=False):
    """Returns (codebook fp16, packed codes uint32, scales fp16).

    tail_pow > 0 is E112's magnitude-weighted k-means: each subvector's
    weight in every Lloyd update is its norm in ORIGINAL weight units raised
    to tail_pow, normalized to mean 1 (vq_397b_codes.py --tail-weight-pow).
    That norm is independent of the per-group scale, so the same weights
    apply in the alternation rounds, where they multiply the s**2 term --
    applying them only to the first Lloyd pass would let alternation undo
    them. tail_pow = 0 takes the original unweighted path bit-for-bit.

    plain=True is E112's fitter (vq_397b_codes.py): Lloyd on max-abs-scaled
    subvectors, NO scale alternation, max-abs scales at encode. Needed to
    replicate E112 faithfully: with alternation on, least-squares scales
    drift against a tail-weighted codebook and the tail error gets WORSE,
    not better (measured 2026-09-25: top-0.1% relerr 0.224 -> 0.292).
    """
    E_, OUT_, IN_ = W.shape
    NGRP, nsub, bits = IN_ // GSZ, IN_ // D, math.ceil(math.log2(K))
    Wg = np.array(W.reshape(-1, GSZ))
    fidx = rng.choice(Wg.shape[0], min(NFITG, Wg.shape[0]), replace=False)
    Gf = Wg[fidx].astype(np.float32)
    s = np.abs(Gf).max(axis=1) + 1e-8
    Xs = (Gf / s[:, None]).reshape(-1, D)
    if tail_pow:
        wt = np.linalg.norm(Gf.reshape(-1, D), axis=1).astype(np.float64) ** tail_pow
        wt = (wt / max(wt.mean(), 1e-20)).astype(np.float32)
    else:
        wt = np.ones(Xs.shape[0], np.float32)
    C = _lloyd(Xs, mx.array(_kmeanspp(Xs, K, rng)), wt, ITERS_LLOYD, K, rng)
    for _ in range(0 if plain else ITERS_ALT):       # F80/F81 alternation
        Xs = (Gf / s[:, None]).reshape(-1, D)
        rec = np.array(C)[_assign(Xs, C)].reshape(-1, GSZ)
        s = np.clip((Gf * rec).sum(1) / ((rec * rec).sum(1) + 1e-12),
                    1e-8, None).astype(np.float32)
        C = _lloyd((Gf / s[:, None]).reshape(-1, D), C,
                   (np.repeat(s ** 2, GSZ // D) * wt).astype(np.float32), 2, K, rng)
    C_np = np.array(C)
    codes = np.empty((Wg.shape[0], GSZ // D), dtype=np.uint16)
    scales = np.empty(Wg.shape[0], dtype=np.float32)
    B = _chunk(K) // (GSZ // D)
    for i in range(0, Wg.shape[0], B):
        wb = Wg[i:i + B].astype(np.float32)
        sb = np.abs(wb).max(axis=1) + 1e-8
        for it in range(1 if plain else 2):
            a = mx.argmin(_dists(mx.array((wb / sb[:, None]).reshape(-1, D)), C), axis=1)
            mx.eval(a)
            a = np.array(a)
            rec = C_np[a].reshape(wb.shape[0], GSZ)
            if plain:
                break                                # keep the max-abs scale
            sb = np.clip((wb * rec).sum(1) / ((rec * rec).sum(1) + 1e-12),
                         1e-8, None).astype(np.float32)
        codes[i:i + B] = a.reshape(wb.shape[0], GSZ // D).astype(np.uint16)
        scales[i:i + B] = sb
    return (mx.array(C_np.astype(np.float16)),
            mx.array(pack(codes.reshape(E_, OUT_, NGRP * (GSZ // D)), bits)),
            mx.array(scales.reshape(E_, OUT_, NGRP).astype(np.float16)))


def _method(a):
    """The fitter settings a refit here actually used -- including the ones
    that are module constants, not flags, so no record depends on reading
    this file at the right commit to know them."""
    return {"init": "kmeans++ (pool 2^18)", "lloyd_iters": ITERS_LLOYD,
            "sample_groups": NFITG, "group": GSZ,
            "alternation": not a.plain_lloyd,
            "alternation_rounds": 0 if a.plain_lloyd else ITERS_ALT,
            "scales": "max-abs" if a.plain_lloyd else "least-squares (F80/F81)",
            "tail_weight_pow": a.tail_weight_pow,
            "tail_weight_from": a.tail_weight_from,
            "seed": a.seed if a.seed >= 0 else "unseeded",
            "rng": "numpy default_rng, one stream across modules"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="shipped artifact to diff from")
    ap.add_argument("--teacher", required=True, help="bf16 teacher dir")
    ap.add_argument("--family", required=True, choices=sorted(FAMILY))
    ap.add_argument("--geomap", required=True, help='{module: {"dim":d,"k":K}}')
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default=None, help="checkpoint dir (default <out>_parts)")
    ap.add_argument("--reuse", action="append", default=[],
                    help="parts dir to reuse fits from; repeatable")
    ap.add_argument("--memory-limit-gb", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234,
                    help="RNG seed. Default 1234, the lab-wide default, so an "
                         "unflagged build is reproducible. -1 = unseeded (an "
                         "explicit choice to draw fresh). Builds before "
                         "2026-09-25 defaulted to 13: pass --seed 13 to "
                         "reproduce one; its build record or log names it.")
    ap.add_argument("--tail-weight-pow", type=float, default=0.0,
                    help="E112 magnitude-weighted k-means (see fit_module). "
                         "0 = the unweighted objective, bit-identical.")
    ap.add_argument("--tail-weight-from", type=int, default=0,
                    help="apply --tail-weight-pow only to layers >= this "
                         "index (shallow layers are heavy-tailed, E110)")
    ap.add_argument("--plain-lloyd", action="store_true",
                    help="E112's fitter: no scale alternation, max-abs scales. "
                         "Use for BOTH arms of a tail-weighting pair.")
    a = ap.parse_args()

    mx.set_wired_limit(0)
    mx.set_memory_limit(a.memory_limit_gb * 1024 ** 3)
    rng = np.random.default_rng(None if a.seed < 0 else a.seed)
    geo = json.load(open(a.geomap))
    parts = a.parts or (a.out.rstrip("/") + "_parts")
    os.makedirs(parts, exist_ok=True)

    t_index = json.load(open(os.path.join(
        a.teacher, "model.safetensors.index.json")))["weight_map"]
    a_index = {}
    for f in sorted(glob.glob(os.path.join(a.artifact, "*" + EXT))):
        rp = os.path.realpath(f)
        for k in mx.load(rp):
            a_index[k] = rp

    # ---- reuse: verified by CODEBOOK SHAPE *and* MODULE SHAPE ----
    # The codebook shape only pins (K, d). It does NOT pin the module the fit
    # came from, so a fit for a DIFFERENT model with the same geometry passes
    # it: the 2026-09-15 Flash pool holds 240 d4-K2048 parts whose codes are
    # [256, 512, ...] against Flash's [512, 640, ...] -- another family's
    # experts entirely. Only the `language_model.` name prefix kept those out,
    # which is luck, not verification. The shipped artifact already carries
    # this module at its OLD geometry, and (experts, out) do not change with
    # K or d -- so the shipped codes give a free, exact identity check.
    # Per-part ORIGIN ledger, kept in the parts dir so it survives a resume:
    # a checkpoint that exists says nothing about whether it was fit here, by
    # which recipe, or harvested from another rung (docs/PROVENANCE.md).
    origins_path = os.path.join(parts, "origins.json")
    origins = (json.load(open(origins_path))
               if os.path.exists(origins_path) else {})

    def _note(fname, rec):
        origins[fname] = rec
        json.dump(origins, open(origins_path, "w"), indent=1)

    reused = 0
    rejected = []
    for src in a.reuse:
        for f in glob.glob(os.path.join(src, "*" + EXT)):
            n, fd, fk = parse_part(os.path.basename(f))
            if n not in geo:
                continue
            want = part_name(n, geo[n]["dim"], geo[n]["k"])
            if os.path.exists(os.path.join(parts, want)):
                continue
            # A geometry-stamped name that disagrees is the wrong fit: skip it
            # without paying the load. Legacy unstamped parts fall through to
            # the shape check, which is why that check stays.
            if fd is not None and (fd, fk) != (geo[n]["dim"], geo[n]["k"]):
                continue
            part = mx.load(f)
            cb = part.get(n + ".codebook")
            if cb is None or tuple(cb.shape) != (geo[n]["k"], geo[n]["dim"]):
                continue
            codes = part.get(n + ".codes")
            shipped = a_index.get(n + ".codes")
            if codes is None or shipped is None:
                rejected.append((os.path.basename(f), "no codes to compare"))
                continue
            want_eo = tuple(mx.load(shipped)[n + ".codes"].shape[:2])
            got_eo = tuple(codes.shape[:2])
            if got_eo != want_eo:
                rejected.append((os.path.basename(f),
                                 f"codes {got_eo} != shipped {want_eo}"))
                continue
            shutil.copy(f, os.path.join(parts, want))
            # The reused part's OWN recipe is whatever its source dir recorded;
            # carry that forward rather than letting it pass as a fit here.
            src_orig = {}
            op = os.path.join(src, "origins.json")
            if os.path.exists(op):
                src_orig = json.load(open(op)).get(os.path.basename(f), {})
            _note(want, {"origin": "reuse", "from": os.path.realpath(f),
                         "sha256": provenance._sha(f), "source_origin": src_orig})
            reused += 1
    for fn, why in rejected[:10]:
        _log(f"  REJECTED reuse {fn}: {why}")
    if rejected:
        _log(f"rejected {len(rejected)} reuse candidates on module shape")
    if a.reuse:
        _log(f"reused {reused}/{len(geo)} fits (codebook-shape verified)")

    # ---- fit what is missing ----
    done = 0
    for n in sorted(geo):
        part = os.path.join(parts, part_name(n, geo[n]["dim"], geo[n]["k"]))
        if os.path.exists(part):
            # RESUME MUST VERIFY, NOT ASSUME. A part that exists is not a part
            # that is finished: a fit killed mid-save leaves a truncated file
            # with a valid name, resume skips it, and the failure surfaces
            # much later in assemble as "invalid data offsets ... exceeding
            # the size of the file" -- a corrupt-download message for a file
            # nobody downloaded. Cheap to check, and checkpoints exist
            # precisely because these runs get interrupted.
            try:
                mx.eval(list(mx.load(part).values()))
                done += 1
                continue
            except Exception as e:
                _log(f"  REFITTING {os.path.basename(part)}: unreadable "
                     f"checkpoint ({str(e)[:60]})")
                os.remove(part)
        D, K = int(geo[n]["dim"]), int(geo[n]["k"])
        t0 = time.time()
        W = teacher_weight(a.teacher, t_index, a.family, n)
        check_exact(n, W.shape[2], D, K)
        li = int(n.split("layers.")[1].split(".")[0])
        P = a.tail_weight_pow if li >= a.tail_weight_from else 0.0
        cb, codes, scales = fit_module(W, D, K, rng, tail_pow=P, plain=a.plain_lloyd)
        del W
        mx.save_safetensors(part, {n + ".codebook": cb, n + ".codes": codes,
                                   n + ".vq_scales": scales})
        _note(os.path.basename(part), {
            "origin": "fit", "tool": "geo-build",
            "commit": provenance.code_state()["commit"],
            "fitter": _method(a), "tail_pow": P,
            "sha256": provenance._sha(part)})
        mx.clear_cache()
        done += 1
        _log(f"  {done}/{len(geo)} {n} d{D}-K{K}"
             f"{f' tailw{P:g}' if P else ''} [{time.time()-t0:.0f}s]")

    # ---- assemble: shipped bytes + refit parts + config ----
    new = {}
    for f in glob.glob(os.path.join(parts, "*" + EXT)):
        new.update(mx.load(f))
    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(os.path.join(a.artifact, "*")):
        b = os.path.basename(f)
        # never inherit the PARENT's build record: it describes other bytes
        if b.endswith(EXT) or b == "__pycache__" or b == provenance.RECORD:
            continue
        shutil.copy(os.path.realpath(f), os.path.join(a.out, b))
    cfg_path = os.path.join(a.out, "config.json")
    cfg = json.load(open(cfg_path))
    vm = cfg.get("vq_modules") or cfg.get("vq_linear") or {}
    for n, g in geo.items():
        # A module ABSENT from vq_modules is the RESTORE case: it shipped
        # unquantized (affine) and is being given a VQ fit for the first
        # time. 397B rungs 2.4/2.6/3.1 shipped layers 57-59 that way -- 171
        # of 180 modules quantized -- so `vm[n]` raised KeyError and the
        # build died after doing all the work. Derive the entry from the
        # tensors actually being written; every field is a shape, so there
        # is nothing to guess.
        if n not in vm:
            c_ = new[n + ".codes"]
            sc_ = new[n + ".vq_scales"]
            E_, OUT_ = int(c_.shape[0]), int(c_.shape[1])
            if c_.dtype == mx.uint32:
                bits_ = int(math.ceil(math.log2(int(g["k"]))))
                nsub_ = int(c_.shape[2]) // bits_ * 32
            else:
                nsub_ = int(c_.shape[2])
            IN_ = nsub_ * int(g["dim"])
            vm[n] = {"experts": E_, "out": OUT_, "in": IN_,
                     "group": IN_ // int(sc_.shape[2])}
            _log(f"  RESTORED {n}: was not in vq_modules "
                 f"(experts={E_} out={OUT_} in={IN_} "
                 f"group={vm[n]['group']})")
        ent = vm[n]
        ent["dim"], ent["k"] = int(g["dim"]), int(g["k"])
        # pack_bits must describe the BYTES, not the geometry: the loader
        # derives the codes shape from it. Fits made here are packed uint32,
        # but a part harvested from a shipped rung may carry that rung's raw
        # uint8 codes (every shipped d2-K256 module does) -- stamping
        # pack_bits on those makes the runtime expect (.., IN/d*bits/32)
        # words and refuse the (.., IN/d) bytes it gets.
        codes = new[n + ".codes"]
        if codes.dtype == mx.uint32:
            ent["pack_bits"] = int(math.ceil(math.log2(int(g["k"]))))
        else:
            ent.pop("pack_bits", None)
    # A RESTORED module must also leave the affine `quantization` map. The
    # loader walks that map and calls mlx's quantizer on every entry, so a
    # module that is now VQ but still listed affine dies at load with
    # "Unable to quantize model of type VQSwitchLinear" -- measured, not
    # theorised: the first restored 3.1 build assembled cleanly, passed its
    # shape audit, and could not be loaded at all.
    qmap = cfg.get("quantization")
    if isinstance(qmap, dict):
        gone = [n for n in geo if n in qmap]
        for n in gone:
            qmap.pop(n)
        if gone:
            _log(f"  removed {len(gone)} restored modules from the affine "
                 f"quantization map")
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    # ---- RESTORE bookkeeping -------------------------------------------
    # A module that shipped UNQUANTIZED has no `.codes` to swap, so the
    # plain swap loop below would write nothing for it (measured: 9 modules
    # "RESTORED" in config and "ASSEMBLE-DONE: 0 tensors"). Restoring one
    # means three edits the swap path never needs: ADD its VQ tensors, DROP
    # the affine tensors they supersede, and REWRITE the index, because the
    # key set changes. The VQ tensors go into the shard that held the
    # affine ones so the module's bytes stay co-located.
    a_wm = json.load(open(os.path.join(
        a.artifact, "model.safetensors.index.json")))["weight_map"]
    restored = [n for n in geo if (n + ".codes") not in a_wm]
    drop, add_to = set(), {}
    for n in restored:
        host = None
        for suf in (".weight", ".scales", ".biases"):
            if n + suf in a_wm:
                drop.add(n + suf)
                host = host or a_wm[n + suf]
        if host is None:
            raise SystemExit(
                f"FAIL: {n} has neither codes nor affine tensors; there is "
                "nothing to restore and nowhere to put it")
        for suf in (".codes", ".codebook", ".vq_scales"):
            add_to.setdefault(host, []).append(n + suf)
    if restored:
        _log(f"restoring {len(restored)} modules: +{sum(len(v) for v in add_to.values())} "
             f"VQ tensors, -{len(drop)} affine tensors, index rewritten")

    swapped = 0
    wm_out = {k: v for k, v in a_wm.items() if k not in drop}
    for f in sorted(glob.glob(os.path.join(a.artifact, "*" + EXT))):
        rp, b = os.path.realpath(f), os.path.basename(f)
        tens = mx.load(rp)
        hit = [k for k in tens if k in new]
        adds = add_to.get(b, [])
        dels = [k for k in tens if k in drop]
        dst = os.path.join(a.out, b)
        if os.path.lexists(dst):
            os.remove(dst)
        if not hit and not adds and not dels:
            os.symlink(rp, dst)          # unchanged shard: shipped bytes, no copy
            continue
        out = {}
        for k, v in tens.items():
            if k in drop:
                continue
            out[k] = new[k] if k in new else v
            swapped += k in new
        for k in adds:
            out[k] = new[k]
            wm_out[k] = b
        mx.save_safetensors(dst, out)
        del tens, out
        mx.clear_cache()
    if restored:
        ip = os.path.join(a.out, "model.safetensors.index.json")
        idx = json.load(open(ip))
        idx["weight_map"] = wm_out
        idx.pop("metadata", None) if False else None
        json.dump(idx, open(ip, "w"), indent=1)
    modules = {}
    for n in sorted(geo):
        fn = part_name(n, geo[n]["dim"], geo[n]["k"])
        modules[n] = {"dim": int(geo[n]["dim"]), "k": int(geo[n]["k"]),
                      "restored": n in restored,
                      **origins.get(fn, {"origin": "resume",
                                         "note": "checkpoint predates origin ledger"})}
    provenance.write_build_record(
        a.out, tool="geo-build", script=__file__, ap=ap, args=a,
        method=_method(a),
        inputs=[("base", a.artifact), ("teacher", a.teacher),
                *[("reuse", r) for r in a.reuse]],
        modules=modules,
        full_hash={os.path.basename(p) for p in glob.glob(os.path.join(a.out, "*"))
                   if not os.path.islink(p)})
    _log(f"ASSEMBLE-DONE: {swapped} swapped, "
         f"{sum(len(v) for v in add_to.values())} added, {len(drop)} dropped "
         f"-> {a.out} (config pack_bits updated)")


if __name__ == "__main__":
    main()
