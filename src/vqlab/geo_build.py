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


def _assign(Xs, C):
    out = []
    for i in range(0, Xs.shape[0], CHUNK):
        a = mx.argmin(_dists(mx.array(Xs[i:i + CHUNK]), C), axis=1)
        mx.eval(a)
        out.append(np.array(a))
    return np.concatenate(out)


def _lloyd(Xs, C, w, iters, K, rng):
    D = Xs.shape[1]
    for _ in range(iters):
        num = np.zeros((K, D), dtype=np.float64)
        den = np.zeros(K, dtype=np.float64)
        for i in range(0, Xs.shape[0], CHUNK):
            xb, wb = Xs[i:i + CHUNK], w[i:i + CHUNK]
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


def fit_module(W, D, K, rng):
    """Returns (codebook fp16, packed codes uint32, scales fp16)."""
    E_, OUT_, IN_ = W.shape
    NGRP, nsub, bits = IN_ // GSZ, IN_ // D, math.ceil(math.log2(K))
    Wg = np.array(W.reshape(-1, GSZ))
    fidx = rng.choice(Wg.shape[0], min(NFITG, Wg.shape[0]), replace=False)
    Gf = Wg[fidx].astype(np.float32)
    s = np.abs(Gf).max(axis=1) + 1e-8
    Xs = (Gf / s[:, None]).reshape(-1, D)
    C = _lloyd(Xs, mx.array(_kmeanspp(Xs, K, rng)),
               np.ones(Xs.shape[0], np.float32), ITERS_LLOYD, K, rng)
    for _ in range(ITERS_ALT):                       # F80/F81 alternation
        Xs = (Gf / s[:, None]).reshape(-1, D)
        rec = np.array(C)[_assign(Xs, C)].reshape(-1, GSZ)
        s = np.clip((Gf * rec).sum(1) / ((rec * rec).sum(1) + 1e-12),
                    1e-8, None).astype(np.float32)
        C = _lloyd((Gf / s[:, None]).reshape(-1, D), C,
                   np.repeat(s ** 2, GSZ // D).astype(np.float32), 2, K, rng)
    C_np = np.array(C)
    codes = np.empty((Wg.shape[0], GSZ // D), dtype=np.uint16)
    scales = np.empty(Wg.shape[0], dtype=np.float32)
    B = CHUNK // (GSZ // D)
    for i in range(0, Wg.shape[0], B):
        wb = Wg[i:i + B].astype(np.float32)
        sb = np.abs(wb).max(axis=1) + 1e-8
        for _ in range(2):
            a = mx.argmin(_dists(mx.array((wb / sb[:, None]).reshape(-1, D)), C), axis=1)
            mx.eval(a)
            a = np.array(a)
            rec = C_np[a].reshape(wb.shape[0], GSZ)
            sb = np.clip((wb * rec).sum(1) / ((rec * rec).sum(1) + 1e-12),
                         1e-8, None).astype(np.float32)
        codes[i:i + B] = a.reshape(wb.shape[0], GSZ // D).astype(np.uint16)
        scales[i:i + B] = sb
    return (mx.array(C_np.astype(np.float16)),
            mx.array(pack(codes.reshape(E_, OUT_, NGRP * (GSZ // D)), bits)),
            mx.array(scales.reshape(E_, OUT_, NGRP).astype(np.float16)))


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
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    mx.set_wired_limit(0)
    mx.set_memory_limit(a.memory_limit_gb * 1024 ** 3)
    rng = np.random.default_rng(a.seed)
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
        cb, codes, scales = fit_module(W, D, K, rng)
        del W
        mx.save_safetensors(part, {n + ".codebook": cb, n + ".codes": codes,
                                   n + ".vq_scales": scales})
        mx.clear_cache()
        done += 1
        _log(f"  {done}/{len(geo)} {n} d{D}-K{K} [{time.time()-t0:.0f}s]")

    # ---- assemble: shipped bytes + refit parts + config ----
    new = {}
    for f in glob.glob(os.path.join(parts, "*" + EXT)):
        new.update(mx.load(f))
    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(os.path.join(a.artifact, "*")):
        b = os.path.basename(f)
        if b.endswith(EXT) or b == "__pycache__":
            continue
        shutil.copy(os.path.realpath(f), os.path.join(a.out, b))
    cfg_path = os.path.join(a.out, "config.json")
    cfg = json.load(open(cfg_path))
    vm = cfg.get("vq_modules") or cfg.get("vq_linear") or {}
    for n, g in geo.items():
        ent = vm[n]
        ent["dim"], ent["k"] = int(g["dim"]), int(g["k"])
        ent["pack_bits"] = int(math.ceil(math.log2(int(g["k"]))))
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    swapped = 0
    for f in sorted(glob.glob(os.path.join(a.artifact, "*" + EXT))):
        rp, b = os.path.realpath(f), os.path.basename(f)
        tens = mx.load(rp)
        hit = [k for k in tens if k in new]
        dst = os.path.join(a.out, b)
        if os.path.lexists(dst):
            os.remove(dst)
        if not hit:
            os.symlink(rp, dst)          # unchanged shard: shipped bytes, no copy
            continue
        out = {}
        for k, v in tens.items():
            out[k] = new[k] if k in new else v
            swapped += k in new
        mx.save_safetensors(dst, out)
        del tens, out
        mx.clear_cache()
    _log(f"ASSEMBLE-DONE: {swapped} tensors -> {a.out} (config pack_bits updated)")


if __name__ == "__main__":
    main()
