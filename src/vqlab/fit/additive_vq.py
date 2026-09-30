"""vqlab fit-additive -- two small codebooks per module, EXPANDED to one big one.

THE IDEA. Additive VQ reconstructs a subvector as C1[i] + C2[j] from two
small codebooks (default K1 = K2 = 128 at d4). Two 7-bit codes carry 14 bits,
the same as ONE d4-K16384 code, but the fitter only has 256 centroids to
learn instead of 16384 -- so per learned parameter it may reach K16384
quality, and a runtime could decode it from 2 KB of codebook that sits in
threadgroup memory (d4-K16384 is 128 KB, 4x over the IV.1 ceiling).

WHY THE EXPANSION MAKES THIS A ZERO-KERNEL TEST. Every (i, j) pair names a
point of the Minkowski sum, so the fit is written out as an ORDINARY
d4-K16384 fit: entry i*K2 + j = C1[i] + C2[j], code = i*K2 + j. The packer,
loader, runtime and KL gate consume it unchanged, and `geo-build --reuse`
assembles it like any other part. It is then KL-comparable against a true
d4-K16384 fit at IDENTICAL bytes (same code width, same codebook shape). The
only thing the expanded form cannot show is the speed win.

So: the additive DECODE kernel is only worth writing if this KL comparison
shows parity (or better) with the true K16384 fit. Weight-space relerr is NOT
that comparison (law I.6) -- it is printed here for sanity only.

Recipe (method()): residual init (k-means++ + Lloyd for C1 on the scaled
subvectors, then for C2 on the residual), then alternation rounds of
  joint re-assignment (top-8 C1 candidates x all K2 by EXACT error) ->
  per-group least-squares scale (F80/F81) -> refit C1 given j -> refit C2 given i,
and a final encode of every group with the same joint search and two
scale<->assignment passes, as geo_build.fit_module does. Its `init` says
"additive", so `geo-build --pool` never pools one of these into an ordinary
build (the pool matches on RECIPE_KEYS).

    vqlab fit-additive --teacher <bf16 dir> --family qwen3_5 \
        --modules <geomap.json | a,b,c> --out-parts <dir> [--k1 128 --k2 128 --dim 4]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401  one module object per name
import geo_build as gb      # reuse the helpers; never copy them
from geo_build import GSZ, NFITG, ITERS_LLOYD, _assign, _kmeanspp, _lloyd, pack

ROUNDS = 8          # alternation rounds (joint assign -> scale -> C1 -> C2)
BEAM = 8            # C1 candidates searched per subvector in the joint assign
CH = 1 << 15        # subvectors per joint-assign chunk (x BEAM x K2 fp32)


def _joint_assign(X, C1, C2, beam=BEAM):
    """(i, j) minimizing ||x - C1[i] - C2[j]||^2 over the top-`beam` C1
    candidates (nearest to x) x ALL of C2 -- exact error, a superset of the
    top-8 x top-8 search. Returns int arrays (i, j)."""
    C1m, C2m = mx.array(C1), mx.array(C2)
    c2n = mx.sum(C2m * C2m, 1)
    b = min(beam, C1.shape[0])
    I_out, J_out = [], []
    for s in range(0, X.shape[0], CH):
        x = mx.array(X[s:s + CH])
        top = mx.argpartition(gb._dists(x, C1m), b - 1, axis=1)[:, :b]   # [n,b]
        r = x[:, None, :] - C1m[top]                                     # [n,b,D]
        e = (mx.sum(r * r, -1)[:, :, None] - 2 * (r @ C2m.T)
             + c2n[None, None, :])                                       # [n,b,K2]
        flat = mx.argmin(e.reshape(e.shape[0], -1), axis=1)
        bi, j = flat // C2.shape[0], flat % C2.shape[0]
        i = mx.take_along_axis(top, bi[:, None], axis=1)[:, 0]
        mx.eval(i, j)
        I_out.append(np.array(i)); J_out.append(np.array(j))
    return np.concatenate(I_out).astype(np.int64), np.concatenate(J_out).astype(np.int64)


def _mean_update(T, idx, w, C_old, rng):
    """Weighted centroid of targets T per index; dead entries reseeded from T
    (as _lloyd does)."""
    K, D = C_old.shape
    num = np.zeros((K, D)); den = np.zeros(K)
    np.add.at(num, idx, T * w[:, None]); np.add.at(den, idx, w)
    dead = den == 0
    if dead.any():
        num[dead] = T[rng.choice(T.shape[0], int(dead.sum()))]; den[dead] = 1
    return (num / den[:, None]).astype(np.float32)


def _ls_scale(G, rec):
    return np.clip((G * rec).sum(1) / ((rec * rec).sum(1) + 1e-12),
                   1e-8, None).astype(np.float32)


def expand(C1, C2):
    """[K1*K2, D] fp32: entry i*K2 + j = C1[i] + C2[j]."""
    return (C1[:, None, :] + C2[None, :, :]).reshape(-1, C1.shape[1])


def fit_module_additive(W, D, K1, K2, rng, rounds=ROUNDS, return_parts=False):
    """Same (codebook fp16 [K1*K2, D], packed codes uint32, scales fp16)
    triple as geo_build.fit_module. return_parts=True also returns
    (C1, C2, i, j) in fp32/int for exactness checks."""
    E_, OUT_, IN_ = W.shape
    K = K1 * K2
    NGRP, bits = IN_ // GSZ, math.ceil(math.log2(K))
    Wg = np.array(W.reshape(-1, GSZ))
    fidx = rng.choice(Wg.shape[0], min(NFITG, Wg.shape[0]), replace=False)
    Gf = Wg[fidx].astype(np.float32)
    s = np.abs(Gf).max(axis=1) + 1e-8
    Xs = (Gf / s[:, None]).reshape(-1, D)
    ones = np.ones(Xs.shape[0], np.float32)
    # residual init
    C1 = _lloyd(Xs, mx.array(_kmeanspp(Xs, K1, rng)), ones, ITERS_LLOYD, K1, rng)
    R = Xs - np.array(C1)[_assign(Xs, C1)]
    C2 = _lloyd(R, mx.array(_kmeanspp(R, K2, rng)), ones, ITERS_LLOYD, K2, rng)
    C1, C2 = np.array(C1), np.array(C2)
    # alternation: joint assign -> LS scale -> C1 | j -> C2 | i  (weights s^2,
    # so the scaled-space means are least squares in weight units)
    for _ in range(rounds):
        Xs = (Gf / s[:, None]).reshape(-1, D)
        i, j = _joint_assign(Xs, C1, C2)
        s = _ls_scale(Gf, (C1[i] + C2[j]).reshape(-1, GSZ))
        Xs = (Gf / s[:, None]).reshape(-1, D)
        w = np.repeat(s ** 2, GSZ // D).astype(np.float64)
        C1 = _mean_update(Xs - C2[j], i, w, C1, rng)
        C2 = _mean_update(Xs - C1[i], j, w, C2, rng)
    # final encode, every group: max-abs start, two assign<->scale passes
    codes = np.empty((Wg.shape[0], GSZ // D), dtype=np.uint16)
    scales = np.empty(Wg.shape[0], dtype=np.float32)
    I_all = np.empty((Wg.shape[0], GSZ // D), np.int64); J_all = I_all.copy()
    B = CH * 4 // (GSZ // D)
    for a in range(0, Wg.shape[0], B):
        wb = Wg[a:a + B].astype(np.float32)
        sb = np.abs(wb).max(axis=1) + 1e-8
        for _ in range(2):
            i, j = _joint_assign((wb / sb[:, None]).reshape(-1, D), C1, C2)
            sb = _ls_scale(wb, (C1[i] + C2[j]).reshape(wb.shape[0], GSZ))
        codes[a:a + B] = (i * K2 + j).reshape(wb.shape[0], -1).astype(np.uint16)
        I_all[a:a + B] = i.reshape(wb.shape[0], -1); J_all[a:a + B] = j.reshape(wb.shape[0], -1)
        scales[a:a + B] = sb
    out = (mx.array(expand(C1, C2).astype(np.float16)),
           mx.array(pack(codes.reshape(E_, OUT_, NGRP * (GSZ // D)), bits)),
           mx.array(scales.reshape(E_, OUT_, NGRP).astype(np.float16)))
    return out + ((C1, C2, I_all, J_all),) if return_parts else out


def method(k1=128, k2=128, rounds=ROUNDS, seed=1234):
    """Recipe dict under geo-build's RECIPE_KEYS (plus additive specifics)."""
    return {"init": f"additive {k1}x{k2}: residual kmeans++ (pool 2^18) + Lloyd, "
                    f"expanded to K{k1 * k2}",
            "lloyd_iters": ITERS_LLOYD, "sample_groups": NFITG, "group": GSZ,
            "alternation": True, "alternation_rounds": rounds,
            "scales": "least-squares (F80/F81)",
            "tail_weight_pow": 0.0, "tail_weight_from": 0,
            "additive": {"k1": k1, "k2": k2, "beam": BEAM,
                         "assign": f"joint: top-{BEAM} C1 x all C2, exact error"},
            "seed": seed if seed >= 0 else "unseeded",
            "rng": "numpy default_rng, one stream across modules"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--family", required=True, choices=sorted(gb.FAMILY))
    ap.add_argument("--modules", required=True,
                    help="geomap.json (entries must say dim=--dim, k=k1*k2) or a comma list")
    ap.add_argument("--out-parts", required=True, help="a geo-build --reuse dir")
    ap.add_argument("--k1", type=int, default=128)
    ap.add_argument("--k2", type=int, default=128)
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--seed", type=int, default=1234, help="-1 = unseeded")
    ap.add_argument("--cpu", action="store_true", help="fit on the CPU device")
    ap.add_argument("--no-store", action="store_true", help="do not file into the fit store")
    ap.add_argument("--memory-limit-gb", type=int, default=40)
    a = ap.parse_args()
    if a.cpu:
        mx.set_default_device(mx.cpu)
    else:
        mx.set_wired_limit(0)
        mx.set_memory_limit(a.memory_limit_gb * 1024 ** 3)
    K = a.k1 * a.k2
    if a.modules.endswith(".json"):
        geo = json.load(open(a.modules))
        bad = [n for n, g in geo.items() if (int(g["dim"]), int(g["k"])) != (a.dim, K)]
        if bad:
            raise SystemExit(f"geomap entries not at d{a.dim}-K{K}: {bad[:3]}")
        mods = sorted(geo)
    else:
        mods = sorted(m for m in a.modules.split(",") if m)
    rng = np.random.default_rng(None if a.seed < 0 else a.seed)
    rec_method = method(a.k1, a.k2, a.rounds, a.seed)
    os.makedirs(a.out_parts, exist_ok=True)
    op = os.path.join(a.out_parts, "origins.json")
    origins = json.load(open(op)) if os.path.exists(op) else {}
    t_index = json.load(open(os.path.join(a.teacher, "model.safetensors.index.json")))["weight_map"]
    teacher = gb.fitstore.teacher_slug(a.teacher)
    for n in mods:
        part = os.path.join(a.out_parts, gb.part_name(n, a.dim, K))
        if os.path.exists(part):
            gb._log(f"  have {os.path.basename(part)}"); continue
        t0 = time.time()
        W = gb.teacher_weight(a.teacher, t_index, a.family, n)
        gb.check_exact(n, W.shape[2], a.dim, K)
        cb, codes, scales = fit_module_additive(W, a.dim, a.k1, a.k2, rng, rounds=a.rounds)
        del W
        mx.save_safetensors(part, {n + ".codebook": cb, n + ".codes": codes,
                                   n + ".vq_scales": scales})
        rec = {"origin": "fit", "tool": "fit-additive",
               "commit": gb.provenance.code_state()["commit"],
               "fitter": rec_method, "tail_pow": 0.0,
               "run_id": os.environ.get("VQLAB_RUN_ID"),
               "sha256": gb.provenance._sha(part)}
        origins[os.path.basename(part)] = rec
        json.dump(origins, open(op, "w"), indent=1)
        if not a.no_store:
            st = gb.fitstore.put(part, n, a.family, teacher, recipe=rec)
            gb._log(f"    stored {st['fit_id']}")
        mx.clear_cache()
        gb._log(f"  {n} additive d{a.dim}-{a.k1}x{a.k2} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
