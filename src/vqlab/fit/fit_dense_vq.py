#!/usr/bin/env python3
"""VQ-fit a DENSE mlp trio from a bf16 source (family-aware: gemma4_e4b, qwen3_8).

WHY A SEPARATE FITTER. fit_moe.py targets modules that a prepared
struct BASE marked as 2-bit placeholders (is_vq_target checks bits==2);
dense e4b has no such base. This is the minimum honest experiment for
"does VQ transfer to a small dense model": fit gate/up/down_proj
(6.15 GiB = 42% of bf16 bytes) at d2-K2048 with the SAME contract as the
main fitter — group-64 max-abs fp16 scales along IN, kmeans++ init,
scatter-add Lloyd iterations — and write an artifact that
verify_artifact.py (--family gemma4_e4b) can decode independently.

Known-healthy reference for the geometry: 26b d2-K2048 mean relerr ~0.032.
A small dense model has less redundancy, so worse is EXPECTED; the
question is how much. This artifact is NOT runnable (no model.py splice) —
fit quality first, runtime integration only if the numbers earn it.

    ./fit_e4b_vq.py --src <bf16 snapshot> --out <dir> [--k 2048 --dim 2]
"""
import argparse
import os
import pathlib
import sys
import time

import mlx.core as mx

from vqlab.core.source import fit_store as fitstore
from vqlab.family.registry import DENSE_FAMILIES
from vqlab.fit.kmeans import assign_mx as assign
from vqlab.records import provenance
from vqlab.records.provenance import write_build_record
from vqlab.util.fs import read_json, write_json
from vqlab.core.artifacts.artifact import Artifact


def _assert_family_layout(src_idx, sample_key, family, registry_name):
    """Fail at STARTUP, helpfully, when the source does not match the family.

    Without this the first miss is a bare KeyError deep in the fit loop. A
    new model family is SUPPORTED — via a families.py entry — but the two-hour
    characterisation pass in docs/ONBOARDING.md comes first: establish a
    deterministic instrument, profile the weight geometry, run the init
    sweep. Fitting an uncharacterised family at defaults produces numbers
    nobody should believe, including you.
    """
    if sample_key in src_idx:
        return
    import difflib
    frag = sample_key.split(".")[-2]
    near = [k for k in src_idx if frag in k][:3] or         difflib.get_close_matches(sample_key, list(src_idx), n=3, cutoff=0.3)
    raise SystemExit(
        f"FAIL: this source does not match --family {family!r}.\n"
        f"  expected key:      {sample_key}\n"
        f"  closest in source: " + ("\n                     ".join(near) or "(nothing similar)") + "\n"
        f"To onboard a new model family: add its key template to {registry_name} "
        f"(src/vqlab/families.py) AND run the characterisation pass in "
        f"docs/ONBOARDING.md first — a fit at unexamined defaults produces "
        f"numbers nobody should believe. If the family is right, the source "
        f"dir is probably the wrong checkpoint (quantized vs bf16, or HF vs "
        f"mlx layout).")


def normalize(T, D, G):
    """group-64 max-abs scales along IN, mirrored from the main fitter."""
    out_d, in_d = T.shape
    Wg = T.reshape(out_d, in_d // G, G)
    scale = mx.maximum(mx.abs(Wg).max(axis=-1, keepdims=True), 1e-8)
    Xn = (Wg / scale).reshape(-1, D)
    return Xn, scale.astype(mx.float16)


def kmeanspp(X, k, cap):
    n = X.shape[0]
    idx = mx.random.randint(0, n, (min(cap, n),))
    P = X[idx]
    C = P[mx.random.randint(0, P.shape[0], (1,))]
    d2 = mx.sum((P - C[0]) ** 2, axis=1)
    for _ in range(k - 1):
        probs = d2 / mx.maximum(d2.sum(), 1e-12)
        r = float(mx.random.uniform().item())
        cum = mx.cumsum(probs)
        j = int(mx.argmax(cum >= r).item())
        C = mx.concatenate([C, P[j:j + 1]], axis=0)
        d2 = mx.minimum(d2, mx.sum((P - P[j]) ** 2, axis=1))
        mx.eval(C, d2)
    return C


def fit_tensor(T, K, D, G, iters, init_cap):
    """Fit one tensor -> (codes, codebook fp16, scales fp16, relerr)."""
    Xn, scale = normalize(T.astype(mx.float32), D, G)
    C = kmeanspp(Xn, K, init_cap)
    for _ in range(iters):
        a = assign(Xn, C)
        oh = mx.zeros((K, D)).at[a].add(Xn)
        cnt = mx.zeros((K,)).at[a].add(mx.ones((Xn.shape[0],)))
        C = mx.where((cnt > 0)[:, None], oh / mx.maximum(cnt, 1)[:, None], C)
        mx.eval(C)
    a = assign(Xn, C)
    R = C[a].reshape(T.shape[0], T.shape[1] // G, G) * scale.astype(mx.float32)
    R = R.reshape(T.shape)
    num = float(mx.sum((R - T) ** 2).item())
    den = float(mx.sum(T.astype(mx.float32) ** 2).item())
    relerr = (num / max(den, 1e-12)) ** 0.5
    # Code width MUST follow K, not default to uint16: the runtime shim picks
    # uint8 for K<=256 (build_e4b_vq.py:161), so a uint16 artifact at K256 both
    # DOUBLES the stored bytes -- turning a 2.0 bpw fit into 4.0 bpw on disk --
    # and mismatches the dtype the loader allocates. Caught 2026-08-21 when the
    # assembled 27B came out only 5.6% smaller than its 4-bit base.
    code_dtype = mx.uint8 if K <= 256 else mx.uint16
    codes = a.reshape(1, T.shape[0], T.shape[1] // D).astype(code_dtype)
    return codes, C.astype(mx.float16), scale.reshape(1, T.shape[0], -1), relerr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--family", default="gemma4_e4b",
                    choices=["gemma4_e4b", "qwen3_8_dense"],
                    help="selects tensor-name template and default layer range")
    ap.add_argument("--init-cap", type=int, default=200_000,
                    help="rows subsampled for the k-means++ INIT draw. Default "
                         "200_000 is the historical value. At K=65536 that is ~3 "
                         "samples per centroid for seeding (the Lloyd loop still "
                         "refines over ALL rows, measured 22.28M / 340 per "
                         "centroid, 0 dead codes). Cost is ~O(k*cap) and is <1%% of "
                         "one Lloyd iteration, so raising it is nearly free. "
                         "Added 2026-08-24 to TEST whether the thin init starves "
                         "high-K fits (E141).")
    ap.add_argument("--seed", type=int, default=1234,
                    help="RNG seed for the k-means++ subsample and centroid draws. "
                         "DEFAULT IS SEEDED (1234) so fits are REPRODUCIBLE. "
                         "Before 2026-08-22 nothing seeded these and every fit was "
                         "a fresh draw: measured spread at 397B d4/K256, same "
                         "stack, n=2, was 0.0256 wikitext ppl while mean relerr "
                         "moved only 0.0001 (E136/E136b). Pass --seed -1 to "
                         "restore the old unseeded behaviour, e.g. to MEASURE a "
                         "draw distribution on purpose (III.12 floors).")
    ap.add_argument("--relerr-abort", type=float, default=0.90,
                    help="stop the fit if any tensor exceeds this relative error. "
                         "Default 0.90 catches the degenerate case (relerr 1.0 = "
                         "reconstruction is exactly zero) without tripping on "
                         "legitimately hard low-K geometries.")
    args = ap.parse_args(argv)


    # SEEDING (2026-08-22). mx.random is process-global; seeding here covers the
    # subsample draw at kmeanspp():73, the first-centroid draw at :75 and the
    # k-means++ selection at :79 — the only randomness in this fitter (no numpy
    # RNG is used). -1 means "do not seed", which is the pre-2026-08-22 behaviour.
    if args.seed >= 0:
        mx.random.seed(args.seed)
        print(f"SEEDED mx.random with {args.seed} — this fit is reproducible", flush=True)
    else:
        print("UNSEEDED (--seed -1) — this fit is a fresh draw", flush=True)
    SRC, OUT = pathlib.Path(args.src), pathlib.Path(args.out)
    K, D, G = args.k, args.dim, args.group
    KEY_TMPL, _default_layers = DENSE_FAMILIES[args.family]
    LO, HI = (int(x) for x in (args.layers or _default_layers).split("-"))
    OUT.mkdir(parents=True, exist_ok=True)
    src_idx = Artifact(SRC).index
    PROJ = ["gate_proj", "up_proj", "down_proj"]
    KEY = KEY_TMPL
    _assert_family_layout(src_idx, KEY.format(li=LO, key=PROJ[0]), args.family,
                          "DENSE_FAMILIES")

    weights, vq_modules, report = {}, {}, []
    t0 = time.time()
    for li in range(LO, HI + 1):
        for proj in PROJ:
            sk = KEY.format(li=li, key=proj)
            sh = src_idx[sk]
            # MATERIALISE THE READ HERE, on the cpu stream (FINDINGS IV.1).
            # mx.load is LAZY: left unevaluated, the source read is paid inside
            # the kmeans GPU command buffer, and it can silently yield ZEROS.
            # Measured 2026-08-21 (E123) by instrumenting this exact loop:
            #   Xn all-zero, codebook all-zero, scales all-zero (fp16 underflow of
            #   the 1e-8 floor), reconstruction all-zero -> relerr exactly 1.0000,
            #   while the SAME lazy T re-evaluated moments later for the relerr
            #   denominator read back correctly at norm 104.28.
            # One tensor in ~90 on this path. The abort catches it; this prevents
            # it. Do NOT "optimise" this eval away.
            with mx.stream(mx.cpu):
                T = mx.load(str(SRC / sh))[sk]
                mx.eval(T)
            codes, cb, sc, rel = fit_tensor(T, K, D, G, args.iters, args.init_cap)
            mod = KEY_TMPL.replace(".weight", "").format(li=li, key=proj)
            weights[mod + ".codes"] = codes
            weights[mod + ".codebook"] = cb
            weights[mod + ".vq_scales"] = sc
            vq_modules[mod] = {"experts": 1, "out": int(T.shape[0]),
                               "in": int(T.shape[1]), "k": K, "dim": D, "group": G}
            report.append(rel)
            print(f"L{li:02d} {proj:10s} relerr {rel:.4f}  "
                  f"[{time.time()-t0:6.0f}s]", flush=True)
            # ABORT on a degenerate fit. fit_moe.py has had --relerr-abort
            # for weeks; this fitter only PRINTED relerr and carried on. Measured
            # 2026-08-21: the E95 dense 27B shipped L60 up_proj with codebook,
            # codes AND scales all exactly zero against a real source tensor
            # (absmax 0.283) — relerr 1.0000, printed, ignored, and invisible until
            # the outlier gate learned to read dense artifacts hours later. A fit
            # that reconstructs NOTHING must stop the run, not decorate the log.
            if rel > args.relerr_abort:
                raise SystemExit(
                    f"FATAL: L{li} {proj} relerr {rel:.4f} > {args.relerr_abort} "
                    f"— degenerate fit, do not ship this artifact. "
                    f"(relerr 1.0 means the reconstruction is exactly zero.)")
            del T
            mx.clear_cache()

    mx.save_safetensors(str(OUT / "model-00001-of-00001.safetensors"), weights)
    idx = {"metadata": {}, "weight_map":
           {k: "model-00001-of-00001.safetensors" for k in weights}}
    write_json(OUT / "model.safetensors.index.json", idx, indent=1)
    cfg = read_json(SRC / "config.json")
    cfg["vq_modules"] = vq_modules
    write_json(OUT / "config.json", cfg, indent=1)
    # BUILD RECORD (docs/PROVENANCE.md). Last write, so it hashes the final bytes.
    write_build_record(
        OUT, tool="fit-dense", script=__file__, ap=ap, args=args,
        method={"init": "kmeans++", "init_cap": args.init_cap,
                "lloyd_iters": args.iters, "sample": "all rows",
                "scales": "max-abs per group", "alternation": False,
                "tail_weight_pow": 0.0,
                "seed": args.seed if args.seed >= 0 else "unseeded",
                "k": K, "dim": D, "group": G,
                "mean_relerr": round(sum(report) / len(report), 6),
                "worst_relerr": round(max(report), 6)},
        inputs=[("source", SRC)],
        modules={m: {**v, "origin": "fit"} for m, v in vq_modules.items()})
    # File every module's fit in the fit store (core/source/fit_store.py): findable by
    # family / teacher / layer / geometry, and never paid for twice.
    _teacher = fitstore.teacher_slug(SRC)
    _recipe = {"origin": "fit", "tool": "fit-dense", "run_id": os.environ.get("VQLAB_RUN_ID"),
               "commit": provenance.code_state()["commit"],
               "fitter": {"init": "kmeans++", "init_cap": args.init_cap,
                          "lloyd_iters": args.iters, "alternation": False,
                          "scales": "max-abs per group",
                          "seed": args.seed if args.seed >= 0 else "unseeded"}}
    for _m in vq_modules:
        fitstore.put(OUT / "model-00001-of-00001.safetensors", _m, args.family,
                     _teacher, recipe=_recipe)
    print(f"stored {len(vq_modules)} fits in {fitstore.roots()[0]}")
    print(f"\nfit {len(report)} tensors, mean relerr "
          f"{sum(report)/len(report):.4f}, worst {max(report):.4f}, "
          f"{time.time()-t0:.0f}s total -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
