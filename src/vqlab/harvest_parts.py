"""Harvest VQ fits OUT of a shipped artifact into a geo-build parts dir.

A shipped VQ artifact already stores, per module, exactly the three tensors
a `geo-build` part carries:

    <module>.codebook   <module>.codes   <module>.vq_scales

So any geometry that some rung already ships is a fit we have ALREADY PAID
FOR, and it can be fed back to `geo-build --reuse` instead of refitting from
the bf16 teacher. That is what makes a same-family v2 rebuild cheap: the
Flash-5.5 rung is uniform d2-K1024 over all 144 expert modules, which is
precisely the promotion width a Flash-4.4 v2 wants, and d2 fits appear
nowhere in the HDD fit pool.

Why this is a separate tool and not a flag on geo-build: the artifact is the
DIFF BASE there ("keep these shipped bytes"), and a reuse SOURCE is a
different role ("borrow this other rung's codebooks"). Conflating them makes
it too easy to reuse from the artifact you are rebuilding, which is a no-op
that silently pins the old geometry.

Parts are written with `geo_build.part_name(..., tag=<content hash>)`,
because (module, d, K) is NOT unique -- fitting is stochastic, and a pool
that drops the tag silently collapses two different valid codebooks
(geo_build docstring point 3).

    vqlab harvest-parts --artifact <shipped rung> --out <parts dir>
        [--geometry d2-K1024] [--modules-like switch_mlp]

The geometry of each module is READ FROM THE BYTES (codebook shape), never
from the config: a config records intent, the tensors record what shipped.
"""
import argparse
import glob
import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geo_build import EXT, part_name  # noqa: E402

TRIPLE = ("codebook", "codes", "vq_scales")


def content_tag(cb):
    """Short stable tag over the codebook, matching the fit pool's scheme.

    Cast to float32 first: MLX codebooks are commonly bf16, which numpy has
    no dtype for, and `.tobytes()` on the raw buffer would raise.
    """
    import numpy as np
    import mlx.core as mx
    a = np.asarray(cb.astype(mx.float32))
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="shipped rung to harvest FROM")
    ap.add_argument("--out", required=True, help="parts dir to write")
    ap.add_argument("--geometry", default=None,
                    help="only harvest this geometry, e.g. d2-K1024")
    ap.add_argument("--modules-like", default=None,
                    help="only modules whose name contains this substring")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import mlx.core as mx

    ipath = os.path.join(a.artifact, "model.safetensors.index.json")
    if not os.path.exists(ipath):
        raise SystemExit(f"FAIL: no index at {ipath}")
    wmap = json.load(open(ipath))["weight_map"]

    mods = sorted({k.rsplit(".", 1)[0] for k in wmap
                   if k.rsplit(".", 1)[-1] == "codebook"})
    if a.modules_like:
        mods = [m for m in mods if a.modules_like in m]
    if not mods:
        raise SystemExit("FAIL: no VQ modules found (no *.codebook in index)")

    os.makedirs(a.out, exist_ok=True)
    shards = {}

    def shard(f):
        # mx.load, not safetensors.numpy: artifacts carry bf16 tensors, which
        # numpy cannot represent, and load_file materialises the WHOLE shard.
        if f not in shards:
            shards[f] = mx.load(os.path.join(a.artifact, f))
        return shards[f]

    written, skipped = 0, 0
    manifest = []
    for m in mods:
        keys = [f"{m}.{s}" for s in TRIPLE]
        if any(k not in wmap for k in keys):
            skipped += 1
            continue
        tensors = {}
        for k in keys:
            tensors[k] = shard(wmap[k])[k]
        K, d = tensors[f"{m}.codebook"].shape
        geom = f"d{d}-K{K}"
        if a.geometry and geom != a.geometry:
            skipped += 1
            continue
        tag = content_tag(tensors[f"{m}.codebook"])
        name = part_name(m, d, K, tag=tag)
        dst = os.path.join(a.out, name)
        rec = {"file": name, "module": m, "d": int(d), "k": int(K),
               "codes_shape": [int(x) for x in tensors[f"{m}.codes"].shape],
               "origin": os.path.basename(a.artifact.rstrip("/"))}
        manifest.append(rec)
        if a.dry_run:
            print(f"[dry] {name}  codes={rec['codes_shape']}")
            written += 1
            continue
        if not os.path.exists(dst):
            mx.save_safetensors(dst, tensors)
        written += 1
        print(f"  {written}/{len(mods)} {m} {geom}", flush=True)
        # one shard at a time: these are multi-GB
        if len(shards) > 2:
            shards.clear()

    if not a.dry_run:
        pathlib.Path(a.out, "manifest.json").write_text(json.dumps(
            {"harvested_from": a.artifact, "count": written,
             "note": "fits harvested from a shipped artifact; names carry "
                     "module.d{D}-K{K}.<content tag> because fitting is "
                     "stochastic and (module,d,K) is not unique",
             "fits": manifest}, indent=1))
    print(f"harvested {written} modules, skipped {skipped} -> {a.out}")


if __name__ == "__main__":
    main()
