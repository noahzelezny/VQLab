#!/usr/bin/env python3
"""Re-quantize AFFINE modules of a VQ artifact at a different bit width.

WHY. The 397B's affine side is ~6.4 GiB (outside the layers-57-59 defect)
and NO arc has ever questioned its allocation: linear-attention input
projections at 4 bits, attention/shared-experts/embeddings/lm_head at 6.
Those numbers were chosen once and inherited by every rung. Nobody has
measured whether lm_head deserves 6 bits or whether the shared expert --
which EVERY token passes through, unlike a routed expert seen by ~10 in
512 -- deserves more than one at 6.

Affine re-quantization needs no k-means: load the bf16, `mx.quantize`,
write. Minutes per candidate instead of the ~20 a VQ fit costs. This is
the "quantizing and scoring" tooling the project has always had, wired to
the splice contract the VQ tools use.

NULL TEST FIRST. `--bits` equal to the module's current width must
reproduce the shipped tensors and score identically. Run that before
trusting any row -- same discipline as demote_fit's vintage gate.

  ./affine_requant.py --base <artifact> --src <bf16> \
      --pattern language_model.lm_head --bits 4 --out <dir>
"""
import argparse
import json
import os
import pathlib
import shutil
import sys

import mlx.core as mx

GiB = 2 ** 30
AFFINE_SUFFIXES = ("weight", "scales", "biases")


def to_src_name(module):
    """artifact module -> bf16 tensor name.

    language_model.model.X   -> model.language_model.X
    language_model.lm_head   -> lm_head
    (verified against the bf16 index, 2026-09-06)
    """
    if module.startswith("language_model.model."):
        return "model.language_model." + module[len("language_model.model."):] + ".weight"
    if module == "language_model.lm_head":
        return "lm_head.weight"
    raise SystemExit(f"unmapped module name: {module}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--src", required=True, help="bf16 source dir")
    ap.add_argument("--pattern", required=True,
                    help="substring matched against affine module names")
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    bp, out = pathlib.Path(a.base), pathlib.Path(a.out)
    sp = pathlib.Path(a.src)
    idx = json.load(open(bp / "model.safetensors.index.json"))
    wm = idx["weight_map"]
    cfg = json.load(open(bp / "config.json"))
    src_wm = json.load(open(sp / "model.safetensors.index.json"))["weight_map"]

    q = cfg.get("quantization", {})
    targets = sorted(m for m, v in q.items()
                     if isinstance(v, dict) and "bits" in v and a.pattern in m)
    if not targets:
        raise SystemExit(f"no affine modules match {a.pattern!r}")
    cur = {q[m]["bits"] for m in targets}
    print(f"{len(targets)} module(s) match {a.pattern!r}: "
          f"{sorted(cur)}-bit -> {a.bits}-bit", flush=True)
    if a.bits in cur and len(cur) == 1:
        print("  NULL TEST: re-quantizing at the CURRENT width; the result "
              "must score identically to the base.", flush=True)

    # shards we must rewrite, and where each new tensor lands
    touched = sorted({wm[m + "." + s] for m in targets
                      for s in AFFINE_SUFFIXES if m + "." + s in wm})
    print(f"  rewrites {len(touched)} shard(s) of {len(set(wm.values()))}",
          flush=True)
    if a.dry_run:
        for m in targets[:5]:
            print(f"    {m}  <- {to_src_name(m)}")
        print("DRY RUN — nothing loaded, nothing written.")
        return

    # quantize each target from the bf16, grouped by SOURCE shard so the
    # remote bf16 is read once per shard rather than once per tensor
    by_src = {}
    for m in targets:
        by_src.setdefault(src_wm[to_src_name(m)], []).append(m)
    new = {}
    mx.set_default_device(mx.cpu)
    for i, (sh, mods) in enumerate(sorted(by_src.items()), 1):
        t = mx.load(str(sp / sh))
        for m in mods:
            src_t = t[to_src_name(m)]
            # PRESERVE THE SOURCE DTYPE. mx.quantize returns scales/biases in
            # the INPUT dtype, so upcasting to fp32 here emitted fp32 scales
            # for embed_tokens -- which makes the embedding output fp32, which
            # makes the WHOLE forward pass fp32, which at head_dim 256 asks
            # Metal for a 53 KB threadgroup attention kernel against a 32 KB
            # limit and crashes every exo prefill. (2026-09-07: this shipped
            # to both boxes and broke serving before it was caught.)
            wq, sc, bi = mx.quantize(src_t, group_size=a.group, bits=a.bits)
            want = src_t.dtype
            sc, bi = sc.astype(want), bi.astype(want)
            mx.eval(wq, sc, bi)
            new[m + ".weight"], new[m + ".scales"], new[m + ".biases"] = wq, sc, bi
            del src_t
        del t
        mx.clear_cache()
        print(f"  [{i}/{len(by_src)}] {sh}: {len(mods)} module(s)", flush=True)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # non-shard files come across as-is. A .safetensors file that the INDEX
    # does not reference is a SIDECAR (mtp-head-q6, vision grafts), not a
    # shard -- skipping every .safetensors dropped the 397B's published
    # 5.4 GiB MTP head from a rebuild (caught 2026-09-07, before it could
    # be uploaded over the live repo).
    _idx_shards = set(wm.values())
    for f in bp.iterdir():
        if f.is_dir():
            continue
        if f.suffix == ".safetensors" and f.name in _idx_shards:
            continue
        shutil.copy2(f, out / f.name)
    for sh in sorted(set(wm.values())):
        if sh not in touched:
            os.link(bp / sh, out / sh)
            continue
        t = mx.load(str(bp / sh))
        n = 0
        for name, arr in new.items():
            if wm.get(name) == sh:
                t[name] = arr
                n += 1
        tmp = out / (sh[:-len(".safetensors")] + ".tmp.safetensors")
        mx.save_safetensors(str(tmp), t)
        os.replace(tmp, out / sh)
        del t
        print(f"    rewrote {sh}: {n} tensors", flush=True)

    for m in targets:
        for d in ("quantization", "quantization_config"):
            if m in cfg.get(d, {}):
                cfg[d][m] = {"group_size": a.group, "bits": a.bits,
                             "mode": "affine"}
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    idx.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(out / s) for s in sorted(set(wm.values())))
    json.dump(idx, open(out / "model.safetensors.index.json", "w"), indent=1)
    base_gib = json.load(open(bp / "model.safetensors.index.json"))["metadata"]["total_size"] / GiB
    got = idx["metadata"]["total_size"] / GiB
    print(f"DONE {out.name}: {got:.3f} GiB  ({got - base_gib:+.3f} vs base)",
          flush=True)


if __name__ == "__main__":
    main()
