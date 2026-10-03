#!/usr/bin/env python3
"""vqlab reskeleton: put a VQ build's expert tensors on another build's skeleton.

A matched-skeleton comparison. A VQ artifact is two things: VQ-coded expert
(or MLP) modules, and an affine SKELETON -- attention, shared experts,
embeddings, head -- at some bit map. Comparing it against an affine build
with a DIFFERENT skeleton compares two variables at once (paper v5, 2026-09-28:
the 35B's apparent 2x advantage over uniform q4 was mostly the skeleton). This
tool removes the second variable: the output carries the VQ build's expert
tensors verbatim and the SKELETON build's non-expert tensors verbatim, so it
differs from the skeleton build only in how the experts are stored.

    vqlab reskeleton --vq <vq artifact> --skeleton <affine artifact> --out <dir>

Rules, each refused loudly rather than guessed:
  * expert modules = the VQ artifact's `vq_modules`; each must exist in the
    skeleton build too (as affine .weight/.scales/.biases, which are dropped);
  * every other language_model.* module must exist in BOTH builds -- the
    skeleton's copy is used; a module in only one of them is a FAIL;
  * tensors outside language_model.* (vision tower, MTP sidecar) come from the
    VQ artifact unchanged, so the output loads exactly like the VQ build;
  * config: the VQ config, with every skeleton module's quantization entry
    replaced by the skeleton build's (per-module entry or its default width).
All non-weight files (model.py runtime, tokenizer, templates) come from the VQ
artifact. Tensors are copied bit-for-bit, never re-quantized. --out must not
exist; nothing is deleted.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import mlx.core as mx

from vqlab.core.artifact import Artifact, write_index

LM = "language_model."
VQ_SUFFIX = (".codes", ".codebook", ".vq_scales")
AFF_SUFFIX = (".weight", ".scales", ".biases")
SHARD = 5 * 2**30


def _qmap(cfg):
    return cfg.get("quantization") or cfg.get("text_config", {}).get("quantization") or {}


def _module(key):
    for s in VQ_SUFFIX + AFF_SUFFIX:
        if key.endswith(s):
            return key[: -len(s)]
    return key


def plan(vq: pathlib.Path, sk: pathlib.Path):
    """(sources {key: (dir, file)}, new quantization map, report). Raises on mismatch."""
    va, sa = Artifact.open(vq), Artifact.open(sk)
    vcfg, scfg, vidx, sidx = va.config, sa.config, va.index, sa.index
    experts = set(vcfg.get("vq_modules") or vcfg.get("text_config", {}).get("vq_modules") or {})
    if not experts:
        raise SystemExit(f"FAIL: {vq} has no vq_modules; not a VQ artifact")
    smods = {_module(k) for k in sidx}
    missing = sorted(e for e in experts if e not in smods)
    if missing:
        raise SystemExit(f"FAIL: {len(missing)} VQ expert modules absent from the skeleton "
                         f"build, e.g. {missing[:3]}")
    v_lm = {_module(k) for k in vidx if k.startswith(LM)} - experts
    s_lm = {_module(k) for k in sidx if k.startswith(LM)} - experts
    if v_lm != s_lm:
        only_v, only_s = sorted(v_lm - s_lm), sorted(s_lm - v_lm)
        raise SystemExit(f"FAIL: skeleton module sets differ: only in VQ {only_v[:5]} "
                         f"({len(only_v)}), only in skeleton {only_s[:5]} ({len(only_s)})")
    src = {}
    for k, f in vidx.items():
        if not k.startswith(LM) or _module(k) in experts:
            src[k] = (vq, f)                         # VQ experts + vision/MTP
    for k, f in sidx.items():
        if k.startswith(LM) and _module(k) not in experts:
            src[k] = (sk, f)                         # skeleton, verbatim
    sq, vqmap = _qmap(scfg), dict(_qmap(vcfg))
    sdefault = {"group_size": sq.get("group_size", 64), "bits": sq.get("bits"),
                "mode": sq.get("mode", "affine")}
    quantized = {k[: -len(".scales")] for k in sidx
                 if k.endswith(".scales") and _module(k) not in experts}
    changed = 0
    for m in quantized:
        new = sq.get(m) if isinstance(sq.get(m), dict) else sdefault
        if vqmap.get(m) != new:
            changed += 1
        vqmap[m] = new
    report = {"experts": len(experts), "skeleton_modules": len(quantized),
              "quant_entries_changed": changed,
              "from_vq": sum(1 for d, _ in src.values() if d == vq),
              "from_skeleton": sum(1 for d, _ in src.values() if d == sk)}
    return src, vqmap, vcfg, report


def write(vq, sk, out, src, qmap, vcfg):
    out.mkdir(parents=True)
    for f in vq.iterdir():                           # runtime, tokenizer, ...
        if f.suffix != ".safetensors" and f.name not in (
                "model.safetensors.index.json", "config.json", "vqlab_provenance.json",
                "vqlab_pin.json") and not f.is_dir():
            shutil.copy2(f, out / f.name)
    cfg = dict(vcfg)
    for key in ("quantization", "quantization_config"):
        if key in cfg:
            cfg[key] = qmap
    if "text_config" in cfg and "quantization" in cfg["text_config"]:
        cfg["text_config"] = {**cfg["text_config"], "quantization": qmap}
    json.dump(cfg, open(out / "config.json", "w"), indent=2)

    by_file = {}
    for k, (d, f) in src.items():
        by_file.setdefault((d, f), []).append(k)
    index, shard, size, n = {}, {}, 0, 1

    def flush():
        nonlocal shard, size, n
        if not shard:
            return
        fn = f"model-{n:05d}.safetensors"
        mx.save_safetensors(str(out / fn), shard, metadata={"format": "mlx"})
        for k in shard:
            index[k] = fn
        shard, size, n = {}, 0, n + 1
        mx.clear_cache()

    with mx.stream(mx.cpu):                          # Metal rule IV: bind reads to CPU
        for (d, f), keys in sorted(by_file.items(), key=lambda x: (str(x[0][0]), x[0][1])):
            arrs = mx.load(str(d / f))
            for k in sorted(keys):
                t = arrs[k]
                mx.eval(t)
                shard[k] = t
                size += t.nbytes
                if size >= SHARD:
                    flush()
            del arrs
    flush()
    total = sum((out / fn).stat().st_size for fn in set(index.values()))
    write_index(out, dict(sorted(index.items())), total_size=total)
    return len(index), total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab reskeleton", description=__doc__.split("\n")[0])
    ap.add_argument("--vq", required=True, help="VQ artifact (supplies experts + runtime)")
    ap.add_argument("--skeleton", required=True, help="affine artifact (supplies the skeleton)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="plan and report only")
    a = ap.parse_args(argv)
    vq, sk, out = pathlib.Path(a.vq), pathlib.Path(a.skeleton), pathlib.Path(a.out)
    if out.exists() and not a.dry_run:
        raise SystemExit(f"FAIL: {out} exists; reskeleton never overwrites")
    src, qmap, vcfg, rep = plan(vq, sk)
    print(f"reskeleton plan: {rep}")
    if a.dry_run:
        return 0
    n, total = write(vq, sk, out, src, qmap, vcfg)
    print(f"wrote {n} tensors, {total / 2**30:.2f} GiB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
