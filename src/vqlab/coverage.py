#!/usr/bin/env python3
"""Which kernel path does each module of this artifact actually take?

Artifacts increasingly mix geometries — one file can carry d2-K256
unpacked shared experts, d4-K2048 packed, and d8-K16384 all at once, and
each dispatches independently. "The artifact is covered" is therefore not
a property of the artifact; it is a property of every module in it. This
reads config.json only (no weights, no GPU) and reports the prefill path
each module resolves to, using the runtime's OWN gate functions rather
than a restatement of them — so the report cannot drift from the code.

    vqlab coverage --artifact <dir> [--json]
"""
import argparse
import collections
import json
import pathlib
import sys


def classify(D, K, G, bits, IN, vq):
    """Ask the runtime itself. Returns (path, why)."""
    if not vq.gemmseg_fits(D, K, G, bits, IN):
        if D == 8 and not vq._FUSED_GEMM_D8:
            return "legacy", "d8 arm written but unarmed (VQ_MOE_FUSED_GEMM_D8=1)"
        if not vq._FUSED_GEMM:
            return "legacy", "fused GEMM disabled (VQ_MOE_FUSED_GEMM=0)"
        if G != 64:
            return "legacy", f"group={G}, kernel requires 64"
        if (IN // D) % 32 != 0:
            return "legacy", f"NSUB={IN // D} not a multiple of 32"
        over = K * 2 * D + 3 * 4096 > 32768
        if over and not vq._FUSED_GEMM_BIGK:
            return "legacy", "big-K needs CB_DEV arm (VQ_MOE_FUSED_GEMM_BIGK=1)"
        return "legacy", "geometry outside gemmseg_fits"
    cb_bytes = K * 2 * D
    # ASK the runtime, never restate it: this label read "threadgroup" for
    # d4-K2048 for the first hour after the 16 KB device-arm preference
    # landed, because it carried its own copy of the rule (2026-09-08).
    arm = (f"CB_DEV (device codebook, {cb_bytes // 1024} KB)"
           if vq.gemmseg_cb_dev(D, K)
           else f"threadgroup ({cb_bytes // 1024} KB cb)")
    fetch = "packed" if bits else "unpacked (BITS=0)"
    return "gemmseg v2", f"{arm}, {fetch}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    art = pathlib.Path(a.artifact)
    cfg = json.loads((art / "config.json").read_text())
    mods = cfg.get("vq_modules") or {}
    dense = cfg.get("vq_linear") or {}
    if dense and not mods:
        print(f"{art.name}: DENSE artifact ({len(dense)} vq_linear modules) — "
              "prefill is wdec + GEMM; gemmseg is MoE-only.")
        return 0
    if not mods:
        print(f"{art.name}: no vq_modules in config — not a VQ MoE artifact?")
        return 1

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from vqlab import vq_switch as vq

    buckets = collections.Counter()
    detail = {}
    for name, m in mods.items():
        D, K = m.get("dim"), m.get("k")
        G, bits = m.get("group", 64), m.get("pack_bits", 0) or 0
        IN = m.get("in")
        if None in (D, K, IN):
            continue
        path, why = classify(D, K, G, bits, IN, vq)
        key = (f"d{D} K{K} G{G} " + (f"bits{bits}" if bits else "unpacked"),
               path, why)
        buckets[key] += 1
        detail[name] = {"geometry": key[0], "path": path, "why": why}

    if a.json:
        print(json.dumps({"artifact": art.name, "modules": detail}, indent=1))
        return 0

    total = sum(buckets.values())
    fused = sum(n for (g, p, w), n in buckets.items() if p != "legacy")
    print(f"{art.name}: {total} VQ modules, {len(buckets)} distinct geometries")
    print(f"  fused: {fused}/{total} "
          f"({100 * fused / max(total, 1):.0f}%)\n")
    for (geo, path, why), n in sorted(buckets.items(), key=lambda kv: -kv[1]):
        mark = "OK " if path != "legacy" else "-> "
        print(f"  {mark}{n:4d} x {geo:26s} {path:12s} {why}")
    if fused < total:
        print("\n  modules on 'legacy' are CORRECT, just not accelerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
