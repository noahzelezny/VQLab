#!/usr/bin/env python3
"""Preflight: does a comparator artifact hold the SAME core tensor set as the
teacher it is being scored against?

Motivation: a comparator that silently loads with dropped tensors produces a
plausible-looking WORSE score, which flatters us. Nothing in the scoring path
asserts tensor completeness, so a benchmark can be wrong in our favour and look
fine. Metadata only -- reads safetensors headers, never loads weights, so this
is safe to run beside a live fit.

Usage: check_comparator.py --artifact DIR --teacher DIR [--quiet]
"""
import argparse, json, re, struct, sys, pathlib

# quantisation companions collapse back onto the weight site they belong to
COMPANION = re.compile(r"\.(scales|biases|codes|codebook|vq_scales)$")

def core_tensors(d: pathlib.Path):
    d = pathlib.Path(d)
    idx = d / "model.safetensors.index.json"
    shards = (sorted(set(json.load(open(idx))["weight_map"].values()))
              if idx.exists() else
              sorted(p.name for p in d.glob("*.safetensors")))
    names = set()
    for sh in shards:
        with open(d / sh, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k in hdr:
            if k == "__metadata__":
                continue
            names.add(COMPANION.sub(".weight", k))
    return names

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True)
ap.add_argument("--teacher", required=True)
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()

art, tea = core_tensors(a.artifact), core_tensors(a.teacher)
extra, missing = sorted(art - tea), sorted(tea - art)

if not a.quiet:
    for label, s in (("EXTRA (in artifact, not teacher)", extra),
                     ("MISSING (in teacher, not artifact)", missing)):
        if s:
            print(f"  {label}: {len(s)}")
            for t in s[:5]:
                print(f"      {t}")
            if len(s) > 5:
                print(f"      ... and {len(s) - 5} more")

if extra or missing:
    print(f"FAIL: tensor sets differ — {len(extra)} extra, {len(missing)} missing "
          f"(artifact {len(art)} core sites, teacher {len(tea)}). A comparator that "
          f"loads with dropped tensors scores WORSE and flatters us; do not trust "
          f"a benchmark row from this pairing until resolved.")
    sys.exit(1)
print(f"PASS: {len(art)} core tensor sites, identical to teacher")
