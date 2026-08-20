#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Release gate: vision tensor count in artifact == count in source.

WHY. Two Qwen3.6 publish artifacts shipped 40 minutes into upload with ZERO
vision tensors and passed every existing gate (guard, KL, smoke, cards) —
because every gate exercises the TEXT path. Noah caught it by asking.
This is the gate that would have caught it mechanically: count tensors by
prefix in both safetensors indexes and fail on mismatch, unless the card
made an explicit text-only decision (--allow-text-only).

    ./check_vision.py --artifact <dir> --src <bf16 dir>
    ./check_vision.py --artifact <dir> --src <dir> --allow-text-only
"""
import argparse
import json
import pathlib
import sys

# model.visual is the 397B/qwen3.5 layout — its absence from the first
# version made the gate PASS VACUOUSLY on the family it was most needed for
# (overnight 08-19: "source has no vision tensors" printed for a 751G bf16
# with 333 visual tensors). A gate that cannot see the tensors it guards is
# worse than no gate: it stamps approval.
PREFIXES = ("vision_tower", "embed_vision", "vision_model", "multi_modal",
            "model.visual", "visual")

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--allow-text-only", action="store_true",
                help="pass with 0 vision tensors IF the card states it")
args = ap.parse_args()


def vcount(d):
    """Count distinct WEIGHT SITES, not raw tensors. A quantized artifact
    carries .scales/.biases companions per weight, so raw counts read HIGHER
    than a bf16 source and the gate cried missing-towers on an artifact with
    MORE vision tensors than its source (e4b VQ-PLE, 661 vs 659, 08-20).
    Stripping the quantization suffixes makes the site count representation-
    invariant."""
    m = json.load(open(pathlib.Path(d) / "model.safetensors.index.json"))
    sites = set()
    for k in m["weight_map"]:
        for suf in (".weight", ".scales", ".biases", ".codes", ".codebook",
                    ".vq_scales"):
            if k.endswith(suf):
                k = k[:-len(suf)]
                break
        sites.add(k)
    return {p: sum(k.startswith(p) for k in sites) for p in PREFIXES}


a, s = vcount(args.artifact), vcount(args.src)
ok = True
for p in PREFIXES:
    if s[p] == 0 and a[p] == 0:
        continue
    mark = "OK" if a[p] == s[p] else "MISMATCH"
    if a[p] != s[p]:
        ok = False
    print(f"{p:15s} src {s[p]:4d}  artifact {a[p]:4d}  {mark}")
tot_s, tot_a = sum(s.values()), sum(a.values())
if tot_s == 0:
    print("source has no vision tensors — text-only family, PASS")
elif ok:
    print(f"PASS: all {tot_a} vision tensors present")
elif tot_a == 0 and args.allow_text_only:
    print(f"PASS (text-only by declaration): source has {tot_s} vision "
          f"tensors, artifact ships none — the CARD must say so")
else:
    print(f"FAIL: source {tot_s} vision tensors, artifact {tot_a}. "
          f"Graft (graft_vision.py) or pass --allow-text-only with a card "
          f"statement.")
    sys.exit(1)
