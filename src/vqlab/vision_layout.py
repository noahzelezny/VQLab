"""Report -- and optionally rewrite -- the on-disk LAYOUT of a grafted tower's
patch-embed conv. A DIAGNOSTIC, not a defect detector.

mlx wants a 5-D patch-embed conv channels-last, (out, T, H, W, C); HF ships
(out, C, T, H, W). Eight artifacts (397B x4, Flash-Next x4) carry the HF form
on disk under `model.visual.*`; the 35B, 27B and GLM rungs carry channels-last.

WHAT THIS IS NOT (F161): it is NOT a bug. Both real loaders -- mlx_vlm's
`sanitize_weights` and exo's worker calling `VisionModel.sanitize` -- map BOTH
layouts to the same channels-last tensor, measured to produce the identical
embedding. The first cut of this tool called the HF form "transposed ...
computes garbage" because `vision-smoke --tower-only` had bypassed sanitize()
and failed in conv3d; the instrument's bypass was reported as the artifact's
defect, eight shards were needlessly rewritten, and all eight were then
restored to the published bytes. Do not "fix" a layout on this tool's say-so.

Keep it for what it is good for: knowing which form an artifact carries before
writing code that reads the tensor directly (anything that skips sanitize()).

    vqlab vision-layout <artifact>          # report
    vqlab vision-layout <artifact> --fix    # rewrite to channels-last (rarely wanted)
"""
import argparse
import json
import pathlib
import shutil

import mlx.core as mx

# A 5-D conv weight is channels-last iff its LAST axis is a channel count.
# Real images give 1, 3 or 4; a patch grid never does.
_CHANNELS = (1, 3, 4)


def classify(shape):
    if len(shape) != 5:
        return "not-5d"
    return "ok" if shape[-1] in _CHANNELS else "hf-layout"


def scan(art: pathlib.Path):
    """[(key, shard, shape, verdict)] for every 5-D vision conv weight."""
    idx = art / "model.safetensors.index.json"
    wm = json.loads(idx.read_text())["weight_map"]
    vis = [k for k in wm if any("vis" in s for s in k.split("."))]
    out = []
    for sh in sorted({wm[k] for k in vis}):
        data = mx.load(str(art / sh))
        for k in vis:
            if wm[k] != sh:
                continue
            t = data[k]
            if t.ndim == 5:
                out.append((k, sh, tuple(t.shape), classify(tuple(t.shape))))
        del data
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--fix", action="store_true",
                    help="rewrite the offending shard with the tensor permuted")
    ap.add_argument("--out", default=None,
                    help="write the repaired artifact here instead of in "
                         "place (a dir of symlinks plus the new shard)")
    a = ap.parse_args()
    art = pathlib.Path(a.artifact)

    found = scan(art)
    if not found:
        print("no 5-D vision conv weights found -- nothing this tool checks. "
              "(gemma-style towers use a 2-D patch embed.)")
        return 0
    bad = [f for f in found if f[3] == "hf-layout"]
    for k, sh, shape, verdict in found:
        flag = "OK" if verdict == "ok" else "HF LAYOUT -- WRONG"
        print(f"  {k}\n    shape={shape} in {sh}  -> {flag}")
    if not bad:
        print("\nPASS: every 5-D vision conv weight is channels-last.")
        return 0
    if not a.fix:
        print(f"\nNOTE: {len(bad)} tensor(s) in HF layout. Both real loaders "
              "sanitize this to channels-last on load (F161), so this is a "
              "layout REPORT, not a defect. Only code that reads the tensor "
              "without sanitize() needs to care. --fix rewrites it anyway.")
        return 0

    dest = pathlib.Path(a.out) if a.out else art
    shards = sorted({f[1] for f in bad})
    if a.out:
        dest.mkdir(parents=True, exist_ok=True)
        for f in art.iterdir():
            if f.name in shards or f.name.startswith("model.py"):
                continue
            tgt = dest / f.name
            if not tgt.exists():
                tgt.symlink_to(f)
        for n in ("model.py",):
            if (art / n).exists() and not (dest / n).exists():
                shutil.copy2(art / n, dest / n)

    keys = {f[0] for f in bad}
    for sh in shards:
        data = mx.load(str(art / sh))
        fixed = {}
        for k, v in data.items():
            # (out, C, T, H, W) -> (out, T, H, W, C). Same permutation
            # graft_vision.py --permute-conv5 applies.
            fixed[k] = mx.transpose(v, (0, 2, 3, 4, 1)) if k in keys else v
        mx.eval(list(fixed.values()))
        # mx.save_safetensors APPENDS .safetensors if the name lacks it, so
        # a bare ".tmp" suffix lands somewhere you did not name. Keep the
        # extension in the temp name and resolve what was actually written.
        tmp = dest / (sh.replace(".safetensors", "") + ".tmp.safetensors")
        mx.save_safetensors(str(tmp), fixed)
        if not tmp.exists():
            raise SystemExit(f"FAIL: expected {tmp} to be written; nothing "
                             "was replaced.")
        tmp.replace(dest / sh)
        print(f"rewrote {sh}: permuted {sum(k in keys for k in data)} of "
              f"{len(data)} tensors")
        del data, fixed

    after = [f for f in scan(dest) if f[3] == "hf-layout"]
    if after:
        print("FAIL: still HF layout after the rewrite -- nothing trustworthy "
              "happened; do not ship this.")
        return 1
    print(f"\nPASS: {dest} now carries a channels-last patch embedding. "
          "Re-run `vqlab vision-smoke` before believing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
