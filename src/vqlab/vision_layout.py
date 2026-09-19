#!/usr/bin/env python3
"""Check -- and repair -- the MEMORY LAYOUT of a grafted vision tower.

mlx stores a vision patch-embed convolution CHANNELS-LAST: HF ships
(out, C, T, H, W) and mlx wants (out, T, H, W, C). `graft_vision.py` has
carried a `--permute-conv5` flag for this since the Qwen3.5-35B pair was
measured, where exactly one tensor of 333 -- patch_embed.proj.weight --
needs it. The flag is OPT-IN, and a graft that omits it writes a tower that
is complete, correctly named, and silently transposed.

That is what happened. Measured 2026-09-19 across the fleet: the four 397B
rungs and the four Flash-Next rungs carry (1152, 3, 2, 16, 16) -- HF layout.
The 35B, 27B and GLM rungs carry the permuted form and are correct. Eight
artifacts, all grafted without the flag.

Nothing caught it. check_vision.py counts tensors, check_release checks the
index, and the new vision-smoke surface arm proves the tower is REACHABLE,
not that it is right. A layout error is invisible to every byte-level gate by
construction: the bytes are all present and the names are all correct.

Repair is a permutation of one tensor. No refit, no re-quantization: the
vision shard is rewritten with that tensor transposed and every other tensor
copied through unchanged.

    vqlab vision-layout <artifact>          # report
    vqlab vision-layout <artifact> --fix    # rewrite the shard in place
    vqlab vision-layout <artifact> --fix --out <dir>   # write elsewhere
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
        print(f"\nFAIL: {len(bad)} tensor(s) in HF layout. The tower is "
              "complete and correctly named but its patch embedding is "
              "transposed, so it computes garbage (or raises a conv shape "
              "error). Re-run with --fix, or re-graft with --permute-conv5.")
        return 1

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
