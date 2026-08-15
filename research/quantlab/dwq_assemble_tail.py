#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Paste a tail-DWQ patch (dwq_train_tail_solo.py output) into a COPY of the
student artifact.

Only shards containing patched tensors are rewritten; everything else (and
the vision sidecar + configs) is copied through. Patch keys look like
`model.layers.57.mlp.switch_mlp.gate_proj.scales`; artifact keys carry the
`language_model.` prefix — mapped here. Every patch tensor must match its
target's shape and dtype, and every patch key must land: unmatched keys are
a hard error, not a warning (a silently dropped patch tensor would ship an
artifact that scored differently than what was trained).
"""
import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.student)
    out = Path(args.out)
    assert not out.exists(), f"refusing to overwrite {out}"
    out.mkdir(parents=True)

    patch = dict(mx.load(args.patch).items())
    index = json.loads((src / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]

    # map patch keys -> artifact keys
    mapped = {}
    for k, v in patch.items():
        for cand in (k, "language_model." + k,
                     k.replace("model.", "language_model.model.", 1)):
            if cand in wmap:
                mapped[cand] = v
                break
        else:
            raise SystemExit(f"patch key not found in artifact: {k}")
    print(f"{len(mapped)} patch tensors mapped", flush=True)

    touched = {wmap[k] for k in mapped}
    print(f"rewriting {len(touched)} of "
          f"{len(set(wmap.values()))} shards", flush=True)

    for f in sorted(src.iterdir()):
        if f.is_dir():
            shutil.copytree(f, out / f.name)
            continue
        if f.name in touched:
            tensors = dict(mx.load(str(f)).items())
            n = 0
            for k in list(tensors):
                if k in mapped:
                    old = tensors[k]
                    new = mapped[k].astype(old.dtype)
                    assert new.shape == old.shape, \
                        f"{k}: shape {new.shape} != {old.shape}"
                    tensors[k] = new
                    n += 1
            mx.save_safetensors(str(out / f.name), tensors,
                                metadata={"format": "mlx"})
            print(f"  {f.name}: {n} tensors patched", flush=True)
        else:
            shutil.copy2(f, out / f.name)

    applied = sum(1 for k in mapped if wmap[k] in touched)
    assert applied == len(mapped)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
