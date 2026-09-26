"""vqlab unpack-dense — rewrite a packed dense VQ artifact with PLAIN codes.

WHY THIS EXISTS. F141 found a clean 27B d2 twin (K256 vs K512) where the
bigger codebook is 20-28% better on code and literary and 10% WORSE on
prose, and could not say why: the two rungs differ in K *and* in packing
(K256 ships unpacked uint8, K512 ships packed 9-bit). F139 independently
found packed and unpacked rungs behaving differently under VQ_DENSE_SS.
PACKING is a variable in its own right and nothing in the lab has ever
isolated it.

This isolates it exactly. The codes and the codebook are UNCHANGED -- every
subvector still points at the same centroid -- and only the STORAGE changes,
from bit-packed uint32 words to one uint16 per subvector. Same K, same d,
same group, same fit. A KL difference between an artifact and its unpacked
twin can therefore come from ONE place: the packed and plain kernels do not
agree numerically.

That is a sharper instrument than a refit, and it needs no teacher, which
matters because this family's bf16 teacher blobs are gone.

`vq_pack.unpack` is the exact inverse of `vq_pack.pack` and is the reference
the Metal reader is required to agree with, so the round trip is lossless by
construction -- and this verifies that on every tensor rather than trusting
it.

    vqlab unpack-dense --artifact <packed rung> --out <dir>

The output is bigger (9-bit -> 16-bit is ~1.78x on the code tensors). It is
a DIAGNOSTIC twin, not a shipping candidate.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401  one module object per name
import vq_pack  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true", default=True,
                    help="re-pack every unpacked tensor and require it to "
                         "reproduce the original words EXACTLY (default on)")
    a = ap.parse_args()

    src = pathlib.Path(a.artifact).resolve()
    out = pathlib.Path(a.out).resolve()
    cfg = json.load(open(src / "config.json"))
    vql = cfg.get("vq_linear")
    if not vql:
        raise SystemExit("REFUSED: no vq_linear block; this is not a dense "
                         "VQ artifact")
    packed = {k: v for k, v in vql.items() if v.get("pack_bits")}
    if not packed:
        raise SystemExit("REFUSED: no module carries pack_bits -- this "
                         "artifact is ALREADY unpacked, and writing a copy "
                         "would produce a twin that tests nothing")
    if len(packed) != len(vql):
        raise SystemExit(
            f"REFUSED: MIXED packing ({len(packed)} of {len(vql)} modules "
            "packed). Unpacking only some of them makes an artifact that "
            "differs from its parent in two ways at once, which is the "
            "confound this tool exists to remove")

    out.mkdir(parents=True, exist_ok=True)
    index = json.load(open(src / "model.safetensors.index.json"))
    wm = index["weight_map"]

    # by-shard so peak memory is one shard, not the model
    shards = sorted(set(wm.values()))
    for shard in shards:
        # mlx, not safetensors.numpy: these shards carry bf16 tensors and
        # numpy has no bfloat16, so a numpy reader dies on the first weight
        # it is supposed to pass through untouched.
        with mx.stream(mx.cpu):
            data = mx.load(str(src / shard))
            mx.eval(data)                      # INSIDE the block (rule IV)
        changed = 0
        for k in list(data):
            mod = k[: -len(".codes")] if k.endswith(".codes") else None
            spec = vql.get(mod) if mod else None
            if spec is None or not spec.get("pack_bits"):
                continue
            bits = spec["pack_bits"]
            nsub = spec["in"] // spec["dim"]
            e = spec.get("experts", 1)
            arr = np.array(data[k])
            p = arr.reshape(e, spec["out"], arr.shape[-1])
            codes = vq_pack.unpack(p, nsub, bits)
            if a.verify:
                # Round-trip on the REAL tensor, not a sample. An unpacker
                # that is subtly wrong would otherwise ship a twin whose KL
                # delta reads as a kernel disagreement.
                re = vq_pack.pack(codes, bits)
                if not np.array_equal(re.reshape(p.shape), p):
                    raise SystemExit(
                        f"REFUSED: round trip changed {k}; the unpacked "
                        "twin would not be code-identical to its parent")
            data[k] = mx.array(
                codes.reshape(spec["out"], nsub) if e == 1 else codes)
            changed += 1
        mx.save_safetensors(str(out / shard), data)
        del data
        print(f"  {shard}: {changed} code tensors unpacked", flush=True)

    for f in src.iterdir():
        if f.suffix == ".safetensors" or f.name in ("__pycache__",):
            continue
        dst = out / f.name
        if f.is_dir():
            shutil.copytree(f, dst, dirs_exist_ok=True)
        else:
            shutil.copy(f.resolve(), dst)

    for spec in cfg["vq_linear"].values():
        spec.pop("pack_bits", None)
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    shutil.copy(src / "model.safetensors.index.json",
                out / "model.safetensors.index.json")
    print(f"\nunpack-dense: {len(packed)} modules, pack_bits dropped -> {out}")
    print("DIAGNOSTIC TWIN: same codes, same codebook, plain storage. Any KL "
          "difference from the parent is a packed-vs-plain KERNEL "
          "disagreement, not a fit difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
