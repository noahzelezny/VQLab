#!/usr/bin/env python3
"""Pack PLE codes to true 11-bit width, row-aligned.

The block packer needs 32-aligned rows; PLE rows are 40 codes. But
40 x 11 = 440 bits = exactly 55 bytes, so each row packs byte-aligned with
zero padding — perfect random access for the embedding gather
(row r starts at byte r*55). uint16 [rows, 40] -> uint8 [rows, 55].

Every tensor is verified by ROUND-TRIP before its shard is swapped in:
unpacking the packed bytes must reproduce the original codes exactly.

    python -m vqlab.pack_ple --artifact <dir>
"""
import argparse
import json
import pathlib

import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True)
a = ap.parse_args()
ART = pathlib.Path(a.artifact)

cfg = json.load(open(ART / "config.json"))
ple = cfg.get("vq_ple") or {}
if not ple:
    raise SystemExit("no vq_ple block in config — nothing to pack")
g = ple["geometry"]
if g.get("row_bytes"):
    raise SystemExit("vq_ple already packed (row_bytes present)")
bits = (g["k"] - 1).bit_length()

idx_p = ART / "model.safetensors.index.json"
idx = json.load(open(idx_p)); wm = idx["weight_map"]
shards = sorted({wm[k + ".codes"] for k in ple["keys"]})
nsub = None

def pack_rows(c16):
    rows, n = c16.shape
    total_bits = n * bits
    assert total_bits % 8 == 0, f"{n}x{bits} bits not byte-aligned"
    rb = total_bits // 8
    out = np.zeros((rows, rb + 2), dtype=np.uint32)   # +2 slack for windows
    v = c16.astype(np.uint32)
    for i in range(n):
        b0, sh = (i * bits) // 8, (i * bits) % 8
        w = v[:, i] << sh
        out[:, b0] |= w & 0xFF
        out[:, b0 + 1] |= (w >> 8) & 0xFF
        out[:, b0 + 2] |= (w >> 16) & 0xFF
    assert not out[:, rb:].any(), "codes overflowed the row"
    return out[:, :rb].astype(np.uint8), rb

def unpack_rows(p8, n):
    b = np.concatenate([p8.astype(np.uint32),
                        np.zeros((p8.shape[0], 2), np.uint32)], axis=1)
    outs = []
    for i in range(n):
        b0, sh = (i * bits) // 8, (i * bits) % 8
        w = b[:, b0] | (b[:, b0 + 1] << 8) | (b[:, b0 + 2] << 16)
        outs.append((w >> sh) & ((1 << bits) - 1))
    return np.stack(outs, axis=1)

for sh_name in shards:
    data = dict(mx.load(str(ART / sh_name)))
    changed = 0
    for k in list(data):
        if not k.endswith(".codes") or data[k].ndim != 2:
            continue
        c = np.array(data[k], copy=False)
        if c.dtype != np.uint16:
            continue
        packed, rb = pack_rows(c)
        rt = unpack_rows(packed, c.shape[1])
        assert (rt == c).all(), f"round-trip mismatch in {k}"
        data[k] = mx.array(packed)
        nsub = c.shape[1]
        changed += 1
    if changed:
        tmp = ART / sh_name.replace(".safetensors", ".tmp.safetensors")
        mx.save_safetensors(str(tmp), data, metadata={"format": "mlx"})
        tmp.replace(ART / sh_name)
    print(f"{sh_name}: packed {changed} tensors", flush=True)

g["row_bytes"] = nsub * bits // 8
cfg["vq_ple"]["geometry"] = g
json.dump(cfg, open(ART / "config.json", "w"), indent=1)
idx["metadata"]["total_size"] = sum((ART / f).stat().st_size
                                    for f in set(wm.values()))
json.dump(idx, open(idx_p, "w"), indent=1)
tot = idx["metadata"]["total_size"]
print(f"row_bytes={g['row_bytes']}; artifact now {tot/2**30:.1f} GiB")
