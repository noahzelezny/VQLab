#!/usr/bin/env python
"""Pack an existing (unpacked) VQ codes artifact into the sub-byte format.

    ./pack_artifact.py --src <unpacked-dir> --out <packed-dir>

This is a pure REPRESENTATION change: every decoded weight is bit-identical
to the source artifact's, so the packed model must referee to exactly the
same perplexity. Only `codes` tensors change (uint8/uint16 -> uint32 words);
codebooks, scales, and every non-VQ tensor are copied through untouched.

Sizes this unlocks (see vq_pack.py's table): d4 K128 100.1 GiB, d4 K2048
142.9 GiB — the points between the byte-aligned sizes, which is the whole
reason the format exists.

Writes model.py + config (model_file, vq_modules with pack_bits) via
add_model_file.py, so the result is still zero-patch loadable by stock
mlx_lm.

MIXED ARTIFACTS. A tensor whose NSUB is not a multiple of 32 cannot use the
block layout and is copied through UNPACKED (gemma-4-26b's down_proj, NSUB
176). The result is a legal artifact: the reader dispatches per tensor on
codes.dtype, so packed and unpacked tensors coexist. Such tensors keep
paying full uint16, so the achieved size will miss the analytic target —
the run prints exactly which ones and why.
"""
import argparse
import json
import pathlib
import shutil
import struct
import subprocess
import sys

import mlx.core as mx
import numpy as np

import vq_pack

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--vision-config-from", default="/Volumes/Thunderbay SSD/"
                "Exo Models/Qwen--Qwen3.5-397B-A17B-bf16",
                help="artifact dir to copy vision_config/image_token_id from "
                     "when the packed config lacks them. Defaults to the Qwen "
                     "3.5 bf16 source since that is what this lab packs; pass "
                     "a different path for another family, or \"\" to skip "
                     "and get a warning instead.")
args = ap.parse_args()

SRC, OUT = pathlib.Path(args.src), pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

idx_path = SRC / "model.safetensors.index.json"
weight_map = json.load(open(idx_path))["weight_map"]
shards = sorted(set(weight_map.values()))

# non-tensor files (config, tokenizer, template, README...) come along
for f in SRC.iterdir():
    if f.is_file() and f.suffix not in (".safetensors",):
        shutil.copy2(f, OUT / f.name)

vq_meta = {}
skipped = []
for si, sh in enumerate(shards, 1):
    # Materialize the shard ON THE CPU STREAM and force it here. Tensors we
    # pass through untouched (every non-.codes key, and every byte-aligned
    # or NSUB-skipped codes tensor) go into out_data STILL LAZY, so without
    # this the only pending work at mx.save_safetensors is a read of SRC --
    # and it is then paid inside a GPU command buffer. Over a fast local
    # disk that finishes; over SMB it trips the Metal watchdog and the pack
    # dies at the write step with a GPU Timeout that names the save, not the
    # read. Measured 2026-08-21: M4 packing the share died on shard 1 (K512)
    # and shard 5 (K256, where n_packed=0 makes EVERY tensor a lazy
    # passthrough); same artifacts pack clean on M3's local disk. The eval
    # is LOAD-BEARING -- creation-binding to the CPU stream alone is
    # measured-insufficient. Same family as vq_397b_codes.py's load path.
    with mx.stream(mx.cpu):
        data = mx.load(str(SRC / sh))
        mx.eval(list(data.values()))
    out_data = {}
    n_packed = 0
    for key, val in data.items():
        if not key.endswith(".codes"):
            out_data[key] = val
            continue
        mod = key[:-len(".codes")]
        k = data[mod + ".codebook"].shape[0]
        dim = data[mod + ".codebook"].shape[1]
        bits = vq_pack.bits_for_k(k)
        codes = np.array(val, copy=False)
        nsub = codes.shape[2]
        # The block layout needs NSUB % 32 == 0 (vq_pack's BLOCK). Qwen shapes
        # always satisfy it; gemma-4-26b's down_proj does not (NSUB=176 at
        # d=4). Such a tensor is COPIED THROUGH UNPACKED rather than fataling:
        # mixed artifacts are safe by construction, because add_model_file
        # dispatches on codes.dtype (uint32 = packed) and so reads a skipped
        # tensor by the original path. Absent pack_bits IS the unpacked signal.
        if nsub % vq_pack.BLOCK:
            out_data[key] = val
            skipped.append((mod, nsub))
            continue
        # BYTE-ALIGNED skip (08-20, found by the 397B session's A/B): when
        # bits %% 8 == 0, packing saves ZERO bytes (32 codes x 8 bits = 32
        # bytes either way) but routes the tensor through the packed
        # kernel's bit-field extraction — measured 37%% decode tax on the
        # cheap-shallow 397B, whose 141 K256 tensors were "packed" at 8
        # bits for nothing. Same pass-through mechanism as the NSUB skip:
        # absent pack_bits IS the unpacked signal, so mixed artifacts stay
        # safe by construction.
        if bits % 8 == 0:
            out_data[key] = val
            continue
        packed = vq_pack.pack(codes.astype(np.uint16), bits)
        # verify THIS tensor round-trips before we let it out of the process:
        # a silent packing error decodes to plausible garbage, not an error.
        back = vq_pack.unpack(packed, nsub, bits)
        if not np.array_equal(back.astype(np.uint16), codes.astype(np.uint16)):
            sys.exit(f"FATAL: {mod} did not round-trip through the packer")
        out_data[key] = mx.array(packed)
        vq_meta[mod] = {"pack_bits": bits, "in": nsub * dim}
        n_packed += 1
    mx.save_safetensors(str(OUT / sh), out_data, metadata={"format": "mlx"})
    print(f"[{si}/{len(shards)}] {sh}: packed {n_packed} code tensors",
          flush=True)
    del data, out_data

# The index must be REWRITTEN, not copied: metadata.total_size describes the
# SOURCE (unpacked) tensors, and copying it verbatim makes a packed artifact
# declare a size it does not have. Measured 2026-08-21: flatk2048-refit-packed
# claimed 197.12 GiB for 143.68 GiB (+37%), flatk512-packed +61%. exo reads
# this field to size the model and REFUSED to place the flagship — "No cycles
# found with sufficient memory" — and any downloader would be misled the same
# way. Recomputed below from the packed shards' own headers.
_idx = json.load(open(idx_path))
_total = 0
for _sh in sorted(set(_idx["weight_map"].values())):
    with open(OUT / _sh, "rb") as _f:
        _n = struct.unpack("<Q", _f.read(8))[0]
        _hdr = json.loads(_f.read(_n))
    for _name, _meta in _hdr.items():
        if _name != "__metadata__":
            _s, _e = _meta["data_offsets"]
            _total += _e - _s
_idx.setdefault("metadata", {})["total_size"] = _total
json.dump(_idx, open(OUT / idx_path.name, "w"), indent=1)
print(f"index total_size recomputed from packed shards: {_total} "
      f"({_total / 2**30:.2f} GiB)")

# seed config with pack_bits/in so add_model_file can read them back (it
# refuses to guess a bit width, by design)
cfg = json.load(open(OUT / "config.json"))

# ---- vision keys.
# The packer copies the fit output's config verbatim, and the fit output never
# had vision_config: the FIT's base (struct6-tail3x3) has no such key, and
# mlx_lm.convert strips it for text-only artifacts. Result until 2026-08-21:
# every chain-built 397B artifact grafted all 333 vision tensors and still
# shipped a config exo cannot build a VisionCardConfig from. graft_vision.py
# had a flag for it that no chain ever passed. Fixed in both places — here so
# a packed artifact is correct even if nobody runs the graft, and there so the
# graft repairs one that slipped through.
VISION_KEYS = ("vision_config", "image_token_id")
_missing = [k for k in VISION_KEYS if k not in cfg]
if _missing:
    _vsrc = pathlib.Path(args.vision_config_from) if args.vision_config_from else None
    if _vsrc and (_vsrc / "config.json").exists():
        _vc = json.load(open(_vsrc / "config.json"))
        _copied = [k for k in _missing if k in _vc]
        for k in _copied:
            cfg[k] = _vc[k]
        if _copied:
            print(f"copied vision keys from {_vsrc.name}: {_copied}")
        if [k for k in _missing if k not in _vc]:
            print(f"WARNING: {[k for k in _missing if k not in _vc]} absent "
                  f"from {_vsrc.name} too — artifact will not be exo vision-ready.")
    else:
        print(f"WARNING: config lacks {_missing} and no --vision-config-from "
              f"given. exo will not build a VisionCardConfig. Re-run with "
              f"--vision-config-from <bf16 source>, or run graft_vision.py "
              f"(its --copy-config-keys now defaults ON).")
merged = cfg.get("vq_modules", {})
for mod, meta in vq_meta.items():
    merged.setdefault(mod, {}).update(meta)
cfg["vq_modules"] = merged
json.dump(cfg, open(OUT / "config.json", "w"), indent=1)

subprocess.run([sys.executable, str(pathlib.Path(__file__).parent /
                                    "add_model_file.py"),
                "--artifact", str(OUT), "--group", str(args.group)], check=True)

src_gib = sum(f.stat().st_size for f in SRC.glob("*.safetensors")) / 1024 ** 3
out_gib = sum(f.stat().st_size for f in OUT.glob("*.safetensors")) / 1024 ** 3
print(f"\npacked {len(vq_meta)} tensors: {src_gib:.1f} -> {out_gib:.1f} GiB "
      f"({out_gib / src_gib:.3f}x)")
if skipped:
    # Loud, because these tensors keep paying full uint16 and are the reason
    # the achieved size misses the analytic target.
    shapes = sorted({n for _, n in skipped})
    print(f"NOT PACKED: {len(skipped)} tensors with NSUB % {vq_pack.BLOCK} "
          f"!= 0 (NSUB {shapes}) — copied through unpacked, still uint16.")
    for mod, n in skipped[:3]:
        print(f"  e.g. {mod} (NSUB={n})")
