#!/usr/bin/env python3
"""graft-extras: copy tensors a text-only VQ artifact lacks (vision tower,
aligner, image tokens, image-routing gate biases) from the official release,
byte for byte, under the names the serving runtime expects.

    vqlab graft-extras --artifact <VQ artifact> --src <official release dir>
                       --preset deepseek-v4-vision [--dry-run]

`vqlab graft` (Qwen/GLM/gemma towers) proves the source is the artifact's
base by finding byte-identical tensors under a SHARED name; an official
release and its MLX conversion share no names, so that probe cannot run.
This tool probes through the preset's own name map instead (norms and the
hash-routing tables pass through conversion unchanged), and refuses unless
every probed tensor is byte-identical.

Bytes are copied raw (no MLX, no dtype change) into one new shard,
`model-extras-graft.safetensors`, registered in the index. Idempotent:
re-running replaces that shard and its index entries. Config keys the
runtime needs are copied from --src only when the artifact lacks them.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
import sys

from vqlab.core.artifact import Artifact, write_index

SHARD = "model-extras-graft.safetensors"

# Each preset: (select regex, rename (pattern, repl)) rules, identity probe
# map (src key -> artifact key), config keys the runtime needs.
PRESETS = {
    # Layout agreed with Knurlogic's Vision-Exp loader (2026-10-02):
    # tower/aligner keep HF names; image tokens as model.image_*; every layer
    # gets ffn.gate.bias_vl (fp32 [256]); hash layers 0-2 also get their HF
    # gate bias as e_score_correction_bias (image tokens route by bias there).
    "deepseek-v4-vision": {
        "rules": [
            (r"^(vision\..*|aligner\..*)$", r"\1"),
            (r"^(image_(start|end|newline|pad))$", r"model.\1"),
            (r"^layers\.(\d+)\.ffn\.gate\.bias_vl$", r"model.layers.\1.ffn.gate.bias_vl"),
            (r"^layers\.([0-2])\.ffn\.gate\.bias$", r"model.layers.\1.ffn.gate.e_score_correction_bias"),
        ],
        "probe": (r"^layers\.(\d+)\.(attn_norm\.weight|ffn_norm\.weight|ffn\.gate\.tid2eid)$",
                  r"model.layers.\1.\2"),
        "config_keys": ["vision_n_layers", "vision_dim", "vision_n_heads", "vision_inter_dim",
                        "vision_patch_size", "vision_rope_theta", "vision_downsample_ratio",
                        "vision_max_n_token", "vision_min_pixels", "vision_max_wh_ratio"],
    },
}


def _raw(path, key, hdr):
    h, start = hdr
    a, b = h[key]["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(start + a)
        return fh.read(b - a)


def _hdr(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def plan(art: Artifact, src: pathlib.Path, preset: dict):
    sw = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    pick = {}
    for k in sw:
        for pat, repl in preset["rules"]:
            if re.match(pat, k):
                pick[k] = re.sub(pat, repl, k)
                break
    probe = {}
    pat, repl = preset["probe"]
    for k in sw:
        if re.match(pat, k):
            t = re.sub(pat, repl, k)
            if t in art.index:
                probe[k] = t
    return sw, pick, probe


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab graft-extras", description=__doc__.split("\n")[0])
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    art, src, preset = Artifact.open(a.artifact), pathlib.Path(a.src), PRESETS[a.preset]
    sw, pick, probe = plan(art, src, preset)
    if not pick:
        raise SystemExit(f"{src}: nothing matches preset {a.preset}")
    if len(probe) < 3:
        raise SystemExit(f"identity probe found only {len(probe)} mapped tensors; "
                         "the preset's names do not fit this artifact. Nothing written.")
    hdrs = {}
    def hdr(d, f):
        if (d, f) not in hdrs:
            hdrs[(d, f)] = _hdr(d / f)
        return hdrs[(d, f)]
    # probe: a spread of norms + routing tables, every one byte-identical
    keys = sorted(probe)
    sample = keys[:: max(1, len(keys) // 12)][:12]
    bad = [k for k in sample
           if _raw(src / sw[k], k, hdr(src, sw[k]))
           != _raw(art.dir / art.index[probe[k]], probe[k], hdr(art.dir, art.index[probe[k]]))]
    if bad:
        raise SystemExit(f"FAIL: {len(bad)}/{len(sample)} probed tensors differ between --src and "
                         f"the artifact (e.g. {bad[0]} vs {probe[bad[0]]}): not this artifact's base. "
                         "Nothing written.")
    print(f"base-identity OK: {len(sample)}/{len(sample)} probed tensors byte-identical "
          f"(e.g. {sample[0]} == {probe[sample[0]]})")
    clash = [t for t in pick.values() if t in art.index and art.index[t] != SHARD]
    if clash:
        raise SystemExit(f"FAIL: {len(clash)} target names already exist in the artifact "
                         f"outside {SHARD}, e.g. {clash[0]}. Nothing written.")
    nbytes = 0
    for k in pick:
        h, _ = hdr(src, sw[k])
        s, e = h[k]["data_offsets"]
        nbytes += e - s
    cfg_add = {k: v for k, v in json.load(open(src / "config.json")).items()
               if k in preset["config_keys"] and k not in art.config}
    print(f"{len(pick)} tensors, {nbytes / 2**30:.2f} GiB -> {SHARD}; "
          f"config keys added: {sorted(cfg_add) or 'none (already present)'}")
    if a.dry_run:
        return 0
    # write the shard from raw bytes, sorted by target name
    meta, blobs, off = {}, [], 0
    for k, t in sorted(pick.items(), key=lambda kv: kv[1]):
        h, _ = hdr(src, sw[k])
        b = _raw(src / sw[k], k, hdr(src, sw[k]))
        meta[t] = {"dtype": h[k]["dtype"], "shape": h[k]["shape"], "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    meta["__metadata__"] = {"format": "mlx", "grafted_from": src.name, "preset": a.preset}
    hb = json.dumps(meta).encode()
    hb += b" " * (-len(hb) % 8)
    tmp = art.dir / (SHARD + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)) + hb)
        for b in blobs:
            fh.write(b)
    tmp.rename(art.dir / SHARD)
    wm = {k: f for k, f in art.index.items() if f != SHARD}
    wm.update({t: SHARD for t in pick.values()})
    total = sum((art.dir / f).stat().st_size for f in set(wm.values()))
    write_index(art.dir, wm, total_size=total)
    if cfg_add:
        cfg = {**art.config, **cfg_add}
        (art.dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"grafted into {art.dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
