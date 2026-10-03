#!/usr/bin/env python3
"""slice: a small REAL slice of any teacher, for preflighting a writer.

    vqlab slice <teacher> --layers 0-3 --out <dir> [--copy] [--with-extras]

The 4-layer DeepSeek slice (night-20261002/ds4-teacher-4L) caught two
stream-convert bugs in minutes that would each have wasted a 30-minute
conversion (lab/docs/OPERATOR-NOTES-2026-10-03.md, sec 4). This makes that
slice a tool for every family: run the new step on the slice first.

The slice holds the chosen layers' tensors plus every top-level TEXT tensor
(embeddings, final norm, head, DeepSeek's hc_head), renumbered 0..n-1, with
an index written by `core.artifact.write_index` and a config whose
`num_hidden_layers` (top level and `text_config`) is n. Per-layer config
lists (DeepSeek `compress_ratios`, Qwen `layer_types`, ...) keep the chosen
entries, plus any trailing entries past num_hidden_layers (an MTP layer's);
per-module config maps (`quantization`, `vq_modules`, ...) follow the
renumbering. Vision tower and MTP tensors are left out unless --with-extras.

Bytes: a source shard whose tensors ALL go into the slice, unrenamed, is
symlinked (or APFS-cloned with --copy); any other shard contributes a new
shard holding byte copies of just the selected tensors (header entries
verbatim, any dtype, no mlx, no GPU). Only those tensors' byte ranges are
read.

Refuses an --out outside the configured storage, on the system disk (set
VQLAB_ALLOW_SYSTEM_DISK=1 for a deliberately tiny slice there), or on a
volume without room for the copies.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import struct
import sys

from vqlab import config as C
from vqlab.core.artifact import (Artifact, CONFIG, layer_of, place, read_header,
                                 tensor_class, write_config, write_index)

_LAYER_RX = re.compile(r"((?:^|\.)layers\.)(\d+)(\.)")
MAP_KEYS = ("quantization", "quantization_config", "vq_modules", "vq_skipzero")
CHUNK = 1 << 26


def parse_layers(spec: str, n: int) -> list:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    out = sorted(set(out))
    if not out or out[0] < 0 or out[-1] >= n:
        raise SystemExit(f"--layers {spec}: the teacher has layers 0-{n - 1}")
    return out


def rename(key: str, remap: dict) -> str:
    """layers.{old} -> layers.{new} (first occurrence: the decoder layer)."""
    return _LAYER_RX.sub(lambda m: f"{m.group(1)}{remap[int(m.group(2))]}{m.group(3)}",
                         key, count=1)


def _text_cfgs(cfg: dict):
    """The config dicts that carry num_hidden_layers (top level, text_config)."""
    out = [cfg] if "num_hidden_layers" in cfg else []
    tc = cfg.get("text_config")
    if isinstance(tc, dict) and "num_hidden_layers" in tc:
        out.append(tc)
    return out


def slice_config(cfg: dict, layers: list, remap: dict) -> tuple:
    """-> (new config, [changes]); config edits for a slice of `layers`."""
    cfg = json.loads(json.dumps(cfg))
    changes = []
    for c in _text_cfgs(cfg):
        n = int(c["num_hidden_layers"])
        extra = int(c.get("num_nextn_predict_layers") or c.get("mtp_num_hidden_layers") or 0)
        for k, v in list(c.items()):
            if isinstance(v, list) and len(v) in (n, n + extra) and k != "architectures":
                c[k] = [v[i] for i in layers] + v[n:]
                changes.append(f"{k}: {len(v)} -> {len(c[k])} entries")
        c["num_hidden_layers"] = len(layers)
        changes.append(f"num_hidden_layers: {n} -> {len(layers)}")
    for name in MAP_KEYS:
        m = cfg.get(name)
        if not isinstance(m, dict):
            continue
        new = {}
        for k, v in m.items():
            if not isinstance(v, dict):
                new[k] = v                   # group_size / bits / mode / ...
                continue
            li = layer_of(k)
            if li == -1:
                new[k] = v
            elif li in remap:
                new[rename(k, remap)] = v
        if len(new) != len(m):
            changes.append(f"{name}: {len(m)} -> {len(new)} entries")
        cfg[name] = new
    return cfg, changes


def write_subshard(src: pathlib.Path, dst: pathlib.Path, names: dict) -> int:
    """New safetensors file at `dst` holding byte copies of `names` ({old:
    new}) from `src`. Header entries (dtype, shape) are copied verbatim, so
    any dtype the source carries survives. Returns data bytes written."""
    with open(src, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    md = hdr.pop("__metadata__", None)
    order = sorted(names, key=lambda k: hdr[k]["data_offsets"][0])
    out_hdr, off = {}, 0
    if md is not None:
        out_hdr["__metadata__"] = md
    for k in order:
        a, b = hdr[k]["data_offsets"]
        out_hdr[names[k]] = {"dtype": hdr[k]["dtype"], "shape": hdr[k]["shape"],
                             "data_offsets": [off, off + b - a]}
        off += b - a
    blob = json.dumps(out_hdr, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        for k in order:
            a, b = hdr[k]["data_offsets"]
            fi.seek(8 + n + a)
            left = b - a
            while left:
                buf = fi.read(min(CHUNK, left))
                if not buf:
                    raise SystemExit(f"{src}: truncated reading {k}")
                fo.write(buf)
                left -= len(buf)
    os.replace(tmp, dst)
    return off


def _on_system_disk(p: pathlib.Path) -> bool:
    while not p.exists() and p != p.parent:
        p = p.parent
    return os.stat(p).st_dev == os.stat("/").st_dev


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab slice", description=__doc__.split("\n")[0])
    ap.add_argument("teacher", help="teacher dir, or a name under the configured teachers root")
    ap.add_argument("--layers", required=True, help="LO-HI or a comma list, e.g. 0-3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", action="store_true",
                    help="clone whole shards (cp -c) instead of symlinking them")
    ap.add_argument("--with-extras", action="store_true",
                    help="also carry vision-tower and MTP tensors (unchanged)")
    a = ap.parse_args(argv)

    src = pathlib.Path(a.teacher)
    if not (src / CONFIG).exists() and (C.teachers() / a.teacher / CONFIG).exists():
        src = C.teachers() / a.teacher
    t = Artifact.open(src)
    out = pathlib.Path(a.out)
    C.require_storage(out)
    if _on_system_disk(out) and os.environ.get("VQLAB_ALLOW_SYSTEM_DISK") != "1":
        raise SystemExit(f"REFUSED: {out} is on the system disk; artifacts never go there "
                         "(VQLAB_ALLOW_SYSTEM_DISK=1 overrides for a deliberately tiny slice)")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"REFUSED: {out} exists and is not empty")

    tcs = _text_cfgs(t.config)
    if not tcs:
        raise SystemExit(f"{src}: config has no num_hidden_layers (top level or text_config)")
    n = int(tcs[-1]["num_hidden_layers"])
    layers = parse_layers(a.layers, n)
    remap = {old: new for new, old in enumerate(layers)}

    # tensor -> its name in the slice (None: left out)
    pick = {}
    for k in t.index:
        cls = tensor_class(k)
        if cls != "text":
            pick[k] = k if a.with_extras else None
        elif (li := layer_of(k)) == -1:
            pick[k] = k
        else:
            pick[k] = rename(k, remap) if li in remap else None

    by_shard = {}
    for k, f in t.index.items():
        by_shard.setdefault(f, []).append(k)
    whole, partial = [], {}
    for f, keys in sorted(by_shard.items()):
        chosen = {k: pick[k] for k in keys if pick[k] is not None}
        if not chosen:
            continue
        if len(chosen) == len(keys) and all(k == v for k, v in chosen.items()):
            whole.append(f)
        else:
            partial[f] = chosen

    need = 0
    for f, chosen in partial.items():
        h = t.header(f)
        need += sum(h[k]["data_offsets"][1] - h[k]["data_offsets"][0] for k in chosen)
    if a.copy:
        need += sum(os.path.getsize(t.shard_path(f)) for f in whole)
    C.require_free(out, need, "slice", margin_gib=1.0)

    out.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for f in whole:
        place(t.shard_path(f), out / f, copy=a.copy)
        weight_map.update({k: f for k in by_shard[f]})
    for f, chosen in partial.items():
        write_subshard(pathlib.Path(t.shard_path(f)), out / f, chosen)
        weight_map.update({v: f for v in chosen.values()})
    write_index(out, dict(sorted(weight_map.items())))
    cfg, changes = slice_config(t.config, layers, remap)
    write_config(out, cfg)
    for f in t.other_files():
        shutil.copy2(f, out)

    got = {layer_of(k) for k in weight_map if tensor_class(k) == "text"} - {-1}
    if got != set(range(len(layers))):
        raise SystemExit(f"FAIL: slice holds layers {sorted(got)}, expected 0-{len(layers) - 1}")
    for f in sorted({*whole, *partial}):
        read_header(out / f)                       # every shard parses
    print(f"{out}: layers {layers[0]}-{layers[-1]} of {n} -> 0-{len(layers) - 1}; "
          f"{len(weight_map)} tensors, {len(whole)} shard(s) "
          f"{'cloned' if a.copy else 'symlinked'}, {len(partial)} written "
          f"({need / 2**20:.1f} MiB copied)")
    for c in changes:
        print(f"  config {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
