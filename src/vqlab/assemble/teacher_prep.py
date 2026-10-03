#!/usr/bin/env python3
"""teacher-prep: an official release -> an exact MLX-layout teacher, one command.

    vqlab teacher-prep --src <downloaded release dir> --out <teacher dir> [--layers 0-0]

Steps (each refuses loudly rather than guessing):
1. Guard: the source must be a real download (`hf download --local-dir`), never
   the HF cache. Cache blobs are content-addressed and shared; rewriting a
   header there corrupts every snapshot that points at it.
2. Relabel F8_E8M0 as U8 in the safetensors headers, in place: bytes untouched,
   header length unchanged (JSON whitespace pad), idempotent. mx.load refuses
   F8_E8M0; as U8 the E8M0 exponents load and the family sanitize decodes them.
3. Disk preflight: the output volume must hold about the source size again.
4. sanitize-stream (one layer at a time through the runtime's Model.sanitize;
   tensors the runtime model has no parameter for are dropped AND listed).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import struct
import sys

from vqlab.assemble import sanitize_stream


def relabel_e8m0(src: pathlib.Path) -> int:
    n_t = 0
    for p in sorted(src.glob("*.safetensors")):
        with open(p, "r+b") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
            hit = [k for k, v in h.items()
                   if k != "__metadata__" and v["dtype"] == "F8_E8M0"]
            if not hit:
                continue
            for k in hit:
                h[k]["dtype"] = "U8"
            new = json.dumps(h, separators=(",", ":")).encode()
            if len(new) > n:
                raise SystemExit(f"{p.name}: relabelled header grew ({len(new)} > {n})")
            fh.seek(8)
            fh.write(new + b" " * (n - len(new)))
            n_t += len(hit)
    return n_t


def guard_source(src: pathlib.Path) -> None:
    if not (src / "config.json").exists():
        raise SystemExit(f"{src}: no config.json")
    if "/hub/models--" in str(src.resolve()) or any(
            p.is_symlink() for p in src.glob("*.safetensors")):
        raise SystemExit(f"{src}: looks like the HF cache (symlinked blobs). "
                         "Download with `hf download <repo> --local-dir <dir>` "
                         "and point --src there; cache blobs are never rewritten.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab teacher-prep")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", help="LO-HI subset (preflight); default all")
    ap.add_argument("--no-relabel", action="store_true")
    a, rest = ap.parse_known_args(argv)
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    guard_source(src)
    need = sum(p.stat().st_size for p in src.glob("*.safetensors"))
    if a.layers:
        need = 0
    out.mkdir(parents=True, exist_ok=True)
    have = shutil.disk_usage(out).free
    if have < need * 1.05:
        raise SystemExit(f"disk preflight: {out} has {have / 2**30:.0f} GiB free, "
                         f"needs ~{need * 1.05 / 2**30:.0f} GiB")
    if not a.no_relabel:
        print(f"relabelled {relabel_e8m0(src)} F8_E8M0 tensors as U8", flush=True)
    args = ["--src", str(src), "--out", str(out)] + (
        ["--layers", a.layers] if a.layers else []) + rest
    return sanitize_stream.main(args)


if __name__ == "__main__":
    sys.exit(main())
