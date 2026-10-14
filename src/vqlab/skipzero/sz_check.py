"""sz-check (EXPERIMENTAL): prove a vq-skipzero artifact expands back exactly.

CPU / numpy only, no GPU. For every module in config `vq_skipzero.modules`:
expands the compact tensors with the PACKED artifact's own skipzero_load.py
(the file the loader will run), then checks against the original artifact:
  * every live row's codes and scales are byte-identical;
  * every dead row expands to code 0 / scale 0 (an exact-zero weight row), and
    the original's dead rows really were all < tiny;
  * every other tensor in a rewritten shard is byte-identical (skip with
    --skip-passthrough); symlinked shards must resolve to the original's file.
Reports bytes saved.

    vqlab sz-check <packed> <original> [--skip-passthrough]
    vqlab sz-check --selftest            # synthetic artifact, pack, expand, assert (CPU)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

from vqlab.core.source import fit_store as fitstore

from vqlab.skipzero import sz_pack
from vqlab.util.fs import GIB, write_json



def _load_shim(d):
    spec = importlib.util.spec_from_file_location("skipzero_load_art", pathlib.Path(d) / "skipzero_load.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _index(d):
    out = {}
    for f in sorted(pathlib.Path(d).glob("*.safetensors")):
        h, base = fitstore.read_header(f)
        for k in h:
            out[k] = (f, h, base)
    return out


def _arr(ix, key):
    f, h, base = ix[key]
    t = h[key]
    raw = fitstore.read_tensor_bytes(f, h, base, key)
    return raw, t


def _rows(raw, E, OUT):
    return np.frombuffer(raw, np.uint8).reshape(E * OUT, -1)


def check(packed, orig, skip_passthrough=False):
    packed, orig = pathlib.Path(packed), pathlib.Path(orig)
    cfg = json.loads((packed / "config.json").read_text())
    sz = cfg.get("vq_skipzero")
    assert sz and sz.get("format") == "vq-skipzero", "not a vq-skipzero artifact"
    shim = _load_shim(packed)
    pix, oix = _index(packed), _index(orig)
    fails, tiny = [], float(sz["tiny"])
    # An INCREMENTAL pack (sz-pack on a source that was already skipzero):
    # modules the original already holds as sz tensors are passthrough and
    # must match it tensor for tensor; only modules the original holds as
    # plain codes are expanded and compared row by row.
    prior = {p for p in sz["modules"] if p + ".sz_codes" in oix}
    new_mods = {p: m for p, m in sz["modules"].items() if p not in prior}
    for p, meta in new_mods.items():
        (shp_raw, _), (rm_raw, rmt) = _arr(pix, p + ".sz_shape"), _arr(pix, p + ".sz_rowmask")
        shape = np.frombuffer(shp_raw, np.int32)
        rmask = np.frombuffer(rm_raw, np.uint8).reshape(rmt["shape"])
        E, OUT = int(shape[0]), int(shape[1])
        live = shim.live_mask_np(rmask, E, OUT).reshape(-1)
        if int(live.sum()) != meta["live_rows"]:
            fails.append(f"{p}: live_rows {int(live.sum())} != config {meta['live_rows']}")
        for leaf, sleaf in (("codes", "sz_codes"), ("vq_scales", "sz_scales")):
            craw, ct = _arr(pix, f"{p}.{sleaf}")
            oraw, ot = _arr(oix, f"{p}.{leaf}")
            compact = np.frombuffer(craw, np.uint8).reshape(ct["shape"][0], -1)
            full = shim.expand_np(shape, rmask, compact).reshape(E * OUT, -1)
            o = _rows(oraw, E, OUT)
            if full.shape != o.shape:
                fails.append(f"{p}.{leaf}: shape {full.shape} vs {o.shape}")
                continue
            if ct["dtype"] != ot["dtype"]:
                fails.append(f"{p}.{leaf}: dtype {ct['dtype']} vs {ot['dtype']}")
            if not np.array_equal(full[live], o[live]):
                fails.append(f"{p}.{leaf}: live rows differ")
            if full[~live].any():
                fails.append(f"{p}.{leaf}: dead rows not zero")
            if leaf == "vq_scales":
                s = sz_pack.abs_scales(oraw, ot["dtype"]).reshape(E * OUT, -1)
                if not (s[~live] < tiny).all():
                    fails.append(f"{p}: a dropped row had a scale >= tiny")
                if (s[live] < tiny).all(axis=-1).any():
                    fails.append(f"{p}: a kept row was fully dead")
    # every other tensor
    sz_keys = {f"{p}.{s}" for p in new_mods for s in shim.SUFFIXES}
    dropped = {f"{p}.{s}" for p in new_mods for s in ("codes", "vq_scales")}
    missing = set(oix) - dropped - set(pix)
    extra = set(pix) - sz_keys - set(oix)
    if missing or extra:
        fails.append(f"key sets differ: missing {sorted(missing)[:5]} extra {sorted(extra)[:5]}")
    passthrough = 0
    for k, (f, h, base) in pix.items():
        if k in sz_keys:
            continue
        if f.is_symlink():
            if os.path.realpath(f) != os.path.realpath(oix[k][0]):
                fails.append(f"{k}: symlinked shard points elsewhere")
            continue
        if skip_passthrough:
            continue
        a, _ = _arr(pix, k)
        b, _ = _arr(oix, k)
        passthrough += 1
        if a != b:
            fails.append(f"{k}: passthrough tensor differs")
    size = lambda d: sum(os.path.getsize(os.path.realpath(f)) for f in pathlib.Path(d).glob("*.safetensors"))
    so, sp = size(orig), size(packed)
    rep = {"packed": str(packed), "original": str(orig), "modules": len(sz["modules"]),
           "modules_expanded_and_checked": len(new_mods), "modules_already_sz_in_original": len(prior),
           "passthrough_tensors_compared": passthrough,
           "text_bytes_orig": so, "text_bytes_packed": sp, "saved_bytes": so - sp,
           "saved_pct": 100 * (so - sp) / max(so, 1), "fails": fails,
           "verdict": "PASS" if not fails else "FAIL"}
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab sz-check", description=__doc__.split("\n\n")[0])
    ap.add_argument("packed", nargs="?")
    ap.add_argument("original", nargs="?")
    ap.add_argument("--skip-passthrough", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.selftest:
        # moved to vqlab.selftest (untangle step 14); `vqlab sz-check
        # --selftest` is routed there by the CLI (commands.SELFTEST_ROUTES)
        sys.exit("sz-check's self-check lives in vqlab.selftest.sz_check_fixture: "
                 "run `vqlab sz-check --selftest`")
    if not (a.packed and a.original):
        ap.error("packed and original are required")
    rep = check(a.packed, a.original, a.skip_passthrough)
    if a.json:
        write_json(a.json, rep, indent=1)
    print(json.dumps({k: (v[:20] if k == "fails" else v) for k, v in rep.items()}, indent=1))
    print(f"sz-check {rep['verdict']}: {rep['modules']} modules, saved "
          f"{rep['saved_bytes'] / GIB:.3f} GiB ({rep['saved_pct']:.2f}%)")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
