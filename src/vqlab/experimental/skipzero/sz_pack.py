"""sz-pack (EXPERIMENTAL): write a vq-skipzero copy of a VQ MoE artifact.

For every VQ switch module (config `vq_modules`), a row [expert, out_row] is
DEAD when every one of its vq_scales is < 6.2e-5 (the teacher's ~1e-29
weights; `vqlab zero-groups` prices them). sz-pack drops the dead rows' packed
code words and scales and writes, per module, the live rows plus a bit-packed
[E, OUT] row mask (format: skipzero_load.py). Shards with no dead rows are
SYMLINKED; every other file is symlinked except config.json (gets a
`vq_skipzero` block), model.py (source bytes + an appended expansion hook) and
the index. Reads raw bytes with numpy only: no mlx, no GPU.

    vqlab sz-pack <artifact> --out <scratch>/<new>
                  [--layers A-B] [--dry-run] [--json plan.json]

--dry-run reads only vq_scales and prints the exact packed size; writes nothing.
The output directory must NOT exist and must be under vqlab-scratch/.
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

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401
from vqlab import config  # noqa: E402
import fitstore  # noqa: E402
import registry  # noqa: E402
import provenance  # noqa: E402

sys.path.insert(0, str(HERE))
import skipzero_load  # noqa: E402

TINY = 6.2e-5                       # same threshold as plan/zero_groups.py
GIB = 2 ** 30
ESIZE = {"U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
         "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8, "BOOL": 1}
SKIP_FILES = {".DS_Store", "config.json", "model.py", "model.safetensors.index.json"}
_LAYER = re.compile(r"\.layers\.(\d+)\.")


def _log(*a):
    print("[sz-pack]", *a, flush=True)


def abs_scales(raw, dtype):
    if dtype == "BF16":
        return np.abs((np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32))
    return np.abs(np.frombuffer(raw, {"F16": np.float16, "F32": np.float32}[dtype]).astype(np.float32))


def parse_layers(s):
    if not s:
        return None
    a, _, b = s.partition("-")
    return int(a), int(b or a)


def in_layers(name, rng):
    if rng is None:
        return True
    m = _LAYER.search(name)
    return bool(m) and rng[0] <= int(m.group(1)) <= rng[1]


def _nbytes(t):
    a, b = t["data_offsets"]
    return b - a


def plan_shard(path, specs, rng, tiny=TINY, skipped=None):
    """Which modules in this shard have dead rows, and their live masks.

    Modules whose codebook dim the runtime cannot serve compact are left
    untouched and recorded in `skipped` {module: dim}."""
    skipped = {} if skipped is None else skipped
    h, base = fitstore.read_header(path)
    mods = {}
    for key in h:
        if not key.endswith(".vq_scales"):
            continue
        p = key[: -len(".vq_scales")]
        sp = specs.get(p)
        if sp is None or sp.get("kind") != "expert" or not in_layers(p, rng):
            continue
        if not skipzero_load.servable(sp):
            d = sp.get("dim")                # no skipzero kernel serves this dim
            skipped[p] = d if d not in skipzero_load.SUPPORTED_DIMS else f"{d} unpacked"
            continue
        if p + ".codes" not in h:
            _log(f"WARN {p}: codes not in the same shard as scales; left unchanged")
            continue
        st, ct = h[key], h[p + ".codes"]
        if len(st["shape"]) != 3 or len(ct["shape"]) != 3 or st["shape"][:2] != ct["shape"][:2]:
            continue
        E, OUT, G = st["shape"]
        s = abs_scales(fitstore.read_tensor_bytes(path, h, base, key), st["dtype"]).reshape(E, OUT, G)
        live = ~(s < tiny).all(axis=-1)
        nd = int(live.size - live.sum())
        if nd == 0:
            continue
        rb_c, rb_s = _nbytes(ct) // (E * OUT), _nbytes(st) // (E * OUT)
        mask_b = E * ((OUT + 7) // 8)
        mods[p] = {"E": E, "OUT": OUT, "live": live, "dead_rows": nd,
                   "live_rows": int(live.sum()), "code_row_bytes": rb_c, "scale_row_bytes": rb_s,
                   "saved": nd * (rb_c + rb_s) - mask_b - 16,
                   "dead_experts": int((~live).all(axis=1).sum())}
    return h, base, mods


def build_tensors(h, mods):
    """Ordered [(name, dtype, shape, source)] for the rewritten shard."""
    out = []
    for key in sorted(h, key=lambda k: h[k]["data_offsets"][0]):
        t = h[key]
        p = key.rsplit(".", 1)[0]
        leaf = key.rsplit(".", 1)[-1]
        if p in mods and leaf in ("codes", "vq_scales"):
            m = mods[p]
            if leaf == "codes":
                out.append((p + ".sz_shape", "I32", [4], ("shape", p)))
                out.append((p + ".sz_rowmask", "U8", [m["E"], (m["OUT"] + 7) // 8], ("mask", p)))
                out.append((p + ".sz_codes", t["dtype"], [m["live_rows"], t["shape"][2]], ("rows", key)))
            else:
                out.append((p + ".sz_scales", t["dtype"], [m["live_rows"], t["shape"][2]], ("rows", key)))
        else:
            out.append((key, t["dtype"], t["shape"], ("copy", key)))
    return out


def _size(dtype, shape):
    n = ESIZE[dtype]
    for s in shape:
        n *= s
    return n


def header_bytes(tensors, metadata):
    hdr, off = {"__metadata__": metadata}, 0
    for name, dt, shape, _ in tensors:
        n = _size(dt, shape)
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    js = json.dumps(hdr, separators=(",", ":")).encode()
    js += b" " * ((8 - len(js) % 8) % 8)
    return struct.pack("<Q", len(js)) + js, off


def write_shard(src, dst, h, base, mods, tensors, metadata):
    hb, total = header_bytes(tensors, metadata)
    with open(src, "rb") as fi, open(dst, "xb") as fo:     # "x": never overwrite
        fo.write(hb)
        for name, dt, shape, (kind, ref) in tensors:
            if kind == "copy":
                a, b = h[ref]["data_offsets"]
                fi.seek(base + a)
                left = b - a
                while left:
                    buf = fi.read(min(left, 256 << 20))
                    fo.write(buf)
                    left -= len(buf)
            elif kind == "shape":
                m = mods[ref]
                w = h[ref + ".codes"]["shape"][2]
                g = h[ref + ".vq_scales"]["shape"][2]
                fo.write(np.array([m["E"], m["OUT"], w, g], np.int32).tobytes())
            elif kind == "mask":
                fo.write(np.packbits(mods[ref]["live"], axis=-1, bitorder="little").tobytes())
            else:  # rows
                p = ref.rsplit(".", 1)[0]
                m = mods[p]
                raw = np.frombuffer(fitstore.read_tensor_bytes(src, h, base, ref), np.uint8)
                rows = raw.reshape(m["E"] * m["OUT"], -1)[m["live"].reshape(-1)]
                fo.write(np.ascontiguousarray(rows).tobytes())
                del raw, rows
        assert fo.tell() == len(hb) + total, (dst, fo.tell(), len(hb) + total)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab sz-pack", description=__doc__.split("\n\n")[0])
    ap.add_argument("artifact")
    ap.add_argument("--out", help="new dir under vqlab-scratch/ (must not exist)")
    ap.add_argument("--layers", help="only modules in layers A-B (inclusive); others unchanged")
    ap.add_argument("--dry-run", action="store_true", help="price only: read scales, write nothing")
    ap.add_argument("--tiny", type=float, default=TINY)
    ap.add_argument("--json", help="write the per-module plan here")
    ap.add_argument("--allow-any-out", action="store_true",
                    help="selftest only: permit --out outside vqlab-scratch/")
    a = ap.parse_args(argv)
    src = pathlib.Path(a.artifact).resolve()
    rng = parse_layers(a.layers)
    cfg = json.loads((src / "config.json").read_text())
    if "vq_skipzero" in cfg:
        ap.error("source is already vq-skipzero")
    specs = registry.vq_specs(cfg)
    shards = sorted(src.glob("*.safetensors"))

    plans, text_orig, text_new, skipped = {}, 0, 0, {}
    for f in shards:
        h, base, mods = plan_shard(f, specs, rng, a.tiny, skipped)
        size = os.path.getsize(os.path.realpath(f))
        text_orig += size
        if mods:
            hb, total = header_bytes(build_tensors(h, mods), {"format": "mlx"})
            new = len(hb) + total
        else:
            new = size
        text_new += new
        plans[f.name] = (h, base, mods, size, new)
        if mods:
            _log(f"{f.name}: {len(mods)} modules, {size / GIB:.3f} -> {new / GIB:.3f} GiB")
    if skipped:
        by_dim = {}
        for d in skipped.values():
            by_dim[d] = by_dim.get(d, 0) + 1
        _log(f"skipped {len(skipped)} module(s) left unpacked: the runtime serves "
             f"skipzero only at dim {list(skipzero_load.SUPPORTED_DIMS)} "
             f"(d{skipzero_load.PACKED_ONLY_DIMS} packed only) "
             f"(found {', '.join(f'dim={d}: {n}' for d, n in sorted(by_dim.items(), key=str))}), "
             f"e.g. {sorted(skipped)[0]}")
    rewrite = {k: v for k, v in plans.items() if v[2]}
    new_bytes = sum(v[4] for v in rewrite.values())
    allm = {p: m for v in plans.values() for p, m in v[2].items()}
    dead = sum(m["dead_rows"] for m in allm.values())
    rows = sum(m["E"] * m["OUT"] for m in allm.values())
    summary = {"artifact": str(src), "layers": a.layers, "tiny": a.tiny,
               "modules_packed": len(allm), "dead_rows": dead, "rows_in_packed_modules": rows,
               "dead_experts": sum(m["dead_experts"] for m in allm.values()),
               "shards_rewritten": len(rewrite), "shards_symlinked": len(plans) - len(rewrite),
               "text_bytes_orig": text_orig, "text_bytes_packed": text_new,
               "saved_bytes": text_orig - text_new,
               "saved_pct": 100 * (text_orig - text_new) / max(text_orig, 1),
               "new_bytes_written": new_bytes,
               "modules_skipped_unsupported_dim": len(skipped),
               "skip_reason": (f"runtime skipzero serves dim {list(skipzero_load.SUPPORTED_DIMS)} only"
                               if skipped else None)}
    _log(f"text {text_orig / GIB:.3f} -> {text_new / GIB:.3f} GiB, saved "
         f"{summary['saved_bytes'] / GIB:.3f} GiB ({summary['saved_pct']:.2f}%); "
         f"{len(allm)} modules, {dead} dead rows; rewrites {len(rewrite)} shards = "
         f"{new_bytes / GIB:.3f} GiB of NEW bytes")
    if not allm:
        reason = (f"; {len(skipped)} module(s) skipped for an unsupported dim"
                  if skipped else "")
        print(f"[sz-pack] REFUSED: no module qualifies for skipzero (needs dim in "
              f"{list(skipzero_load.SUPPORTED_DIMS)} and at least one dead row{reason})",
              file=sys.stderr)
        return 1
    if a.json:
        json.dump({**summary, "modules": {p: {k: v for k, v in m.items() if k != "live"}
                                          for p, m in allm.items()}},
                  open(a.json, "w"), indent=1)
    if a.dry_run:
        print(json.dumps(summary, indent=1))
        return 0

    if not a.out:
        ap.error("--out is required unless --dry-run")
    out = pathlib.Path(os.path.abspath(a.out))
    if out.exists():
        ap.error(f"{out} exists; sz-pack never overwrites")
    if not a.allow_any_out:
        config.require_storage(out.parent)
    if str(out).startswith(str(src) + os.sep):
        ap.error("--out may not be inside the source artifact")
    free = shutil.disk_usage(out.parent).free
    if new_bytes + 5 * GIB > free:
        ap.error(f"needs {new_bytes / GIB:.1f} GiB + 5 GiB margin, only {free / GIB:.1f} GiB free")
    out.mkdir(parents=False)

    wmap = {}
    for name, (h, base, mods, _, _) in plans.items():
        dst = out / name
        if mods:
            _log(f"writing {name}")
            tens = build_tensors(h, mods)
            write_shard(src / name, dst, h, base, mods, tens, {"format": "mlx"})
            keys = [t[0] for t in tens]
        else:
            os.symlink(os.path.realpath(src / name), dst)
            keys = list(h)
        for k in keys:
            wmap[k] = name
    for f in sorted(src.iterdir()):
        if f.is_dir() or f.name in SKIP_FILES or f.name.endswith(".safetensors") \
                or f.name.startswith("vqlab_provenance"):
            continue
        os.symlink(os.path.realpath(f), out / f.name)
    json.dump({"metadata": {"total_size": text_new}, "weight_map": dict(sorted(wmap.items()))},
              open(out / "model.safetensors.index.json", "x"), indent=2)
    shutil.copyfile(HERE / "skipzero_load.py", out / "skipzero_load.py")
    mf = cfg.get("model_file") or "model.py"
    src_model = (src / mf).read_bytes()
    with open(out / mf, "xb") as fo:
        fo.write(src_model + skipzero_load.MODEL_HOOK.encode())
    cfg["vq_skipzero"] = {
        "format": skipzero_load.FORMAT, "version": skipzero_load.VERSION,
        "experimental": True,
        "note": "EXPERIMENTAL, not a shipped format. Dead VQ rows dropped on disk; "
                "skipzero_load.py re-expands at load. RAM is unchanged (stage 1).",
        "tiny": a.tiny, "layers": a.layers, "source": str(src),
        "source_model_py_bytes": len(src_model),
        "modules": {p: {"experts": m["E"], "out": m["OUT"], "live_rows": m["live_rows"],
                        "dead_rows": m["dead_rows"]} for p, m in sorted(allm.items())},
        "saved_bytes": summary["saved_bytes"],
    }
    with open(out / "config.json", "x") as fo:
        json.dump(cfg, fo, indent=2)
    provenance.write_build_record(
        out, tool="sz-pack", script=__file__, ap=ap, args=a,
        method={"format": "vq-skipzero", "version": skipzero_load.VERSION,
                "experimental": True, "tiny": a.tiny, "layers": a.layers,
                "dead_row_rule": "all vq_scales of [expert, out_row] < tiny"},
        inputs=[("source", str(src))],
        modules={p: {"origin": "skipzero", "dead_rows": m["dead_rows"],
                     "live_rows": m["live_rows"], "saved_bytes": m["saved"]}
                 for p, m in sorted(allm.items())},
        full_hash={"config.json", mf, "skipzero_load.py", "model.safetensors.index.json"})
    print(json.dumps(summary, indent=1))
    _log(f"DONE -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
