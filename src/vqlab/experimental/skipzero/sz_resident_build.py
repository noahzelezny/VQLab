"""sz-resident (EXPERIMENTAL): turn a stage-1 vq-skipzero pack into a STAGE-2
(compact-resident) artifact. Writes a NEW dir; never modifies the input.

    vqlab sz-resident <stage1_pack> --out <scratch>/<new>

The new dir symlinks every shard and file of the stage-1 pack (the on-disk
format is unchanged), and writes:
  * config.json  -- vq_skipzero gains resident=true, resident_version, and per
                    module code_words / scale_groups read from the shards'
                    sz_shape, so every compact shape is known before load;
  * model.py     -- the SOURCE bundle's model.py bytes (stage-1 hook stripped,
                    verified) + sz_resident.MODEL_HOOK;
  * sz_resident.py (the runtime fork) and skipzero_load.py.
Refuses modules that are not packed d4 (the only geometry forked).
Numpy/stdlib only; no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401
import fitstore  # noqa: E402
import provenance  # noqa: E402

sys.path.insert(0, str(HERE))
import skipzero_load  # noqa: E402
import sz_resident  # noqa: E402

SCRATCH = "<scratch>/"
OWN = {"config.json", "model.py", "sz_resident.py", "skipzero_load.py", ".DS_Store"}


def _log(*a):
    print("[sz-resident]", *a, flush=True)


def plan(src):
    cfg = json.loads((src / "config.json").read_text())
    sz = cfg.get("vq_skipzero")
    if not sz:
        raise SystemExit("not a vq-skipzero (stage-1) artifact: no vq_skipzero block")
    if sz.get("resident"):
        raise SystemExit("already resident")
    shapes = {}
    for f in sorted(src.glob("*.safetensors")):
        h, base = fitstore.read_header(f)
        for k in h:
            if k.endswith(".sz_shape"):
                p = k[: -len(".sz_shape")]
                shp = np.frombuffer(fitstore.read_tensor_bytes(f, h, base, k), np.int32)
                nl = h[p + ".sz_codes"]["shape"][0]
                if h[p + ".sz_codes"]["dtype"] != "U32" or h[p + ".sz_scales"]["dtype"] != "F16":
                    raise SystemExit(f"{p}: codes {h[p + '.sz_codes']['dtype']} / scales "
                                     f"{h[p + '.sz_scales']['dtype']}; only packed U32 + F16 forked")
                shapes[p] = (int(shp[0]), int(shp[1]), int(shp[2]), int(shp[3]), int(nl))
    mods = sz["modules"]
    if set(shapes) != set(mods):
        raise SystemExit(f"shard sz_ modules {len(shapes)} != config modules {len(mods)}")
    for p, (E, OUT, W, G, NL) in shapes.items():
        g, m = cfg["vq_modules"][p], mods[p]
        if g.get("dim") != 4 or not g.get("pack_bits"):
            raise SystemExit(f"{p}: d{g.get('dim')} pack_bits {g.get('pack_bits')}; only packed d4 forked")
        if (E, OUT, NL) != (m["experts"], m["out"], m["live_rows"]):
            raise SystemExit(f"{p}: shard shape {(E, OUT, NL)} != config {m}")
        if W != (g["in"] // 4 + 31) // 32 * g["pack_bits"] or G != g["in"] // g["group"]:
            raise SystemExit(f"{p}: W={W} G={G} inconsistent with vq_modules")
    return cfg, shapes


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab sz-resident", description=__doc__.split("\n\n")[0])
    ap.add_argument("pack", help="stage-1 vq-skipzero artifact (sz-pack output)")
    ap.add_argument("--out", required=True, help="new dir under vqlab-scratch/ (must not exist)")
    ap.add_argument("--allow-any-out", action="store_true", help="selftest only")
    a = ap.parse_args(argv)
    src = pathlib.Path(a.pack).resolve()
    out = pathlib.Path(os.path.abspath(a.out))
    if out.exists():
        ap.error(f"{out} exists; sz-resident never overwrites")
    if not a.allow_any_out and not str(out).startswith(SCRATCH):
        ap.error(f"--out must be under {SCRATCH}")
    cfg, shapes = plan(src)
    mf = cfg.get("model_file") or "model.py"
    stage1 = (src / mf).read_bytes()
    nsrc = cfg["vq_skipzero"]["source_model_py_bytes"]
    if stage1[nsrc:] != skipzero_load.MODEL_HOOK.encode():
        raise SystemExit("stage-1 model.py tail is not this repo's skipzero_load.MODEL_HOOK; "
                         "refusing to guess where the source bundle ends")
    out.mkdir(parents=False)
    for f in sorted(src.iterdir()):
        if f.is_dir() or f.name in OWN or f.name == mf or f.name.startswith("vqlab_provenance"):
            continue
        os.symlink(os.path.realpath(f), out / f.name)
    shutil.copyfile(HERE / "skipzero_load.py", out / "skipzero_load.py")
    shutil.copyfile(HERE / "sz_resident.py", out / "sz_resident.py")
    with open(out / mf, "xb") as fo:
        fo.write(stage1[:nsrc] + sz_resident.MODEL_HOOK.encode())
    sz = cfg["vq_skipzero"]
    sz["resident"] = True
    sz["resident_format"] = sz_resident.FORMAT
    sz["resident_version"] = sz_resident.VERSION
    sz["stage1_source"] = str(src)
    sz["note"] = ("EXPERIMENTAL, not a shipped format. STAGE 2: compact live rows + "
                  "[E, OUT] int32 row table stay RESIDENT (sz_resident.py); dead rows "
                  "read no code bytes and produce exact zeros.")
    for p, (E, OUT, W, G, NL) in shapes.items():
        sz["modules"][p].update({"code_words": W, "scale_groups": G})
    row_tbl = sum(E * OUT * 4 for E, OUT, _, _, _ in shapes.values())
    saved = sum((E * OUT - NL) * (W * 4 + G * 2) for E, OUT, W, G, NL in shapes.values())
    sz["resident_saved_bytes"] = saved - row_tbl
    with open(out / "config.json", "x") as fo:
        json.dump(cfg, fo, indent=2)
    provenance.write_build_record(
        out, tool="sz-resident", script=__file__, ap=ap, args=a,
        method={"format": sz_resident.FORMAT, "version": sz_resident.VERSION,
                "experimental": True},
        inputs=[("stage1", str(src))],
        modules={p: {"origin": "skipzero-resident", "live_rows": v[4]} for p, v in shapes.items()},
        full_hash={"config.json", mf, "sz_resident.py", "skipzero_load.py"})
    _log(f"{len(shapes)} modules; resident saving {saved / 2**30:.4f} GiB rows - "
         f"{row_tbl / 2**30:.4f} GiB row tables = {(saved - row_tbl) / 2**30:.4f} GiB net")
    _log(f"DONE -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
