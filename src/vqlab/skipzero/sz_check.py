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
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # src/
from vqlab import _layout  # noqa: E402,F401
import fitstore  # noqa: E402

sys.path.insert(0, str(HERE))
import sz_pack  # noqa: E402

GIB = 2 ** 30


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


# ------------------------------------------------------------------ selftest
def _write_st(path, tensors):
    """tensors: {name: np.ndarray} -> an mlx-format safetensors file."""
    dt = {np.dtype(np.uint32): "U32", np.dtype(np.uint8): "U8", np.dtype(np.float16): "F16",
          np.dtype(np.uint16): "U16"}
    specs = [(k, dt[v.dtype], list(v.shape), None) for k, v in tensors.items()]
    hb, _ = sz_pack.header_bytes(specs, {"format": "mlx"})
    with open(path, "wb") as f:
        f.write(hb)
        for v in tensors.values():
            f.write(np.ascontiguousarray(v).tobytes())


def selftest(root=None):
    """Synthetic artifact with dead rows: pack, expand (numpy AND mlx-CPU),
    assert live rows byte-identical and dead rows exactly zero."""
    rng = np.random.default_rng(1234)
    root = pathlib.Path(root or tempfile.mkdtemp(prefix="szself-"))
    src, out = root / "src", root / "packed"
    src.mkdir(parents=True)
    E, OUT, IN, D, G, PB = 4, 12, 128, 4, 64, 8
    W = (IN // D + 31) // 32 * PB
    p = "model.layers.0.mlp.switch_mlp.up_proj"
    codes = rng.integers(0, 2 ** 32, (E, OUT, W), dtype=np.uint32)
    scales = rng.uniform(0.01, 1.0, (E, OUT, IN // G)).astype(np.float16)
    scales[0, 3] = 0                                   # exact-zero row
    scales[2, 5] = np.float16(1e-6)                    # tiny row
    scales[3] = np.float16(3e-5)                       # whole dead expert
    scales[1, 7, 0] = 0                                # partly zero: LIVE
    cb = rng.standard_normal((256, D)).astype(np.float16)
    other = rng.standard_normal((8, 8)).astype(np.float16)
    _write_st(src / "model-00001-of-00001.safetensors",
              {p + ".codebook": cb, p + ".codes": codes, p + ".vq_scales": scales,
               "model.norm.weight": other})
    (src / "config.json").write_text(json.dumps({"model_type": "synthetic", "model_file": "model.py",
        "vq_modules": {p: {"experts": E, "out": OUT, "in": IN, "k": 256, "dim": D,
                           "group": G, "pack_bits": PB}}}))
    (src / "model.py").write_text("import pathlib as _pathlib\nclass Model:\n"
                                  "    def load_weights(self, w, strict=True):\n"
                                  "        self.w = dict(w)\n")
    (src / "tokenizer.json").write_text("{}")
    assert sz_pack.main([str(src), "--out", str(out), "--allow-any-out"]) == 0
    rep = check(out, src)
    assert rep["verdict"] == "PASS", rep["fails"]
    assert json.loads((out / "config.json").read_text())["vq_skipzero"]["modules"][p]["dead_rows"] == OUT + 2
    assert os.path.islink(out / "tokenizer.json")
    # the loader hook, through the generated model.py, on the mlx CPU path
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    spec = importlib.util.spec_from_file_location("sz_model", out / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.Model()
    w = mx.load(str(out / "model-00001-of-00001.safetensors"))
    m.load_weights(list(w.items()))
    got_c, got_s = np.array(m.w[p + ".codes"]), np.array(m.w[p + ".vq_scales"])
    dead = (scales < 6.2e-5).all(-1)
    assert got_c.dtype == codes.dtype and got_c.shape == codes.shape
    assert np.array_equal(got_c[~dead], codes[~dead]) and not got_c[dead].any()
    assert got_s[~dead].tobytes() == scales[~dead].tobytes() and not got_s[dead].any()
    assert not any(".sz_" in k for k in m.w) and "model.norm.weight" in m.w
    print(f"sz-check selftest PASS ({int(dead.sum())} dead rows of {E * OUT}; "
          f"numpy + mlx-CPU expansion; saved {rep['saved_bytes']} B) in {root}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab sz-check", description=__doc__.split("\n\n")[0])
    ap.add_argument("packed", nargs="?")
    ap.add_argument("original", nargs="?")
    ap.add_argument("--skip-passthrough", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.packed and a.original):
        ap.error("packed and original are required")
    rep = check(a.packed, a.original, a.skip_passthrough)
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    print(json.dumps({k: (v[:20] if k == "fails" else v) for k, v in rep.items()}, indent=1))
    print(f"sz-check {rep['verdict']}: {rep['modules']} modules, saved "
          f"{rep['saved_bytes'] / GIB:.3f} GiB ({rep['saved_pct']:.2f}%)")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
