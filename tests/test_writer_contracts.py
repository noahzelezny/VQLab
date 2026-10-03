"""Characterization tests: what every artifact WRITER produces today.

Step 1 of the Artifact-API work (lab/docs/DEVLIST-2026-10-02.md, structural
#1). Every writer re-implements "read index, map modules to shards, swap
tensors, rewrite vq_modules / quantization / vq_skipzero, write index". Before
any of that is unified, this file pins each writer's CURRENT output so a port
onto a shared object must reproduce it exactly.

One deterministic fixture (numpy, seed 1234): a qwen3_5-shaped MoE, 3 layers,
2 shards, with L1.gate_proj's tensors deliberately STRADDLING the shard
boundary (the case DeepSeek hit). Each writer runs through the real CLI; its
output is reduced to a snapshot:
  files (name, symlink or file), index weight_map, the config maps
  (vq_modules, quantization, vq_skipzero, model_file, pack_bits),
  every tensor's dtype + shape + shard, and -- for writers that do not fit --
  the sha256 of every tensor's bytes.
Fitting writers (fit-moe, geo-build) are pinned on STRUCTURE only: their
k-means is seeded but GPU reductions can differ in the last place.

Regenerate goldens deliberately:  VQLAB_UPDATE_GOLDEN=1 pytest tests/test_writer_contracts.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys

import numpy as np
import pytest

GOLD = pathlib.Path(__file__).parent / "golden" / "writers"
SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
E, H, I, G, L = 2, 128, 128, 64, 3
PROJS = ("gate_proj", "up_proj", "down_proj")
MOD = "model.language_model.layers.{li}.mlp.switch_mlp.{p}"
NORM = "model.language_model.layers.{li}.input_layernorm.weight"


# ----------------------------------------------------------------- fixture io
def _save(path, tensors):
    """Write a safetensors file from numpy arrays (no MLX: bytes are exact)."""
    dt = {np.dtype("uint8"): "U8", np.dtype("uint16"): "U16", np.dtype("uint32"): "U32",
          np.dtype("float16"): "F16", np.dtype("float32"): "F32"}
    hdr, off, blobs = {}, 0, []
    for k in sorted(tensors):
        a = np.ascontiguousarray(tensors[k])
        b = a.tobytes()
        hdr[k] = {"dtype": dt[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _bf16(a):
    """float32 -> bf16 bit pattern stored as a 'BF16' tensor."""
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _save_bf16(path, tensors):
    hdr, off, blobs = {}, 0, []
    for k in sorted(tensors):
        b = _bf16(tensors[k]).tobytes()
        hdr[k] = {"dtype": "BF16", "shape": list(tensors[k].shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _art(d, shards, cfg):
    d.mkdir(parents=True)
    wm = {}
    for f, t in shards.items():
        _save(d / f, t)
        wm.update({k: f for k in t})
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (d / "config.json").write_text(json.dumps(cfg, indent=1))


def _place(li, p, suffix):
    """Shard of a tensor: L0 -> s1, L2 -> s2, L1 splits: gate_proj's FIRST
    tensor in s1 and the rest in s2 (the straddle)."""
    if li == 0:
        return "model-00001-of-00002.safetensors"
    if li == 2:
        return "model-00002-of-00002.safetensors"
    first = suffix in (".codes", ".weight")
    return "model-00001-of-00002.safetensors" if (p == "gate_proj" and first) or p == "norm" \
        else "model-00002-of-00002.safetensors"


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    r = np.random.default_rng(1234)
    root = tmp_path_factory.mktemp("writers")
    # teacher: HF-layout bf16, fused gate_up [E, 2I, H] + down [E, H, I]
    centres = r.standard_normal((32, 4)).astype(np.float32) * 0.05
    def wgt(o, i):
        pick = r.integers(0, 32, (E, o, i // 4))
        return (centres[pick] + r.standard_normal((E, o, i // 4, 4)) * 0.002).reshape(E, o, i)
    teacher = root / "teacher"; teacher.mkdir()
    tt = {}
    for li in range(L):
        tt[f"model.language_model.layers.{li}.mlp.experts.gate_up_proj"] = wgt(2 * I, H)
        tt[f"model.language_model.layers.{li}.mlp.experts.down_proj"] = wgt(H, I)
    _save_bf16(teacher / "t.safetensors", tt)
    (teacher / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "t.safetensors" for k in tt}}))
    (teacher / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))

    # affine base (2-bit group 64, uint32 words), the fit-moe / reskeleton skeleton
    shards, q = {}, {"group_size": G, "bits": 4}
    for li in range(L):
        for p in PROJS:
            m = MOD.format(li=li, p=p)
            ins = I if p == "down_proj" else H
            outs = H if p == "down_proj" else I
            for suf, arr in ((".weight", r.integers(0, 2**32, (E, outs, ins * 2 // 32), dtype=np.uint64).astype(np.uint32)),
                             (".scales", (r.random((E, outs, ins // G)) * .01 + .001).astype(np.float16)),
                             (".biases", (r.standard_normal((E, outs, ins // G)) * .01).astype(np.float16))):
                shards.setdefault(_place(li, p, suf), {})[m + suf] = arr
            q[m] = {"group_size": G, "bits": 2}
        shards.setdefault(_place(li, "norm", ""), {})[NORM.format(li=li)] = np.ones(H, np.float16)
    base = root / "affine"
    _art(base, shards, {"model_type": "qwen3_5_moe", "quantization": q})

    # VQ artifact (unpacked d4/K16), built directly: deterministic bytes
    def vq(K, seed):
        rr = np.random.default_rng(seed)
        sh, vqm = {}, {}
        for li in range(L):
            for p in PROJS:
                m = MOD.format(li=li, p=p)
                ins = I if p == "down_proj" else H
                outs = H if p == "down_proj" else I
                sc = (rr.random((E, outs, ins // G)) + .5).astype(np.float16)
                if li == 0 and p == "up_proj":
                    sc[0, :5] = 0          # dead rows for sz-pack
                for suf, arr in ((".codes", rr.integers(0, K, (E, outs, ins // 4)).astype(np.uint8 if K <= 256 else np.uint16)),
                                 (".codebook", (rr.standard_normal((K, 4)) * .05).astype(np.float16)),
                                 (".vq_scales", sc)):
                    sh.setdefault(_place(li, p, suf), {})[m + suf] = arr
                vqm[m] = {"experts": E, "out": outs, "in": ins, "k": K, "dim": 4, "group": G}
            sh.setdefault(_place(li, "norm", ""), {})[NORM.format(li=li)] = np.ones(H, np.float16)
        return sh, vqm
    sh, vqm = vq(16, 1)
    vqa = root / "vq16"
    _art(vqa, sh, {"model_type": "qwen3_5_moe", "model_file": "model.py",
                   "quantization": {"group_size": G, "bits": 4}, "vq_modules": vqm})
    (vqa / "model.py").write_text("# fixture runtime\n")
    sh, vqm = vq(32, 2)
    vqb = root / "vq32"
    _art(vqb, sh, {"model_type": "qwen3_5_moe", "model_file": "model.py",
                   "quantization": {"group_size": G, "bits": 4}, "vq_modules": vqm})
    (vqb / "model.py").write_text("# fixture runtime\n")
    return {"root": root, "teacher": teacher, "affine": base, "vq16": vqa, "vq32": vqb}


# ------------------------------------------------------------------ snapshot
def _header(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def snapshot(d, values=True):
    d = pathlib.Path(d)
    cfg = json.loads((d / "config.json").read_text())
    idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for f in sorted(set(idx.values())):
        h, start = _header(d / f)
        data = (d / f).read_bytes() if values else b""
        for k, v in h.items():
            e = {"shard": f, "dtype": v["dtype"], "shape": v["shape"]}
            if values:
                a, b = v["data_offsets"]
                e["sha"] = hashlib.sha256(data[start + a:start + b]).hexdigest()[:16]
            tensors[k] = e
    files = sorted(f"{p.name}{' ->' if p.is_symlink() else ''}" for p in d.iterdir()
                   if p.name not in ("vqlab_provenance.json", "vqlab_provenance.history.jsonl")
                   and not p.name.startswith("."))
    keep = {k: cfg.get(k) for k in ("model_file", "vq_modules", "quantization", "vq_skipzero") if k in cfg}
    return {"files": files, "index": dict(sorted(idx.items())), "config": keep, "tensors": tensors}


def check(name, snap, root=None):
    GOLD.mkdir(parents=True, exist_ok=True)
    g = GOLD / f"{name}.json"
    text = json.dumps(snap, indent=1, sort_keys=True) + "\n"
    if root is not None:                       # recorded source paths are per-run
        text = text.replace(str(root), "<fixture>")
    if os.environ.get("VQLAB_UPDATE_GOLDEN") == "1" or not g.exists():
        g.write_text(text)
        if os.environ.get("VQLAB_UPDATE_GOLDEN") != "1":
            pytest.fail(f"golden {g.name} was missing and has been written; review and commit it")
        return
    assert json.loads(text) == json.loads(g.read_text()), f"{name}: output differs from golden {g}"


def cli(*args, ok=True):
    env = {**os.environ, "PYTHONPATH": str(SRC), "VQLAB_SKIP_DISK_CHECK": "1"}
    p = subprocess.run([sys.executable, "-m", "vqlab.cli", *map(str, args)],
                       capture_output=True, text=True, env=env)
    if ok:
        assert p.returncode == 0, f"{args[0]} failed:\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}"
    return p


# ------------------------------------------------------------------- writers
def test_mix(fx):
    # L1.gate_proj straddles both shards, so both hold layer 1: the only legal
    # band covering any VQ layer of shard 2 is the whole model.
    out = fx["root"] / "mix"
    cli("mix", "--out", out, "--base", fx["vq16"], "--band", f"{fx['vq32']}:0-2")
    check("mix", snapshot(out))


@pytest.mark.parametrize("band", ["1-2", "2-2", "1-1"])
def test_mix_straddle_refused(fx, band):
    """Current behavior, pinned: a band is whole shards, and a module split
    across shards ties both shards to its layer. A port may LIFT this (move
    the module's tensors as a unit) but must then say so here."""
    p = cli("mix", "--out", fx["root"] / f"mix-bad-{band}", "--base", fx["vq16"],
            "--band", f"{fx['vq32']}:{band}", ok=False)
    assert p.returncode != 0 and "straddles" in (p.stderr + p.stdout)


def test_loo_bands(fx):
    od = fx["root"] / "loo"
    cli("loo-bands", "--vq", fx["vq16"], "--exact", fx["affine"], "--bands", "0-2", "--out-dir", od)
    check("loo_bands", snapshot(od / "loo-L0-2"))


def test_minibase(fx):
    # Pinned as-is: a band-2 minibase takes shard 2 whole, so its index also
    # carries HALF of L1.gate_proj (.scales/.biases, not .weight, which is in
    # shard 1). fit-moe passes non-band modules through, so a band-2 fit is
    # unaffected, but the minibase is not a loadable model on its own.
    out = fx["root"] / "minibase"
    cli("minibase", "--base", fx["affine"], "--layers", "2-2", "--out", out)
    check("minibase", snapshot(out))


def test_pack_straddle(fx):
    # Was a KeyError on .codebook until the Artifact port (2026-10-02): pack now
    # takes K/dim from vq_modules, so a straddling module packs in place.
    out = fx["root"] / "packed-straddle"
    cli("pack", "--src", fx["vq16"], "--out", out)
    s = snapshot(out)
    s["files"] = [f for f in s["files"] if f != "model.py"]
    check("pack_straddle", s)


def test_pack(fx, tmp_path):
    # pack on the same artifact WITHOUT the straddle (L1.gate_proj whole in s2)
    src = tmp_path / "vq16-whole"
    src.mkdir()
    for f in fx["vq16"].iterdir():
        if f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            (src / f.name).write_bytes(f.read_bytes())
    idx = json.loads((fx["vq16"] / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {}
    for k, f in idx.items():
        h, start = _header(fx["vq16"] / f)
        a, b = h[k]["data_offsets"]
        raw = (fx["vq16"] / f).read_bytes()[start + a:start + b]
        np_dt = {"U8": np.uint8, "U16": np.uint16, "F16": np.float16}[h[k]["dtype"]]
        arr = np.frombuffer(raw, np_dt).reshape(h[k]["shape"])
        tgt = "model-00002-of-00002.safetensors" if ".layers.1.mlp.switch_mlp.gate_proj" in k else f
        shards.setdefault(tgt, {})[k] = arr
    wm = {}
    for f, t in shards.items():
        _save(src / f, t)
        wm.update({k: f for k in t})
    (src / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    out = fx["root"] / "packed"
    cli("pack", "--src", src, "--out", out)
    s = snapshot(out)
    s["files"] = [f for f in s["files"] if f != "model.py"]   # pack writes the live runtime
    check("pack", s)


def test_sz_pack(fx):
    out = fx["root"] / "sz"
    p = cli("sz-pack", fx["vq16"], "--out", out, "--allow-any-out")
    # the straddling L1.gate_proj is left unpacked, and said so (not silently)
    assert "straddle a shard boundary" in p.stdout + p.stderr
    s = snapshot(out)
    s["files"] = [f for f in s["files"] if f != "model.py"]
    check("sz_pack", s, fx["root"])


def test_reskeleton(fx):
    out = fx["root"] / "resk"
    cli("reskeleton", "--vq", fx["vq16"], "--skeleton", fx["affine"], "--out", out)
    check("reskeleton", snapshot(out))


def test_fit_moe(fx):
    out = fx["root"] / "fitmoe"
    cli("fit-moe", "--base", fx["affine"], "--src", fx["teacher"], "--out", out,
        "--vq-layers", "0-2", "--k", "16", "--dim", "4", "--iters", "2", "--sample", "2000",
        "--family", "qwen3_5", "--relerr-abort", "1.0")
    s = snapshot(out, values=False)
    s["files"] = [f for f in s["files"] if f != "model.py"]
    check("fit_moe", s)


def test_geo_build(fx):
    gm = fx["root"] / "geomap.json"
    m = MOD.format(li=2, p="down_proj")
    gm.write_text(json.dumps({m: {"dim": 4, "k": 32}}))
    out = fx["root"] / "geo"
    cli("geo-build", "--artifact", fx["vq16"], "--teacher", fx["teacher"], "--family", "qwen3_5",
        "--geomap", gm, "--out", out)
    s = snapshot(out, values=False)
    s["files"] = [f for f in s["files"] if f != "model.py"]
    check("geo_build", s)


def test_sizes_three_ways(fx, tmp_path):
    from vqlab.core.artifact import sizes
    d = tmp_path / "sized"
    _art(d, {"a.safetensors": {"model.layers.0.w": np.zeros(256, np.uint8),
                               "model.visual.blocks.0.w": np.zeros(64, np.uint8),
                               "block.mlp.w": np.zeros(32, np.uint8)}}, {"model_type": "x"})
    _save(d / "mtp-head-q6.safetensors", {"x": np.zeros(16, np.uint8)})
    s = sizes(d)
    assert (s["text"], s["tower"], s["mtp"]) == (256, 64, 32)
    assert s["mtp_files"] == ["mtp-head-q6.safetensors"] and s["mtp_sidecar"] > 16
