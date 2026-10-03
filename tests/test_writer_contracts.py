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

ADVERSARIAL INPUTS (operator notes 2026-10-03, section 4): wherever a writer
chooses between inputs, the inputs DIFFER, so a golden can only pass if the
writer took the right one. The reskeleton golden once passed before AND after
a fundamental fix because the skeleton tensors were byte-identical in both
builds. So every artifact carries its OWN skeleton bytes (norms, and an
affine attention o_proj) and its own spelling of the o_proj quantization
entry (affine/vq32 say mode=affine, vq16 does not): mix, loo-bands and
reskeleton must show the bytes AND the config entry of the source they took.

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
CPU_SITE = pathlib.Path(__file__).resolve().parent / "cpu_site"
E, H, I, G, L = 2, 128, 128, 64, 3
PROJS = ("gate_proj", "up_proj", "down_proj")
MOD = "model.language_model.layers.{li}.mlp.switch_mlp.{p}"
NORM = "model.language_model.layers.{li}.input_layernorm.weight"
OPROJ = "model.language_model.layers.{li}.self_attn.o_proj"


def _skeleton(r, li, shards, q, mode):
    """One layer's non-expert tensors, bytes drawn from THIS artifact's rng:
    a norm and a 4-bit affine o_proj whose quant entry spells `mode` or not."""
    shards.setdefault(_place(li, "norm", ""), {})[NORM.format(li=li)] = \
        (1 + r.standard_normal(H) * .01).astype(np.float16)
    m = OPROJ.format(li=li)
    for suf, arr in ((".weight", r.integers(0, 2**32, (H, H * 4 // 32), dtype=np.uint64).astype(np.uint32)),
                     (".scales", (r.random((H, H // G)) * .01 + .001).astype(np.float16)),
                     (".biases", (r.standard_normal((H, H // G)) * .01).astype(np.float16))):
        shards.setdefault(_place(li, "o_proj", suf), {})[m + suf] = arr
    q[m] = {"group_size": G, "bits": 4, **({"mode": "affine"} if mode else {})}


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
    rs = np.random.default_rng(99)       # skeleton bytes: own stream, experts unchanged
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
        _skeleton(rs, li, shards, q, mode=True)
    base = root / "affine"
    _art(base, shards, {"model_type": "qwen3_5_moe", "quantization": q})

    # VQ artifact (unpacked d4/K16), built directly: deterministic bytes
    def vq(K, seed, mode):
        rr = np.random.default_rng(seed)
        sh, vqm, qv = {}, {}, {"group_size": G, "bits": 4}
        rs = np.random.default_rng(seed + 100)
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
            _skeleton(rs, li, sh, qv, mode)
        return sh, vqm, qv
    sh, vqm, qv = vq(16, 1, mode=False)
    vqa = root / "vq16"
    _art(vqa, sh, {"model_type": "qwen3_5_moe", "model_file": "model.py",
                   "quantization": qv, "vq_modules": vqm})
    (vqa / "model.py").write_text("# fixture runtime\n")
    sh, vqm, qv = vq(32, 2, mode=True)
    vqb = root / "vq32"
    _art(vqb, sh, {"model_type": "qwen3_5_moe", "model_file": "model.py",
                   "quantization": qv, "vq_modules": vqm})
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
    # cpu_site/sitecustomize.py pins MLX to the CPU in the child: no GPU use
    env = {**os.environ, "PYTHONPATH": f"{CPU_SITE}{os.pathsep}{SRC}",
           "VQLAB_SKIP_DISK_CHECK": "1"}
    p = subprocess.run([sys.executable, "-m", "vqlab.cli", *map(str, args)],
                       capture_output=True, text=True, env=env)
    if ok:
        assert p.returncode == 0, f"{args[0]} failed:\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}"
    return p


def _shas(snap):
    return {k: v["sha"] for k, v in snap["tensors"].items()}


# ------------------------------------------------------------------- writers
def test_mix(fx):
    # L1.gate_proj straddles both shards, so both hold layer 1: the only legal
    # band covering any VQ layer of shard 2 is the whole model.
    out = fx["root"] / "mix"
    cli("mix", "--out", out, "--base", fx["vq16"], "--band", f"{fx['vq32']}:0-2")
    snap = snapshot(out)
    check("mix", snap)
    # adversarial: vq16 and vq32 differ in EVERY tensor and in the o_proj
    # quant spelling, so these hold only if the bytes and maps came from vq32
    got, want, other = _shas(snap), _shas(snapshot(fx["vq32"])), _shas(snapshot(fx["vq16"]))
    assert got == want and all(got[k] != other[k] for k in got)
    assert snap["config"]["quantization"][OPROJ.format(li=0)].get("mode") == "affine"


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
        np_dt = {"U8": np.uint8, "U16": np.uint16, "U32": np.uint32,
                 "F16": np.float16}[h[k]["dtype"]]
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
    snap = snapshot(out)
    check("reskeleton", snap)
    # adversarial: the two builds' skeletons differ byte for byte and in the
    # o_proj quant spelling; experts come from the VQ build, everything else
    # from the skeleton build, and the config entry follows the bytes
    got = _shas(snap)
    vq, sk = _shas(snapshot(fx["vq16"])), _shas(snapshot(fx["affine"]))
    experts = {k for k in got if ".switch_mlp." in k}
    assert experts and all(got[k] == vq[k] for k in experts)
    skel = set(got) - experts
    assert skel and all(got[k] == sk[k] != vq[k] for k in skel)
    for li in range(L):
        assert snap["config"]["quantization"][OPROJ.format(li=li)].get("mode") == "affine"


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


@pytest.mark.parametrize("name", ["mix", "pack", "reskeleton", "geo", "fitmoe"])
def test_total_size_is_tensor_bytes(fx, name):
    """Every writer's index total_size is the sum of its shards' tensor bytes
    (HF's definition), measured from the output, never copied from a parent."""
    from vqlab.core.artifact import Artifact, tensor_bytes
    d = {"mix": "mix", "pack": "packed", "reskeleton": "resk", "geo": "geo", "fitmoe": "fitmoe"}[name]
    out = fx["root"] / d
    if not out.exists():
        pytest.skip(f"{d} not built in this session (run the whole file)")
    a = Artifact.open(out)
    md = json.loads((out / "model.safetensors.index.json").read_text()).get("metadata", {})
    assert md.get("total_size") == tensor_bytes(out, a.shards)


def test_size_fix_index(fx, tmp_path):
    import shutil
    from vqlab.records import size_cmd
    d = tmp_path / "stale"
    shutil.copytree(fx["vq16"], d)
    p = d / "model.safetensors.index.json"
    doc = json.loads(p.read_text())
    doc["metadata"] = {"total_size": 10 ** 12}
    p.write_text(json.dumps(doc))
    assert size_cmd.main([str(d), "--check-index"]) == 1
    assert size_cmd.main([str(d), "--fix-index"]) == 0
    assert size_cmd.main([str(d), "--check-index"]) == 0
    assert json.loads(p.read_text())["weight_map"] == doc["weight_map"]
    # a second --fix-index has nothing to fix: a no-op writer exits non-zero
    before = p.read_bytes()
    assert size_cmd.main([str(d), "--fix-index"]) == 1
    assert p.read_bytes() == before


@pytest.mark.parametrize("key,cls", [
    ("model.language_model.layers.0.mlp.switch_mlp.gate_proj.codes", "text"),
    ("language_model.model.layers.1.self_attn.q_proj.weight", "text"),
    ("model.layers.3.ffn.experts.w1.codes", "text"),
    ("lm_head.weight", "text"),
    ("model.visual.blocks.0.attn.qkv.weight", "tower"),         # Qwen3.5 HF
    ("vision_tower.blocks.0.attn.qkv.weight", "tower"),         # Qwen3.6 / gemma
    ("embed_vision.embedding_projection.weight", "tower"),      # gemma's 2nd half
    ("model.vision_model.encoder.layers.0.w", "tower"),         # GLM
    ("vision.blocks.0.w", "tower"),                             # DeepSeek release
    ("aligner.layers.0.weight", "tower"),
    ("image_newline", "tower"),                                 # release name
    ("model.image_newline", "tower"),                           # runtime name
    ("model.layers.7.ffn.gate.bias_vl", "tower"),               # image routing bias
    ("block.mlp.switch_mlp.up_proj.weight", "mtp"),             # 397B indexed head
    ("fc.weight", "mtp"),
    ("mtp.0.attn.wq_a.weight", "mtp"),                          # DeepSeek source
    ("language_model.mtp.layers.0.mlp.gate.weight", "mtp"),     # native runtime layout
])
def test_tensor_class(key, cls):
    """The one classifier (operator notes 2026-10-03, 1.5): every spelling the
    fleet uses, including DeepSeek Vision-Exp's tower, image tokens and the
    per-layer image-routing bias."""
    from vqlab.core.artifact import tensor_class
    assert tensor_class(key) == cls


# ---------------------------------------------------- no-op writers refuse
# Operator notes 2026-10-03, 1.6: a writer prints what it changed and exits
# non-zero when it changed nothing it was asked to change -- and a refused
# run never leaves an output behind or a build record.
def _copy(src, d):
    import shutil
    shutil.copytree(src, d, symlinks=False)
    return d


def _refused(p, *words):
    out = p.stdout + p.stderr
    assert p.returncode != 0, out[-2000:]
    for w in words:
        assert w in out, out[-2000:]


def test_noop_pack_twice_refused(fx, tmp_path):
    once = tmp_path / "once"
    cli("pack", "--src", fx["vq16"], "--out", once)
    p = cli("pack", "--src", once, "--out", tmp_path / "twice", ok=False)
    _refused(p, "nothing in")
    assert not (tmp_path / "twice").exists()


@pytest.mark.parametrize("bands,why", [
    (["{vq32}:7-9"], "selects no shard"),
    (["{vq16}:0-2"], "is --base itself"),
    ([], "nothing to mix"),
])
def test_noop_mix_refused(fx, tmp_path, bands, why):
    args = []
    for b in bands:
        args += ["--band", b.format(vq16=fx["vq16"], vq32=fx["vq32"])]
    p = cli("mix", "--out", tmp_path / "m", "--base", fx["vq16"], *args, ok=False)
    _refused(p, why)
    assert not (tmp_path / "m").exists()


@pytest.mark.parametrize("geomap,why", [
    ({}, "names no module"),
    ({"model.language_model.layers.9.mlp.switch_mlp.up_proj": {"dim": 4, "k": 32}},
     "not in"),
])
def test_noop_geo_build_refused(fx, tmp_path, geomap, why):
    gm = tmp_path / "g.json"
    gm.write_text(json.dumps(geomap))
    p = cli("geo-build", "--artifact", fx["vq16"], "--teacher", fx["teacher"], "--family",
            "qwen3_5", "--geomap", gm, "--out", tmp_path / "geo", ok=False)
    _refused(p, why)
    assert not (tmp_path / "geo").exists()


def test_noop_vision_layout_fix_refused_and_unrecorded(fx, tmp_path):
    d = _copy(fx["vq16"], tmp_path / "a")
    p = cli("vision-layout", d, "--fix", ok=False)
    _refused(p, "nothing written")
    assert not (d / "vqlab_provenance.json").exists()


def test_noop_pack_dense_refused(fx, tmp_path):
    p = cli("pack-dense", "--src", fx["vq16"], "--out", tmp_path / "pd", ok=False)
    _refused(p, "nothing in")
    assert not (tmp_path / "pd").exists()


def test_noop_pack_ple_refused(tmp_path):
    d = tmp_path / "ple"
    m = "model.layers.0.ngram_embedding.0"
    _art(d, {"model.safetensors": {m + ".codes": np.zeros((4, 8), np.uint8),
                                   m + ".codebook": np.zeros((16, 4), np.float16)}},
         {"model_type": "x", "vq_ple": {"geometry": {"k": 16, "dim": 4}, "keys": [m]}})
    before = (d / "config.json").read_bytes()
    p = cli("pack-ple", "--artifact", d, ok=False)
    _refused(p, "nothing written")
    assert (d / "config.json").read_bytes() == before


def test_noop_splice_ple_refused(fx, tmp_path):
    d = _copy(fx["vq16"], tmp_path / "a")
    fit = tmp_path / "fit"
    fit.mkdir()
    (fit / "ple_manifest.json").write_text(json.dumps({"geometry": {}, "tensors": {}}))
    before = (d / "model.safetensors.index.json").read_bytes()
    p = cli("splice-ple", "--artifact", d, "--ple-fit", fit, ok=False)
    _refused(p, "nothing to splice")
    assert (d / "model.safetensors.index.json").read_bytes() == before


def test_noop_ple_swap_refused(fx, tmp_path):
    p = cli("ple-swap", "--artifact", fx["vq16"], "--donor", fx["vq32"],
            "--out", tmp_path / "o", ok=False)
    _refused(p, "change nothing")
    assert not (tmp_path / "o").exists()


def test_noop_harvest_refused(fx, tmp_path):
    p = cli("harvest-parts", "--artifact", fx["vq16"], "--out", tmp_path / "parts",
            "--geometry", "d4-K9999", ok=False)
    _refused(p, "nothing harvested")


def test_bundle_refuses_empty_and_says_unchanged(fx, tmp_path):
    p = cli("bundle", "--artifact", _copy(fx["affine"], tmp_path / "aff"), ok=False)
    _refused(p, "no VQ expert modules")
    d = _copy(fx["vq16"], tmp_path / "a")
    first = cli("bundle", "--artifact", d)
    assert "changed: model.py" in first.stdout
    again = cli("bundle", "--artifact", d)     # idempotent: rc 0, but SAID
    assert "UNCHANGED" in again.stdout


# ------------------------------------------- read-only modes never record
@pytest.mark.parametrize("cmd,flag", [
    ("bundle", "--artifact"), ("pack-ple", "--artifact"), ("graft", "--artifact"),
    ("graft-extras", "--artifact")])
def test_help_never_records(fx, tmp_path, cmd, flag):
    """--help exits 0, and the CLI's automatic build record fired on any clean
    exit of a build command: `vqlab bundle --artifact X --help` stamped X."""
    d = _copy(fx["vq16"], tmp_path / "a")
    for h in ("--help", "-h"):
        p = cli(cmd, flag, d, h)
        assert p.returncode == 0
    assert not (d / "vqlab_provenance.json").exists()


def test_read_only_flags_cover_the_cli():
    from vqlab import cli as vcli
    assert {"--dry-run", "--plan", "-h", "--help"} <= vcli.READ_ONLY_FLAGS
