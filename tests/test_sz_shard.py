"""Tensor-parallel split of skipzero modules (vq_switch.skipzero_shard).

gate_proj/up_proj split on OUTPUT rows: each rank keeps its slice of every
expert's rows, taken through the row mask (never through the compact list).
The rank outputs, concatenated, must be BYTE-EQUAL to the unsplit module.
down_proj splits on its INPUT axis: rows stay whole and the ordinary cut of
codes/vq_scales applies; the summed rank outputs must be byte-equal to the
non-sz EXPANDED module split and summed the same way (same reduction).
Checked at d2, d4 and d8, decode and prefill N, with many dead rows.
"""
import json
import struct

import numpy as np
import mlx.core as mx
import pytest

from vqlab import vq_switch as VS
from vqlab import vq_pack as VP


@pytest.fixture(autouse=True)
def _legacy_prefill_flags(monkeypatch):
    """These tests certify byte-equality against reference paths / earlier
    revisions that predate the rebundle defaults (VQ_FUSED_MAX_N 512,
    VQ_GEMMSEG_ACCS=1, VQ_GEMMSEG_CVEC=1; F200-F204). Those defaults change
    numerics for N in 512..4096 by design, so pin the legacy values here."""
    monkeypatch.setattr(VS, "VQ_FUSED_MAX_N", 4096)
    monkeypatch.setattr(VS, "_GEMMSEG_ACCS", False)
    monkeypatch.setattr(VS, "_GEMMSEG_CVEC", False)


def _bits(a):
    return np.array(a.astype(mx.float16)).view(np.uint16)


GEOM = {2: 1024, 4: 2048, 8: 16384}          # d -> K (fleet geometries)


def _ondisk(E, OUT, IN, D, dead_frac, seed):
    """(expanded codes/scales, sz on-disk tensors, codebook, bits, live)."""
    r = np.random.default_rng(seed)
    K, G = GEOM[D], 64
    bits = VP.bits_for_k(K)
    codes = r.integers(0, K, (E, OUT, IN // D)).astype(np.uint32)
    cb = (r.standard_normal((K, D)) * 0.05).astype(np.float16)
    sc = (r.standard_normal((E, OUT, IN // G)) * 0.1 + 1).astype(np.float16)
    live = r.random((E, OUT)) >= dead_frac
    live[1, :] = False
    live[2, :] = True
    codes[~live] = 0
    sc[~live] = 0
    full = VP.pack(codes, bits)
    sz = {"sz_codes": mx.array(full[live]), "sz_scales": mx.array(sc[live]),
          "sz_rowmask": mx.array(np.packbits(live, axis=-1, bitorder="little")),
          "sz_shape": mx.array(np.array([E, OUT, full.shape[-1], IN // G], np.int32))}
    return full, sc, sz, cb, bits, live


def _mod(codes, cb, sc, bits, IN, rt=None):
    return VS.VQSwitchLinear(mx.array(codes) if not isinstance(codes, mx.array) else codes,
                             mx.array(cb), sc if isinstance(sc, mx.array) else mx.array(sc),
                             group_size=64, pack_bits=bits, in_features=IN,
                             row_table=rt)


def _entry(E, OUT, IN, D, bits, live):
    return {"experts": E, "out": OUT, "live_rows": int(live.sum()),
            "in": IN, "dim": D, "group": 64, "pack_bits": bits, "k": GEOM[D]}


def _sz_module(w, p, modules, cb, bits, IN):
    w = VS.skipzero_weights(w, modules)
    return _mod(w[p + ".codes"], cb, w[p + ".vq_scales"], bits, IN,
                rt=w[p + ".row_table"])


def _xi(T, top, E, IN, seed):
    r = np.random.default_rng(seed)
    x = mx.array(r.standard_normal((T, 1, 1, IN)).astype(np.float32)).astype(mx.bfloat16)
    idx = mx.array(r.integers(0, E, (T, top)).astype(np.uint32))
    return x, idx


@pytest.mark.parametrize("D", [2, 4, 8])
@pytest.mark.parametrize("leaf", ["gate_proj", "up_proj"])
@pytest.mark.parametrize("OUT,n", [(96, 2), (96, 4), (102, 2), (64, 1)])
@pytest.mark.parametrize("T,top", [(1, 8), (3, 10), (600, 8)])
def test_row_split_concat_byte_equal(D, leaf, OUT, n, T, top):
    E, IN = 8, 512
    p = f"model.layers.3.mlp.switch_mlp.{leaf}"
    _, _, sz, cb, bits, live = _ondisk(E, OUT, IN, D, 0.7, seed=D * 7 + n)
    w = {p + "." + k: v for k, v in sz.items()}
    mods = {p: _entry(E, OUT, IN, D, bits, live)}
    x, idx = _xi(T, top, E, IN, seed=T)
    ref = _sz_module(dict(w), p, mods, cb, bits, IN)(x, idx)
    parts, total = [], 0
    for rank in range(n):
        wr, mr = VS.skipzero_shard(dict(w), mods, rank, n)
        assert mr[p]["out"] == OUT // n
        assert wr[p + ".sz_codes"].shape[0] == mr[p]["live_rows"]
        assert np.array(wr[p + ".sz_shape"]).tolist()[1] == OUT // n
        total += mr[p]["live_rows"]
        m = _sz_module(wr, p, mr, cb, bits, IN)
        assert m.output_dims == OUT // n
        parts.append(m(x, idx))
    assert total == int(live.sum())
    got = mx.concatenate(parts, axis=-1)
    mx.eval(ref, got)
    assert np.array_equal(_bits(ref), _bits(got))


@pytest.mark.parametrize("D", [2, 4, 8])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("T,top", [(1, 8), (3, 10), (600, 8)])
def test_down_input_split_sum_byte_equal(D, n, T, top):
    E, OUT, IN = 8, 97, 1024
    p = "model.layers.3.mlp.switch_mlp.down_proj"
    full, sc, sz, cb, bits, live = _ondisk(E, OUT, IN, D, 0.7, seed=D + n)
    w = {p + "." + k: v for k, v in sz.items()}
    mods = {p: _entry(E, OUT, IN, D, bits, live)}
    wr, mr = VS.skipzero_shard(dict(w), mods, 1, n)
    assert mr[p] == mods[p]
    for k in sz:                                 # down_proj: untouched
        assert wr[p + "." + k] is w[p + "." + k]
    w = VS.skipzero_weights(w, mods)
    cs, ss = w[p + ".codes"], w[p + ".vq_scales"]
    W, NG, INr = cs.shape[-1] // n, ss.shape[-1] // n, IN // n
    x, idx = _xi(T, top, E, IN, seed=T + 1)
    ysz = yfull = None
    for r in range(n):
        xr = x[..., r * INr:(r + 1) * INr]
        a = _mod(cs[:, r * W:(r + 1) * W], cb, ss[:, r * NG:(r + 1) * NG], bits,
                 INr, rt=w[p + ".row_table"])(xr, idx)
        b = _mod(full[..., r * W:(r + 1) * W], cb, sc[..., r * NG:(r + 1) * NG],
                 bits, INr)(xr, idx)
        ysz = a if ysz is None else ysz + a
        yfull = b if yfull is None else yfull + b
    mx.eval(ysz, yfull)
    assert np.array_equal(_bits(ysz), _bits(yfull))


def test_refusals():
    E, IN = 4, 512
    _, _, sz, cb, bits, live = _ondisk(E, 96, IN, 4, 0.5, seed=1)
    g = "model.layers.0.mlp.switch_mlp.gate_proj"
    w = {g + "." + k: v for k, v in sz.items()}
    with pytest.raises(ValueError, match="does not divide"):
        VS.skipzero_shard(dict(w), {g: _entry(E, 96, IN, 4, bits, live)}, 0, 5)
    dp = "model.layers.0.mlp.switch_mlp.down_proj"
    wd = {dp + "." + k: v for k, v in sz.items()}
    # IN=512 d4 4-way: 128 inputs = 32 codes per rank, aligned; 8-way is 16
    VS.skipzero_shard(dict(wd), {dp: _entry(E, 96, IN, 4, bits, live)}, 0, 4)
    with pytest.raises(ValueError, match="not aligned"):
        VS.skipzero_shard(dict(wd), {dp: _entry(E, 96, IN, 4, bits, live)}, 0, 8)
    with pytest.raises(ValueError, match="not aligned"):
        VS.skipzero_shard(dict(wd), {dp: _entry(E, 96, IN, 4, bits, live)}, 0, 3)
    with pytest.raises(KeyError, match="geometry"):
        VS.skipzero_shard(dict(wd), {dp: {"experts": E, "out": 96}}, 0, 2)
    o = "model.layers.0.mlp.switch_mlp.gate_up_proj"
    wo = {o + "." + k: v for k, v in sz.items()}
    with pytest.raises(ValueError, match="no tensor split rule"):
        VS.skipzero_shard(dict(wo), {o: _entry(E, 96, IN, 4, bits, live)}, 0, 2)


def test_read_rowmasks(tmp_path):
    rm = np.arange(24, dtype=np.uint8).reshape(4, 6)
    other = np.ones((3, 3), np.float32)
    hdr, blobs, off = {}, [], 0
    for name, a, dt in (("a.other", other, "F32"), ("m.x.sz_rowmask", rm, "U8")):
        b = a.tobytes()
        hdr[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    js = json.dumps(hdr).encode()
    (tmp_path / "s.safetensors").write_bytes(struct.pack("<Q", len(js)) + js + b"".join(blobs))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a.other": "s.safetensors", "m.x.sz_rowmask": "s.safetensors"}}))
    got = VS.skipzero_read_rowmasks(tmp_path, ["m.x"])
    assert np.array_equal(got["m.x"], rm)


# ---------------------------------------------------------------- the shim
# A tiny synthetic qwen3_moe artifact, bundled by add_model_file and loaded
# through mlx_lm's own load_model with the shard request in model_config --
# the path a tensor-parallel loader takes.
TINY = dict(model_type="qwen3_moe", hidden_size=128, num_hidden_layers=1,
            intermediate_size=128, num_attention_heads=2, num_experts=4,
            num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
            moe_intermediate_size=256, rms_norm_eps=1e-6, vocab_size=64,
            num_key_value_heads=1, head_dim=64, rope_theta=1e4,
            tie_word_embeddings=False, max_position_embeddings=128,
            norm_topk_prob=True)


def _tiny_artifact(root):
    import os
    import subprocess
    import sys
    from pathlib import Path
    from mlx.utils import tree_flatten
    q = pytest.importorskip("mlx_lm.models.qwen3_moe")
    base = q.Model(q.ModelArgs(**TINY))
    w = {k: v for k, v in tree_flatten(base.parameters()) if ".switch_mlp." not in k}
    vqm, szm = {}, {}
    for leaf, OUT, IN in (("gate_proj", 256, 128), ("up_proj", 256, 128),
                          ("down_proj", 128, 256)):
        p = f"model.layers.0.mlp.switch_mlp.{leaf}"
        _, _, sz, cb, bits, live = _ondisk(4, OUT, IN, 4, 0.6, seed=len(leaf))
        for k, v in sz.items():
            w[f"{p}.{k}"] = v
        w[f"{p}.codebook"] = mx.array(cb)
        e = _entry(4, OUT, IN, 4, bits, live)
        vqm[p] = {k: e[k] for k in ("experts", "out", "in", "k", "dim", "group", "pack_bits")}
        szm[p] = {k: e[k] for k in ("experts", "out", "live_rows")}
    mx.save_safetensors(str(root / "model.safetensors"), w, metadata={"format": "mlx"})
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model.safetensors" for k in w}}))
    (root / "config.json").write_text(json.dumps(
        dict(TINY, vq_modules=vqm, vq_skipzero={"loader": "runtime", "modules": szm})))
    src = Path(__file__).resolve().parents[1] / "src"
    r = subprocess.run([sys.executable, str(src / "vqlab/bundle/add_model_file.py"),
                        "--artifact", str(root)], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(src)))
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads((root / "config.json").read_text())


def test_shim_loads_rank_shards(tmp_path):
    utils = pytest.importorskip("mlx_lm.utils")
    cfg = _tiny_artifact(tmp_path)
    sw = lambda m: m.model.layers[0].mlp.switch_mlp  # noqa: E731
    import inspect
    kw = ({"trust_remote_code": True} if "trust_remote_code" in
          inspect.signature(utils.load_model).parameters else {})
    full, _ = utils.load_model(tmp_path, lazy=True, **kw)
    assert not hasattr(sw(full).gate_proj, "_vq_sharded")
    mx.eval(full(mx.array([[1, 2, 3]])))                 # the whole forward runs
    x, idx = _xi(3, 2, 4, 128, seed=9)
    n = 2
    ranks = []
    for r in range(n):
        shard = dict(cfg["vq_skipzero"], shard={"rank": r, "n": n})
        m, _ = utils.load_model(tmp_path, lazy=True, **kw,
                                model_config={"vq_skipzero": shard})
        ranks.append(m)
    assert type(ranks[0]).__init__.__globals__["SKIPZERO_SHARD"] == 1
    for leaf in ("gate_proj", "up_proj"):
        ref = getattr(sw(full), leaf)(x, idx)
        parts = []
        for r, m in enumerate(ranks):
            lin = getattr(sw(m), leaf)
            assert lin._vq_sharded == (r, n) and lin.output_dims == 128
            assert "_vq_sharded" not in lin
            parts.append(lin(x, idx))
        got = mx.concatenate(parts, axis=-1)
        mx.eval(ref, got)
        assert np.array_equal(_bits(ref), _bits(got)), leaf
    xd = mx.concatenate([x, x], axis=-1)
    ref = sw(full).down_proj(xd, idx)
    for m in ranks:                                      # built whole
        assert not hasattr(sw(m).down_proj, "_vq_sharded")
        assert np.array_equal(_bits(ref), _bits(sw(m).down_proj(xd, idx)))
