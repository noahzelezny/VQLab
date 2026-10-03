"""deepseek_v4 reference parity (operator notes 1.4), CPU only, tiny shapes.

Each test transcribes the reference (DeepSeek's inference/model.py) in
numpy and compares mlx-lm's deepseek_v4 against it; PARITY in
vqlab/family/deepseek_v4.py names these tests. Also: the plugin's fused
projection table (stream-convert reads it), `vqlab parity`, and the
act-stats clamp counter."""
import pathlib
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
A = pytest.importorskip("mlx_lm.models.deepseek_v4")

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _ref_expert_act(gate, up, limit):
    """model.py Expert.forward: clamp up both sides, gate from above."""
    if limit > 0:
        up = np.clip(up, -limit, limit)
        gate = np.minimum(gate, limit)
    return _silu(gate) * up


def test_routed_clamp_matches_reference():
    rng = np.random.default_rng(1234)
    g = (rng.standard_normal((4, 64)) * 12).astype(np.float32)
    u = (rng.standard_normal((4, 64)) * 12).astype(np.float32)
    act = A._DSV4SwiGLU(10.0)                       # what DeepseekV4MoE builds
    got = np.array(act(mx.array(u), mx.array(g)))   # SwitchGLU passes (up, gate)
    np.testing.assert_allclose(got, _ref_expert_act(g, u, 10.0), rtol=1e-5, atol=1e-5)
    assert (np.abs(g) > 10).any() and (np.abs(u) > 10).any()   # the clamp was exercised
    # and the MoE wires the routed experts at args.swiglu_limit
    src = pathlib.Path(A.__file__).read_text()
    assert "activation=_DSV4SwiGLU(args.swiglu_limit)" in src


def _ref_gate(x, W, bias, tid2eid, ids, topk, route_scale, hash_):
    """model.py Gate.forward (text tokens, score_func sqrtsoftplus)."""
    scores = x.astype(np.float64) @ W.astype(np.float64).T
    scores = np.sqrt(np.logaddexp(scores, 0.0))
    orig = scores
    if hash_:
        idx = tid2eid[ids]
    else:
        idx = np.argsort(-(scores + bias), axis=-1)[:, :topk]
    w = np.take_along_axis(orig, idx, axis=-1)
    w = w / w.sum(-1, keepdims=True)
    return idx, w * route_scale


def _gate(layer_id, n_hash=1):
    args = types.SimpleNamespace(n_routed_experts=8, num_experts_per_tok=2,
                                 num_hash_layers=n_hash, scoring_func="sqrtsoftplus",
                                 routed_scaling_factor=1.5, norm_topk_prob=True,
                                 hidden_size=16, vocab_size=32)
    return A.MoEGate(args, layer_id)


def _run(g, x, ids, monkeypatch):
    monkeypatch.setattr(A, "_moe_gate_kernel", None)   # the reference-shaped fp32 path
    inds, w = g(mx.array(x[None]), mx.array(ids[None]))
    return np.array(inds[0]), np.array(w[0].astype(mx.float32))


def test_gate_matches_reference(monkeypatch):
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((5, 16)).astype(np.float32)
    W = rng.standard_normal((8, 16)).astype(np.float32)
    bias = (rng.standard_normal(8) * 2).astype(np.float32)   # big enough to change the pick
    g = _gate(layer_id=1)
    assert not g.hash
    g.weight, g.e_score_correction_bias = mx.array(W), mx.array(bias)
    inds, w = _run(g, x, np.zeros(5, np.int32), monkeypatch)
    ri, rw = _ref_gate(x, W, bias, None, None, 2, 1.5, False)
    assert (np.sort(inds, -1) == np.sort(ri, -1)).all()
    o, ro = np.argsort(inds, -1), np.argsort(ri, -1)
    np.testing.assert_allclose(np.take_along_axis(w, o, -1),
                               np.take_along_axis(rw, ro, -1), rtol=1e-5)
    # bias moved selection on at least one token, yet weights stayed unbiased
    plain = np.argsort(-np.sqrt(np.logaddexp(x @ W.T, 0)), -1)[:, :2]
    assert (np.sort(plain, -1) != np.sort(ri, -1)).any()


def test_hash_gate_matches_reference(monkeypatch):
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((5, 16)).astype(np.float32)
    W = rng.standard_normal((8, 16)).astype(np.float32)
    t2e = rng.integers(0, 8, (32, 2)).astype(np.int32)
    ids = np.array([3, 7, 0, 31, 12], np.int32)
    g = _gate(layer_id=0, n_hash=3)
    assert g.hash and _gate(layer_id=3, n_hash=3).hash is False
    g.weight, g.tid2eid = mx.array(W), mx.array(t2e)
    inds, w = _run(g, x, ids, monkeypatch)
    ri, rw = _ref_gate(x, W, None, t2e, ids, 2, 1.5, True)
    assert (inds == ri).all()
    np.testing.assert_allclose(w, rw, rtol=1e-5)


def test_fused_table_lives_in_plugin():
    from vqlab.family import alias_fused, fused_projections
    t = fused_projections("deepseek_v4")
    assert ("attn.wq_a", "attn.wkv", "attn.wqkv_a") in t
    assert ("compressor.wkv", "compressor.wgate", "compressor.wkv_gate") in t
    assert fused_projections("qwen3_5") == ()
    q8 = {"bits": 8, "group_size": 64}
    skel = {"model.layers.2.attn.wq_a": q8, "model.layers.2.attn.wkv": dict(q8),
            "model.layers.2.attn.compressor.wkv": q8, "model.layers.2.attn.compressor.wgate": q8,
            "model.layers.3.attn.wq_a": q8}               # half missing: no alias
    alias_fused(skel, t)
    assert skel["model.layers.2.attn.wqkv_a"] is q8
    assert skel["model.layers.2.attn.compressor.wkv_gate"] is q8
    assert "model.layers.3.attn.wqkv_a" not in skel
    with pytest.raises(ValueError, match="different widths"):
        alias_fused({"l.attn.wq_a": q8, "l.attn.wkv": {"bits": 4, "group_size": 64}}, t)
    src = (ROOT / "src/vqlab/assemble/stream_convert.py").read_text()
    assert "fused_projections(a.family)" in src and "wqkv_a\")" not in src


def test_parity_checklist():
    from vqlab.family import Parity, plugins
    from vqlab.gate import parity as P
    p = plugins()["deepseek_v4"]
    assert "inference/model.py" in p.reference
    names = {x.name for x in p.parity}
    assert {"shared_expert_swiglu_clamp", "routed_expert_swiglu_clamp",
            "routing_score_and_bias", "hash_routing_layers", "norm_eps"} <= names
    for it in p.parity:                         # a named test must exist
        assert not it.test or P.has_test(it.test, ROOT), it.test
        assert it.test or not it.required, f"{it.name}: required but untested"
    lines = []
    assert P.check(p, ROOT, out=lines.append) == 0
    assert any("[untested" in s for s in lines)
    bad = types.SimpleNamespace(name="x", reference="", parity=(
        Parity("a", "r", "m"), Parity("b", "r", "m", test="tests/nope.py::t"),
        Parity("c", "r", "m", required=False)))
    assert P.check(bad, ROOT, out=lambda s: None) == 2


def test_clamp_counter():
    from vqlab.score.act_stats import ClampCounter
    c = ClampCounter(10.0)
    c.add(mx.array([[11.0, -20.0, 1.0]]), mx.array([[0.0, -11.0, 10.0]]))
    r = c.result()
    assert (r["elements"], r["gate_over"], r["up_abs_over"], r["any_over"]) == (3, 1, 1, 2)
    assert r["frac_any_over"] == pytest.approx(2 / 3)
