"""The shared-expert clamp (DeepSeek's reference, the DEFAULT) really changes the shared expert: setting
swiglu_limit on mlx-lm's DeepseekV4MLP clamps gate/up exactly as DeepSeek's
reference does, and the plugin stamps the variant; =0 opts out."""
import types

import mlx.core as mx
import pytest

A = pytest.importorskip("mlx_lm.models.deepseek_v4")


def test_clamp_attribute_takes_effect():
    mlp = A.DeepseekV4MLP(8, 16, swiglu_limit=0.0)
    for lin in (mlp.gate_proj, mlp.up_proj):
        lin.weight = mx.ones_like(lin.weight) * 3.0          # 8 inputs x 3 = 24 > 10
    x = mx.ones((1, 8))
    y0 = mlp(x)
    mlp.swiglu_limit = 10.0
    y1 = mlp(x)
    assert not mx.array_equal(y0, y1).item()


def test_plugin_applies_and_stamps(monkeypatch):
    from vqlab.family import deepseek_v4 as P
    sh = types.SimpleNamespace(swiglu_limit=0.0)
    core = types.SimpleNamespace(args=types.SimpleNamespace(swiglu_limit=10.0),
                                 layers=[types.SimpleNamespace(ffn=types.SimpleNamespace(shared_experts=sh))])
    monkeypatch.setenv("VQLAB_DS4_SHARED_CLAMP", "0")
    P._apply_variant(core)
    assert sh.swiglu_limit == 0.0 and P._variant() is None
    monkeypatch.delenv("VQLAB_DS4_SHARED_CLAMP", raising=False)
    P._apply_variant(core)
    assert sh.swiglu_limit == 10.0 and P._variant() == "shared-clamp"
