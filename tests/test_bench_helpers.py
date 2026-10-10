"""bench/timeline_common.py, bench/power.py and kernel_truth's public helpers."""
from types import SimpleNamespace as NS

from vqlab.bench import power, timeline_common as tc


def test_trunk_needs_layers_and_embed():
    inner = NS(layers=[1], embed_tokens=0)
    outer = NS(language_model=NS(model=inner), layers=[2])
    assert tc.trunk(outer) is inner
    assert tc.trunk(NS(x=1)) is None


def test_layer_kind():
    assert tc.layer_kind(NS(is_linear=True, mlp=NS(switch_mlp=1))) == ("lin", "moe")
    assert tc.layer_kind(NS(mlp=NS(up=1))) == ("attn", "dense")


def test_gpu_watts_without_macmon(monkeypatch):
    monkeypatch.setattr(power.shutil, "which", lambda _: None)
    monkeypatch.setenv("PATH", "")
    assert power.gpu_watts() == -1.0


def test_kernel_truth_public_names_and_aliases():
    from vqlab.score import kernel_truth as kt
    assert kt._parse_arm is kt.parse_arm and kt._attr is kt.module_attr
    assert kt.parse_arm("a=VQ_X:1,VQ_Y:true") == ("a", {"VQ_X": 1, "VQ_Y": True})
    assert kt.module_attr({"_X": 0}, "VQ_X") == "_X"
