"""build/build_dense.py: no affine quantization entry survives for a module
the build replaced with VQ."""
import pytest

from vqlab.build.build_dense import drop_replaced_quantization

MLP = "model.layers.0.mlp.up_proj"


def test_replaced_modules_lose_their_entries():
    cfg = {"vq_linear": {MLP: {}}, "quantization": {"bits": 4, MLP: {"bits": 4}, "lm_head": {"bits": 8}}}
    drop_replaced_quantization(cfg)
    assert cfg["quantization"] == {"bits": 4, "lm_head": {"bits": 8}}


def test_a_leftover_entry_under_a_vq_module_fails_loudly():
    cfg = {"vq_linear": {MLP: {}}, "quantization_config": {MLP + ".inner": {"bits": 4}}}
    with pytest.raises(SystemExit, match="FAIL"):
        drop_replaced_quantization(cfg)
