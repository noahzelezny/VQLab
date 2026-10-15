"""mtp.registry: the pack recipe's one home (default widths, sidecar names,
the qwen3_5 wiring choices)."""
from vqlab.mtp import registry


def test_a_sidecar_is_named_for_its_recipe():
    spec = registry.FAMILIES["qwen3_5_moe"]
    assert spec.sidecar_for() == "mtp-head-q6.safetensors"
    assert spec.sidecar_for(8) == "mtp-head-q8.safetensors"
    assert spec.sidecar_for(8, 3) == "mtp-head-e3q8.safetensors"


def test_qwen35_wiring_options_are_the_choices():
    assert registry.FAMILIES["qwen3_5_moe"].wiring == tuple(registry.WIRING_CHOICES)
