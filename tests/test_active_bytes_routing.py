"""active-bytes reads the routed-expert count under every family's spelling."""

from vqlab.bench.active_bytes import _component, _routing


def test_glm_n_routed_experts():
    assert _routing({"n_routed_experts": 288, "num_experts_per_tok": 8}) == (288, 8)


def test_glm_nested_text_config():
    cfg = {"text_config": {"n_routed_experts": 288, "num_experts_per_tok": 8}}
    assert _routing(cfg) == (288, 8)


def test_qwen_and_mixtral_spellings():
    assert _routing({"num_experts": 512, "num_experts_per_tok": 10}) == (512, 10)
    assert _routing({"num_local_experts": 8, "num_experts_per_tok": 2}) == (8, 2)


def test_dense_config_has_no_routing():
    assert _routing({"hidden_size": 4096}) == (None, None)


def test_glm_shared_experts_billed_dense():
    assert _component("model.layers.3.mlp.shared_experts.up_proj.weight")[1] == "dense"
    assert _component("model.layers.3.mlp.switch_mlp.up_proj.weight")[1] == "routed"
