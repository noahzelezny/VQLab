"""plan/family_profile.py: a profile's byte classes and layers, read from a
teacher's headers -- here a tiny GLM-shaped teacher whose draft head is the
trailing layer, and a Flash-Next-shaped one with n-gram tables."""
import json

import pytest

mx = pytest.importorskip("mlx.core")
from vqlab.plan import family_profile as fp  # noqa: E402


def _teacher(d, cfg, tensors):
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(str(d / "model-00001.safetensors"),
                        {k: mx.zeros(s, dtype=mx.bfloat16)
                         for k, s in tensors.items()})
    return d


def _profile(tmp_path, teacher, family):
    out = tmp_path / "out"
    args = ["--teacher", str(teacher), "--family", family, "--out", str(out)]
    assert fp.main(args) == 0
    return json.loads((out / "profile.json").read_text())


def test_a_trailing_head_layer_is_the_head_not_a_layer_to_fit(tmp_path):
    """GLM stores its draft head as layers.<num_hidden_layers>: that layer is
    the mtp class, recorded as arch.mtp_layer, and never a VQ layer."""
    lm, t = "model.language_model", {}
    projs = (("gate_proj", (8, 16)), ("up_proj", (8, 16)), ("down_proj", (16, 8)))
    for li in (0, 1, 2):                       # 2 == num_hidden_layers: the head
        for e in range(2):
            for k, s in projs:
                t[f"{lm}.layers.{li}.mlp.experts.{e}.{k}.weight"] = s
    t[f"{lm}.layers.2.eh_proj.weight"] = (16, 32)
    t[f"{lm}.embed_tokens.weight"] = (32, 16)
    cfg = {"model_type": "glm5_next", "num_hidden_layers": 2, "n_routed_experts": 2}
    teacher = _teacher(tmp_path / "glm", cfg, t)
    prof = _profile(tmp_path, teacher, "glm5_next")
    assert prof["arch"]["vq_layers"] == [0, 1] and prof["arch"]["mtp_layer"] == 2
    assert prof["arch"]["mtp_head"] is True
    head = (2 * 3 * 8 * 16 + 16 * 32) * 2      # its experts and eh_proj, bf16
    assert prof["bytes"]["by_class_gib"]["mtp"] == round(head / 2**30, 3)
    assert set(prof["bytes"]["vq_target_gib_per_layer"]) == {"0", "1"}


def test_ngram_tables_are_their_own_class():
    """A family with n-gram tables (registry "ple") counts them apart from carry."""
    import re
    ple = re.compile("ngram_embedding")
    lm = "model.language_model.layers.3"
    key = f"{lm}.ple.ple_embedding.ngram_embedding.shard_0.weight"
    assert fp.classify(key, None, ple) == "ple"
    assert fp.classify(key, None) == "carry"
    assert fp.classify(f"{lm}.self_attn.q_proj.weight", None, ple) == "carry"
