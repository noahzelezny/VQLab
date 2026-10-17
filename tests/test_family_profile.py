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



def test_ngram_tables_are_signed_so_their_fits_attribute(tmp_path):
    """Flash-Next's n-gram tables are fitted on their own: the profile signs
    each table under the name the fit store files it with (shard_N), so a
    filed table fit is attributed; the expert geometries are unchanged."""
    lm, t = "model.language_model", {}
    for li in (0, 1):
        t[f"{lm}.layers.{li}.mlp.experts.gate_up_proj"] = (2, 8, 16)
        t[f"{lm}.layers.{li}.mlp.experts.down_proj"] = (2, 16, 4)
        t[f"{lm}.layers.{li}.ple.ple_embedding.ngram_embedding.shard_0.weight"] = (10, 4)
        t[f"{lm}.layers.{li}.ple.ple_embedding.ngram_embedding.shard_1.weight"] = (12, 4)
    cfg = {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 2, "num_experts": 2}}
    prof = _profile(tmp_path, _teacher(tmp_path / "fn", cfg, t), "qwen4_exp")
    sig = prof["modules"]["signatures"]
    assert sig["shard_0"] == [[1, 10, 4]] and sig["shard_1"] == [[1, 12, 4]]
    assert not {"shard_0", "shard_1"} & set(prof["geometries"])

def test_a_one_table_family_profiles_the_table_as_its_vq_target(tmp_path):
    """gemma4_e4b (registry "embed"): the per-layer table is the one VQ
    module, so the geometries are the table's and its bytes are vq_target."""
    lm = "language_model.model"
    t = {f"{lm}.embed_tokens_per_layer.weight": (64, 256),
         f"{lm}.embed_tokens.weight": (64, 32),
         f"{lm}.layers.0.mlp.gate_proj.weight": (64, 32)}
    cfg = {"model_type": "gemma4", "text_config": {"num_hidden_layers": 1}}
    prof = _profile(tmp_path, _teacher(tmp_path / "e4b", cfg, t), "gemma4_e4b")
    assert prof["arch"]["kind"] == "dense" and prof["arch"]["vq_layers"] == []
    assert prof["arch"]["vq_table"] == {"key": f"{lm}.embed_tokens_per_layer",
                                        "rows": 64, "in": 256}
    assert prof["modules"]["signatures"] == {"embed": [[1, 64, 256]]}
    assert prof["bytes"]["vq_target_params"] == 64 * 256
    assert prof["bytes"]["by_class_gib"]["vq_target"] == round(64 * 256 * 2 / 2**30, 3)
    legal = {(g["d"], g["K"]) for g in prof["geometries"]["embed"] if g["legal"]}
    assert (2, 2048) in legal and (2, 4096) in legal and (4, 4096) not in legal


def test_an_audio_tower_is_its_own_class():
    assert fp.classify("audio_tower.layers.0.w", None) == "audio"
    assert fp.classify("embed_audio.embedding_projection.weight", None) == "audio"
