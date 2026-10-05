"""qwen4_exp: the n-gram hash multipliers the loaded architecture rebuilds
from config must equal the ones the checkpoint stores (2026-10-03: a seed
default of 0 vs the reference's 1234 sent every n-gram to the wrong rows)."""
import json
import pathlib
from vqlab import config as _cfg

import pytest

pytest.importorskip("knurlogic")

FLASH_NEXT = _cfg.models() / "Qwen--Qwen3.8-Flash-Next-3bit"


def _fixture(tmp_path, stored, seed=None):
    import mlx.core as mx
    cfg = {"model_type": "qwen4_exp", "vocab_size": 248320, "ngram_size": 3,
           "num_hidden_layers": 2, "ple_layer_ids": [2]}
    if seed is not None:
        cfg["seed"] = seed
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    k = "model.layers.1.ple.ple_embedding.layer_multipliers"
    with mx.stream(mx.cpu):
        mx.save_safetensors(str(tmp_path / "m.safetensors"), {k: mx.array(stored, dtype=mx.int64)})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "m.safetensors"}}))
    return tmp_path


def test_check_catches_a_wrong_seed(tmp_path):
    from vqlab.family import qwen4_exp as Q
    d = _fixture(tmp_path, [1, 3, 5], seed=1234)
    bad = Q.ple_multiplier_check(d, runtime=False)
    assert bad and bad[0][0] == 1 and bad[0][1] == [1, 3, 5]
    good = _fixture(tmp_path, Q.rebuilt(d)[1], seed=1234)
    assert Q.ple_multiplier_check(good, runtime=False) == []


@pytest.mark.lab
def test_ple_multipliers_match_stored():
    """The parity item: on Qwen's own Flash-Next checkpoint (no `seed` in its
    config), the loaded architecture's rebuild matches what Qwen stored."""
    if not FLASH_NEXT.exists():
        pytest.skip("Flash-Next maker checkpoint not on this machine")
    from vqlab.family import qwen4_exp as Q
    assert Q.ple_multiplier_check(FLASH_NEXT) == [], (
        "rebuilt PLE hash multipliers differ from the checkpoint's stored ones: "
        "the architecture's `seed` default is wrong (reference: 1234)")


@pytest.mark.lab
def test_runtime_adopts_stored_multipliers_over_a_wrong_seed(tmp_path):
    """Edit 4 is what makes this pass: with the config's seed forced WRONG (0),
    the formula rebuild differs from the checkpoint, yet the loaded runtime
    still hashes with the stored multipliers. Proves the check exercises the
    adopt-from-checkpoint path, not a lucky default."""
    if not FLASH_NEXT.exists():
        pytest.skip("Flash-Next maker checkpoint not on this machine")
    from vqlab.family import qwen4_exp as Q
    cfg = json.load(open(FLASH_NEXT / "config.json"))
    (cfg.get("text_config") or cfg)["seed"] = 0
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    idx = json.load(open(FLASH_NEXT / "model.safetensors.index.json"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(idx))
    for f in {v for k, v in idx["weight_map"].items() if k.endswith("layer_multipliers")}:
        (tmp_path / f).symlink_to(FLASH_NEXT / f)
    assert Q.ple_multiplier_check(tmp_path, runtime=False) != []   # the seed-0 formula is wrong
    assert Q.ple_multiplier_check(tmp_path) == []                   # the runtime is not fooled
