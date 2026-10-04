"""qwen4_exp: the n-gram hash multipliers the loaded architecture rebuilds
from config must equal the ones the checkpoint stores (2026-10-03: a seed
default of 0 vs the reference's 1234 sent every n-gram to the wrong rows)."""
import json
import pathlib

import pytest

pytest.importorskip("knurlogic")

FLASH_NEXT = pathlib.Path("/Volumes/Storage SSD/Exo Models/Qwen--Qwen3.8-Flash-Next-3bit")


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
    bad = Q.ple_multiplier_check(d)
    assert bad and bad[0][0] == 1 and bad[0][1] == [1, 3, 5]
    good = _fixture(tmp_path, Q.rebuilt(d)[1], seed=1234)
    assert Q.ple_multiplier_check(good) == []


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
