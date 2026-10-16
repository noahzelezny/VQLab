"""fit/fit_ple.py --rows: a validation fit keeps every table at full shape,
so a preflight's splice, pack and load run as the full build's do."""
import json

import pytest

mx = pytest.importorskip("mlx.core")
from vqlab.fit import fit_ple  # noqa: E402

KEY = "language_model.model.embed_tokens_per_layer.weight"


def test_rows_fits_the_first_rows_and_zeroes_the_rest_at_full_shape(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    mx.random.seed(0)
    mx.save_safetensors(str(src / "model.safetensors"),
                        {KEY: mx.random.normal((8, 64)).astype(mx.bfloat16)})
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {KEY: "model.safetensors"}}))
    assert fit_ple.main(["--src", str(src), "--out", str(out), "--key-regex", "embed_tokens",
                         "--k", "16", "--dim", "2", "--group", "32", "--iters", "2",
                         "--sample", "64", "--rows-chunk", "2", "--rows", "4"]) == 0
    man = json.loads((out / "ple_manifest.json").read_text())
    meta = man["tensors"][KEY]
    assert meta["shape"] == [8, 64] and meta["rows_fitted"] == 4
    t = mx.load(str(out / meta["shard"]))
    codes, scales = t[f"{KEY}.codes"], t[f"{KEY}.vq_scales"]
    assert codes.shape == (8, 32) and scales.shape == (8, 2)
    assert bool(mx.all(scales[4:] == 0)) and bool(mx.all(scales[:4] > 0))
