"""build/build_dense.py: no affine quantization entry survives for a module
the build replaced with VQ."""
import json

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


# ---------------------------------------------------------------- --embed
EMB = "language_model.model.embed_tokens_per_layer"


def _st(d, name, tensors):
    import mlx.core as mx
    mx.save_safetensors(str(d / name), tensors)
    return dict.fromkeys(tensors, name)


def _base(tmp_path):
    """A tiny e4b-shaped 8-bit base: the table, one carried module, and a
    KV-shared layer (1 of 2) still carrying its dead k_proj."""
    import mlx.core as mx
    d = tmp_path / "base"
    d.mkdir()
    ones = mx.ones((4, 2), dtype=mx.uint32)
    wm = _st(d, "model-00001.safetensors", {
        f"{EMB}.weight": ones, f"{EMB}.scales": ones, f"{EMB}.biases": ones,
        "language_model.model.layers.0.self_attn.k_proj.weight": ones,
        "language_model.model.layers.1.self_attn.k_proj.weight": ones,
        "language_model.model.layers.1.self_attn.q_proj.weight": ones})
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    (d / "config.json").write_text(json.dumps({
        "model_type": "gemma4", "quantization": {"bits": 8, "group_size": 64},
        "text_config": {"num_hidden_layers": 2, "num_kv_shared_layers": 1}}))
    return d


def _fit(tmp_path, key=EMB + ".weight", codes=None):
    """A fit-ple output dir for one [4, 128] table at d2/K2048, group 64."""
    import mlx.core as mx
    d = tmp_path / "fit"
    d.mkdir()
    codes = mx.ones((4, 64), dtype=mx.uint16) if codes is None else codes
    _st(d, "ple-0000.safetensors", {f"{key}.codes": codes,
                                    f"{key}.codebook": mx.ones((2048, 2), dtype=mx.float16),
                                    f"{key}.vq_scales": mx.ones((4, 2), dtype=mx.float16)})
    (d / "ple_manifest.json").write_text(json.dumps({
        "geometry": {"k": 2048, "dim": 2, "group": 64},
        "tensors": {key: {"shape": [4, 128], "relerr": 0.03, "shard": "ple-0000.safetensors"}}}))
    return d


def _build(base, fit, out, *extra):
    from vqlab.build import build_dense
    return build_dense.main(["--family", "gemma4_e4b", "--base", str(base),
                             "--embed", str(fit), "--out", str(out), *extra])


def test_embed_splices_the_table_as_vq_embed(tmp_path):
    pytest.importorskip("mlx.core")
    from vqlab.core.artifacts.artifact import Artifact
    out = tmp_path / "out"
    assert _build(_base(tmp_path), _fit(tmp_path), out) == 0
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["vq_embed"] == {EMB: {"rows": 4, "in": 128, "k": 2048, "dim": 2, "group": 64}}
    assert cfg["vq_linear"] == {}
    idx = Artifact(out).index
    assert {f"{EMB}.{leaf}" for leaf in ("codes", "codebook", "vq_scales")} <= set(idx)
    assert not any(k.startswith(f"{EMB}.") and k.rsplit(".", 1)[1] in
                   ("weight", "scales", "biases") for k in idx)
    # the KV-shared layer's k_proj is dropped, the non-shared layer's kept
    assert "language_model.model.layers.1.self_attn.k_proj.weight" not in idx
    assert "language_model.model.layers.0.self_attn.k_proj.weight" in idx
    assert "language_model.model.layers.1.self_attn.q_proj.weight" in idx


def test_embed_dry_run_writes_nothing(tmp_path):
    pytest.importorskip("mlx.core")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as e:
        _build(_base(tmp_path), _fit(tmp_path), out, "--dry-run")
    assert e.value.code == 0 and not out.exists()


def test_a_fitted_table_the_base_lacks_is_refused(tmp_path):
    pytest.importorskip("mlx.core")
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="name no tensor in the base"):
        _build(_base(tmp_path), _fit(tmp_path, key="language_model.model.nope.weight"), out)
    assert not out.exists()


def test_an_all_zero_fit_is_refused_before_writing(tmp_path):
    mx = pytest.importorskip("mlx.core")
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="ALL ZERO"):
        _build(_base(tmp_path), _fit(tmp_path, codes=mx.zeros((4, 64), dtype=mx.uint16)), out)
    assert not list(out.glob("*.safetensors"))


def test_mlp_or_embed_is_required(tmp_path):
    from vqlab.build import build_dense
    with pytest.raises(SystemExit):
        build_dense.main(["--family", "gemma4_e4b", "--base", str(tmp_path), "--out",
                          str(tmp_path / "o")])


def test_mlp_remap_splices_under_base_names(tmp_path):
    """qwen3_8_dense --mlp, unchanged by the --embed split: fitter names
    remapped to the base's, codes squeezed to 2D and narrowed to uint8 at
    K<=256, the affine tensors dropped, no vq_embed."""
    mx = pytest.importorskip("mlx.core")
    from vqlab.build import build_dense
    from vqlab.core.artifacts.artifact import Artifact
    base, fit, out = tmp_path / "base", tmp_path / "fit", tmp_path / "out"
    base.mkdir(), fit.mkdir()
    ones = mx.ones((8, 4), dtype=mx.uint32)
    projs = ("gate_proj", "up_proj")
    wm = _st(base, "model-00001.safetensors", {
        **{f"language_model.model.layers.0.mlp.{p}.{leaf}": ones
           for p in projs for leaf in ("weight", "scales", "biases")},
        "language_model.model.norm.weight": ones})
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_5",
                                                  "quantization": {"bits": 4}}))
    meta = {"k": 256, "dim": 4, "group": 64, "in": 128, "out": 8}
    mods = [f"model.language_model.layers.0.mlp.{p}" for p in projs]
    (fit / "config.json").write_text(json.dumps({"vq_modules": dict.fromkeys(mods, meta)}))
    _st(fit, "model-00001-of-00001.safetensors", {
        f"{m}.{leaf}": t for m in mods for leaf, t in (
            ("codes", mx.ones((1, 8, 32), dtype=mx.uint16)),
            ("codebook", mx.ones((256, 4), dtype=mx.float16)),
            ("vq_scales", mx.ones((1, 8, 2), dtype=mx.float16)))})
    assert build_dense.main(["--family", "qwen3_8_dense", "--base", str(base),
                             "--mlp", str(fit), "--out", str(out)]) == 0
    cfg = json.loads((out / "config.json").read_text())
    want = [f"language_model.model.layers.0.mlp.{p}" for p in projs]
    assert sorted(cfg["vq_linear"]) == want and cfg["vq_embed"] == {}
    idx = Artifact(out).index
    assert not any(k.endswith((".weight", ".scales", ".biases")) and ".mlp." in k for k in idx)
    codes = mx.load(str(out / idx[f"{want[0]}.codes"]))[f"{want[0]}.codes"]
    assert codes.shape == (8, 32) and codes.dtype == mx.uint8
