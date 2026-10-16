"""stream-convert's per-module quantization: defaults unchanged, --group and
default-mode --protect-bits (mixed affine) reach the predicate."""
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
from vqlab.build.stream_convert import body_spec  # noqa: E402
from vqlab.core.artifacts.artifact import copy_tokenizer_files  # noqa: E402

T = "switch_mlp"
EXP = "model.layers.3.mlp.switch_mlp.gate_proj"
ATT = "model.layers.3.self_attn.q_proj"
SHE = "model.layers.3.mlp.shared_expert.up_proj"


def test_defaults_are_the_old_uniform_g64():
    for p in (EXP, ATT, SHE, "model.embed_tokens", "lm_head"):
        assert body_spec(p, target=T, bits=3) == {"group_size": 64, "bits": 3}
    assert body_spec("model.layers.3.mlp.gate", target=T, bits=3) is False
    assert body_spec("model.ngram_embedding", target=T, bits=3) is True


def test_group_reaches_the_body():
    assert body_spec(ATT, target=T, bits=4, group=128) == {"group_size": 128, "bits": 4}
    assert body_spec(EXP, target=T, bits=4, group=32) == {"group_size": 32, "bits": 4}


def test_protect_in_default_mode_is_mixed_affine():
    kw = dict(target=T, bits=2, group=128, protect_bits=8)
    assert body_spec(EXP, **kw) == {"group_size": 128, "bits": 2}
    for p in (ATT, SHE, "model.embed_tokens", "lm_head"):
        assert body_spec(p, **kw) == {"group_size": 64, "bits": 8}
    assert body_spec("model.layers.3.mlp.gate", **kw) is False


def test_struct_unchanged():
    kw = dict(target=T, bits=4, group=128, protect_bits=8, struct=True)
    assert body_spec(EXP, **kw) == {"group_size": 64, "bits": 2}
    assert body_spec(ATT, **kw) == {"group_size": 64, "bits": 8}


def test_slabbed_quantize_is_exactly_the_whole_stack(monkeypatch):
    """An expert stack quantized a slab of experts at a time (the 397B's 4 GiB
    stacks ran past the Metal watchdog in one kernel) is the same bytes."""
    import mlx.core as mx
    from vqlab.build import stream_convert as sc
    monkeypatch.setattr(sc, "SLAB_BYTES", 3 * 64 * 128 * 2)       # 3 experts per slab
    w = mx.random.normal((8, 64, 128), key=mx.random.key(0)).astype(mx.bfloat16)
    want = mx.quantize(w, group_size=64, bits=2)
    slabs = sc._SlabQuantizer()
    got = slabs.wrap(mx.quantize)(w, group_size=64, bits=2)
    src, parts = slabs.pop(got[1])
    assert src is w and len(parts) == 3
    assert all(slabs.pop(g) is None for g in got)      # one pop releases all three
    for g, x in zip(got, want):
        assert mx.array_equal(g, x).item()
    assert slabs.pop(mx.quantize(w[:1], group_size=64, bits=2)[0]) is None


def test_teacher_tokenizer_files_are_copied_verbatim(tmp_path):
    """A re-serialized tokenizer.json is not the teacher's: transformers 5
    flipped the Qwen3.5 ByteLevel pre_tokenizer/decoder flags on save, and
    check-release failed the artifact against its own teacher."""
    src, dst = tmp_path / "teacher", tmp_path / "out"
    src.mkdir(), dst.mkdir()
    teacher = b'{"pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false}}'
    (src / "tokenizer.json").write_bytes(teacher)
    (src / "merges.txt").write_bytes(b"a b\n")
    (src / "config.json").write_text("{}")                   # not a tokenizer file
    (dst / "tokenizer.json").write_bytes(b'{"pre_tokenizer": {"add_prefix_space": true}}')
    (dst / "tokenizer_config.json").write_text("{}")        # teacher has none: kept
    assert copy_tokenizer_files(src, dst) == ["tokenizer.json", "merges.txt"]
    assert (dst / "tokenizer.json").read_bytes() == teacher
    assert (dst / "tokenizer_config.json").read_text() == "{}"
    assert not (dst / "config.json").exists()


def test_family_keeps_its_gates_full_precision():
    """qwen3_5 keeps the shared-expert gate bf16 like every published rung;
    a struct base once quantized it to 8-bit and check-release refused the
    draft head beside it."""
    from vqlab.family import keep_fp
    gate = "model.layers.3.mlp.shared_expert_gate"
    kw = {"target": T, "bits": 2, "struct": True, "protect_bits": 8}
    assert body_spec(gate, keep=keep_fp("qwen3_5"), **kw) is False
    assert body_spec(gate, keep=keep_fp("qwen4_exp"), **kw) == {"group_size": 64,
                                                                 "bits": 8}
    assert keep_fp("no-such-family") == ("mlp.gate",)
