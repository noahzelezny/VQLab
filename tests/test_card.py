"""vqlab card: every number equals its source; TODOs, a foreign fingerprint
and mixed builds refuse. CPU only, a tiny fixture artifact, no mlx."""
import json
import struct

import pytest

from vqlab.core.artifact import sizes
from vqlab.records import card as C
from vqlab.records import card_tables as CT
from vqlab.records import provenance

GIB = 2 ** 30
CORP = ("prose", "code", "lit")


def _st(path, tensors):
    hdr, off = {}, 0
    for name, n in tensors.items():
        hdr[name] = {"dtype": "U8", "shape": [n], "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"\0" * off)


def _artifact(tmp_path):
    a = tmp_path / "TheDrainFlorist--Tiny-VQ-3.0bpw"
    a.mkdir()
    text = {"model.embed_tokens.weight": 4000, "model.layers.0.mlp.switch_mlp.up_proj.codes": 3000,
            "model.layers.1.mlp.switch_mlp.up_proj.codes": 3000}
    _st(a / "model-00001-of-00001.safetensors", text)
    _st(a / "model-vision-graft.safetensors", {"vision_tower.patch.weight": 900})
    _st(a / "mtp-head-q6.safetensors", {"mtp.0.proj.weight": 500})
    (a / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 10000},
         "weight_map": {**{k: "model-00001-of-00001.safetensors" for k in text},
                        "vision_tower.patch.weight": "model-vision-graft.safetensors"}}))
    (a / "config.json").write_text(json.dumps({
        "model_type": "tiny_moe", "num_hidden_layers": 2,
        "quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
        "vq_modules": {
            "model.layers.0.mlp.switch_mlp.up_proj":
                {"experts": 8, "out": 16, "in": 16, "k": 2048, "dim": 4, "pack_bits": 11},
            "model.layers.1.mlp.switch_mlp.up_proj":
                {"experts": 8, "out": 16, "in": 16, "k": 4096, "dim": 4, "pack_bits": 12}}}))
    (a / "LICENSE").write_text("MIT License\n\nCopyright ...")
    return a


def _kl(tmp_path, art, fp=None, numerics=("x", "x")):
    def per(vals, num):
        return {c: {"mean_kl_millinats": v, "kl_positions": 12288, "scorer_variant": "v",
                    "numerics": {"mlx": num, "mlx_lm": "0.1", "scorer_variant": "v"},
                    "measured": {"fingerprint": fp} if fp else {}}
                for c, v in zip(CORP, vals)}
    j = tmp_path / "kl.json"
    j.write_text(json.dumps({
        "table": {"ref": per((90.0, 30.0, 20.0), numerics[0]),
                  "cand": per((60.0, 18.0, 6.0), numerics[1])},
        "rungs": {"ref": "/nope", "cand": str(art)}, "reference": "ref",
        "measured": {"cand": {"fingerprint": fp}} if fp else {},
        "changed_during_run": [],
        "paired": {"cand": {c: {"delta": -12.34, "t": -4.5} for c in CORP}}}))
    return j


def _extra(tmp_path, body):
    p = tmp_path / "extra.md"
    p.write_text(body)
    return p


FULL = "A tiny build.\n\n## Changelog\n\n2026-10-03: first.\n\n## Limitations\n\n- Small.\n"


def test_numbers_come_from_sources(tmp_path, capsys):
    art = _artifact(tmp_path)
    j = _kl(tmp_path, art, fp=provenance.shard_fingerprint(art))
    rc = C.main(["--artifact", str(art), "--kl", str(j), "--this", "cand",
                 "--base-model", "org/Tiny", "--extra-md", str(_extra(tmp_path, FULL))])
    out = capsys.readouterr().out
    assert rc == 0
    s = sizes(art)
    assert f"**{s['text'] / GIB:.1f} GiB text weights.**" in out
    assert s["tower"] and f"Vision tower +{s['tower'] / GIB:.2f} GiB" in out
    assert f"MTP draft head +{(s['mtp'] + s['mtp_sidecar']) / GIB:.2f} GiB" in out
    assert f"full download {s['download'] / GIB:.1f} GiB" in out
    assert CT.render([str(j)], {}, "cand") in out           # the card-tables output, verbatim
    assert "-12.3 (t -4.5)" in out
    assert "| d4/K2048 (11-bit codes, 2.75 code bits per weight) | 0 | 1 | 8 |" in out
    assert "| d4/K4096 (12-bit codes, 3 code bits per weight) | 1 | 1 | 8 |" in out
    assert "8-bit affine (group 64)" in out
    assert "license: mit" in out and "base_model: org/Tiny" in out
    assert "hf download TheDrainFlorist/Tiny-VQ-3.0bpw" in out
    assert "TODO" not in out
    assert "## Support this work" in out


def test_todo_refuses_unless_draft(tmp_path, capsys):
    art = _artifact(tmp_path)
    j = _kl(tmp_path, art)
    args = ["--artifact", str(art), "--kl", str(j), "--this", "cand", "--base-model", "org/Tiny"]
    assert C.main(args) == 1
    assert "TODO(card)" in capsys.readouterr().out
    assert C.main(args + ["--draft"]) == 0


def test_spelling_fails(tmp_path, capsys):
    art = _artifact(tmp_path)
    j = _kl(tmp_path, art)
    rc = C.main(["--artifact", str(art), "--kl", str(j), "--this", "cand", "--base-model", "o/T",
                 "--extra-md", str(_extra(tmp_path, FULL.replace("A tiny", "An optimised")))])
    assert rc == 1
    assert "optimised -> optimized" in capsys.readouterr().err


def test_fingerprint_mismatch_refuses(tmp_path):
    art = _artifact(tmp_path)
    j = _kl(tmp_path, art, fp="0000000000000000")
    with pytest.raises(SystemExit) as e:
        C.main(["--artifact", str(art), "--kl", str(j), "--this", "cand", "--draft"])
    assert e.value.code == 2


def test_fingerprint_before_text_free_sidecar_accepted(tmp_path, capsys):
    art = _artifact(tmp_path)
    side = art / "mtp-head-q6.safetensors"
    side.rename(tmp_path / "held")
    fp = provenance.shard_fingerprint(art)               # scored before the MTP head arrived
    (tmp_path / "held").rename(side)
    j = _kl(tmp_path, art, fp=fp)
    assert C.main(["--artifact", str(art), "--kl", str(j), "--this", "cand", "--draft"]) == 0
    assert "`mtp-head-q6.safetensors`" in capsys.readouterr().out


def test_mixed_builds_refuse(tmp_path):
    art = _artifact(tmp_path)
    j = _kl(tmp_path, art, numerics=("0.31", "0.32"))
    with pytest.raises(SystemExit) as e:
        C.main(["--artifact", str(art), "--kl", str(j), "--this", "cand", "--draft"])
    assert e.value.code == 2
