"""`vqlab scratch reclaimable`: lists an unpacked dir only when its packed
counterpart's build record names it AND both sides still verify; prints the
rm lines; never deletes. Each refusal is gated in both directions on the same
fixture (a gate must fail on a known-bad input before its pass means
anything)."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

from vqlab.records import provenance
from vqlab.records import scratch_cmd

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _art(d: pathlib.Path, shard_bytes: bytes):
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model-00001-of-00001.safetensors").write_bytes(shard_bytes)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))


def _fixture(root: pathlib.Path):
    fit, packed, other = root / "fit-k2048", root / "fit-k2048-packed", root / "fit-k256"
    _art(fit, b"u" * 4000)
    provenance.write_build_record(fit, tool="fit-moe")
    _art(packed, b"p" * 2750)
    provenance.write_build_record(packed, tool="pack", inputs=[("src", fit)])
    _art(other, b"x" * 100)                     # unpacked, never packed: not listed
    provenance.write_build_record(other, tool="fit-moe")
    return fit, packed, other


def test_lists_verified_pairs_and_prints_rm(tmp_path, capsys):
    fit, packed, other = _fixture(tmp_path)
    cands, kept = scratch_cmd.reclaimable([tmp_path])
    assert [c["path"] for c in cands] == [str(fit)] and not kept
    assert cands[0]["packed"] == str(packed)
    assert cands[0]["bytes"] == provenance_bytes(fit)
    capsys.readouterr()                                      # drop the fixture's PROVENANCE lines
    scratch_cmd.main(["reclaimable", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert f"rm -rf {fit}" in out and "Nothing was deleted" in out
    assert str(other) not in out
    assert fit.is_dir() and other.is_dir()                   # nothing deleted
    cands_full, _ = scratch_cmd.reclaimable([tmp_path], full=True)
    assert [c["path"] for c in cands_full] == [str(fit)]


def provenance_bytes(d):
    return sum(f.stat().st_size for f in d.iterdir() if f.is_file() and not f.is_symlink())


def test_refuses_when_packed_does_not_verify(tmp_path):
    fit, packed, _ = _fixture(tmp_path)
    (packed / "model-00001-of-00001.safetensors").write_bytes(b"p" * 10)   # truncated pack
    cands, kept = scratch_cmd.reclaimable([tmp_path])
    assert not cands and "does not verify" in kept[0]["reason"]


def test_refuses_when_source_changed(tmp_path):
    fit, packed, _ = _fixture(tmp_path)
    (fit / "model-00001-of-00001.safetensors").write_bytes(b"v" * 4000)    # same size, new bytes
    cands, kept = scratch_cmd.reclaimable([tmp_path])
    assert not cands and "changed" in kept[0]["reason"]


def test_refuses_when_something_links_into_source(tmp_path):
    fit, packed, _ = _fixture(tmp_path)
    pin = tmp_path / "pin"
    pin.mkdir()
    os.symlink(fit / "model-00001-of-00001.safetensors", pin / "model-00001-of-00001.safetensors")
    cands, kept = scratch_cmd.reclaimable([tmp_path])
    assert not cands and "symlink" in kept[0]["reason"]


def test_cli_routes_and_has_no_delete(tmp_path):
    _fixture(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(SRC), "VQLAB_SCRATCH": str(tmp_path),
           "VQLAB_CONFIG": str(tmp_path / "none.toml"), "VQLAB_LOG_DIR": str(tmp_path / "log")}
    p = subprocess.run([sys.executable, "-m", "vqlab.cli", "scratch", "reclaimable"],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    assert "rm -rf" in p.stdout and (tmp_path / "fit-k2048").is_dir()
    p = subprocess.run([sys.executable, "-m", "vqlab.cli", "scratch", "reclaimable", "--delete"],
                       capture_output=True, text=True, env=env)
    assert p.returncode != 0
